import os
import time
import logging
import shutil
import requests
import tempfile
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from celery import Celery, states
from celery.exceptions import MaxRetriesPerTaskError

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# KONFIGURACJA I INICJALIZACJA
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Konfiguracja Celery (Redis jako Broker i Backend)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery_app = Celery(app.name, broker=REDIS_URL, backend=REDIS_URL)

MODEL = "veo-3.1-lite-generate-preview"
STORAGE_DIR = os.getenv('STORAGE_DIR', '/app/data')
os.makedirs(STORAGE_DIR, exist_ok=True)

def get_gemini_client():
    return genai.Client(
        http_options={"api_version": "v1beta"}, 
        api_key=os.getenv("GEMINI_API_KEY")
    )

# ---------------------------------------------------------------------------
# ZADANIA CELERY (PRACA W TLE)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=5)
def download_video_with_backoff(self, video_uri, temp_path):
    """Pobieranie wideo z chmury z automatycznym retry/backoff w przypadku błędów sieciowych"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        headers = {"x-goog-api-key": api_key} if api_key else {}
        
        response = requests.get(video_uri, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as exc:
        # Wykładnicze cofanie: 2s, 4s, 8s, 16s...
        countdown = 2 ** self.request.retries
        logger.warning(f"Błąd pobierania. Ponowna próba za {countdown}s...")
        raise self.retry(exc=exc, countdown=countdown)


def generate_video_segment(client, prompt, aspect_ratio="9:16"):
    """Generuje pojedynczy klip wideo i zwraca lokalną ścieżkę tymczasową"""
    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=5, # Krótsze segmenty dynamiczne pod rolki
            resolution="1080p",
        ),
    )
    
    # Polling statusu zadania w API Google
    attempt = 0
    while not operation.done and attempt < 60:
        time.sleep(10)
        operation = client.operations.get(operation)
        attempt += 1
        
    if not operation.done:
        raise TimeoutError("Veo API timeout podczas generowania segmentu.")
        
    result = operation.result
    video_uri = result.generated_videos[0].video.uri
    
    temp_file = os.path.join(tempfile.gettempdir(), f"seg_{os.urandom(4).hex()}.mp4")
    # Wywołanie pobierania (synchronicznie wewnątrz wątku robotnika)
    download_video_with_backoff.run(video_uri, temp_file)
    return temp_file


@celery_app.task(bind=True)
def render_sequence_task(self, raw_data, webhook_url=None):
    """Główny proces montażu sekwencji w tle"""
    client = get_gemini_client()
    job_id = self.request.id
    
    topic = raw_data.get("topic", "Finanse osobiste")
    aspect_ratio = raw_data.get("aspectRatio", "9:16")
    
    # 1. SZABLON MARKETINGOWY: Hook -> Problem -> Rozwiązanie -> CTA
    prompts = {
        "hook": f"Dynamic cinematic shot, extreme close up, shock and stress, concept of {topic}, corporate finance style, 4k",
        "problem": f"A person looking anxiously at bills and charts on a screen, dark moody lighting, financial stress, 4k",
        "rozwiązanie": f"Bright clean studio lighting, a smartphone screen displaying green rising financial growth charts, relief, 4k",
        "cta": f"Clean minimalist background, text placeholder, elegant financial advisor theme, 4k. Professional look."
    }
    
    # Automatyczne wstrzyknięcie domeny do części CTA
    cta_text = f"Wejdź na raport-finansowy24.pl i odbierz swój darmowy audyt finansowy!"
    
    logger.info(f"Rozpoczynanie generowania sekwencji dla Job ID: {job_id}")
    segment_files = []
    
    try:
        # 2. GENEROWANIE POSZCZEGÓLNYCH KLIPÓW
        for key, prompt_text in prompts.items():
            self.update_state(state='PROGRESS', meta={'status': f'Generowanie segmentu: {key}'})
            file_path = generate_video_segment(client, prompt_text, aspect_ratio)
            segment_files.append(file_path)
            
        # 3. GENEROWANIE AUDIO I NAPISÓW (FFMPEG MONTAGE)
        # Przygotowujemy plik listy dla demuxera FFmpeg
        list_file_path = os.path.join(tempfile.gettempdir(), f"list_{job_id}.txt")
        with open(list_file_path, "w") as f:
            for file_path in segment_files:
                f.write(f"file '{file_path}'\n")
                
        final_filename = f"render_{job_id}.mp4"
        final_output_path = os.path.join(STORAGE_DIR, final_filename)
        
        # Składanie klipów, nakładanie dynamicznych napisów z CTA za pomocą FFmpeg
        # Generujemy filtr rysowania tekstu (drawtext) dla sekcji CTA na końcu filmu (od 15 do 20 sekundy)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path,
            '-vf', f"drawtext=text='{cta_text}':x=(w-text_w)/2:y=h-200:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:enable='between(t,15,20)'",
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', final_output_path
        ]
        
        self.update_state(state='PROGRESS', meta={'status': 'Montaż końcowy wideo (FFmpeg)...'})
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        video_url = f"https://{raw_data.get('host')}/videos/{final_filename}"
        logger.info(f"Render zakończony sukcesem! Plik: {final_output_path}")
        
        # 4. WEBHOOK PO ZAKOŃCZENIU
        if webhook_url:
            self.update_state(state='PROGRESS', meta={'status': 'Wysyłanie powiadomienia webhook...'})
            try:
                requests.post(webhook_url, json={"job_id": job_id, "status": "success", "video_url": video_url}, timeout=10)
            except requests.RequestException as e:
                logger.error(f"Nie udało się dostarczyć Webhooka: {e}")
                
        return {"status": "success", "video_url": video_url}
        
    except Exception as e:
        logger.error(f"Błąd krytyczny w zadaniu {job_id}: {e}", exc_info=True)
        if webhook_url:
            try:
                requests.post(webhook_url, json={"job_id": job_id, "status": "failed", "error": str(e)}, timeout=10)
            except Exception: pass
        raise e
        
    finally:
        # CLEANUP STORAGE (Czyszczenie plików tymczasowych segmentów)
        for path in segment_files:
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
        if os.path.exists(list_file_path):
            try: os.remove(list_file_path)
            except Exception: pass

# ---------------------------------------------------------------------------
# ENDPOINTY FLASK (API)
# ---------------------------------------------------------------------------

@app.route("/render-sequence", methods=["POST"])
def start_render_sequence():
    """Przyjmuje zlecenie, wrzuca je do kolejki i natychmiast zwraca ID zadania"""
    data = request.json or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Missing or empty 'topic'"}), 400
        
    webhook_url = data.get("webhookUrl")
    
    # Przekazujemy hosta, aby pracownik tła wiedział, jak zbudować finalny URL
    data['host'] = request.host
    
    # Wywołanie asynchroniczne Celery (.delay())
    task = render_sequence_task.delay(data, webhook_url=webhook_url)
    
    return jsonify({
        "status": "queued",
        "job_id": task.id,
        "status_url": f"https://{request.host}/tasks/{task.id}"
    }), 202


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Sprawdzanie aktualnego stanu renderowania przez klienta"""
    task = render_sequence_task.AsyncResult(task_id)
    response = {"job_id": task_id, "state": task.state}
    
    if task.state == states.PENDING:
        response["status"] = "Oczekiwanie w kolejce..."
    elif task.state == 'PROGRESS':
        response["status"] = task.info.get('status', 'Przetwarzanie...')
    elif task.state == states.SUCCESS:
        response["status"] = "Zakończono sukcesem"
        response["result"] = task.result
    elif task.state == states.FAILURE:
        response["status"] = "Błąd wykonania"
        response["error"] = str(task.info)
        
    return jsonify(response)


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Dystrybucja gotowych filmów wideo"""
    return send_from_directory(STORAGE_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
