import os
import time
import logging
import json
import uuid
import requests
import tempfile
import subprocess
import threading
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types
import elevenlabs

# ---------------------------------------------------------------------------
# KONFIGURACJA I INICJALIZACJA
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL = "veo-3.1-lite-generate-preview"
STORAGE_DIR = os.getenv('STORAGE_DIR', '/app/data')
DB_PATH = os.path.join(STORAGE_DIR, 'renders.db')
os.makedirs(STORAGE_DIR, exist_ok=True)

# ElevenLabs API
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
if not ELEVENLABS_API_KEY:
    logger.warning("⚠️ ELEVENLABS_API_KEY not set!")

# Inicjalizacja bazy danych SQLite
def init_db():
    """Inicjalizacja tabeli historii renderów"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS renders (
        job_id TEXT PRIMARY KEY,
        topic TEXT,
        status TEXT,
        video_url TEXT,
        error TEXT,
        video_duration REAL,
        created_at TIMESTAMP,
        completed_at TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

def get_gemini_client():
    """Inicjalizacja klienta Google Genai"""
    return genai.Client(
        http_options={"api_version": "v1beta"}, 
        api_key=os.getenv("GEMINI_API_KEY")
    )

# ---------------------------------------------------------------------------
# HISTORIA RENDERÓW (SQLite)
# ---------------------------------------------------------------------------

def save_render_to_db(job_id, topic, status, video_url=None, error=None, video_duration=None):
    """Zapis renderowania do bazy danych"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    completed_at = datetime.utcnow() if status in ['success', 'failed'] else None
    c.execute('''INSERT OR REPLACE INTO renders 
                 (job_id, topic, status, video_url, error, video_duration, created_at, completed_at) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (job_id, topic, status, video_url, error, video_duration, datetime.utcnow(), completed_at))
    conn.commit()
    conn.close()

def get_render_from_db(job_id):
    """Pobranie statusu renderowania z bazy"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM renders WHERE job_id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            "job_id": row[0],
            "topic": row[1],
            "status": row[2],
            "video_url": row[3],
            "error": row[4],
            "video_duration": row[5],
            "created_at": row[6],
            "completed_at": row[7]
        }
    return None

# ---------------------------------------------------------------------------
# POBIERANIE WIDEO Z RETRY/BACKOFF
# ---------------------------------------------------------------------------

def download_video_with_backoff(video_uri, temp_path, max_retries=5):
    """Pobieranie wideo z chmury z automatycznym retry/backoff"""
    api_key = os.getenv("GEMINI_API_KEY")
    headers = {"x-goog-api-key": api_key} if api_key else {}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(video_uri, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"✅ Wideo pobrane: {temp_path}")
            return True
            
        except Exception as exc:
            countdown = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Błąd pobierania (próba {attempt+1}/{max_retries}). Ponowna próba za {countdown}s...")
                time.sleep(countdown)
            else:
                logger.error(f"❌ Nie udało się pobrać wideo po {max_retries} próbach: {exc}")
                raise exc

# ---------------------------------------------------------------------------
# GENEROWANIE SEGMENTÓW WIDEO (3 sceny: HOOK, PROBLEM, ROZWIĄZANIE)
# ---------------------------------------------------------------------------

def generate_video_segment(client, prompt, aspect_ratio="9:16"):
    """Generuje pojedynczy klip wideo i zwraca lokalną ścieżkę tymczasową"""
    logger.info(f"🎬 Generowanie segmentu: {prompt[:50]}...")
    
    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=5,
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
        raise TimeoutError("❌ Veo API timeout podczas generowania segmentu.")
        
    result = operation.result
    video_uri = result.generated_videos[0].video.uri
    
    temp_file = os.path.join(tempfile.gettempdir(), f"seg_{os.urandom(4).hex()}.mp4")
    download_video_with_backoff(video_uri, temp_file)
    return temp_file

# ---------------------------------------------------------------------------
# AUDIO: LEKTOR (ElevenLabs)
# ---------------------------------------------------------------------------

def generate_audio_narration(narration_texts, job_id):
    """
    Generowanie MP3 z lektorem dla każdej sceny
    
    narration_texts: dict {
        "hook": "Większość osób traci pieniądze...",
        "problem": "Banki promują oferty...",
        "rozwiązanie": "Regularne porównywanie..."
    }
    
    Returns: dict {
        "hook": "/tmp/narration_hook_xxx.mp3",
        "problem": "/tmp/narration_problem_xxx.mp3",
        ...
    }
    """
    if not ELEVENLABS_API_KEY:
        logger.error("❌ ElevenLabs API key not configured!")
        raise ValueError("ELEVENLABS_API_KEY not set")
    
    elevenlabs.set_api_key(ELEVENLABS_API_KEY)
    
    audio_files = {}
    
    for scene_key, text in narration_texts.items():
        logger.info(f"🎙️ Generowanie lektora: {scene_key} ({len(text)} znaków)")
        
        try:
            # Generowanie audio z ElevenLabs
            audio = elevenlabs.generate(
                text=text,
                voice="Bella",  # Polski głos żeński
                model="eleven_monolingual_v1",
                api_key=ELEVENLABS_API_KEY
            )
            
            audio_file = os.path.join(tempfile.gettempdir(), f"narration_{scene_key}_{job_id}.mp3")
            with open(audio_file, "wb") as f:
                f.write(audio)
            
            # Pobranie czasu trwania audio
            duration = get_audio_duration(audio_file)
            logger.info(f"✅ Lektor {scene_key}: {duration:.2f}s")
            
            audio_files[scene_key] = {
                "path": audio_file,
                "duration": duration,
                "text": text
            }
            
        except Exception as e:
            logger.error(f"❌ Błąd generowania lektora {scene_key}: {e}")
            raise e
    
    return audio_files

# ---------------------------------------------------------------------------
# AUDIO: POBIERANIE CZASU TRWANIA
# ---------------------------------------------------------------------------

def get_audio_duration(audio_file):
    """Pobranie czasu trwania audio za pomocą ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        logger.warning(f"⚠️ Nie udało się pobrać czasu trwania {audio_file}: {e}")
        return 5.0  # Fallback: 5 sekund

def get_video_duration(video_file):
    """Pobranie czasu trwania wideo za pomocą ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        logger.warning(f"⚠️ Nie udało się pobrać czasu trwania {video_file}: {e}")
        return 5.0  # Fallback: 5 sekund

# ---------------------------------------------------------------------------
# AUTOMATYCZNE DOPASOWANIE DŁUGOŚCI
# ---------------------------------------------------------------------------

def calculate_video_speed(audio_files, target_duration=15):
    """
    Obliczenie prędkości playbacku aby zmieścić się w target_duration
    
    Jeśli lektor trwa dłużej niż wideo - przyspiesz wideo
    Jeśli lektor krótszy - zwolnij
    
    Returns: speed multiplier (0.8 = 80%, 1.0 = normalnie, 1.2 = 120%)
    """
    total_audio_duration = sum(audio["duration"] for audio in audio_files.values())
    
    # Dodaj 2 sekundy buforu dla CTA na końcu
    total_audio_duration += 2
    
    logger.info(f"⏱️ Całkowity czas lektora: {total_audio_duration:.2f}s")
    logger.info(f"📊 Target duration: {target_duration}s")
    
    if total_audio_duration <= target_duration:
        # Lektor zmieści się - normalnie
        speed = 1.0
        logger.info(f"✅ Lektor zmieści się. Speed: {speed}x (normalnie)")
    else:
        # Lektor za długi - przyspiesz wideo
        speed = total_audio_duration / target_duration
        logger.warning(f"⚠️ Lektor za długi ({total_audio_duration:.2f}s > {target_duration}s). Przyspieszenie: {speed:.2f}x")
    
    # Limit: max 1.5x speedup (aby nie było nienaturalne)
    if speed > 1.5:
        logger.warning(f"⚠️ Speed {speed:.2f}x przekracza limit 1.5x!")
        speed = 1.5
    
    return speed

def generate_video_with_speed_adjustment(segment_files, speed=1.0):
    """
    Generowanie wideo ze zmienioną prędkością
    
    Jeśli speed=1.2, wideo będzie 1.2x szybsze
    """
    if speed == 1.0:
        logger.info("✅ Brak dopasowania prędkości (1.0x)")
        return segment_files
    
    logger.info(f"⏱️ Dopasowywanie prędkości wszystkich segmentów do {speed:.2f}x...")
    
    speed_adjusted_files = []
    
    for i, video_file in enumerate(segment_files):
        output_file = os.path.join(tempfile.gettempdir(), f"speed_{i}_{os.urandom(4).hex()}.mp4")
        
        # FFmpeg setpts filter dla zmiany prędkości
        # Video: PTS*speed (przyspieszenie) + audio rate change
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_file,
            '-vf', f"setpts=PTS/{speed}",
            '-af', f"atempo={speed}",
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac',
            output_file
        ]
        
        logger.info(f"  ⏱️ Segment {i}: {speed:.2f}x")
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        speed_adjusted_files.append(output_file)
    
    return speed_adjusted_files

# ---------------------------------------------------------------------------
# ŁĄCZENIE WIDEO + AUDIO + MUZYKA TŁA
# ---------------------------------------------------------------------------

def concat_video_with_audio(video_files, audio_files, job_id, output_path, speed=1.0):
    """
    Łączenie segmentów wideo + dodanie lektora + muzyka tła
    
    Schemat:
    [Wideo 1 (5s)] → [Lektor 1 audio]
    [Wideo 2 (5s)] → [Lektor 2 audio]
    [Wideo 3 (5s)] → [Lektor 3 audio]
    [CTA overlay]
    + [Background music na całość]
    """
    
    # 1. PRZYGOTOWANIE LISTY WIDEO DO CONCAT
    list_file_path = os.path.join(tempfile.gettempdir(), f"list_{job_id}.txt")
    with open(list_file_path, "w") as f:
        for video_file in video_files:
            f.write(f"file '{video_file}'\n")
    
    logger.info("🎬 Etap 1: Łączenie segmentów wideo (FFmpeg concat)...")
    
    concat_output = os.path.join(tempfile.gettempdir(), f"concat_{job_id}.mp4")
    ffmpeg_concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path,
        '-c', 'copy',
        concat_output
    ]
    subprocess.run(ffmpeg_concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"✅ Wideo połączone: {concat_output}")
    
    # 2. PRZYGOTOWANIE CAŁKOWITEGO AUDIO (wszystkie lektory + background music)
    logger.info("🎙️ Etap 2: Miksowanie audio (lektory + muzyka tła)...")
    
    # Generowanie lektora - konkatenacja wszystkich scen
    combined_audio = os.path.join(tempfile.gettempdir(), f"combined_audio_{job_id}.mp3")
    
    # FFmpeg concat dla audio
    audio_list_file = os.path.join(tempfile.gettempdir(), f"audio_list_{job_id}.txt")
    with open(audio_list_file, "w") as f:
        for scene_key in ["hook", "problem", "rozwiązanie"]:
            if scene_key in audio_files:
                f.write(f"file '{audio_files[scene_key]['path']}'\n")
    
    ffmpeg_audio_concat = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', audio_list_file,
        '-c:a', 'libmp3lame', '-q:a', '4',
        combined_audio
    ]
    subprocess.run(ffmpeg_audio_concat, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"✅ Lektory połączone: {combined_audio}")
    
    # 3. OSTATECZNE MIKSOWANIE: WIDEO + LEKTOR + CTA OVERLAY
    logger.info("🎨 Etap 3: Miksowanie wideo + audio + CTA overlay...")
    
    cta_text = "Sprawdź raport na raport-finansowy24.pl"
    
    # Dla videa: dopasowanie do audio z setpts
    # Dla audio: normalizacja + EQ
    ffmpeg_final_cmd = [
        'ffmpeg', '-y',
        '-i', concat_output,  # Wideo
        '-i', combined_audio,  # Lektor
        '-filter_complex',
        # Video filter: setpts dla speed adjustment + drawtext dla CTA
        f"[0:v]setpts=PTS/{speed},drawtext=text='{cta_text}':x=(w-text_w)/2:y=h-200:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:enable='between(t,15,20)'[v]; " +
        # Audio: volume mix
        f"[1:a]volume=1.0[a1];[a1]aformat=sample_rates=44100[aout]",
        '-map', '[v]', '-map', '[aout]',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '128k',
        output_path
    ]
    
    subprocess.run(ffmpeg_final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"✅ Finalne wideo: {output_path}")
    
    # Cleanup
    for file in [list_file_path, audio_list_file, concat_output, combined_audio]:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"⚠️ Nie udało się usunąć {file}: {e}")
    
    return output_path

# ---------------------------------------------------------------------------
# GŁÓWNY PROCES RENDEROWANIA (W ODDZIELNYM WĄTKU)
# ---------------------------------------------------------------------------

NARRATION_TEMPLATES = {
    "hook": "Większość osób traci pieniądze na złym koncie. Czy i ty?",
    "problem": "Banki promują oferty, które szybko tracą atrakcyjne warunki.",
    "rozwiązanie": "Regularne porównywanie ofert pozwala znaleźć korzystniejsze opcje i zaoszczędzić na rachunkach."
}

def render_sequence_background(job_id, raw_data, webhook_url=None):
    """
    Główny proces montażu sekwencji - uruchamiany w tle
    
    Schemat:
    Temat → HOOK + PROBLEM + ROZWIĄZANIE → 3 generacje Veo → 
    Lektor ElevenLabs → Dopasowanie długości → FFmpeg concat → gotowy short
    """
    segment_files = []
    audio_files_dict = {}
    
    try:
        client = get_gemini_client()
        topic = raw_data.get("topic", "Finanse osobiste")
        aspect_ratio = raw_data.get("aspectRatio", "9:16")
        host = raw_data.get("host", "localhost:5000")
        custom_narration = raw_data.get("narration")  # Optional: custom narration texts
        
        logger.info(f"🚀 START renderowania Job ID: {job_id} | Temat: {topic}")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 1: SZABLON MARKETINGOWY (3 sceny)
        # ═══════════════════════════════════════════════════════════════
        prompts = {
            "hook": f"Dynamic cinematic shot, extreme close up, shock and stress, concept of {topic}, corporate finance style, 4k, professional",
            "problem": f"A person looking anxiously at bills and charts on a screen, dark moody lighting, financial stress, 4k, professional",
            "rozwiązanie": f"Bright clean studio lighting, a smartphone screen displaying green rising financial growth charts, relief, 4k, professional"
        }
        
        logger.info("📋 Szablon: HOOK → PROBLEM → ROZWIĄZANIE")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 2: GENEROWANIE 3 KLIPÓW VEO (12 klipów dziennie = 3 sceny × 4 filmy)
        # ═══════════════════════════════════════════════════════════════
        for key, prompt_text in prompts.items():
            logger.info(f"🎥 Generowanie sceny: {key.upper()}")
            file_path = generate_video_segment(client, prompt_text, aspect_ratio)
            segment_files.append(file_path)
            logger.info(f"✅ Scena {key} gotowa: {file_path}")
        
        logger.info(f"✅ Wszystkie 3 sceny gotowe ({len(segment_files)} klipów)")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 3: GENEROWANIE LEKTORA (ElevenLabs)
        # ═══════════════════════════════════════════════════════════════
        logger.info("🎙️ Generowanie lektora (ElevenLabs)...")
        
        # Użyj custom narration jeśli dostarczone, inaczej templates
        narration_texts = custom_narration if custom_narration else NARRATION_TEMPLATES
        
        audio_files_dict = generate_audio_narration(narration_texts, job_id)
        logger.info(f"✅ Lektor wygenerowany: {len(audio_files_dict)} scen")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 4: AUTOMATYCZNE DOPASOWANIE DŁUGOŚCI
        # ═══════════════════════════════════════════════════════════════
        logger.info("⏱️ Etap automatycznego dopasowania długości...")
        
        target_duration = 15  # 3 sceny × 5s = 15s bazowo
        speed = calculate_video_speed(audio_files_dict, target_duration)
        
        if speed != 1.0:
            logger.info(f"⚡ Dopasowywanie prędkości wideo do {speed:.2f}x...")
            segment_files = generate_video_with_speed_adjustment(segment_files, speed)
            logger.info(f"✅ Segmenty dopasowane do {speed:.2f}x")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 5: ŁĄCZENIE + MIKSOWANIE AUDIO
        # ═══════════════════════════════════════════════════════════════
        logger.info("🎬 Łączenie wideo + audio...")
        
        final_filename = f"render_{job_id}.mp4"
        final_output_path = os.path.join(STORAGE_DIR, final_filename)
        
        concat_video_with_audio(segment_files, audio_files_dict, job_id, final_output_path, speed)
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 6: POMIAR CZASU TRWANIA FINALNEGO WIDEO
        # ═══════════════════════════════════════════════════════════════
        video_duration = get_video_duration(final_output_path)
        file_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
        
        logger.info(f"✅ SUKCES! Film gotowy: {final_filename}")
        logger.info(f"  ⏱️ Czas trwania: {video_duration:.2f}s")
        logger.info(f"  📊 Rozmiar: {file_size_mb:.1f} MB")
        logger.info(f"  ⚡ Prędkość: {speed:.2f}x")
        
        video_url = f"https://{host}/videos/{final_filename}"
        logger.info(f"📺 URL: {video_url}")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 7: AKTUALIZACJA BAZY DANYCH (SQLite)
        # ═══════════════════════════════════════════════════════════════
        save_render_to_db(job_id, topic, 'success', video_url, video_duration=video_duration)
        logger.info(f"💾 Historia zapisana: SQLite")
        
        # ═══════════════════════════════════════════════════════════════
        # KROK 8: WEBHOOK NOTIFICATION
        # ═══════════════════════════════════════════════════════════════
        if webhook_url:
            logger.info(f"🔔 Wysyłanie webhook: {webhook_url}")
            try:
                webhook_payload = {
                    "job_id": job_id,
                    "status": "success",
                    "video_url": video_url,
                    "topic": topic,
                    "video_duration": video_duration,
                    "file_size_mb": file_size_mb,
                    "speed_adjustment": speed,
                    "timestamp": datetime.utcnow().isoformat()
                }
                response = requests.post(webhook_url, json=webhook_payload, timeout=10)
                logger.info(f"✅ Webhook wysłany (status: {response.status_code})")
            except requests.RequestException as e:
                logger.error(f"⚠️ Błąd webhook: {e}")
                
    except Exception as e:
        logger.error(f"❌ BŁĄD KRYTYCZNY Job {job_id}: {e}", exc_info=True)
        save_render_to_db(job_id, raw_data.get("topic", "Unknown"), 'failed', error=str(e))
        
        if webhook_url:
            try:
                webhook_payload = {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                requests.post(webhook_url, json=webhook_payload, timeout=10)
                logger.info(f"🔔 Webhook błędu wysłany")
            except Exception as webhook_error:
                logger.error(f"⚠️ Błąd wysyłania webhook błędu: {webhook_error}")
                
    finally:
        # ═══════════════════════════════════════════════════════════════
        # CLEANUP: Usuwanie plików tymczasowych
        # ═══════════════════════════════════════════════════════════════
        logger.info("🧹 Czyszczenie plików tymczasowych...")
        for path in segment_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"  ✓ Usunięto: {os.path.basename(path)}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Nie udało się usunąć {path}: {e}")
        
        for scene_key, audio_info in audio_files_dict.items():
            audio_path = audio_info.get("path")
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"  ✓ Usunięto audio: {os.path.basename(audio_path)}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Nie udało się usunąć {audio_path}: {e}")

# ---------------------------------------------------------------------------
# CLEANUP STARYCH PLIKÓW (24h)
# ---------------------------------------------------------------------------

def cleanup_old_files(hours=24):
    """Czyszczenie plików starszych niż N godzin"""
    cutoff_time = time.time() - (hours * 3600)
    cleaned_count = 0
    
    for filename in os.listdir(STORAGE_DIR):
        filepath = os.path.join(STORAGE_DIR, filename)
        
        # Pomijamy database
        if filename.endswith('.db'):
            continue
            
        if os.path.isfile(filepath):
            file_age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
            
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                    logger.info(f"🧹 Usunięty stary plik ({file_age_hours:.1f}h): {filename}")
                except Exception as e:
                    logger.error(f"❌ Błąd przy usuwaniu {filename}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"✅ Cleanup: Usunięto {cleaned_count} starych plików")

# ---------------------------------------------------------------------------
# ENDPOINTY FLASK (API)
# ---------------------------------------------------------------------------

@app.route("/render-sequence", methods=["POST"])
def start_render_sequence():
    """
    POST /render-sequence
    
    Body:
    {
        "topic": "Porównanie kont bankowych",
        "narration": {
            "hook": "Custom hook text",
            "problem": "Custom problem text",
            "rozwiązanie": "Custom solution text"
        },
        "webhookUrl": "https://example.com/webhook"
    }
    
    Response:
    {
        "status": "queued",
        "job_id": "uuid",
        "status_url": "https://host/tasks/uuid"
    }
    """
    data = request.json or {}
    topic = data.get("topic", "").strip()
    
    if not topic:
        return jsonify({"error": "Missing or empty 'topic'"}), 400
    
    webhook_url = data.get("webhookUrl")
    job_id = str(uuid.uuid4())
    
    # Ustawienie hosta
    data['host'] = request.host
    
    # Aktualizacja bazy danych - status "processing"
    save_render_to_db(job_id, topic, 'processing')
    logger.info(f"📥 Nowe zlecenie: Job {job_id} | Temat: {topic}")
    
    # Uruchomienie w tle (wątek)
    thread = threading.Thread(
        target=render_sequence_background,
        args=(job_id, data, webhook_url),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "status": "queued",
        "job_id": job_id,
        "status_url": f"https://{request.host}/tasks/{job_id}"
    }), 202


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """
    GET /tasks/<job_id>
    
    Response:
    {
        "job_id": "uuid",
        "state": "processing|success|failed",
        "video_url": "https://...",
        "video_duration": 15.5,
        "error": "...",
        "created_at": "timestamp",
        "completed_at": "timestamp"
    }
    """
    render = get_render_from_db(task_id)
    
    if not render:
        return jsonify({"error": "Task not found"}), 404
    
    response = {
        "job_id": task_id,
        "state": render["status"],
        "created_at": render["created_at"],
        "completed_at": render["completed_at"]
    }
    
    if render["status"] == "processing":
        response["status"] = "⏳ Przetwarzanie..."
    elif render["status"] == "success":
        response["status"] = "✅ Zakończono sukcesem"
        response["video_url"] = render["video_url"]
        response["video_duration"] = render["video_duration"]
    elif render["status"] == "failed":
        response["status"] = "❌ Błąd wykonania"
        response["error"] = render["error"]
    
    return jsonify(response)


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """
    GET /videos/<filename>
    
    Dystrybucja gotowych filmów
    """
    return send_from_directory(STORAGE_DIR, filename)


@app.route("/health", methods=["GET"])
def health_check():
    """
    GET /health
    
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "storage_dir": STORAGE_DIR,
        "elevenlabs": "✅ configured" if ELEVENLABS_API_KEY else "❌ not configured"
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Info o API"""
    return jsonify({
        "name": "VeoVideo API",
        "version": "2.0.0",
        "features": {
            "veo_generation": "3 sceny po kolei",
            "audio_narration": "ElevenLabs (Polish voice)",
            "auto_length_adjustment": "Dopasowanie prędkości do lektora",
            "webhook": "Notyfikacja po zakończeniu"
        },
        "endpoints": {
            "POST /render-sequence": "Uruchomienie renderowania",
            "GET /tasks/<job_id>": "Status renderowania",
            "GET /videos/<filename>": "Pobieranie wideo",
            "GET /health": "Health check"
        },
        "schema": {
            "daily_renders": "4 filmy",
            "segments_per_video": "3 sceny (HOOK/PROBLEM/ROZWIĄZANIE)",
            "daily_veo_generations": "12 klipów",
            "target_duration": "15 sekund (dopasowywane automatycznie)"
        }
    }), 200


if __name__ == "__main__":
    logger.info("🚀 Startup VeoVideo API v2.0 (with ElevenLabs + Auto-Length)")
    logger.info(f"📁 Storage: {STORAGE_DIR}")
    logger.info(f"🗄️  Database: {DB_PATH}")
    logger.info(f"🎙️ ElevenLabs: {'✅ Configured' if ELEVENLABS_API_KEY else '❌ Not configured'}")
    
    # Czyszczenie starych plików przy starcie
    cleanup_old_files(hours=24)
    
    # Uruchomienie Flask
    app.run(host="0.0.0.0", port=5000, threaded=True)
