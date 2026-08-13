# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import http.server
import socketserver
import urllib.request
import urllib.error
import urllib.parse
import os
import base64
import io
import json
import numpy as np
import wave
import traceback
import socket
import ssl

import threading
from collections import OrderedDict

# Multilingual STT via faster-whisper. One model covers every language, so unlike the
# per-language engines below there is nothing to cache or evict.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UI language codes we hand to Whisper directly; anything else gets auto-detected.
# Any Whisper checkpoint works here, including Swedish-tuned ones (e.g. KBLab/kb-whisper-medium).
SUPPORTED_STT_LANGS = {"sv", "fi", "en", "ar", "es", "ja", "zh", "ko"}
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
# Optimeringar hålls bakom flaggor så att bench kan A/B-testa dem i samma
# körning, och så att produkten behåller sitt ursprungsbeteende tills en
# ändring är uppmätt. Defaultarna flippas när kampanjen är klar.
STT_VAD = os.environ.get("STT_VAD", "0") == "1"
MAX_MODELS = 2
# Loaded at startup so the two default lanes never stall on a first utterance.
# Keep this within MAX_MODELS or the entries just evict each other.
PREWARM_LANGS = ("sv", "en")
_whisper_model = None
# RLock (reentrant): handle_stt holds the lock across get_whisper_model() + inference,
# and get_whisper_model() re-acquires it on the same thread. A plain Lock() self-deadlocks.
_stt_lock = threading.RLock()

# Multilingual TTS via Piper, which is the only offline engine covering both Swedish and
# Finnish. The voice is fixed at load, so we lazily build (and cache) one per language used.
PIPER_VOICE_MAP = {
    "sv": "sv_SE-nst-medium",
    "fi": "fi_FI-harri-medium",
    "en": "en_US-lessac-medium",
    "ar": "ar_JO-kareem-medium",
    "es": "es_ES-davefx-medium",
    "zh": "zh_CN-huayan-medium",
    "ko": "ko_KR-kss-medium",
}
# Piper has no Japanese voice, so `ja` alone still goes through moonshine-voice
# (Kokoro / Piper backed). Maps our UI language codes -> moonshine-voice language codes.
TTS_LANG_MAP = {
    "ja": "ja-jp",
}
PIPER_VOICE_DIR = os.environ.get(
    "PIPER_VOICE_DIR", os.path.join(os.path.expanduser("~"), ".local", "share", "piper-voices")
)
_piper_voices = OrderedDict()  # our-lang-code -> PiperVoice
_tts_engines = OrderedDict()  # our-lang-code -> TextToSpeech
# RLock (reentrant): handle_tts holds the lock across the loaders + synthesis, and the
# loaders re-acquire it on the same thread. A plain Lock() self-deadlocks.
_tts_lock = threading.RLock()

def get_tts_engine(language):
    with _tts_lock:
        if language in _tts_engines:
            _tts_engines.move_to_end(language)
            return _tts_engines[language]
        from moonshine_voice import TextToSpeech
        moon_lang = TTS_LANG_MAP[language]
        print(f"[TTS] Loading moonshine-voice (lang={language} -> {moon_lang})...")
        if len(_tts_engines) >= MAX_MODELS:
            oldest_lang, oldest_engine = _tts_engines.popitem(last=False)
            print(f"[TTS] Evicting model for {oldest_lang}")
            del oldest_engine
        _tts_engines[language] = TextToSpeech(moon_lang)
        return _tts_engines[language]

def get_piper_voice(language):
    with _tts_lock:
        if language in _piper_voices:
            _piper_voices.move_to_end(language)
            return _piper_voices[language]
        from pathlib import Path
        from piper import PiperVoice
        from piper.download_voices import download_voice
        voice_id = PIPER_VOICE_MAP[language]
        voice_dir = Path(PIPER_VOICE_DIR)
        voice_dir.mkdir(parents=True, exist_ok=True)
        if not (voice_dir / f"{voice_id}.onnx").exists():
            print(f"[TTS] Downloading Piper voice {voice_id}...")
            download_voice(voice_id, voice_dir)
        print(f"[TTS] Loading Piper (lang={language} -> {voice_id})...")
        if len(_piper_voices) >= MAX_MODELS:
            oldest_lang, oldest_voice = _piper_voices.popitem(last=False)
            print(f"[TTS] Evicting Piper voice for {oldest_lang}")
            del oldest_voice
        _piper_voices[language] = PiperVoice.load(voice_dir / f"{voice_id}.onnx")
        return _piper_voices[language]

def synthesize(text, language, syn_config=None):
    """Synthesize `text` and return (mono float32 samples in [-1, 1], sample_rate).

    `syn_config` is an optional piper SynthesisConfig handed straight to Piper.
    Leaving it None keeps the product's default voice settings; bench/ passes one
    with the noise scales zeroed, because Piper's VITS decoder otherwise samples
    fresh noise per call and no two renderings of the same text are alike.
    """
    if language in TTS_LANG_MAP:
        if syn_config is not None:
            # moonshine-voice accepts no synthesis config. Refuse rather than
            # silently ignoring it and returning non-deterministic audio.
            raise ValueError(
                f"syn_config is not supported for language {language!r} (moonshine-voice path)"
            )
        return get_tts_engine(language).synthesize(text)
    if language not in PIPER_VOICE_MAP:
        language = "en"
    # syn_config=None is what PiperVoice.synthesize already defaults to, so the
    # product path stays byte-identical to before this parameter existed.
    chunks = list(get_piper_voice(language).synthesize(text, syn_config))
    pcm = b"".join(c.audio_int16_bytes for c in chunks)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, chunks[0].sample_rate

def get_whisper_model():
    global _whisper_model
    with _stt_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading faster-whisper ({WHISPER_MODEL_SIZE}, int8)...")
            _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return _whisper_model

def transcribe(audio_np, language):
    """Transcribe 16 kHz mono float32 samples. Unknown languages are auto-detected."""
    segments, _ = get_whisper_model().transcribe(
        audio_np,
        language=language if language in SUPPORTED_STT_LANGS else None,
        beam_size=1,
        # Klipper bort tystnad före dekodning. Kortar ljudet som når modellen
        # och tar bort de hallucinationer stock-Whisper gärna producerar på
        # tysta partier.
        vad_filter=STT_VAD,
    )
    return " ".join(s.text.strip() for s in segments)


# Överskrivbar så att bench/ kan köra en egen instans parallellt med en
# vanlig utvecklingsserver på 3000.
PORT = int(os.environ.get("PORT", 3000))

class ProxyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, x-target-url, authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def handle_proxy(self):
        # Parse query parameter "url"
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        target_url = query.get('url', [None])[0]

        if not target_url:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Error: Missing "url" query parameter.')
            return

        # Restrict target URL to local LLM endpoint (http/https localhost/127.0.0.1)
        parsed_target = urllib.parse.urlparse(target_url)
        if parsed_target.scheme not in ('http', 'https') or parsed_target.hostname not in ('localhost', '127.0.0.1') or parsed_target.port not in (9379, None):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'Forbidden: Proxy target must be localhost:9379')
            return

        print(f"[Proxy] Routing {self.command} request to: {target_url}")
        
        # Read request body if method is POST/PUT/PATCH
        body = None
        if self.command in ['POST', 'PUT', 'PATCH']:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

        # Build request to target url
        req = urllib.request.Request(
            target_url,
            data=body,
            method=self.command
        )

        # Forward headers (Content-Type, Authorization, etc.)
        for key, val in self.headers.items():
            if key.lower() not in ['host', 'connection', 'content-length', 'x-target-url']:
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                res_body = response.read()
                self.send_response(response.status)
                # Forward response headers
                for key, val in response.headers.items():
                    if key.lower() not in ['content-length', 'connection']:
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(res_body)
        except urllib.error.HTTPError as e:
            print(f"[Proxy Error] HTTP Error {e.code}: {e.reason}")
            try:
                res_body = e.read()
            except Exception:
                res_body = str(e).encode('utf-8')
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(res_body)
        except Exception as e:
            print(f"[Proxy Error] Exception: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def handle_tts(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        text = query.get('text', [None])[0]
        lang = query.get('lang', ['en'])[0]

        if not text:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Error: Missing "text" parameter.')
            return

        print(f"[TTS] Synthesizing: {text[:50]}... (lang: {lang})")

        try:
            with _tts_lock:
                audio, sample_rate = synthesize(text, lang)

            # Both engines return mono float samples in [-1, 1]; encode to 16-bit PCM WAV.
            samples = np.asarray(audio, dtype=np.float32)
            samples = np.clip(samples, -1.0, 1.0)
            pcm16 = (samples * 32767.0).astype('<i2')

            with io.BytesIO() as buf:
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(sample_rate))
                    wf.writeframes(pcm16.tobytes())
                wav_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Length', str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(f"[TTS Error] Exception: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def handle_stt(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            if not body:
                raise ValueError("No body data")
                
            data = json.loads(body.decode('utf-8'))
            audio_b64 = data.get('audio_base64')
            if not audio_b64:
                raise ValueError("Missing audio_base64 parameter")

            language = data.get('language', 'en')
            raw_data = base64.b64decode(audio_b64)

            # The browser sends a raw Float32Array buffer (16 kHz mono).
            audio_np = np.frombuffer(raw_data, dtype=np.float32).copy()
            duration_s = len(audio_np) / 16000.0
            peak = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
            rms = float(np.sqrt(np.mean(audio_np ** 2))) if len(audio_np) else 0.0
            print(
                f"[STT] audio: {duration_s:.2f}s, peak={peak:.4f}, rms={rms:.4f}, lang={language}",
                flush=True,
            )

            with _stt_lock:
                text = transcribe(audio_np, language)
            print(f"[STT] Transcribed: {text!r}", flush=True)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"text": text}).encode('utf-8'))
        except Exception as e:
            traceback.print_exc()
            print(f"[STT Error] Exception: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def handle_volume(self):
        client_ip = self.client_address[0]
        if client_ip not in ('127.0.0.1', '::1', 'localhost'):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'Forbidden: Volume control is only accessible locally')
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8')) if body else {}
                action = data.get('action')
            else:
                action = "get"
            
            import subprocess
            import re
            
            # PipeWire/wpctl needs XDG_RUNTIME_DIR to find its socket.
            # The server process may not have it set (e.g. when launched by systemd).
            env = os.environ.copy()
            if 'XDG_RUNTIME_DIR' not in env:
                uid = os.getuid()
                env['XDG_RUNTIME_DIR'] = f'/run/user/{uid}'
            
            def get_vol():
                # Try wpctl (PipeWire) first - outputs "Volume: 0.75"
                try:
                    out = subprocess.check_output(
                        ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                        text=True, timeout=2, env=env
                    )
                    m = re.search(r'Volume:\s+([0-9.]+)', out)
                    if m:
                        return round(float(m.group(1)) * 100)
                except Exception:
                    pass
                # Try pactl (PulseAudio)
                try:
                    out = subprocess.check_output(
                        ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                        text=True, timeout=2, env=env
                    )
                    m = re.search(r'(\d+)%', out)
                    if m:
                        return int(m.group(1))
                except Exception:
                    pass
                # Try amixer
                try:
                    out = subprocess.check_output(
                        ["amixer", "sget", "Master"],
                        text=True, timeout=2, env=env
                    )
                    m = re.search(r'\[(\d+)%\]', out)
                    if m:
                        return int(m.group(1))
                except Exception:
                    pass
                return None

            def set_vol(direction):
                # Try wpctl first
                try:
                    arg = "5%+" if direction == "up" else "5%-"
                    subprocess.run(
                        ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", arg],
                        check=True, timeout=2, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    return True
                except Exception:
                    pass
                # Try pactl
                try:
                    arg = "+5%" if direction == "up" else "-5%"
                    subprocess.run(
                        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", arg],
                        check=True, timeout=2, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    return True
                except Exception:
                    pass
                # Try amixer
                try:
                    arg = "5%+" if direction == "up" else "5%-"
                    subprocess.run(
                        ["amixer", "sset", "Master", arg],
                        check=True, timeout=2, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    return True
                except Exception:
                    pass
                return False

            success = False
            if action in ("up", "down"):
                success = set_vol(action)
            elif action == "get":
                success = True

            if success:
                current_vol = get_vol()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "volume": current_vol}).encode('utf-8'))
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'Failed to change system volume')
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def do_POST(self):
        if self.path.startswith('/proxy'):
            self.handle_proxy()
            return
        if self.path.startswith('/api/stt'):
            self.handle_stt()
            return
        if self.path.startswith('/api/volume'):
            self.handle_volume()
            return
        
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/proxy'):
            self.handle_proxy()
            return

        if self.path.startswith('/api/tts'):
            self.handle_tts()
            return
            
        if self.path.startswith('/api/volume'):
            self.handle_volume()
            return

        # Clean path to serve static files (strip any ?query cache-buster)
        url_path = self.path.split('?', 1)[0]
        if url_path == '/':
            url_path = '/index.html'

        dist_dir = os.path.realpath(os.path.join(BASE_DIR, '..', 'frontend', 'dist'))
        if not os.path.exists(dist_dir):
            dist_dir = os.path.realpath(os.path.join(BASE_DIR, 'dist'))
        if not os.path.exists(dist_dir):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'dist/ directory not found')
            return

        filename = url_path.lstrip('/')
        filepath = os.path.realpath(os.path.join(dist_dir, filename))
        
        # Check if the file is within dist directory
        if not filepath.startswith(dist_dir + os.sep) and filepath != dist_dir:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'Forbidden')
            return

        if not os.path.exists(filepath) or os.path.isdir(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'File not found')
            return

        # Determine MIME type
        ext = os.path.splitext(filepath)[1].lower()
        mime_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
        }
        content_type = mime_types.get(ext, 'application/octet-stream')

        # Read and serve file
        try:
            with open(filepath, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(f.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

if __name__ == '__main__':
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    local_ip = "localhost"
    try:
        # Create a dummy socket to find local network IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    use_ssl = os.path.exists('cert.pem') and os.path.exists('key.pem')

    with socketserver.ThreadingTCPServer(("", PORT), ProxyHTTPRequestHandler) as httpd:
        if use_ssl:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

        protocol = "https" if use_ssl else "http"
        print(f"===========================================================")
        print(f"LiteRT-LM Audio Testbed client running at:")
        print(f"👉 {protocol}://localhost:{PORT}")
        if local_ip != "localhost":
            print(f"👉 {protocol}://{local_ip}:{PORT} (Local Network)")
        print(f"===========================================================")
        def _prewarm_models():
            try:
                get_whisper_model()
                for lang in PREWARM_LANGS:
                    print(f"[Prewarm] Loading {lang} voice into memory...", flush=True)
                    if lang in TTS_LANG_MAP:
                        get_tts_engine(lang)
                    else:
                        get_piper_voice(lang)
                print("[Prewarm] Models pre-warmed successfully.", flush=True)
            except Exception as e:
                print(f"[Prewarm Error] {e}", flush=True)

        threading.Thread(target=_prewarm_models, daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
