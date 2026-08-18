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

# Per-language STT/TTS with a shared Gemma for translation. STT and TTS each
# keep an LRU of at most MAX_MODELS loaded checkpoints/voices so a Pi-class
# box is not asked to hold every language at once.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UI language codes we hand to Whisper directly; anything else gets auto-detected.
SUPPORTED_STT_LANGS = {"sv", "fi", "en", "es", "fr"}
# Fallback multilingual Whisper for every language that has no specialised entry
# in STT_MODEL_MAP. Override with WHISPER_MODEL_SIZE.
DEFAULT_STT_MODEL = os.environ.get("WHISPER_MODEL_SIZE", "small")
# Per-language STT when a Hub id already ships CTranslate2 weights (faster-whisper
# loads them without a local conversion). Override with STT_MODEL_<LANG>=small to
# A/B against stock Whisper. Finnish uses a medium CT2 fine-tune; es/fr stay on
# multilingual small (see README) but still accept env overrides.
STT_MODEL_MAP = {
    "sv": os.environ.get("STT_MODEL_SV", "KBLab/kb-whisper-small"),
    "fi": os.environ.get("STT_MODEL_FI", "mpasila/faster-whisper-medium-finnish"),
    "es": os.environ.get("STT_MODEL_ES", DEFAULT_STT_MODEL),
    "fr": os.environ.get("STT_MODEL_FR", DEFAULT_STT_MODEL),
}
# Optimeringar hålls bakom flaggor så att bench kan A/B-testa dem i samma
# körning, och så att produkten behåller sitt ursprungsbeteende tills en
# ändring är uppmätt. Defaultarna flippas när kampanjen är klar.
# Hallucinations on silence: skip a segment when Whisper itself thinks it
# heard no speech, even if avg_logprob is high (the usual skip rule requires
# *both* high no_speech_prob and low logprob, which confident radio-copy
# hallucinations fail).
STT_NO_SPEECH_PROB = float(os.environ.get("STT_NO_SPEECH_PROB", "0.5"))
# Trim silence before decode. Left env-gated off during the latency campaign
# (null on time-to-first-audio); continuous listening made the quality win
# the default — set STT_VAD=0 to restore the old path.
STT_VAD = os.environ.get("STT_VAD", "1") == "1"
MAX_MODELS = 2
# Dedicated checkpoint for language-id only, held outside the MAX_MODELS LRU.
#
# Language-id is a full encoder pass, and it used to run on DEFAULT_STT_MODEL —
# so every auto-routed utterance paid two `small` encoder passes instead of one.
# Measured over 27 conversation clips (bench/conversations.json, five samtal),
# deciding between the two lane languages as resolve_lane_language does:
#
#   small  858 ms   27/27   smallest winning margin 17.8x
#   base   274 ms   27/27   smallest winning margin  7.2x
#   tiny   153 ms   26/27   smallest winning margin  1.21x
#
# `base` is the default because tiny's correct and incorrect answers *overlap*
# (its one miss won by 1.19x, its narrowest correct call was 1.21x), so no
# confidence threshold can separate them — the failure mode is a short Swedish
# clip landing on the other lane and being transcribed as English.
#
# Keeping it out of the LRU is what lets a specialised pair (sv/fi) auto-route at
# all: a third `small`-sized entry would have evicted a lane model on every turn.
# base int8 is ~75 MB, so the extra resident memory is affordable on a Pi.
STT_LID_MODEL = os.environ.get("STT_LID_MODEL", "Systran/faster-whisper-base")
# Floors for trusting the two-language comparison. Both sit far below `base`'s
# measured worst correct call (mass 0.048, ratio 7.2) — they exist to catch a
# genuine coin flip, not to second-guess the detector.
LID_MIN_MASS = 0.02
LID_MIN_RATIO = 2.0
_lid_model = None
# Loaded at startup so the two default lanes never stall on a first utterance.
# Keep this within MAX_MODELS or the entries just evict each other.
PREWARM_LANGS = ("sv", "en")
_whisper_models = OrderedDict()  # model_id -> WhisperModel
# RLock (reentrant): handle_stt holds the lock across get_whisper_model() + inference,
# and get_whisper_model() re-acquires it on the same thread. A plain Lock() self-deadlocks.
_stt_lock = threading.RLock()

# Piper voices, one per UI language. The voice is fixed at load, so we lazily
# build (and cache) one per language used, with the same MAX_MODELS LRU as STT.
PIPER_VOICE_MAP = {
    "sv": "sv_SE-nst-medium",
    "fi": "fi_FI-harri-medium",
    "en": "en_US-lessac-medium",
    "es": "es_ES-davefx-medium",
    "fr": "fr_FR-siwis-medium",
}
PIPER_VOICE_DIR = os.environ.get(
    "PIPER_VOICE_DIR", os.path.join(os.path.expanduser("~"), ".local", "share", "piper-voices")
)
_piper_voices = OrderedDict()  # our-lang-code -> PiperVoice
# RLock (reentrant): handle_tts holds the lock across the loader + synthesis, and the
# loader re-acquires it on the same thread. A plain Lock() self-deadlocks.
_tts_lock = threading.RLock()

def stt_model_for(language):
    """Whisper checkpoint id for a UI language code."""
    return STT_MODEL_MAP.get(language, DEFAULT_STT_MODEL)

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
    if language not in PIPER_VOICE_MAP:
        language = "en"
    # syn_config=None is what PiperVoice.synthesize already defaults to, so the
    # product path stays byte-identical to before this parameter existed.
    chunks = list(get_piper_voice(language).synthesize(text, syn_config))
    if not chunks:
        # Piper returns nothing for punctuation-only strings (e.g. "."). A
        # 500 here used to pop an alert in the kiosk. Silence is the right
        # answer: there is nothing to speak.
        return np.zeros(1, dtype=np.float32), 22050
    pcm = b"".join(c.audio_int16_bytes for c in chunks)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, chunks[0].sample_rate

def get_whisper_model(model_id=None):
    """Load (or reuse) a faster-whisper checkpoint. LRU-evicts past MAX_MODELS."""
    model_id = model_id or DEFAULT_STT_MODEL
    with _stt_lock:
        if model_id in _whisper_models:
            _whisper_models.move_to_end(model_id)
            return _whisper_models[model_id]
        from faster_whisper import WhisperModel
        print(f"[STT] Loading faster-whisper ({model_id}, int8)...")
        if len(_whisper_models) >= MAX_MODELS:
            oldest_id, oldest_model = _whisper_models.popitem(last=False)
            print(f"[STT] Evicting model {oldest_id}")
            del oldest_model
        _whisper_models[model_id] = WhisperModel(
            model_id, device="cpu", compute_type="int8"
        )
        return _whisper_models[model_id]

def get_lid_model():
    """Load (or reuse) the language-id checkpoint. Never LRU-evicted."""
    global _lid_model
    with _stt_lock:
        if _lid_model is None:
            from faster_whisper import WhisperModel

            print(f"[STT] Loading language-id model ({STT_LID_MODEL}, int8)...")
            _lid_model = WhisperModel(STT_LID_MODEL, device="cpu", compute_type="int8")
        return _lid_model


def keep_stt_segment(segment):
    """False when Whisper is decoding silence as confident radio/TV copy.

    Built-in skip requires high no_speech_prob *and* low avg_logprob.
    KB-Whisper hallucinations on room tone often have high logprob, so we
    drop on no_speech_prob alone.
    """
    text = (getattr(segment, "text", None) or "").strip()
    if not text:
        return False
    try:
        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
    except (TypeError, ValueError):
        no_speech = 0.0
    return no_speech < STT_NO_SPEECH_PROB


def transcribe(audio_np, language, other_language=None, auto_language=False, fast=False):
    """Transcribe 16 kHz mono float32 samples.

    `fast` decodes on the resident language-id checkpoint instead of the lane
    model. It exists for interim previews, where the text is transient and the
    cost is not: see handle_stt for the arithmetic that made this necessary.

    Returns (text, resolved_language). Unknown languages are auto-detected.
    `auto_language` (product default) runs a cheap language-id pass on the
    dedicated LID checkpoint and re-routes STT to whichever of the two lanes
    was actually spoken. Bench leaves this off so fixture language stays locked.
    """
    expected = language if language in SUPPORTED_STT_LANGS else None
    other = other_language if other_language in SUPPORTED_STT_LANGS else None
    resolved = expected
    if auto_language and expected is not None and len(audio_np) >= int(16000 * 0.6):
        try:
            detected, prob, all_probs = get_lid_model().detect_language(audio_np)
            # Prefer the two-way comparison; fall back to the argmax rule only
            # when the detector genuinely cannot separate the two lanes.
            resolved = resolve_lane_language(all_probs, expected, other)
            if resolved is None:
                resolved = resolve_spoken_language(detected, prob, expected, other)
        except Exception as e:
            print(f"[STT] language detect failed, using {expected}: {e}", flush=True)
            resolved = expected

    if fast:
        model = get_lid_model()
    else:
        model_id = (
            stt_model_for(resolved) if resolved in SUPPORTED_STT_LANGS else DEFAULT_STT_MODEL
        )
        model = get_whisper_model(model_id)
    segments, info = model.transcribe(
        audio_np,
        language=resolved if resolved in SUPPORTED_STT_LANGS else None,
        beam_size=1,
        condition_on_previous_text=False,
        no_speech_threshold=STT_NO_SPEECH_PROB,
        vad_filter=STT_VAD,
    )
    parts = []
    for segment in segments:
        if not keep_stt_segment(segment):
            print(
                f"[STT] drop hallucination no_speech_prob="
                f"{getattr(segment, 'no_speech_prob', 0):.2f}: {segment.text!r}",
                flush=True,
            )
            continue
        parts.append(segment.text.strip())
    text = " ".join(parts)
    out_lang = resolved or getattr(info, "language", None) or expected or "en"
    return text, out_lang


def resolve_lane_language(all_probs, expected, other):
    """Pick between the two lane languages, or None when the detector cannot.

    `detect_language` returns the whole distribution, not just its argmax, and
    the argmax is the wrong question here: the utterance is one of the two
    configured lanes, so this is a two-way decision, not a ~100-way one.

    It matters because of what the argmax path did with a miss. On 1.4 s of
    Swedish the tiny detector answered `zh` at p=0.94; `zh` is not a supported
    language, so resolve_spoken_language fell back to the lane prior — i.e. to
    whoever spoke *last*. In an alternating conversation that prior is wrong
    every time, so one detector miss became a whole turn transcribed in the
    wrong language and "translated" into gibberish. Restricting the choice to
    the two lanes turned four such misses into one across the bench corpus.

    Returning None means "do not guess": the caller keeps the old rule.
    """
    if not all_probs or not expected or not other or expected == other:
        return None
    probs = dict(all_probs)
    p_expected = float(probs.get(expected, 0.0))
    p_other = float(probs.get(other, 0.0))
    if p_expected >= p_other:
        winner, hi, lo = expected, p_expected, p_other
    else:
        winner, hi, lo = other, p_other, p_expected
    if hi < LID_MIN_MASS or hi < lo * LID_MIN_RATIO:
        return None
    return winner


def resolve_spoken_language(detected, probability, expected, other=None):
    """Lane language is the prior; a confident other-lane (or supported) id wins.

    Port-adjacent to the product STT auto-language rule. Short/uncertain clips
    keep `expected` so Swedish-on-KB-Whisper is not flipped by a noisy detect.
    """
    code = (detected or "").split("-")[0].lower()
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        prob = 0.0
    if code not in SUPPORTED_STT_LANGS:
        return expected
    if expected and code == expected:
        return expected
    if other and code == other and prob >= 0.5:
        return other
    if expected and prob < 0.6:
        return expected
    return code


# Överskrivbar så att bench/ kan köra en egen instans parallellt med en
# vanlig utvecklingsserver på 3000.
PORT = int(os.environ.get("PORT", 3000))
# The only port /proxy will forward to (see start.sh: LITERT_PORT).
LLM_PORT = int(os.environ.get("LITERT_PORT", 9379))

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

        # Restrict target URL to the local LLM endpoint (http/https localhost).
        # The port must be stated explicitly: allowing a missing port let
        # `http://localhost/...` (port 80/443) through the check that the error
        # message below claims to enforce.
        parsed_target = urllib.parse.urlparse(target_url)
        if (
            parsed_target.scheme not in ('http', 'https')
            or parsed_target.hostname not in ('localhost', '127.0.0.1')
            or parsed_target.port != LLM_PORT
        ):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(
                f'Forbidden: Proxy target must be localhost:{LLM_PORT}'.encode('utf-8')
            )
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

        headers_sent = False
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                self.send_response(response.status)
                # Content-Length droppas: vi vet inte längden i förväg när vi
                # strömmar, och HTTP/1.0-svar avslutas ändå av connection close.
                for key, val in response.headers.items():
                    if key.lower() not in ['content-length', 'connection', 'transfer-encoding']:
                        self.send_header(key, val)
                self.end_headers()
                headers_sent = True
                # Skriv vidare chunk för chunk i stället för att buffra hela
                # svaret — annars kan klienten inte se en SSE-delta förrän
                # genereringen är klar, och strömningen är meningslös.
                #
                # read1, inte read: read(n) är "läs tills du har n byte eller
                # strömmen tar slut" och blockerar alltså tills flera SSE-event
                # har hunnit samlas. litert-lms event är ~200 byte, så
                # read(1024) buntade ihop 4–5 tokens per utskrivning. Uppmätt
                # mot den riktiga kedjan (sv-multi, 5 anrop vardera): första
                # eventet 488 ms med read(1024) mot 420 ms med read1, och
                # första hela meningen 912 ms mot 809 ms. read1 returnerar det
                # som redan ligger i bufferten, så varje event går vidare i
                # samma ögonblick det kommer in.
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            print(f"[Proxy Error] HTTP Error {e.code}: {e.reason}")
            try:
                res_body = e.read()
            except Exception:
                res_body = str(e).encode('utf-8')
            self._proxy_error(headers_sent, e.code, res_body)
        except Exception as e:
            print(f"[Proxy Error] Exception: {e}")
            self._proxy_error(headers_sent, 500, str(e).encode('utf-8'))

    def _proxy_error(self, headers_sent, status, body):
        """Rapporterar ett proxyfel utan att skada ett redan påbörjat svar.

        Två fällor som båda utlöses först när strömningen har börjat:
        statusraden är redan skickad, så ett andra `send_response` skulle
        injicera "HTTP/1.0 500 ..." mitt i SSE-kroppen och förvirra klientens
        parser; och felet är ofta just att klienten gick sin väg
        (BrokenPipeError/ConnectionResetError), så själva felskrivningen
        kastar igen och eskalerar till en traceback ur handlern.
        """
        try:
            if not headers_sent:
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)
            else:
                # Kroppen är redan påbörjad. Det enda ärliga vi kan göra är
                # att avsluta strömmen; klienten ser ett trunkerat svar.
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

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
            other_language = data.get('other_language')
            auto_language = bool(data.get('auto_language', False))
            # Interim requests paint live text while someone is still speaking.
            # They are decorative; the turn does not wait on them.
            interim = bool(data.get('interim', False))
            raw_data = base64.b64decode(audio_b64)

            # The browser sends a raw Float32Array buffer (16 kHz mono).
            audio_np = np.frombuffer(raw_data, dtype=np.float32).copy()
            duration_s = len(audio_np) / 16000.0
            peak = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
            rms = float(np.sqrt(np.mean(audio_np ** 2))) if len(audio_np) else 0.0
            print(
                f"[STT] audio: {duration_s:.2f}s, peak={peak:.4f}, rms={rms:.4f}, "
                f"lang={language} auto={auto_language}",
                flush=True,
            )

            # Interims are the reason live translation collapsed under its own
            # load. They fire every INTERIM_MS while someone talks, so a 4.3 s
            # utterance produced ~3.7 of them plus the real transcription — and
            # every one was a full 30 s-padded encoder pass on the lane model.
            # Over the café conversation that is 33 calls x ~1.1 s = 36 s of STT
            # for 34 s of speech: utilisation 1.07, so the queue grew without
            # bound and each turn's latency was the sum of every turn before it
            # (measured in the browser: keyup->stt 4 s, 6 s, 8 s, 15 s, ... 28 s).
            #
            # No amount of prioritising fixes an oversubscribed queue, so the
            # offered load has to come down. Two measures, both here:
            #   * `fast` decodes interims on the resident base checkpoint
            #     (~0.27 s against ~1.1 s), taking utilisation to ~0.43;
            #   * a busy interim is dropped outright rather than queued.
            # The second matters because aborting the fetch client-side does not
            # stop this handler — it would keep the lock and transcribe for a
            # reader that has gone away. Skipping a preview costs a repaint;
            # running it costs a turn.
            if interim and not _stt_lock.acquire(blocking=False):
                print("[STT] interim skipped (busy)", flush=True)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(
                    json.dumps({"text": "", "language": language, "skipped": True}).encode("utf-8")
                )
                return

            if interim:
                try:
                    text, resolved = transcribe(
                        audio_np,
                        language,
                        other_language=other_language,
                        auto_language=auto_language,
                        fast=True,
                    )
                finally:
                    _stt_lock.release()
            else:
                with _stt_lock:
                    text, resolved = transcribe(
                        audio_np,
                        language,
                        other_language=other_language,
                        auto_language=auto_language,
                    )
            print(f"[STT] Transcribed: {text!r} (lang={resolved})", flush=True)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(
                json.dumps({"text": text, "language": resolved}).encode("utf-8")
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[STT Error] Exception: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def handle_prewarm(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        lang = query.get("lang", [None])[0]
        if not lang or (
            lang not in SUPPORTED_STT_LANGS and lang not in PIPER_VOICE_MAP
        ):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Error: Missing or unsupported "lang" parameter.')
            return
        try:
            if lang in SUPPORTED_STT_LANGS:
                get_whisper_model(stt_model_for(lang))
            if lang in PIPER_VOICE_MAP:
                get_piper_voice(lang)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "lang": lang}).encode("utf-8"))
        except Exception as e:
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))

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

        if self.path.startswith('/api/prewarm'):
            self.handle_prewarm()
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
                print("[Prewarm] Loading language-id model...", flush=True)
                get_lid_model()
                for lang in PREWARM_LANGS:
                    model_id = stt_model_for(lang)
                    print(f"[Prewarm] Loading STT {lang} -> {model_id}...", flush=True)
                    get_whisper_model(model_id)
                    print(f"[Prewarm] Loading {lang} Piper voice into memory...", flush=True)
                    get_piper_voice(lang)
                print("[Prewarm] Models pre-warmed successfully.", flush=True)
            except Exception as e:
                print(f"[Prewarm Error] {e}", flush=True)

        threading.Thread(target=_prewarm_models, daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
