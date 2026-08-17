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

"""Python-kopior av logik som egentligen bor i frontend.

Bench måste skicka exakt samma payload som appen skickar, annars mäter vi
något annat än produkten. Allt som dupliceras ligger därför här, i en enda
fil, så att det finns precis ett ställe att synka när frontend ändras.

Speglar:
  frontend/src/utils/api.js       splitTextIntoSpeechChunks
  frontend/src/utils/api.js       generatePayloadJSON
  frontend/src/utils/api.js       parseTranslation
  frontend/src/utils/api.js       envelope-vakten i translateTextStreaming
  frontend/src/utils/api.js       isBackchannel / endsWithContinuationCue /
                                  isRepairUtterance / stripRepairCue /
                                  normalizeSttText / routeSpokenTurn
  frontend/src/utils/audioHelpers.js  farEndSampleIndex
  frontend/src/utils/audioHelpers.js  resample
  frontend/src/TranslatorApp.jsx  systemprompten i processTranslation
  frontend/src/TranslatorApp.jsx  speakCompleteSentences
  frontend/src/hooks/useVoiceActivity.js  energi-VAD:en (segmentering)

Ändras någon av dem måste den här filen ändras i samma commit.
"""

import json
import math
import re

import numpy as np

SPEECH_CHUNK_LIMIT = 180


def is_speakable(text):
    """True när texten innehåller minst en bokstav. Port av isSpeakable."""
    return bool(re.search(r"[^\W\d_]", text or "", re.UNICODE))


_WHISPER_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def normalize_stt_text(text):
    """Port av normalizeSttText. Tar bort Whisper-specialtokens som <|nospeech|>."""
    cleaned = _WHISPER_TOKEN_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


CONTINUE_WINDOW_MS = 1500
REPAIR_WINDOW_MS = 4000
CONTINUATION_HOLD_MS = 1400

_BACKCHANNEL_RE = re.compile(
    r"^(m+h?m+|uh+|u+m+|öh+|äh+|eh+|euh+|hmm+|huh+|tsk+)$",
    re.IGNORECASE | re.UNICODE,
)


def is_backchannel(text):
    """Port av isBackchannel — korta fyllnadsljud, inte svar som 'ja'/'yes'."""
    stripped = re.sub(r"[.,!?…]+$", "", (text or "").strip())
    return bool(stripped) and bool(_BACKCHANNEL_RE.match(stripped))


# Only conjunctions people actually trail off on. "så"/"that"/"att" are
# ordinary sentence endings and must not delay translation.
CONTINUATION_CUES = frozenset(
    {
        "och",
        "and",
        "men",
        "but",
        "or",
        "eller",
        "y",
        "et",
        "mais",
    }
)


def route_spoken_turn(detected_code, active_lane, src, dst, lang1, lang2):
    """Port av routeSpokenTurn — attribute the turn to whichever lane language was spoken."""
    del lang1, lang2  # reserved for the JS signature; unused once src/dst are the pair
    code = (detected_code or "").split("-")[0].lower()
    if code and code == dst.get("code"):
        other = 2 if active_lane == 1 else 1
        return {"lane": other, "src": dst, "dst": src, "flipped": True}
    return {"lane": active_lane, "src": src, "dst": dst, "flipped": False}


def far_end_sample_index(elapsed_sec, mic_index, tts_rate, mic_rate):
    """Port av farEndSampleIndex — map mic-frame time onto the TTS PCM clock."""
    if not mic_rate:
        return math.floor(elapsed_sec * tts_rate) + int(mic_index)
    return math.floor(elapsed_sec * tts_rate + mic_index * (tts_rate / mic_rate))


def ends_with_continuation_cue(text):
    """Port av endsWithContinuationCue."""
    words = re.sub(r"[,:]+$", "", (text or "").strip()).split()
    if not words:
        return False
    last = re.sub(r"[.,!?…]+$", "", words[-1].lower())
    return last in CONTINUATION_CUES


_REPAIR_CUE_RE = re.compile(
    r"^(nej|no|non|wait|alltså|sorry|jag menade|i mean|en fait|o sea)\b[,.!?]*\s*",
    re.IGNORECASE | re.UNICODE,
)


def is_repair_utterance(text):
    """Port av isRepairUtterance."""
    return bool(_REPAIR_CUE_RE.match((text or "").strip()))


def strip_repair_cue(text):
    """Port av stripRepairCue. Kort rest (t.ex. bara 'Nej') lämnas orörd."""
    stripped = (text or "").strip()
    rest = _REPAIR_CUE_RE.sub("", stripped, count=1).strip()
    return stripped if len(rest) < 3 else rest


def split_text_into_speech_chunks(text, limit=SPEECH_CHUNK_LIMIT):
    """Ordsäker chunkning så varje /api/tts-anrop håller sig under `limit`.

    Port av splitTextIntoSpeechChunks. JS delar på /\\s+/ och trimmar, vilket
    innebär att tom sträng ger noll chunkar och att ett enskilt ord längre än
    gränsen släpps igenom helt. Punctuation-only (t.ex. ".") ger noll chunkar
    — Piper vägrar syntetisera dem.
    """
    if not is_speakable(text):
        return []
    words = re.split(r"\s+", text)
    chunks = []
    current = ""
    for word in words:
        if len((current + " " + word).strip()) <= limit:
            current = (current + " " + word).strip()
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return [c for c in chunks if is_speakable(c)]


def system_prompt(src_name, dst_name):
    """Speglar prompten produkten faktiskt skickar (processTranslation).

    Detta är standardprompten i bench, precis som i appen. Byts prompten i
    TranslatorApp.jsx måste den bytas här i samma commit — annars mäter bench
    en prompt som inte finns, och promptlängd är enligt kampanjens egen
    mätning en av de största latensfaktorerna på den här modellen (README,
    "kontextklippan" vid ~725 tecken total kontext).
    """
    src = src_name.split(" ")[0]
    dst = dst_name.split(" ")[0]
    return (
        f"Translate the user's text from {src} into {dst}. "
        f"Reply with the translation only — no explanations, no alternatives, "
        f"no quotes, no preamble."
    )


def system_prompt_json(src_name, dst_name):
    """Den gamla JSON-wrapper-prompten. Produkten skickar den INTE längre.

    Kvar enbart som mätvariant (`--prompt json` / `--ab-prompt json`), så att
    Task 12:s mätning går att reproducera och så att kontextklippan går att
    demonstrera igen. `parse_translation` tolererar fortfarande svaret, så en
    körning med den här varianten mäter samma kedja.
    """
    src = src_name.split(" ")[0]
    dst = dst_name.split(" ")[0]
    return (
        f"You are a high-performance translator. Your task is to translate text "
        f"from {src} into {dst}.\n"
        f"You MUST format your response as a valid JSON object matching this "
        f"structure:\n"
        f"{{\n"
        f'  "translation": "High-quality, natural translation into {dst}"\n'
        f"}}\n"
        f"Do NOT return anything else except this JSON object. No Markdown block "
        f"wraps (no ```json), no introductory text, no conversational text. Start "
        f'directly with "{{" and end directly with "}}".'
    )


# Namngivna promptvarianter. "plain" är produktens prompt och bench:s default.
PROMPT_VARIANTS = {"plain": system_prompt, "json": system_prompt_json}
DEFAULT_PROMPT = "plain"


SENTENCE_ENDS = (".", "!", "?", "…")


def looks_like_json_envelope(text):
    """Speglar envelope-vakten i translateTextStreaming (api.js).

    En delvis mottagen `{"translation": "Hej` går inte att tolka, så appen
    skickar inte partiella deltan vidare alls när svaret ser ut att börja på
    den gamla JSON-wrappern — den väntar tills hela svaret är parsat. Bench
    måste göra samma bedömning, annars påstår mätningen att en mening kunde
    talas vid ett tillfälle då produkten hade varit tyst.
    """
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("```")


def first_sentence_end(text, spoken_chars=0):
    """Slutindex (exklusivt) för den sist färdiga meningen, annars None.

    Speglar `speakCompleteSentences` i TranslatorApp.jsx: appen använder
    `lastIndexOf` per skiljetecken och talar allt fram till och med det sista
    av dem. Den avfyrar alltså på det första delta som *innehåller* ett
    skiljetecken, inte bara på ett delta som *slutar* på ett. Bench använde
    tidigare `endswith`, vilket bara råkar avfyra när litert-lm skickar
    skiljetecknet som ett eget event — på sv-multi (första meningen slutar vid
    tecken 43 av 165) hamnade den mätta meningsgränsen på 4726 ms av ett
    6016 ms långt svar i stället för strax efter första meningen.

    Returnerar None när ingen ny färdig mening finns, eller när det som skulle
    talas bara är blanktecken (appens `if (!ready) return`).
    """
    if looks_like_json_envelope(text):
        return None
    last_end = max(text.rfind(mark) for mark in SENTENCE_ENDS)
    if last_end < spoken_chars:
        return None
    upto = last_end + 1
    if not text[spoken_chars:upto].strip():
        return None
    return upto


def build_llm_payload(text, model, system_prompt_text):
    """Speglar generatePayloadJSON (api.js:72-83)."""
    messages = []
    if system_prompt_text and system_prompt_text.strip():
        messages.append({"role": "system", "content": system_prompt_text.strip()})
    messages.append({"role": "user", "content": text})
    return {"model": model or "gemma4-e2b", "messages": messages}


def _reject_js_nonstandard_constant(constant):
    """JSON.parse godtar inte NaN/Infinity/-Infinity — det gör Pythons json."""
    raise ValueError(f"JSON.parse rejects {constant}")


def parse_translation(model_response):
    """Speglar JSON-uttolkningen i translateText (api.js:126-144).

    Modellen ombeds svara med ett naket {"translation": ...}, men appen
    tolererar ```-staket och faller tillbaka på rå text om parsningen
    misslyckas. Bench måste falla tillbaka likadant, annars mäter vi WER på
    en sträng som appen aldrig hade visat.

    JS gör `parsed.translation || ""` inuti try-blocket, vilket ger tre
    beteenden som en naiv `.get("translation", "")` inte återger:

      {"translation": null}  ->  ""   (null är falsy, inte "nyckel saknas")
      5 / [1,2,3] / "hej"    ->  ""   (property-access ger undefined)
      null                   ->  rå   (null.translation kastar TypeError,
                                       som fångas av samma catch som ett
                                       parse-fel)

    Bara ett äkta parse-fel (ValueError) faller alltså tillbaka på rå text.
    """
    clean = model_response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        parsed = json.loads(clean, parse_constant=_reject_js_nonstandard_constant)
    except ValueError:
        return model_response

    if parsed is None:
        # `null.translation` kastar TypeError inne i JS:ens try-block, så
        # appen hamnar i catch och visar rå text.
        return model_response
    if not isinstance(parsed, dict):
        return ""
    # `or ""` speglar JS:ens `|| ""` för null, "", 0 och false. JS och Python
    # är oense om [] och {} (truthy i JS, falsy här), men en tom array/objekt
    # renderas ändå som ingenting i UI:t, så "" är rätt sträng att mäta mot.
    return parsed.get("translation") or ""


# ---------------------------------------------------------------------------
# Energi-VAD:en ur useVoiceActivity.js
#
# Konstanterna måste vara identiska med hookens, annars segmenterar bench ett
# annat samtal än produkten hör. Testet i test_frontend_contract.py läser JS-
# källan och fäller om någon av dem glider isär.
# ---------------------------------------------------------------------------

SPEECH_RMS = 0.015
# En enda hangover — se kommentaren i useVoiceActivity.js för mätningen bakom
# värdet och varför den gamla tvånivåregeln (560/1250 ms) togs bort.
SILENCE_MS = 700
MIN_SPEECH_MS = 400
# Tystnad efter vilken yttrandet transkriberas spekulativt, medan hangovern
# fortfarande löper. Se useVoiceActivity.js för resonemanget.
SPECULATIVE_STT_MS = 340
MAX_UTTERANCE_MS = 15000
PRE_ROLL_CHUNKS = 4
# createScriptProcessor(4096, 1, 1) i startListening.
VAD_FRAME = 4096
# Webbläsarens AudioContext körs typiskt i 48 kHz; frame-granulariteten (85 ms)
# påverkar var VAD:en klipper, så simuleringen måste köra i samma takt.
BROWSER_SAMPLE_RATE = 48000
STT_SAMPLE_RATE = 16000


def rms_of(frame):
    """Port av rmsOf."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def resample(samples, source_rate, target_rate):
    """Port av resample() i audioHelpers.js — linjär interpolation, samma längdformel."""
    samples = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate:
        return samples
    ratio = source_rate / target_rate
    new_length = int(round(len(samples) / ratio))
    if new_length <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.arange(new_length, dtype=np.float64) * ratio
    index = np.floor(positions).astype(np.int64)
    fraction = positions - index
    index = np.clip(index, 0, len(samples) - 1)
    nxt = np.clip(index + 1, 0, len(samples) - 1)
    current = samples[index]
    return (current + fraction * (samples[nxt] - current)).astype(np.float32)


def segment_utterances(samples, sample_rate=BROWSER_SAMPLE_RATE, silence_hold_ms=0):
    """Kör VAD:en över en sammanhängande mikrofonström och returnerar yttranden.

    Speglar handleAudioProcess/finishCapture i useVoiceActivity.js, inklusive
    pre-roll, den adaptiva brusgolvströskeln och den kortare tystnadsgränsen
    efter ett tydligt avslut (`abrupt`). `now` är frame-callbackens tid, alltså
    slutet på framen — precis som performance.now() i hooken.

    Returnerar en lista dictar med start/slut i sekunder (mätt på det ljud som
    faktiskt fångades, pre-roll inräknad) och själva samplen.
    """
    samples = np.asarray(samples, dtype=np.float32)
    frame_ms = VAD_FRAME / sample_rate * 1000.0

    capturing = False
    armed = True
    pre_roll = []
    chunks = []
    noise_floor = 0.006
    speech_started_at = 0.0
    speech_started_frame = 0
    last_loud_at = 0.0
    captured = []

    total_frames = len(samples) // VAD_FRAME
    for f in range(total_frames):
        frame = samples[f * VAD_FRAME : (f + 1) * VAD_FRAME]
        rms = rms_of(frame)
        now = (f + 1) * frame_ms

        if not capturing:
            pre_roll.append(f)
            if len(pre_roll) > PRE_ROLL_CHUNKS:
                pre_roll.pop(0)
            if not armed:
                continue
            threshold = max(SPEECH_RMS * 0.7, noise_floor * 3.5)
            if rms < threshold:
                noise_floor = noise_floor * 0.98 + rms * 0.02
                continue
            capturing = True
            speech_started_at = now
            last_loud_at = now
            chunks = list(pre_roll)
            speech_started_frame = chunks[0]
            pre_roll = []
            continue

        chunks.append(f)
        if rms >= SPEECH_RMS:
            last_loud_at = now

        uttered_ms = now - speech_started_at
        silent_ms = now - last_loud_at
        silence_need = max(SILENCE_MS, silence_hold_ms)
        if uttered_ms >= MAX_UTTERANCE_MS or silent_ms >= silence_need:
            capturing = False
            # Onset -> sista höga framen, inte -> now: annars ingår hela
            # hangovern och MIN_SPEECH_MS kan aldrig falla ut (se finishCapture).
            duration_ms = max(0.0, last_loud_at - speech_started_at)
            if chunks and duration_ms >= MIN_SPEECH_MS:
                start = speech_started_frame * VAD_FRAME
                stop = (chunks[-1] + 1) * VAD_FRAME
                captured.append(
                    {
                        "start_s": start / sample_rate,
                        "end_s": stop / sample_rate,
                        # Tiden VAD:en stänger yttrandet på — det är här
                        # klockan för time-to-first-audio startar (marks.keyup).
                        "closed_at_s": now / 1000.0,
                        "samples": samples[start:stop],
                    }
                )
            chunks = []
            pre_roll = []

    # Strömmen tog slut mitt i ett yttrande: appen hade fortsatt lyssna, men
    # för en färdiginspelad konversation är det rätt att stänga det ändå.
    if capturing and chunks:
        now = total_frames * frame_ms
        if max(0.0, last_loud_at - speech_started_at) >= MIN_SPEECH_MS:
            start = speech_started_frame * VAD_FRAME
            stop = (chunks[-1] + 1) * VAD_FRAME
            captured.append(
                {
                    "start_s": start / sample_rate,
                    "end_s": stop / sample_rate,
                    "closed_at_s": now / 1000.0,
                    "samples": samples[start:stop],
                }
            )
    return captured
