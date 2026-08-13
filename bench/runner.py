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

"""Kör en fixtur genom hela HTTP-kedjan och tar tid på varje steg.

Anropen speglar exakt vad frontend gör: base64-kodad Float32-PCM till
/api/stt, LLM-anropet genom /proxy, och ett /api/tts-anrop per chunk.
Att gå via de riktiga endpointsen är hela poängen — base64-overheaden,
proxyhoppet och låsen i server.py är latenskällor vi vill kunna se.
"""

import base64
import time
import urllib.parse
from dataclasses import dataclass, field

import requests

from bench.fixtures import check_duration, ensure_wav, load_pcm_16k
from bench.frontend_mirror import (
    build_llm_payload,
    parse_translation,
    split_text_into_speech_chunks,
    system_prompt,
)

# Namnen frontend skickar in i prompten (AVAILABLE_LANGUAGES i TranslatorApp).
LANGUAGE_NAMES = {"sv": "Swedish", "en": "English", "fi": "Finnish"}

REQUEST_TIMEOUT = 300


@dataclass
class RunResult:
    fixture_id: str
    ok: bool = True
    error: str = ""
    stt_ms: float = 0.0
    llm_ms: float = 0.0
    tts_first_ms: float = 0.0
    tts_rest_ms: float = 0.0
    transcript: str = ""
    translation: str = ""
    chunk_count: int = 0
    upload_bytes: int = 0

    @property
    def time_to_first_audio_ms(self):
        return self.stt_ms + self.llm_ms + self.tts_first_ms

    @property
    def wall_total_ms(self):
        return self.time_to_first_audio_ms + self.tts_rest_ms


def _post_stt(api_base, samples, language):
    payload_b64 = base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii")
    started = time.perf_counter()
    response = requests.post(
        f"{api_base}/api/stt",
        json={"audio_base64": payload_b64, "language": language},
        timeout=REQUEST_TIMEOUT,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json().get("text", ""), elapsed_ms, len(payload_b64)


def _post_llm(api_base, llm_url, model, text, src_lang, dst_lang, prompt_fn=system_prompt):
    prompt = prompt_fn(LANGUAGE_NAMES[src_lang], LANGUAGE_NAMES[dst_lang])
    payload = build_llm_payload(text, model, prompt)
    proxied = f"{api_base}/proxy?url={urllib.parse.quote(llm_url, safe='')}"
    started = time.perf_counter()
    response = requests.post(proxied, json=payload, timeout=REQUEST_TIMEOUT)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    return parse_translation(raw), elapsed_ms


def _get_tts(api_base, text, language):
    started = time.perf_counter()
    response = requests.get(
        f"{api_base}/api/tts",
        params={"text": text, "lang": language},
        timeout=REQUEST_TIMEOUT,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return elapsed_ms


def run_fixture(api_base, llm_url, model, fixture_id, spec, prompt_fn=system_prompt):
    result = RunResult(fixture_id=fixture_id)
    try:
        samples = load_pcm_16k(ensure_wav(fixture_id, spec))
        # Fäller om någon redigerat fixturtexten utan att uppdatera target_s.
        # Whispers paddningsbeteende är längdkänsligt, så en fixtur som tyst
        # bytt längd skulle göra jämförelser mot äldre körningar meningslösa.
        check_duration(fixture_id, spec, samples)

        result.transcript, result.stt_ms, result.upload_bytes = _post_stt(
            api_base, samples, spec["lang"]
        )
        if not result.transcript.strip():
            raise ValueError("STT gav tom transkription")

        result.translation, result.llm_ms = _post_llm(
            api_base,
            llm_url,
            model,
            result.transcript,
            spec["lang"],
            spec["target"],
            prompt_fn,
        )

        chunks = split_text_into_speech_chunks(result.translation)
        result.chunk_count = len(chunks)
        if not chunks:
            raise ValueError("Översättningen gav noll TTS-chunkar")

        result.tts_first_ms = _get_tts(api_base, chunks[0], spec["target"])
        result.tts_rest_ms = sum(
            _get_tts(api_base, chunk, spec["target"]) for chunk in chunks[1:]
        )
    except Exception as err:  # noqa: BLE001 — en trasig fixtur får inte dölja de andra
        result.ok = False
        result.error = f"{type(err).__name__}: {err}"
    return result


def warmup(api_base, llm_url, model, fixtures, prompt_fn=system_prompt):
    """Kör en runda per språkpar utan att mäta.

    Första anropet per Piper-röst laddar modellen från disk, och första
    LLM-anropet kan behöva ladda vikter. Utan uppvärmning hamnar den
    kostnaden i första fixturens siffror.
    """
    seen = set()
    for fixture_id, spec in fixtures.items():
        pair = (spec["lang"], spec["target"])
        if pair in seen:
            continue
        seen.add(pair)
        print(f"[warmup] {fixture_id} ({spec['lang']}→{spec['target']})", flush=True)
        result = run_fixture(api_base, llm_url, model, fixture_id, spec, prompt_fn)
        if not result.ok:
            raise RuntimeError(f"Uppvärmningen misslyckades för {fixture_id}: {result.error}")
