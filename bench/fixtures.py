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

"""Syntetiserar och cachar testljud.

Fixturerna görs med samma Piper-röster som produkten använder, vilket ger
deterministiskt ljud och ett WER-facit gratis: facit är texten vi matade in.
Filerna cachas på disk och regenereras bara när de saknas.
"""

import json
import pathlib
import sys
import wave

import numpy as np

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
FIXTURE_DIR = BENCH_DIR / "fixtures"
FIXTURE_SPEC = BENCH_DIR / "fixtures.json"

TARGET_SAMPLE_RATE = 16000
# Whisper vill ha 16 kHz; Piper levererar 22,05 kHz för medium-rösterna.
DURATION_TOLERANCE = 0.5  # ±50 % mot target_s


def _import_backend():
    """Importerar backend/server.py utan att starta servern.

    server.py startar bara lyssnaren under `if __name__ == '__main__'`, så en
    vanlig import ger oss synthesize() och röstkartan utan sidoeffekter.
    """
    backend_dir = str(REPO_DIR / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import server

    return server


def load_fixtures():
    with open(FIXTURE_SPEC, encoding="utf-8") as handle:
        return json.load(handle)


def _resample(samples, source_rate, target_rate):
    if source_rate == target_rate:
        return samples.astype(np.float32)
    duration = len(samples) / source_rate
    target_length = int(duration * target_rate)
    source_positions = np.arange(len(samples), dtype=np.float64) / source_rate
    target_positions = np.arange(target_length, dtype=np.float64) / target_rate
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def ensure_wav(fixture_id, spec):
    """Returnerar sökvägen till fixturens wav, och genererar den om den saknas."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{fixture_id}.wav"
    if path.exists():
        return path

    server = _import_backend()
    print(f"[fixtures] Syntetiserar {fixture_id} ({spec['lang']})...", flush=True)
    samples, sample_rate = server.synthesize(spec["text"], spec["lang"])
    samples = _resample(np.asarray(samples, dtype=np.float32), sample_rate, TARGET_SAMPLE_RATE)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())
    return path


def load_pcm_16k(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != TARGET_SAMPLE_RATE:
            raise ValueError(f"{path} har {handle.getframerate()} Hz, förväntade {TARGET_SAMPLE_RATE}")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def check_duration(fixture_id, spec, samples):
    """Fäller om en fixtur inte längre har den längd den utger sig för att ha.

    Redigerar någon texten utan att uppdatera target_s mäter vi plötsligt en
    annan ljudlängd än vi tror, och Whispers padding-beteende är längdkänsligt.
    """
    duration = len(samples) / TARGET_SAMPLE_RATE
    target = spec["target_s"]
    low, high = target * (1 - DURATION_TOLERANCE), target * (1 + DURATION_TOLERANCE)
    if not low <= duration <= high:
        raise ValueError(
            f"{fixture_id}: {duration:.2f}s ligger utanför {low:.2f}–{high:.2f}s "
            f"(target_s={target}). Justera texten eller target_s i fixtures.json."
        )
    return duration


if __name__ == "__main__":
    # `venv/bin/python3 -m bench.fixtures` genererar allt och skriver längderna.
    for fixture_id, spec in load_fixtures().items():
        path = ensure_wav(fixture_id, spec)
        samples = load_pcm_16k(path)
        duration = len(samples) / TARGET_SAMPLE_RATE
        print(f"{fixture_id:12s} {duration:5.2f}s  (target_s={spec['target_s']})")
