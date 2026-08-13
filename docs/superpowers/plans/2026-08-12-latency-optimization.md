# Latensoptimering — implementationsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bygg en mät-harness som kör hela översättningskedjan över HTTP, ta en baseline, och genomför sedan nio optimeringar där varje enskild mäts mot baseline på både latens och kvalitet.

**Architecture:** Ett fristående `bench/`-paket talar HTTP med en egen backend-instans (port 3100) mot den redan körande litert-lm (port 9379). Fixturer syntetiseras med Piper så att facit är gratis och körningarna deterministiska. All logik som måste dubbleras från frontend samlas i en enda fil, `bench/frontend_mirror.py`, så att det finns ett ställe att synka. Optimeringarna läggs in en och en, med en bench-körning per commit.

**Tech Stack:** Python 3.13 (venv finns redan i workspacen), stdlib `unittest` för tester, `requests`/`numpy`/`piper-tts`/`faster-whisper` från `backend/requirements.txt`. Inga nya beroenden.

## Global Constraints

- **Inga nya Python-beroenden.** Allt som behövs finns i `backend/requirements.txt`. Levenshtein skrivs för hand.
- **Testramverk: stdlib `unittest`.** Repot har ingen testsvit idag och vi inför inget nytt beroende. Tester ligger i `bench/tests/test_*.py` och körs med `venv/bin/python3 -m unittest discover -s bench/tests -t .`
- **Python-interpretern är `venv/bin/python3`** — aldrig `python3` (systemets saknar beroendena).
- **Backend under mätning kör på port 3100.** Port 3000 är upptagen av workspacen `rio-de-janeiro` och får inte störas.
- **litert-lm på port 9379 återanvänds** och startas aldrig av bench.
- **Modell-ID är `gemma4-e2b`** i alla körningar utom optimering 9, som använder `gemma4-e2b,gpu`.
- **`*.wav` är gitignorerat** — genererade fixturer committas aldrig. `bench/results/*.json` committas.
- **Alla filer som skapas i `bench/` får samma Apache-2.0-huvud** som resten av repot (se `backend/server.py` rad 1–13, med `# Copyright 2026 Google LLC`).
- **Språkval i fixturerna: sv→en, fi→en, en→sv.** Detta är medvetet: `MAX_MODELS = 2` i `backend/server.py` cachar bara två Piper-röster, och de tre paren behöver exakt två målröster (en, sv). Byter man målspråk börjar röstcachen vräka ut modeller och TTS-siffrorna mäter modelladdning i stället för syntes.
- **Commit-meddelanden på engelska**, i imperativ, som repots historik.

---

## Filstruktur

| Fil | Ansvar |
| :--- | :--- |
| `bench/fixtures.json` | Fixturdefinitioner: text, källspråk, målspråk, förväntad längd |
| `bench/frontend_mirror.py` | Allt som dubbleras från frontend: TTS-chunkning, systemprompt, LLM-payload |
| `bench/wer.py` | Normalisering och ordvis WER |
| `bench/fixtures.py` | Syntetiserar och cachar fixtur-wav, laddar dem som 16 kHz float32 |
| `bench/runner.py` | HTTP-anropen mot backend, med tidtagning per steg |
| `bench/report.py` | Median/min/max, delta mot baseline, markdown-tabell, kvalitetsgrind |
| `bench/bench.py` | CLI: startar backend, kör fixturer, skriver resultat, sätter exitkod |
| `bench/tests/test_*.py` | Enhetstester för de rena modulerna |
| `bench/results/*.json` | Committad historik, en fil per körning |
| `backend/server.py` | Ändras: `PORT` läses från env |
| `frontend/src/utils/api.js` | Ändras i optimering 4, 5, 8 |
| `frontend/src/TranslatorApp.jsx` | Ändras i optimering 4, 6, 8 samt instrumenteringen |
| `frontend/src/hooks/useAudioRecorder.js` | Ändras i optimering 7 |

`runner.py`, `report.py` och `fixtures.py` hålls åtskilda för att de ändras av olika skäl: `runner` när protokollet ändras (optimering 8), `report` aldrig, `fixtures` bara om testmaterialet ändras.

---

## Task 1: `bench/frontend_mirror.py`

**Files:**
- Create: `bench/frontend_mirror.py`
- Create: `bench/tests/test_frontend_mirror.py`
- Reference: `frontend/src/utils/api.js:154-168` (`splitTextIntoSpeechChunks`), `frontend/src/utils/api.js:72-83` (`generatePayloadJSON`), `frontend/src/TranslatorApp.jsx:226-230` (systemprompten)

**Interfaces:**
- Consumes: inget
- Produces:
  - `split_text_into_speech_chunks(text: str, limit: int = 180) -> list[str]`
  - `system_prompt(src_name: str, dst_name: str) -> str`
  - `build_llm_payload(text: str, model: str, system_prompt: str) -> dict`
  - `parse_translation(model_response: str) -> str`

Bench måste skicka exakt samma payload som appen skickar, annars mäter vi något annat än produkten. Den här filen är det enda stället där frontend-logik dupliceras, och dess huvudkommentar ska peka ut vilka JS-rader den speglar.

- [ ] **Step 1: Write the failing test**

Skapa `bench/tests/test_frontend_mirror.py`:

```python
import unittest

from bench.frontend_mirror import (
    build_llm_payload,
    parse_translation,
    split_text_into_speech_chunks,
    system_prompt,
)


class TestSplitTextIntoSpeechChunks(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(
            split_text_into_speech_chunks("Var ligger stationen?"),
            ["Var ligger stationen?"],
        )

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(split_text_into_speech_chunks(""), [])

    def test_splits_on_word_boundary_under_limit(self):
        text = " ".join(["ord"] * 10)  # 10 * 4 - 1 = 39 tecken
        chunks = split_text_into_speech_chunks(text, limit=20)
        self.assertEqual(chunks, ["ord ord ord ord ord", "ord ord ord ord ord"])
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)

    def test_word_longer_than_limit_becomes_its_own_chunk(self):
        # Speglar JS-beteendet: ett enskilt ord delas aldrig, även om det
        # spränger gränsen. Porten måste bete sig likadant.
        chunks = split_text_into_speech_chunks("kort " + "x" * 30, limit=10)
        self.assertEqual(chunks, ["kort", "x" * 30])

    def test_collapses_runs_of_whitespace(self):
        self.assertEqual(
            split_text_into_speech_chunks("ett   två\n\ttre"),
            ["ett två tre"],
        )


class TestSystemPrompt(unittest.TestCase):
    def test_uses_first_word_of_language_names(self):
        prompt = system_prompt("Swedish (Source)", "English (Translation)")
        self.assertIn("from Swedish into English", prompt)
        self.assertNotIn("(Source)", prompt)

    def test_demands_bare_json_object(self):
        prompt = system_prompt("Swedish", "English")
        self.assertIn('"translation"', prompt)


class TestBuildLlmPayload(unittest.TestCase):
    def test_system_message_precedes_user_message(self):
        payload = build_llm_payload("hej", "gemma4-e2b", "SYS")
        self.assertEqual(payload["model"], "gemma4-e2b")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hej"},
            ],
        )

    def test_blank_system_prompt_is_omitted(self):
        payload = build_llm_payload("hej", "gemma4-e2b", "   ")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hej"}])


class TestParseTranslation(unittest.TestCase):
    def test_extracts_field_from_bare_json(self):
        self.assertEqual(parse_translation('{"translation": "Where is it?"}'), "Where is it?")

    def test_tolerates_json_code_fence(self):
        raw = '```json\n{"translation": "Where is it?"}\n```'
        self.assertEqual(parse_translation(raw), "Where is it?")

    def test_falls_back_to_raw_text_when_not_json(self):
        self.assertEqual(parse_translation("Where is it?"), "Where is it?")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: FAIL med `ModuleNotFoundError: No module named 'bench.frontend_mirror'`

- [ ] **Step 3: Write the implementation**

Skapa `bench/__init__.py` (tom, så att `bench.frontend_mirror` importeras som paket) och `bench/tests/__init__.py` (tom). Skapa sedan `bench/frontend_mirror.py`:

```python
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
  frontend/src/utils/api.js       splitTextIntoSpeechChunks, generatePayloadJSON
  frontend/src/TranslatorApp.jsx  systemprompten i processTranslation

Ändras någon av dem måste den här filen ändras i samma commit.
"""

import re

SPEECH_CHUNK_LIMIT = 180


def split_text_into_speech_chunks(text, limit=SPEECH_CHUNK_LIMIT):
    """Ordsäker chunkning så varje /api/tts-anrop håller sig under `limit`.

    Port av splitTextIntoSpeechChunks. JS delar på /\\s+/ och trimmar, vilket
    innebär att tom sträng ger noll chunkar och att ett enskilt ord längre än
    gränsen släpps igenom helt. Båda beteendena bevaras medvetet.
    """
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
    return chunks


def system_prompt(src_name, dst_name):
    """Speglar prompten som byggs i TranslatorApp.processTranslation."""
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


def build_llm_payload(text, model, system_prompt_text):
    """Speglar generatePayloadJSON."""
    messages = []
    if system_prompt_text and system_prompt_text.strip():
        messages.append({"role": "system", "content": system_prompt_text.strip()})
    messages.append({"role": "user", "content": text})
    return {"model": model or "gemma4-e2b", "messages": messages}


def parse_translation(model_response):
    """Speglar JSON-uttolkningen i translateText.

    Modellen ombeds svara med ett naket {"translation": ...}, men appen
    tolererar ```-staket och faller tillbaka på rå text om parsningen
    misslyckas. Bench måste falla tillbaka likadant, annars mäter vi WER på
    en sträng som appen aldrig hade visat.
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
        return json.loads(clean).get("translation", "")
    except (ValueError, AttributeError):
        return model_response
```

Lägg till `import json` överst i filen, bredvid `import re`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: PASS, 12 tester

- [ ] **Step 5: Verify the prompt matches the frontend byte for byte**

Det räcker inte att testerna är gröna — prompten måste vara identisk med den
i `TranslatorApp.jsx:229`, annars mäter vi en annan prefill-kostnad än appen har.

Run:
```bash
venv/bin/python3 -c "
from bench.frontend_mirror import system_prompt
print(system_prompt('Swedish (Source)', 'English (Translation)'))
"
```
Jämför utskriften rad för rad mot template-strängen i `frontend/src/TranslatorApp.jsx:229`. Skillnader i radbrytning eller blanksteg ska rättas i `frontend_mirror.py`.

- [ ] **Step 6: Commit**

```bash
git add bench/__init__.py bench/frontend_mirror.py bench/tests/
git commit -m "Add Python mirror of frontend chunking and prompt logic"
```

---

## Task 2: `bench/wer.py`

**Files:**
- Create: `bench/wer.py`
- Create: `bench/tests/test_wer.py`

**Interfaces:**
- Consumes: inget
- Produces:
  - `normalize(text: str) -> list[str]` — gemener, skiljetecken bort, whitespace kollapsad
  - `word_edit_distance(ref: list[str], hyp: list[str]) -> int`
  - `wer(reference: str, hypothesis: str) -> float` — 0.0 = perfekt

- [ ] **Step 1: Write the failing test**

Skapa `bench/tests/test_wer.py`:

```python
import unittest

from bench.wer import normalize, word_edit_distance, wer


class TestNormalize(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize("Var ligger stationen?"), ["var", "ligger", "stationen"])

    def test_keeps_swedish_and_finnish_letters(self):
        self.assertEqual(normalize("Ursäkta, hyvää!"), ["ursäkta", "hyvää"])

    def test_collapses_whitespace(self):
        self.assertEqual(normalize("ett   två\n tre"), ["ett", "två", "tre"])

    def test_empty_string_yields_no_words(self):
        self.assertEqual(normalize("   "), [])


class TestWordEditDistance(unittest.TestCase):
    def test_identical_sequences(self):
        self.assertEqual(word_edit_distance(["a", "b"], ["a", "b"]), 0)

    def test_single_substitution(self):
        self.assertEqual(word_edit_distance(["a", "b"], ["a", "c"]), 1)

    def test_insertion_and_deletion(self):
        self.assertEqual(word_edit_distance(["a"], ["a", "b"]), 1)
        self.assertEqual(word_edit_distance(["a", "b"], ["a"]), 1)


class TestWer(unittest.TestCase):
    def test_perfect_transcription_scores_zero(self):
        self.assertEqual(wer("Var ligger stationen?", "var ligger stationen"), 0.0)

    def test_one_wrong_word_in_four(self):
        self.assertAlmostEqual(wer("ett två tre fyra", "ett två tre fem"), 0.25)

    def test_empty_hypothesis_scores_one(self):
        self.assertEqual(wer("ett två", ""), 1.0)

    def test_empty_reference_and_hypothesis_scores_zero(self):
        self.assertEqual(wer("", ""), 0.0)

    def test_empty_reference_with_output_scores_one(self):
        # Ingen referens att dela med; allt som sägs är fel. Undvik ZeroDivisionError.
        self.assertEqual(wer("", "hallucination"), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: FAIL med `ModuleNotFoundError: No module named 'bench.wer'`

- [ ] **Step 3: Write the implementation**

Skapa `bench/wer.py` med Apache-huvudet, sedan:

```python
"""Ordvis WER utan externa beroenden.

Kvalitetsgrinden behöver bara ett tal per fixtur, och en klassisk
Levenshtein över ordlistor räcker. Normaliseringen är medvetet grov: vi
jämför Whispers utdata mot texten vi syntetiserade, så skiljetecken och
versaler är brus.
"""

import re

# Behåller bokstäver (inklusive å ä ö ü) och siffror; allt annat är skiljetecken.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text):
    """Gemener, skiljetecken bort, whitespace kollapsad."""
    stripped = _PUNCTUATION.sub("", text.lower())
    return stripped.split()


def word_edit_distance(ref, hyp):
    """Levenshtein över två ordlistor."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,      # deletion
                    current[j - 1] + 1,   # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def wer(reference, hypothesis):
    """Word error rate i intervallet 0.0 (perfekt) och uppåt."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        # Ingen referens att dela med: tyst utdata är rätt, allt annat är fel.
        return 0.0 if not hyp else 1.0
    return word_edit_distance(ref, hyp) / len(ref)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: PASS, 24 tester totalt (12 från Task 1 + 12 nya)

- [ ] **Step 5: Commit**

```bash
git add bench/wer.py bench/tests/test_wer.py
git commit -m "Add dependency-free word error rate for the bench quality gate"
```

---

## Task 3: Fixturgenerering + konfigurerbar backend-port

**Files:**
- Modify: `backend/server.py:140` (`PORT = 3000`)
- Create: `bench/fixtures.json`
- Create: `bench/fixtures.py`

**Interfaces:**
- Consumes: `backend.server.synthesize(text, language) -> (np.ndarray float32, int sample_rate)`
- Produces:
  - `load_fixtures() -> dict[str, dict]` — id → `{"lang", "target", "text", "target_s"}`
  - `ensure_wav(fixture_id: str, spec: dict) -> pathlib.Path` — syntetiserar vid behov, returnerar sökväg
  - `load_pcm_16k(path: pathlib.Path) -> np.ndarray` — mono float32 i [-1, 1]
  - `check_duration(fixture_id: str, spec: dict, samples: np.ndarray) -> float` — fäller om längden avviker mer än ±50 % från `target_s`
  - `FIXTURE_DIR: pathlib.Path`

Bench importerar `synthesize` från backend i stället för att duplicera
Piper-hanteringen. Det ger samma röster som produkten använder och håller
röstkartan på ett ställe.

- [ ] **Step 1: Gör backend-porten konfigurerbar**

I `backend/server.py`, ersätt `PORT = 3000` med:

```python
# Överskrivbar så att bench/ kan köra en egen instans parallellt med en
# vanlig utvecklingsserver på 3000.
PORT = int(os.environ.get("PORT", 3000))
```

- [ ] **Step 2: Verifiera att defaulten är oförändrad och att env-varen slår igenom**

Run:
```bash
venv/bin/python3 -c "
import os, sys
sys.path.insert(0, 'backend')
import server
print('default:', server.PORT)
"
PORT=3100 venv/bin/python3 -c "
import os, sys
sys.path.insert(0, 'backend')
import server
print('override:', server.PORT)
"
```
Expected: `default: 3000` följt av `override: 3100`

- [ ] **Step 3: Skapa `bench/fixtures.json`**

Nio fixturer, tre längder per språk. `target_s` är en första gissning som
mäts upp och rättas i steg 5. Målspråken är valda så att bara två Piper-röster
behövs (en, sv) — se Global Constraints.

```json
{
  "sv-short": {
    "lang": "sv", "target": "en", "target_s": 1.5,
    "text": "Var ligger stationen?"
  },
  "sv-medium": {
    "lang": "sv", "target": "en", "target_s": 4.0,
    "text": "Ursäkta, kan du berätta hur jag tar mig till stationen härifrån?"
  },
  "sv-long": {
    "lang": "sv", "target": "en", "target_s": 9.0,
    "text": "Jag skulle behöva boka ett rum för två nätter, helst med utsikt mot havet, och jag undrar också om frukosten ingår i priset eller om den kostar extra."
  },
  "fi-short": {
    "lang": "fi", "target": "en", "target_s": 1.5,
    "text": "Missä asema on?"
  },
  "fi-medium": {
    "lang": "fi", "target": "en", "target_s": 4.0,
    "text": "Anteeksi, voisitko kertoa miten pääsen asemalle täältä?"
  },
  "fi-long": {
    "lang": "fi", "target": "en", "target_s": 9.0,
    "text": "Haluaisin varata huoneen kahdeksi yöksi, mieluiten merinäköalalla, ja haluaisin myös tietää sisältyykö aamiainen hintaan vai maksaako se erikseen."
  },
  "en-short": {
    "lang": "en", "target": "sv", "target_s": 1.5,
    "text": "Where is the station?"
  },
  "en-medium": {
    "lang": "en", "target": "sv", "target_s": 4.0,
    "text": "Excuse me, could you tell me how to get to the station from here?"
  },
  "en-long": {
    "lang": "en", "target": "sv", "target_s": 9.0,
    "text": "I would like to book a room for two nights, preferably with a view of the sea, and I also want to know whether breakfast is included in the price or costs extra."
  }
}
```

- [ ] **Step 4: Skapa `bench/fixtures.py`**

Med Apache-huvudet, sedan:

```python
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
```

- [ ] **Step 5: Generera fixturerna och rätta `target_s` mot uppmätt längd**

Run: `venv/bin/python3 -m bench.fixtures`
Expected: nio rader med faktiska längder. Första körningen laddar ner
`fi_FI-harri-medium` om den saknas, vilket tar några sekunder.

Uppdatera sedan varje `target_s` i `bench/fixtures.json` till den uppmätta
längden avrundad till en decimal. Gissningarna i steg 3 är just gissningar;
efter det här steget är `target_s` en mätning och toleransen fyller sin
funktion.

- [ ] **Step 6: Verifiera att längdkontrollen fäller på felaktig fixtur**

Run:
```bash
venv/bin/python3 -c "
from bench.fixtures import check_duration
import numpy as np
try:
    check_duration('test', {'target_s': 10.0}, np.zeros(16000, dtype=np.float32))
except ValueError as err:
    print('fäller korrekt:', err)
else:
    raise SystemExit('FEL: 1s-klipp mot target_s=10 borde ha fällt')
"
```
Expected: `fäller korrekt: test: 1.00s ligger utanför 5.00–15.00s ...`

- [ ] **Step 7: Commit**

```bash
git add backend/server.py bench/fixtures.json bench/fixtures.py
git commit -m "Add Piper-synthesized bench fixtures and configurable backend port"
```

---

## Task 4: `bench/runner.py`

**Files:**
- Create: `bench/runner.py`

**Interfaces:**
- Consumes: `bench.frontend_mirror.{split_text_into_speech_chunks, system_prompt, build_llm_payload, parse_translation}`, `bench.fixtures.{load_pcm_16k, ensure_wav, check_duration}`
- Produces:
  - `LANGUAGE_NAMES: dict[str, str]` — språkkod → namn som frontend visar (`{"sv": "Swedish", "en": "English", "fi": "Finnish"}`)
  - `RunResult` (dataclass) med fälten `fixture_id, ok, error, stt_ms, llm_ms, tts_first_ms, tts_rest_ms, transcript, translation, chunk_count, upload_bytes`
  - `run_fixture(api_base: str, llm_url: str, model: str, fixture_id: str, spec: dict) -> RunResult`
  - `warmup(api_base: str, llm_url: str, model: str, fixtures: dict) -> None`

Ingen enhetstest här: modulen är nästan bara I/O, och att mocka HTTP skulle
bara testa mockarna. Den verifieras genom att köras skarpt i steg 3.

- [ ] **Step 1: Write the implementation**

Skapa `bench/runner.py` med Apache-huvudet, sedan:

```python
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


def _post_llm(api_base, llm_url, model, text, src_lang, dst_lang):
    prompt = system_prompt(LANGUAGE_NAMES[src_lang], LANGUAGE_NAMES[dst_lang])
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


def run_fixture(api_base, llm_url, model, fixture_id, spec):
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
            api_base, llm_url, model, result.transcript, spec["lang"], spec["target"]
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


def warmup(api_base, llm_url, model, fixtures):
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
        result = run_fixture(api_base, llm_url, model, fixture_id, spec)
        if not result.ok:
            raise RuntimeError(f"Uppvärmningen misslyckades för {fixture_id}: {result.error}")
```

- [ ] **Step 2: Starta en backend på 3100 för rökprovet**

Run:
```bash
PORT=3100 venv/bin/python3 backend/server.py > /tmp/bench-backend.log 2>&1 &
sleep 25 && tail -3 /tmp/bench-backend.log
```
Expected: loggen visar `[Prewarm] Models pre-warmed successfully.`

- [ ] **Step 3: Rökprov mot en enda fixtur**

Run:
```bash
venv/bin/python3 -c "
from bench.fixtures import load_fixtures
from bench.runner import run_fixture
fixtures = load_fixtures()
result = run_fixture('http://localhost:3100', 'http://localhost:9379/v1/chat/completions',
                     'gemma4-e2b', 'sv-short', fixtures['sv-short'])
print('ok:', result.ok, result.error)
print('transcript:', repr(result.transcript))
print('translation:', repr(result.translation))
print(f'stt={result.stt_ms:.0f}ms llm={result.llm_ms:.0f}ms tts1={result.tts_first_ms:.0f}ms')
print(f'time_to_first_audio={result.time_to_first_audio_ms:.0f}ms chunks={result.chunk_count}')
"
```
Expected: `ok: True`, en svensk transkription som liknar "Var ligger stationen?",
en engelsk översättning, och tre tidssiffror skilda från noll.

Om `transcript` är tom eller vansinnig: kontrollera att fixtur-wav:en låter
rätt med `afplay bench/fixtures/sv-short.wav`.

- [ ] **Step 4: Commit**

```bash
git add bench/runner.py
git commit -m "Add bench runner that times each stage over the real HTTP path"
```

---

## Task 5: `bench/report.py`

**Files:**
- Create: `bench/report.py`
- Create: `bench/tests/test_report.py`

**Interfaces:**
- Consumes: `bench.runner.RunResult`, `bench.wer.{wer, normalize, word_edit_distance}`
- Produces:
  - `METRICS: tuple[str, ...]` — `("stt_ms", "llm_ms", "tts_first_ms", "tts_rest_ms", "time_to_first_audio_ms", "wall_total_ms")`
  - `median(values: list[float]) -> float`
  - `summarize(results: list[RunResult], spec_text: str) -> dict` — per fixtur: median/min/max per mätpunkt, plus `wer`, `edits`, `ref_words`, `translation`, `ok`
  - `build_report(label: str, per_fixture: dict) -> dict` — hela JSON-strukturen som skrivs till disk, inklusive `corpus_wer`
  - `render_markdown(current: dict, baseline: dict | None) -> str`
  - `gate(current: dict, baseline: dict | None, wer_threshold: float = 0.02) -> tuple[bool, list[str]]`

### Grinden mäter korpus-WER, inte per fixtur

Planen sa ursprungligen att grinden skulle fälla när en enskild fixturs WER
steg mer än 2 procentenheter. Det visade sig vara omöjligt att uppfylla:
WER kvantiseras i steg om `1 / antal ord i facit`, och de nio fixturerna har
3–33 ord. Minsta möjliga förändring är alltså 0,030 i bästa fall och 0,333
för treordsfixturerna — varenda ett större än tröskeln 0,02. En per-fixtur-grind
hade i praktiken varit nolltolerans med en missvisande etikett.

Grinden räknar därför **korpus-WER**: summan av alla ordfel delat med summan
av alla facitord över samtliga fixturer. Med 121 ord totalt motsvarar ett
enda ändrat ord 0,8 procentenheter, vilket gör 2-procentströskeln till en
verklig tolerans på ungefär två ord i hela sviten. Det är också så WER
konventionellt beräknas över ett testkorpus: långa yttranden ska väga tyngre
än korta.

Per-fixtur-WER står kvar i tabellen, eftersom det är där man ser *vilken*
fixtur som försämrades. Det styr bara inte utfallet.

- [ ] **Step 1: Write the failing test**

Skapa `bench/tests/test_report.py`:

```python
import unittest

from bench.report import build_report, gate, median, summarize
from bench.runner import RunResult


def make_result(
    fixture_id="sv-short",
    stt=100.0,
    llm=1000.0,
    tts=50.0,
    transcript="var ligger stationen",
    translation="Where is the station?",
):
    return RunResult(
        fixture_id=fixture_id,
        stt_ms=stt,
        llm_ms=llm,
        tts_first_ms=tts,
        tts_rest_ms=0.0,
        transcript=transcript,
        translation=translation,
        chunk_count=1,
    )


class TestMedian(unittest.TestCase):
    def test_odd_count_picks_middle(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)

    def test_even_count_averages_middle_pair(self):
        self.assertEqual(median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_empty_is_zero(self):
        self.assertEqual(median([]), 0.0)


class TestSummarize(unittest.TestCase):
    def test_reports_median_and_spread(self):
        results = [make_result(stt=100.0), make_result(stt=200.0), make_result(stt=300.0)]
        summary = summarize(results, "Var ligger stationen?")
        self.assertEqual(summary["stt_ms"]["median"], 200.0)
        self.assertEqual(summary["stt_ms"]["min"], 100.0)
        self.assertEqual(summary["stt_ms"]["max"], 300.0)

    def test_derives_time_to_first_audio(self):
        summary = summarize([make_result(stt=100.0, llm=1000.0, tts=50.0)], "Var ligger stationen?")
        self.assertEqual(summary["time_to_first_audio_ms"]["median"], 1150.0)

    def test_computes_wer_against_fixture_text(self):
        summary = summarize([make_result(transcript="var ligger stationen")], "Var ligger stationen?")
        self.assertEqual(summary["wer"], 0.0)

    def test_reports_raw_counts_for_corpus_aggregation(self):
        # Korpus-WER kan inte räknas ur per-fixtur-WER i efterhand; den behöver
        # täljare och nämnare var för sig.
        summary = summarize([make_result(transcript="vad ligger stationen")], "Var ligger stationen?")
        self.assertEqual(summary["edits"], 1)
        self.assertEqual(summary["ref_words"], 3)

    def test_failed_run_marks_fixture_not_ok(self):
        failed = make_result()
        failed.ok = False
        failed.error = "boom"
        summary = summarize([failed], "Var ligger stationen?")
        self.assertFalse(summary["ok"])


class TestBuildReport(unittest.TestCase):
    def _fixture(self, edits, ref_words, ok=True):
        return {
            "ok": ok,
            "errors": [],
            "edits": edits,
            "ref_words": ref_words,
            "wer": edits / ref_words if ref_words else 0.0,
            "translation": "x",
            "time_to_first_audio_ms": {"median": 1000.0, "min": 1000.0, "max": 1000.0},
        }

    def test_corpus_wer_weights_by_word_count_not_by_fixture(self):
        # Ett fel i en treordsfixtur och noll fel i en trettioordsfixtur:
        # korpus-WER är 1/33, inte medelvärdet av 0.333 och 0.0.
        report = build_report("x", {
            "short": self._fixture(1, 3),
            "long": self._fixture(0, 30),
        })
        self.assertAlmostEqual(report["corpus_wer"], 1 / 33)

    def test_corpus_wer_ignores_failed_fixtures(self):
        report = build_report("x", {
            "ok-one": self._fixture(1, 10),
            "broken": self._fixture(9, 10, ok=False),
        })
        self.assertAlmostEqual(report["corpus_wer"], 0.1)

    def test_corpus_wer_is_zero_when_nothing_succeeded(self):
        report = build_report("x", {"broken": self._fixture(5, 10, ok=False)})
        self.assertEqual(report["corpus_wer"], 0.0)


class TestGate(unittest.TestCase):
    def _report(self, corpus_wer, translation="Where is the station?", ok=True):
        return {
            "corpus_wer": corpus_wer,
            "fixtures": {
                "sv-short": {
                    "wer": corpus_wer,
                    "translation": translation,
                    "ok": ok,
                    "errors": [] if ok else ["ValueError: boom"],
                }
            },
        }

    def test_passes_without_baseline(self):
        passed, messages = gate(self._report(0.5), None)
        self.assertTrue(passed)
        self.assertEqual(messages, [])

    def test_passes_when_corpus_wer_unchanged(self):
        passed, _ = gate(self._report(0.10), self._report(0.10))
        self.assertTrue(passed)

    def test_passes_when_corpus_wer_rises_within_threshold(self):
        passed, _ = gate(self._report(0.115), self._report(0.10))
        self.assertTrue(passed)

    def test_fails_when_corpus_wer_rises_beyond_threshold(self):
        passed, messages = gate(self._report(0.13), self._report(0.10))
        self.assertFalse(passed)
        self.assertTrue(any("REGRESSION" in message for message in messages))

    def test_improved_corpus_wer_never_fails(self):
        passed, _ = gate(self._report(0.01), self._report(0.30))
        self.assertTrue(passed)

    def test_changed_translation_warns_but_passes(self):
        passed, messages = gate(
            self._report(0.10, translation="Where's the station?"),
            self._report(0.10, translation="Where is the station?"),
        )
        self.assertTrue(passed)
        self.assertTrue(any("översättning" in message for message in messages))

    def test_failed_fixture_fails_the_gate(self):
        passed, messages = gate(self._report(0.0, ok=False), None)
        self.assertFalse(passed)
        self.assertIn("sv-short", messages[0])

    def test_baseline_without_corpus_wer_skips_the_wer_gate(self):
        # Äldre resultatfiler saknar fältet; grinden ska säga det rakt ut
        # i stället för att tyst låtsas att kvaliteten hölls.
        legacy = self._report(0.10)
        del legacy["corpus_wer"]
        passed, messages = gate(self._report(0.90), legacy)
        self.assertTrue(passed)
        self.assertTrue(any("corpus_wer" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: FAIL med `ModuleNotFoundError: No module named 'bench.report'`

- [ ] **Step 3: Write the implementation**

Skapa `bench/report.py` med Apache-huvudet, sedan:

```python
"""Sammanfattar körningar, jämför mot baseline och fäller på regression."""

from bench.wer import normalize, word_edit_distance

METRICS = (
    "stt_ms",
    "llm_ms",
    "tts_first_ms",
    "tts_rest_ms",
    "time_to_first_audio_ms",
    "wall_total_ms",
)

# Absolut ökning i korpus-WER som får passera. Med ~121 facitord i sviten
# motsvarar ett enda ändrat ord ca 0,008, så tröskeln rymmer ungefär två
# ords variation innan den fäller.
DEFAULT_WER_THRESHOLD = 0.02


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize(results, spec_text):
    """Slår ihop repetitionerna för en fixtur till en rad.

    Både `wer` och råtalen `edits`/`ref_words` sparas: per-fixtur-WER visas i
    tabellen, medan korpus-WER måste summeras ur täljare och nämnare var för
    sig — ett medelvärde av per-fixtur-WER hade gett treordsfixturerna lika
    stor vikt som trettioordsfixturerna.
    """
    reference = normalize(spec_text)
    hypothesis = normalize(results[-1].transcript)
    edits = word_edit_distance(reference, hypothesis)
    ref_words = len(reference)

    summary = {
        "ok": all(result.ok for result in results),
        "errors": [result.error for result in results if not result.ok],
        "translation": results[-1].translation,
        "transcript": results[-1].transcript,
        "chunk_count": results[-1].chunk_count,
        "upload_bytes": results[-1].upload_bytes,
        "edits": edits,
        "ref_words": ref_words,
        "wer": (edits / ref_words) if ref_words else (0.0 if not hypothesis else 1.0),
    }
    for metric in METRICS:
        values = [getattr(result, metric) for result in results]
        summary[metric] = {
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def build_report(label, per_fixture):
    ok_fixtures = [data for data in per_fixture.values() if data["ok"]]
    total_edits = sum(data["edits"] for data in ok_fixtures)
    total_words = sum(data["ref_words"] for data in ok_fixtures)
    overall = [data["time_to_first_audio_ms"]["median"] for data in ok_fixtures]
    return {
        "label": label,
        "fixtures": per_fixture,
        "median_time_to_first_audio_ms": median(overall),
        "corpus_wer": (total_edits / total_words) if total_words else 0.0,
        "corpus_edits": total_edits,
        "corpus_ref_words": total_words,
    }


def _delta(current_value, baseline_value):
    if baseline_value in (None, 0):
        return ""
    change = (current_value - baseline_value) / baseline_value * 100
    return f"{change:+.0f}%"


def render_markdown(current, baseline=None):
    baseline_fixtures = (baseline or {}).get("fixtures", {})
    lines = [
        f"### {current['label']}",
        "",
        "| Fixtur | STT | LLM | TTS #1 | Till första ljud | Δ | WER |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fixture_id, data in current["fixtures"].items():
        if not data["ok"]:
            lines.append(f"| {fixture_id} | — | — | — | **FEL** | | {data['errors'][0][:40]} |")
            continue
        first_audio = data["time_to_first_audio_ms"]["median"]
        reference = baseline_fixtures.get(fixture_id, {}).get("time_to_first_audio_ms", {}).get("median")
        lines.append(
            f"| {fixture_id} "
            f"| {data['stt_ms']['median']:.0f} ms "
            f"| {data['llm_ms']['median']:.0f} ms "
            f"| {data['tts_first_ms']['median']:.0f} ms "
            f"| {first_audio:.0f} ms "
            f"| {_delta(first_audio, reference)} "
            f"| {data['wer']:.2f} |"
        )

    overall = current["median_time_to_first_audio_ms"]
    baseline_overall = (baseline or {}).get("median_time_to_first_audio_ms")
    lines.append("")
    lines.append(
        f"**Median till första ljud: {overall:.0f} ms** {_delta(overall, baseline_overall)}"
    )
    lines.append(
        f"**Korpus-WER: {current['corpus_wer']:.3f}** "
        f"({current['corpus_edits']} fel på {current['corpus_ref_words']} ord)"
    )
    return "\n".join(lines)


def gate(current, baseline=None, wer_threshold=DEFAULT_WER_THRESHOLD):
    """Returnerar (passerade, meddelanden).

    Fäller på misslyckade fixturer och på korpus-WER som stigit mer än
    tröskeln. Ändrad översättning noteras men fäller inte — Gemma får
    formulera om sig så länge transkriptionen håller.
    """
    messages = []
    passed = True

    for fixture_id, data in current["fixtures"].items():
        if not data["ok"]:
            passed = False
            messages.append(f"FEL {fixture_id}: {data['errors'][0]}")

    if not baseline:
        return passed, messages

    if "corpus_wer" not in baseline:
        messages.append(
            "VARNING: jämförelsekörningen saknar corpus_wer — WER-grinden hoppades över."
        )
    else:
        rise = current["corpus_wer"] - baseline["corpus_wer"]
        if rise > wer_threshold:
            passed = False
            worse = [
                f"{fixture_id} {baseline['fixtures'][fixture_id]['wer']:.2f}→{data['wer']:.2f}"
                for fixture_id, data in current["fixtures"].items()
                if data["ok"]
                and fixture_id in baseline["fixtures"]
                and data["wer"] > baseline["fixtures"][fixture_id]["wer"]
            ]
            messages.append(
                f"REGRESSION korpus-WER {baseline['corpus_wer']:.3f} → "
                f"{current['corpus_wer']:.3f} (+{rise:.3f}, tröskel {wer_threshold:.3f}). "
                f"Försämrade fixturer: {', '.join(worse) or 'inga enskilda'}"
            )

    for fixture_id, data in current["fixtures"].items():
        reference = baseline["fixtures"].get(fixture_id)
        if not reference or not data["ok"]:
            continue
        if data["translation"] != reference["translation"]:
            messages.append(
                f"NOTERA {fixture_id}: översättning ändrad\n"
                f"  förr: {reference['translation']!r}\n"
                f"  nu:   {data['translation']!r}"
            )

    return passed, messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: PASS, alla tester gröna

- [ ] **Step 5: Commit**

```bash
git add bench/report.py bench/tests/test_report.py
git commit -m "Add bench reporting with corpus-level WER regression gate"
```


## Task 6: `bench/bench.py` och baseline-körningen

**Files:**
- Create: `bench/bench.py`
- Create: `bench/results/` (via körningen)
- Modify: `README.md` (nytt avsnitt om hur bench körs)

**Interfaces:**
- Consumes: allt ovan
- Produces: CLI:t som resten av planen använder — `venv/bin/python3 -m bench.bench --label <namn> [--compare <namn>]`

- [ ] **Step 1: Write the implementation**

Skapa `bench/bench.py` med Apache-huvudet, sedan:

```python
"""CLI för latensmätningen.

Startar en egen backend på en fri port, värmer upp modellerna, kör varje
fixtur N gånger, och skriver både en markdown-tabell till terminalen och en
JSON-fil till bench/results/. Exitkoden är nollskild om kvalitetsgrinden
fäller, så en körning kan användas som grind i ett skript.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import requests

from bench.fixtures import load_fixtures
from bench.report import build_report, gate, render_markdown, summarize
from bench.runner import run_fixture, warmup

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"

BACKEND_STARTUP_TIMEOUT = 180


def start_backend(port):
    """Startar backend/server.py på `port` och väntar tills den svarar."""
    env = dict(os.environ, PORT=str(port), PYTHONUNBUFFERED="1")
    log_path = pathlib.Path(f"/tmp/bench-backend-{port}.log")
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(REPO_DIR / "venv" / "bin" / "python3"), str(REPO_DIR / "backend" / "server.py")],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_DIR),
    )
    print(f"[bench] Startade backend på {port} (pid {process.pid}), logg: {log_path}", flush=True)

    deadline = time.time() + BACKEND_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Backend dog vid start, se {log_path}")
        try:
            # /api/tts utan text ger 400 — vilket räcker som bevis på att den lyssnar.
            requests.get(f"http://localhost:{port}/api/tts", timeout=2)
            return process
        except requests.RequestException:
            time.sleep(1)
    process.terminate()
    raise RuntimeError(f"Backend svarade inte inom {BACKEND_STARTUP_TIMEOUT}s, se {log_path}")


def load_result(label):
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        raise SystemExit(f"Hittar ingen tidigare körning med label {label!r} ({path})")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Mät latens i översättningskedjan.")
    parser.add_argument("--label", required=True, help="Namn på körningen, blir filnamn i bench/results/")
    parser.add_argument("--compare", help="Label att jämföra mot, t.ex. baseline")
    parser.add_argument("--api-port", type=int, default=3100)
    parser.add_argument("--llm-port", type=int, default=9379)
    parser.add_argument("--model", default="gemma4-e2b")
    parser.add_argument("--repeats", type=int, default=3, help="Körningar per fixtur; den första kastas")
    parser.add_argument("--fixtures", help="Kommaseparerad lista med fixtur-id, default alla")
    args = parser.parse_args()

    api_base = f"http://localhost:{args.api_port}"
    llm_url = f"http://localhost:{args.llm_port}/v1/chat/completions"

    fixtures = load_fixtures()
    if args.fixtures:
        wanted = set(args.fixtures.split(","))
        fixtures = {key: value for key, value in fixtures.items() if key in wanted}
        if not fixtures:
            raise SystemExit(f"Inga fixturer matchade {args.fixtures!r}")

    baseline = load_result(args.compare) if args.compare else None

    backend = start_backend(args.api_port)
    try:
        warmup(api_base, llm_url, args.model, fixtures)

        per_fixture = {}
        for fixture_id, spec in fixtures.items():
            runs = []
            for attempt in range(args.repeats):
                result = run_fixture(api_base, llm_url, args.model, fixture_id, spec)
                marker = " (uppvärmning, kastas)" if attempt == 0 else ""
                status = "ok" if result.ok else f"FEL {result.error}"
                print(
                    f"[bench] {fixture_id} {attempt + 1}/{args.repeats}{marker}: "
                    f"{result.time_to_first_audio_ms:.0f} ms till första ljud, {status}",
                    flush=True,
                )
                runs.append(result)
            # Första körningen kastas: den bär cache- och JIT-kostnader.
            measured = runs[1:] if len(runs) > 1 else runs
            per_fixture[fixture_id] = summarize(measured, spec["text"])
    finally:
        backend.terminate()
        backend.wait(timeout=30)
        print("[bench] Backend stoppad.", flush=True)

    report = build_report(args.label, per_fixture)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{args.label}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print(render_markdown(report, baseline))
    print()

    passed, messages = gate(report, baseline)
    for message in messages:
        print(message)
    print(f"\nSkrev {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Stoppa eventuell manuellt startad backend från Task 4**

Bench startar sin egen instans och kolliderar annars på porten.

Run: `lsof -ti:3100 | xargs kill 2>/dev/null; sleep 1; lsof -ti:3100 || echo "3100 ledig"`
Expected: `3100 ledig`

- [ ] **Step 3: Rökprov på två fixturer**

Run: `venv/bin/python3 -m bench.bench --label smoke --fixtures sv-short,en-short --repeats 2`
Expected: backend startar, uppvärmning körs, fyra mätrader, en markdown-tabell,
och `Skrev bench/results/smoke.json`. Exitkod 0.

- [ ] **Step 4: Verifiera att grinden fäller vid WER-regression**

Grinden måste bevisas innan vi litar på den.

Två fällor att undvika, båda funna genom att en tidigare version av det här
steget bevisade fel sak:

- **Mutera `corpus_wer`, inte fixturernas `wer`.** `gate()` läser bara det
  översta `corpus_wer`-fältet. Att sänka per-fixtur-WER påverkar ingenting
  och ger en körning som passerar — vilket ser ut som att grinden är trasig,
  eller värre, som att den fungerar om man inte tittar noga.
- **Jämför samma fixturuppsättning.** Kör man `--fixtures sv-short` mot en
  jämförelsekörning som innehåller två fixturer skiljer sig korpus-WER redan
  av den anledningen, och då fäller grinden av fel skäl.

Run:
```bash
venv/bin/python3 -c "
import json, pathlib
path = pathlib.Path('bench/results/smoke.json')
data = json.loads(path.read_text(encoding='utf-8'))
# Nolla korpusfelen: jämförelsekörningen utger sig då för att ha varit felfri,
# så vilken verklig WER som helst i den nya körningen är en regression.
data['corpus_wer'] = 0.0
data['corpus_edits'] = 0
pathlib.Path('bench/results/smoke-strict.json').write_text(
    json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'skrev en konstlad felfri baseline (riktig korpus-WER var {json.loads(path.read_text(encoding=\"utf-8\"))[\"corpus_wer\"]:.3f})')
"
venv/bin/python3 -m bench.bench --label smoke-check --compare smoke-strict --fixtures sv-short,en-short --repeats 2
echo "exitkod: $?"
```
Expected: utskriften innehåller en `REGRESSION korpus-WER 0.000 → ...`-rad och
`exitkod: 1`. Notera att `--fixtures` är samma lista som i steg 3.

Bevisa sedan motsatsen — att grinden *inte* fäller när kvaliteten hålls:

Run:
```bash
venv/bin/python3 -m bench.bench --label smoke-nochange --compare smoke --fixtures sv-short,en-short --repeats 2
echo "exitkod: $?"
```
Expected: ingen `REGRESSION`-rad och `exitkod: 0`. En grind som alltid fäller
är lika värdelös som en som aldrig gör det.

- [ ] **Step 5: Städa bort rökprovsfilerna**

Run: `rm -f bench/results/smoke.json bench/results/smoke-strict.json bench/results/smoke-check.json`

- [ ] **Step 6: Bekräfta att genereringen är deterministisk**

Specen ville ursprungligen sätta `temperature: 0` för att göra
översättningsdiffen användbar. Sonderingen visade att litert-lm ignorerar
generationsparametrar (`max_tokens: 5` gav ett fullängdssvar), så i stället
för att skicka en parameter som inte gör något verifierar vi utfallet direkt.

Run:
```bash
lsof -ti:3100 | xargs kill 2>/dev/null
venv/bin/python3 -c "
import urllib.parse, requests
from bench.frontend_mirror import build_llm_payload, system_prompt
url = urllib.parse.quote('http://localhost:9379/v1/chat/completions', safe='')
payload = build_llm_payload('Var ligger stationen?', 'gemma4-e2b',
                            system_prompt('Swedish', 'English'))
answers = []
for _ in range(3):
    response = requests.post(f'http://localhost:9379/v1/chat/completions', json=payload, timeout=300)
    answers.append(response.json()['choices'][0]['message']['content'])
for index, answer in enumerate(answers):
    print(index, repr(answer[:80]))
print('deterministiskt:', len(set(answers)) == 1)
"
```
Expected: `deterministiskt: True`.

Blir det `False` är översättningsdiffen i grinden brus snarare än signal.
Notera det i så fall, och läs `NOTERA`-raderna i kommande körningar som
information i stället för som varningar — WER-grinden mäter fortfarande
transkriptionen och påverkas inte.

- [ ] **Step 7: Kör baseline skarpt**

Run: `venv/bin/python3 -m bench.bench --label baseline`
Expected: nio fixturer × 3 körningar. Detta tar flera minuter — LLM-steget är
1–6 s per anrop. Exitkod 0 och `bench/results/baseline.json` skriven.

Läs tabellen och notera: vilket steg dominerar, och hur stor är spridningen
mellan min och max? Om `time_to_first_audio` varierar mer än ±20 % mellan
repetitionerna är mätningen för brusig för en 2-procentsgrind, och `--repeats`
behöver höjas till 5 innan optimeringarna påbörjas.

- [ ] **Step 8: Dokumentera bench i README**

Lägg till ett avsnitt direkt före `## Latency notes & ideas` i `README.md`:

```markdown
## Benchmarking

`bench/` measures the full push-to-talk chain over HTTP and gates on quality
regression. It synthesizes its own fixtures with Piper, so no recordings are
needed:

```bash
# Requires litert-lm already running on :9379 (./start.sh)
venv/bin/python3 -m bench.bench --label baseline
venv/bin/python3 -m bench.bench --label my-change --compare baseline
```

Each run starts its own backend on port 3100, warms the models, and reports
median time-to-first-audio per fixture alongside word error rate. A run exits
non-zero if *corpus* WER — total word errors over total reference words across
all fixtures — rises more than 2 points against the comparison run. Per-fixture
WER is reported too but doesn't gate: with 3-33 word references, one changed
word moves a single fixture's WER by 3 to 33 points, so a per-fixture threshold
would be zero tolerance wearing a percentage sign. Results are committed to
`bench/results/` as a history of the optimization campaign.
```

- [ ] **Step 9: Commit**

```bash
git add bench/bench.py bench/results/baseline.json README.md
git commit -m "Add bench CLI and record the pre-optimization baseline"
```

---

## Task 7: Frontend-instrumentering

**Files:**
- Modify: `frontend/src/TranslatorApp.jsx` (`handleRecordStop`, `processTranslation`, `playTTS`)

**Interfaces:**
- Consumes: inget
- Produces: en `[latency]`-rad i browserkonsolen per runda, och samma siffror i drawerns `metaText`

Bench kan inte mäta browserdelen. Den här instrumenteringen är enda stället
där tid till första ljud är sann på riktigt, inklusive autoplay-fördröjning.
Den körs manuellt vid tre milstolpar, inte per patch.

- [ ] **Step 1: Lägg till en tidsstämpel-ref**

I `TranslatorApp`, bredvid `onlineAudioPlayerRef`:

```jsx
  // Tidsstämplar för latensmätning; nollställs vid varje knappsläpp.
  const timingRef = useRef(null)
```

- [ ] **Step 2: Starta mätningen vid knappsläpp**

I `handleRecordStop`, allra först i funktionen:

```jsx
  const handleRecordStop = useCallback(async () => {
    timingRef.current = { keyup: performance.now() }
    const recordedLane = activeLaneRecording
```

- [ ] **Step 3: Stämpla STT och LLM i `processTranslation`**

Efter `const transcribedText = await transcribeAudio(base64Data, src.code)`:

```jsx
      if (timingRef.current) timingRef.current.stt = performance.now()
```

Efter `const result = await translateText(...)`:

```jsx
      if (timingRef.current) timingRef.current.llm = performance.now()
```

- [ ] **Step 4: Stämpla första ljudet och logga**

I `playTTS`, i `playNextChunk`, direkt efter `onlineAudioPlayerRef.current = player`:

```jsx
        player.onplaying = () => {
          const marks = timingRef.current
          if (!marks || marks.logged) return
          marks.logged = true
          const since = (mark) => `${(mark - marks.keyup) | 0}ms`
          console.log(
            `[latency] keyup→stt ${since(marks.stt)} | →llm ${since(marks.llm)} ` +
              `| →first audio ${since(performance.now())}`,
          )
          setMetaText(
            `STT ${since(marks.stt)} · LLM ${since(marks.llm)} · ljud ${since(performance.now())}`,
          )
        }
```

`marks.logged` behövs för att `playTTS` kedjar chunkar genom samma
`playNextChunk`, så `onplaying` fyrar en gång per chunk — vi vill bara ha
den första.

- [ ] **Step 5: Ta bort den gamla metaText-raden**

Ersätt raden

```jsx
      setMetaText(`Duration: ${result.duration}s | Tokens: ${result.tokens}`)
```

med

```jsx
      // Tokens rapporteras inte av litert-lm (usage saknas i svaret), och
      // latenssiffrorna sätts av playTTS när ljudet faktiskt börjar spela.
```

- [ ] **Step 6: Verifiera i browsern**

Run: `npm --prefix frontend install && npm --prefix frontend run dev -- --port 5273`

Öppna `http://localhost:5273`, ställ in endpoint mot `http://localhost:9379`
i inställningarna, håll Z, säg en mening på svenska, släpp.

Expected: en `[latency] keyup→stt ...ms | →llm ...ms | →first audio ...ms`-rad
i konsolen, och samma tre tal i drawern. Anteckna raden — det är
browser-baseline mot vilken milstolpe två och tre jämförs.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/TranslatorApp.jsx
git commit -m "Instrument the browser path with time-to-first-audio marks"
```

---

# Optimeringarna

Varje task nedan följer samma rytm: gör ändringen, kör bench mot föregående
steg, läs tabellen, committa mätresultatet tillsammans med koden. Etiketterna
kedjas (`--compare` pekar på föregående steg) så att varje siffra visar just
den ändringens bidrag. Task 17 gör en avslutande jämförelse mot `baseline`
för att fånga kvalitetsdrift som krupit fram några hundradelar i taget.

**Om en optimering visar sig vara en försämring:** committa den inte. Notera
resultatet i Task 17:s sammanställning som ett prövat och förkastat spår, och
gå vidare. Mätningen är poängen; alla nio behöver inte vinna.

## Task 8: Optimering 1 — VAD-filter i STT

**Files:**
- Modify: `backend/server.py:130-137` (`transcribe`)

- [ ] **Step 1: Slå på VAD-filtret**

I `transcribe`, lägg till `vad_filter=True`:

```python
def transcribe(audio_np, language):
    """Transcribe 16 kHz mono float32 samples. Unknown languages are auto-detected."""
    segments, _ = get_whisper_model().transcribe(
        audio_np,
        language=language if language in SUPPORTED_STT_LANGS else None,
        beam_size=1,
        # Klipper bort tystnad före dekodning. Kortar ljudet som når modellen
        # och tar bort de hallucinationer stock-Whisper gärna producerar på
        # tysta partier.
        vad_filter=True,
    )
    return " ".join(s.text.strip() for s in segments)
```

- [ ] **Step 2: Mät**

Run: `venv/bin/python3 -m bench.bench --label 01-vad --compare baseline`
Expected: exitkod 0. Läs `STT`-kolumnen och `WER`-kolumnen.

- [ ] **Step 3: Commit**

```bash
git add backend/server.py bench/results/01-vad.json
git commit -m "Enable Whisper VAD filter to trim silence before decoding"
```

---

## Task 9: Optimering 2 — kort mel-padding

**Files:**
- Modify: `backend/server.py` (import-blocket, konstanterna vid rad 38-47, `transcribe`)

Whisper paddar alltid till 30 sekunder, vilket betyder att ett 1,5-sekunders
klipp kostar lika mycket som ett 30-sekunders. `faster_whisper` 1.2.1 tar en
`chunk_length`-parameter som går rakt in i feature-extraktorn (se
`venv/lib/python3.13/site-packages/faster_whisper/transcribe.py:916`), så
ingen monkeypatch behövs — README:s påstående om motsatsen gäller en äldre
version.

- [ ] **Step 1: Lägg till konstanten**

Överst i `backend/server.py`, lägg till `import math` bland de andra
importerna, och efter `WHISPER_MODEL_SIZE`-raden:

```python
# Whisper paddar normalt varje klipp till 30 s mel-spektrogram, så ett kort
# yttrande kostar lika mycket som ett långt. Sätts detta till ett positivt
# tal paddar vi i stället till ljudlängden plus marginalen. Marginalen finns
# för att dekodern behöver lite tystnad efter sista ordet för att avsluta.
# Default 0 = oförändrat 30 s-beteende.
STT_CHUNK_MARGIN_S = float(os.environ.get("STT_CHUNK_MARGIN_S", "0"))
WHISPER_DEFAULT_CHUNK_S = 30
```

- [ ] **Step 2: Använd den i `transcribe`**

```python
def transcribe(audio_np, language):
    """Transcribe 16 kHz mono float32 samples. Unknown languages are auto-detected."""
    # chunk_length skickas alltid explicit: FeatureExtractor.__call__ muterar
    # self.n_samples när den får ett värde, och det värdet ligger kvar till
    # nästa anrop. Utan ett uttryckligt värde varje gång skulle en kort
    # inspelning smitta av sin padding på nästa, längre inspelning.
    chunk_length = WHISPER_DEFAULT_CHUNK_S
    if STT_CHUNK_MARGIN_S > 0:
        duration_s = len(audio_np) / 16000.0
        chunk_length = min(
            WHISPER_DEFAULT_CHUNK_S,
            max(2, math.ceil(duration_s + STT_CHUNK_MARGIN_S)),
        )
    segments, _ = get_whisper_model().transcribe(
        audio_np,
        language=language if language in SUPPORTED_STT_LANGS else None,
        beam_size=1,
        vad_filter=True,
        chunk_length=chunk_length,
    )
    return " ".join(s.text.strip() for s in segments)
```

- [ ] **Step 3: Verifiera att defaulten inte ändrar något**

Run: `venv/bin/python3 -m bench.bench --label 02a-chunklen-default --compare 01-vad`
Expected: STT-siffrorna inom brusnivån från `01-vad`, WER oförändrad. Detta
bevisar att refaktoreringen är beteendeneutral innan vi ändrar beteendet.

- [ ] **Step 4: Mät med kort padding**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 02b-chunklen-5s --compare 02a-chunklen-default`
Expected: märkbart lägre STT, oförändrad WER. Fäller grinden vid WER-höjning.

- [ ] **Step 5: Om WER steg, pröva en större marginal**

Bara om steg 4 fällde grinden:

Run: `STT_CHUNK_MARGIN_S=8 venv/bin/python3 -m bench.bench --label 02c-chunklen-8s --compare 02a-chunklen-default`

Håller inte heller 8 s WER, behåll `STT_CHUNK_MARGIN_S=0` som default och
notera spåret som förkastat i Task 17.

- [ ] **Step 6: Dokumentera env-varen i README**

I `README.md`, i listan över miljövariabler efter `WHISPER_MODEL_SIZE`:

```markdown
- `STT_CHUNK_MARGIN_S`: pad Whisper's mel spectrogram to *audio length + this
  many seconds* instead of a fixed 30 s. Cuts STT time on short utterances.
  Default `0` keeps the stock 30 s behaviour.
```

- [ ] **Step 7: Commit**

```bash
git add backend/server.py README.md bench/results/02a-chunklen-default.json bench/results/02b-chunklen-5s.json
git commit -m "Allow padding Whisper to audio length plus a margin"
```

---

## Task 10: Optimering 3 — trådar och dekodningsflaggor

> **STRUKEN.** Två STT-optimeringar är mätta och båda är nollresultat, och
> mekanismen är förstådd: Whispers kostnad är ett encoder-pass med fast
> storlek (se Task 9). Dekodningsflaggor är det minst lovande som återstår,
> och den kvarvarande insatsen läggs på strömningen i stället, eftersom den
> träffar kampanjens huvudmått.


**Files:**
- Modify: `backend/server.py` (`get_whisper_model`, `transcribe`)

- [ ] **Step 1: Sätt trådantal vid modelladdning**

```python
# Ct2 gissar annars konservativt. M1 har 8 kärnor; på Pi 5 (4 kärnor) sätts
# den här via env i stället.
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "8"))
```

och i `get_whisper_model`:

```python
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                cpu_threads=WHISPER_CPU_THREADS,
            )
```

- [ ] **Step 2: Lägg till dekodningsflaggorna**

I `transcribe`, i anropet till `.transcribe(...)`:

```python
        # Vi kastar tidsstämplarna ändå, och varje yttrande är fristående —
        # att inte betinga på tidigare text gör korta klipp stabilare.
        without_timestamps=True,
        condition_on_previous_text=False,
```

- [ ] **Step 3: Mät**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 03-decode-flags --compare 02b-chunklen-5s`

Använd samma `STT_CHUNK_MARGIN_S` som vann i Task 9, annars jämför du två
saker på en gång.

- [ ] **Step 4: Commit**

```bash
git add backend/server.py bench/results/03-decode-flags.json
git commit -m "Set Whisper cpu_threads and per-utterance decode flags"
```

---

## Task 11: Optimering 4 — meningsvis TTS-chunkning

> **STRUKEN efter mätning.** Bänkens upplösning mättes i Task 18 till ungefär
> 3 %: två steg som en STT-ändring bevisligen inte kan påverka läste 0,985 och
> 0,974 i stället för 1,000. Den här optimeringen träffar TTS respektive
> klientsidan, som tillsammans utgör 5 % av tiden till första ljud. Även en
> halvering där landar under bruskgolvet och kan alltså inte mätas med den här
> riggen. Eftersom kampanjens hela premiss är att varje förbättring ska mätas,
> byggs den inte. Den kan tas upp igen om bruskgolvet sänks.


**Files:**
- Modify: `frontend/src/utils/api.js:154-168` (`splitTextIntoSpeechChunks`)
- Modify: `bench/frontend_mirror.py` (`split_text_into_speech_chunks`)
- Modify: `bench/tests/test_frontend_mirror.py`

Idag är första chunken upp till 180 tecken, vilket för en vanlig översättning
är hela texten — första ljudet väntar alltså på att allt syntetiserats. Delar
vi på meningsgräns blir första chunken kort och ljudet startar långt tidigare.
Båda kopiorna ändras i samma commit, annars mäter bench något annat än appen.

- [ ] **Step 1: Skriv de nya testerna**

Lägg till i `bench/tests/test_frontend_mirror.py`, i klassen
`TestSplitTextIntoSpeechChunks`:

```python
    def test_splits_on_sentence_boundary(self):
        chunks = split_text_into_speech_chunks("Where is the station? It is over there.")
        self.assertEqual(chunks, ["Where is the station?", "It is over there."])

    def test_keeps_terminal_punctuation_with_its_sentence(self):
        chunks = split_text_into_speech_chunks("Hej! Hur mår du?")
        self.assertEqual(chunks, ["Hej!", "Hur mår du?"])

    def test_long_sentence_still_falls_back_to_word_packing(self):
        text = " ".join(["ord"] * 10) + "."
        chunks = split_text_into_speech_chunks(text, limit=20)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))
        self.assertGreater(len(chunks), 1)
```

- [ ] **Step 2: Kör testerna och se dem falla**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: FAIL på `test_splits_on_sentence_boundary`

- [ ] **Step 3: Ändra Python-kopian**

Ersätt `split_text_into_speech_chunks` i `bench/frontend_mirror.py`:

```python
# Meningsslut: punkt, utrop, fråga, ellips. Whisper och Gemma producerar inga
# andra terminatorer på våra språk.
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]*\s*")


def split_text_into_speech_chunks(text, limit=SPEECH_CHUNK_LIMIT):
    """Delar först på meningsgräns, sedan på ord om en mening är för lång.

    Poängen är att första chunken ska vara kort: uppspelningen kan börja så
    snart första meningen är syntetiserad i stället för hela svaret.
    """
    chunks = []
    for sentence in _SENTENCE.findall(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            chunks.append(sentence)
            continue
        current = ""
        for word in re.split(r"\s+", sentence):
            if len((current + " " + word).strip()) <= limit:
                current = (current + " " + word).strip()
            else:
                if current:
                    chunks.append(current)
                current = word
        if current:
            chunks.append(current)
    return chunks
```

- [ ] **Step 4: Kör testerna och se dem passera**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: PASS

- [ ] **Step 5: Ändra JS-kopian likadant**

I `frontend/src/utils/api.js`, ersätt `splitTextIntoSpeechChunks`:

```js
// Sentence-first chunking so the first /api/tts request is short and audio
// starts as soon as sentence one is synthesized. Over-long sentences still
// fall back to word packing under ~`limit` chars.
export function splitTextIntoSpeechChunks(text, limit = 180) {
  const sentences = text.match(/[^.!?…]+[.!?…]*\s*/g) || []
  const chunks = []
  for (const raw of sentences) {
    const sentence = raw.trim()
    if (!sentence) continue
    if (sentence.length <= limit) {
      chunks.push(sentence)
      continue
    }
    let currentChunk = ""
    for (const word of sentence.split(/\s+/)) {
      if ((currentChunk + " " + word).trim().length <= limit) {
        currentChunk = (currentChunk + " " + word).trim()
      } else {
        if (currentChunk) chunks.push(currentChunk)
        currentChunk = word
      }
    }
    if (currentChunk) chunks.push(currentChunk)
  }
  return chunks
}
```

- [ ] **Step 6: Verifiera att de två kopiorna ger samma svar**

Run:
```bash
venv/bin/python3 -c "
from bench.frontend_mirror import split_text_into_speech_chunks as py
cases = [
    'Where is the station? It is over there.',
    'Hej! Hur mår du?',
    'En enda mening utan slutpunkt',
    '',
]
for case in cases:
    print(repr(case), '->', py(case))
"
node -e "
const text = require('fs').readFileSync('frontend/src/utils/api.js', 'utf8');
const body = text.slice(text.indexOf('export function splitTextIntoSpeechChunks'));
const fn = new Function('return ' + body.slice(body.indexOf('function')).split('\n\n')[0])();
for (const c of ['Where is the station? It is over there.', 'Hej! Hur mår du?', 'En enda mening utan slutpunkt', '']) {
  console.log(JSON.stringify(c), '->', JSON.stringify(fn(c)));
}
"
```
Expected: identiska listor från båda. Skiljer de sig är det Python-kopian som
ska rättas — JS är produkten.

- [ ] **Step 7: Mät**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 04-sentence-chunks --compare 03-decode-flags`
Expected: tydligt lägre `TTS #1` och `Till första ljud`; `wall_total`
ungefär oförändrad eller något högre (fler HTTP-anrop).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/utils/api.js bench/frontend_mirror.py bench/tests/test_frontend_mirror.py bench/results/04-sentence-chunks.json
git commit -m "Chunk TTS on sentence boundaries so first audio starts sooner"
```

---

## Task 12: Optimering 5 — slopa JSON-wrappern

**Files:**
- Modify: `frontend/src/TranslatorApp.jsx:226-230` (systemprompten)
- Modify: `bench/frontend_mirror.py` (`system_prompt`)
- Modify: `bench/tests/test_frontend_mirror.py` (`TestSystemPrompt`)

Dagens systemprompt är ~90 tokens och tvingar modellen att rama in svaret i
`{"translation": "..."}`, vilket kostar både prefill och extra output-tokens.
`parse_translation` faller redan tillbaka på rå text, så inget behöver ändras
i uttolkningen.

- [ ] **Step 1: Uppdatera testerna**

Ersätt `TestSystemPrompt` i `bench/tests/test_frontend_mirror.py`:

```python
class TestSystemPrompt(unittest.TestCase):
    def test_uses_first_word_of_language_names(self):
        prompt = system_prompt("Swedish (Source)", "English (Translation)")
        self.assertIn("from Swedish into English", prompt)
        self.assertNotIn("(Source)", prompt)

    def test_asks_for_bare_translation_without_json(self):
        prompt = system_prompt("Swedish", "English")
        self.assertNotIn("JSON", prompt)
        self.assertNotIn("translation\":", prompt)

    def test_stays_short(self):
        # Prompten prefillas vid varje anrop; håll den kort.
        self.assertLess(len(system_prompt("Swedish", "English")), 200)
```

- [ ] **Step 2: Kör testerna och se dem falla**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: FAIL på `test_asks_for_bare_translation_without_json`

- [ ] **Step 3: Korta Python-kopian**

```python
def system_prompt(src_name, dst_name):
    """Speglar prompten som byggs i TranslatorApp.processTranslation."""
    src = src_name.split(" ")[0]
    dst = dst_name.split(" ")[0]
    return (
        f"Translate the user's text from {src} into {dst}. "
        f"Reply with the translation only — no explanations, no alternatives, "
        f"no quotes, no preamble."
    )
```

- [ ] **Step 4: Kör testerna och se dem passera**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: PASS

- [ ] **Step 5: Ändra JS-kopian likadant**

I `frontend/src/TranslatorApp.jsx`, ersätt `systemPrompt`-raden i anropet till
`translateText`:

```jsx
        systemPrompt: `Translate the user's text from ${src.name.split(" ")[0]} into ${dst.name.split(" ")[0]}. Reply with the translation only — no explanations, no alternatives, no quotes, no preamble.`,
```

- [ ] **Step 6: Mät**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 05-plain-prompt --compare 04-sentence-chunks`
Expected: lägre LLM-tid. Grinden flaggar sannolikt gult på ändrad
översättning för varje fixtur — det är väntat och ska granskas manuellt:
läs `NOTERA`-raderna och bekräfta att de nya översättningarna är minst lika
bra. Är någon uppenbart sämre, backa prompten.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/TranslatorApp.jsx bench/frontend_mirror.py bench/tests/test_frontend_mirror.py bench/results/05-plain-prompt.json
git commit -m "Ask Gemma for a bare translation instead of a JSON wrapper"
```

---

## Task 13: Optimering 6 — prefetcha nästa TTS-chunk

> **STRUKEN efter mätning.** Bänkens upplösning mättes i Task 18 till ungefär
> 3 %: två steg som en STT-ändring bevisligen inte kan påverka läste 0,985 och
> 0,974 i stället för 1,000. Den här optimeringen träffar TTS respektive
> klientsidan, som tillsammans utgör 5 % av tiden till första ljud. Även en
> halvering där landar under bruskgolvet och kan alltså inte mätas med den här
> riggen. Eftersom kampanjens hela premiss är att varje förbättring ska mätas,
> byggs den inte. Den kan tas upp igen om bruskgolvet sänks.


**Files:**
- Modify: `frontend/src/TranslatorApp.jsx` (`playTTS`)

Idag skapas nästa chunks `Audio` först i `onended`, så nedladdning och
avkodning sker i glappet mellan chunkarna. Att skapa elementet i förväg
startar hämtningen medan föregående chunk fortfarande spelar.

Detta syns **inte** i bench (som mäter serversvarstider sekventiellt) utan i
`wall_total` i browsern. Mät därför med frontend-instrumenteringen, och kör
bench bara för att bevisa att inget gått sönder.

- [ ] **Step 1: Skriv om `playTTS`**

Ersätt hela `playTTS` i `frontend/src/TranslatorApp.jsx`:

```jsx
  // Speak text via /api/tts, one request per sentence-sized chunk. Chunk N+1's
  // Audio element is created while chunk N plays, so its fetch and decode
  // overlap playback instead of stalling in the gap after `onended`.
  const playTTS = useCallback(
    (text, targetLang) => {
      if (!text) return
      stopSpeaking()

      const chunks = splitTextIntoSpeechChunks(text)
      if (chunks.length === 0) return

      const makePlayer = (index) => {
        const url = `/api/tts?text=${encodeURIComponent(chunks[index])}&lang=${encodeURIComponent(targetLang)}`
        const player = new Audio(url)
        player.preload = "auto"
        player.volume = 1.0
        return player
      }

      let index = 0
      let current = makePlayer(0)
      let upcoming = chunks.length > 1 ? makePlayer(1) : null

      const playCurrent = () => {
        if (!current) {
          stopSpeaking()
          return
        }
        onlineAudioPlayerRef.current = current

        current.onplaying = () => {
          const marks = timingRef.current
          if (!marks || marks.logged) return
          marks.logged = true
          const since = (mark) => `${(mark - marks.keyup) | 0}ms`
          console.log(
            `[latency] keyup→stt ${since(marks.stt)} | →llm ${since(marks.llm)} ` +
              `| →first audio ${since(performance.now())}`,
          )
          setMetaText(
            `STT ${since(marks.stt)} · LLM ${since(marks.llm)} · ljud ${since(performance.now())}`,
          )
        }
        current.onended = () => {
          index++
          current = upcoming
          upcoming = index + 1 < chunks.length ? makePlayer(index + 1) : null
          playCurrent()
        }
        current.onerror = () => {
          stopSpeaking()
          alert("TTS playback failed. Backend server may be offline.")
        }
        current.play().catch((e) => {
          console.error("Audio play error:", e)
          stopSpeaking()
        })
      }

      playCurrent()
    },
    [stopSpeaking],
  )
```

- [ ] **Step 2: Verifiera i browsern med en flerchunksmening**

Run: `npm --prefix frontend run dev -- --port 5273`

Säg något tillräckligt långt för att ge minst tre meningar i översättningen.
Expected: ingen hörbar paus mellan meningarna, och `[latency]`-raden loggas
fortfarande exakt en gång.

- [ ] **Step 3: Bevisa att inget gått sönder**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 06-tts-prefetch --compare 05-plain-prompt`
Expected: siffrorna i stort sett oförändrade (bench mäter inte uppspelning),
exitkod 0.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/TranslatorApp.jsx bench/results/06-tts-prefetch.json
git commit -m "Prefetch the next TTS chunk while the current one plays"
```

---

## Task 14: Optimering 7 — klientsidans ljudväg

> **STRUKEN efter mätning.** Bänkens upplösning mättes i Task 18 till ungefär
> 3 %: två steg som en STT-ändring bevisligen inte kan påverka läste 0,985 och
> 0,974 i stället för 1,000. Den här optimeringen träffar TTS respektive
> klientsidan, som tillsammans utgör 5 % av tiden till första ljud. Även en
> halvering där landar under bruskgolvet och kan alltså inte mätas med den här
> riggen. Eftersom kampanjens hela premiss är att varje förbättring ska mätas,
> byggs den inte. Den kan tas upp igen om bruskgolvet sänks.


**Files:**
- Modify: `frontend/src/hooks/useAudioRecorder.js` (`startRecording`, `finalizeRecording`, `samplesToPayload`)
- Modify: `frontend/src/utils/api.js` (`transcribeAudio`)
- Modify: `frontend/src/TranslatorApp.jsx` (`handleRecordStart`, `handleRecordStop` — fältnamnet ändras)
- Modify: `backend/server.py` (`handle_stt`)
- Modify: `bench/runner.py` (`_post_stt`)

Tre saker på en gång, för att de rör samma väg: be om 16 kHz direkt så
resamplingen försvinner, sluta blockera på `audioContext.close()`, och skicka
binär PCM i stället för base64 (som kostar 33 % extra payload och en
FileReader-runda).

Backenden måste acceptera **båda** formaten, annars slutar den fungera för
en frontend som inte byggts om. Och `bench/runner.py` måste byta till binärt
samtidigt, annars mäter bench den gamla vägen.

- [ ] **Step 1: Låt backenden ta emot binär PCM**

I `handle_stt`, ersätt inledningen fram till och med `raw_data`-raden:

```python
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            if not body:
                raise ValueError("No body data")

            # Two accepted shapes: raw Float32 PCM (application/octet-stream,
            # what the browser sends) or the older JSON+base64 envelope.
            content_type = self.headers.get('Content-Type', '')
            if content_type.startswith('application/octet-stream'):
                language = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query
                ).get('language', ['en'])[0]
                raw_data = body
            else:
                data = json.loads(body.decode('utf-8'))
                audio_b64 = data.get('audio_base64')
                if not audio_b64:
                    raise ValueError("Missing audio_base64 parameter")
                language = data.get('language', 'en')
                raw_data = base64.b64decode(audio_b64)
```

- [ ] **Step 2: Skicka binärt från frontend**

I `frontend/src/utils/api.js`, ersätt `transcribeAudio`:

```js
// POST raw Float32 PCM (16 kHz mono) to the local Whisper STT. Sending the
// buffer as-is avoids base64's 33% size penalty and the FileReader round trip.
export async function transcribeAudio(pcmBuffer, sourceLangCode) {
  const response = await fetch(
    `/api/stt?language=${encodeURIComponent(sourceLangCode)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: pcmBuffer,
    },
  )

  if (!response.ok) {
    throw new Error(`STT failed: ${response.status}`)
  }

  const sttData = await response.json()
  return sttData.text || ""
}
```

- [ ] **Step 3: Låt inspelaren returnera bufferten och be om 16 kHz**

I `frontend/src/hooks/useAudioRecorder.js`, ersätt `samplesToPayload`:

```js
async function samplesToPayload(samples, sampleRate) {
  const targetSampleRate = 16000 // Whisper STT expects 16 kHz mono
  // The AudioContext is opened at 16 kHz, so this is usually a no-op; it
  // stays as a fallback for browsers that ignore the sampleRate hint.
  const resampledSamples =
    sampleRate === targetSampleRate
      ? samples
      : resample(samples, sampleRate, targetSampleRate)

  let peak = 0
  for (let i = 0; i < resampledSamples.length; i++) {
    const a = Math.abs(resampledSamples[i])
    if (a > peak) peak = a
  }
  console.log(
    `[mic] ${((resampledSamples.length / targetSampleRate) * 1000) | 0}ms, peak=${peak.toFixed(3)}`,
  )

  // Slice the exact view — Float32Array.buffer may be a larger pooled buffer.
  const pcmBuffer = resampledSamples.buffer.slice(
    resampledSamples.byteOffset,
    resampledSamples.byteOffset + resampledSamples.byteLength,
  )
  return { pcmBuffer }
}
```

I `startRecording`, be om 16 kHz på båda ställena:

```js
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 16000,
        },
      })
```

```js
      const AudioContext = window.AudioContext || window.webkitAudioContext
      // Ask for 16 kHz up front so samplesToPayload has nothing to resample.
      audioContextRef.current = new AudioContext({ sampleRate: 16000 })
```

I `finalizeRecording`, sluta vänta in nedstängningen:

```js
    // Fire and forget: closing the context takes a few ms and nothing below
    // needs it. Awaiting it here put it straight on the critical path between
    // key release and the STT request.
    const closing = audioContextRef.current
    audioContextRef.current = null
    if (closing && closing.state !== "closed") {
      closing.close().catch(() => {})
    }
```

- [ ] **Step 4: Byt fältnamn i `TranslatorApp`**

Tre ställen refererar `base64Data`:

```jsx
      if (result !== true && result.pcmBuffer) {
        setActiveLaneRecording(null)
        processTranslation(lane, result.pcmBuffer)
      }
```

```jsx
    processTranslation(recordedLane, audioData.pcmBuffer)
```

```jsx
  const processTranslation = async (lane, pcmBuffer) => {
```

och anropet inuti:

```jsx
      const transcribedText = await transcribeAudio(pcmBuffer, src.code)
```

- [ ] **Step 5: Byt bench till binärt**

I `bench/runner.py`, ersätt `_post_stt`:

```python
def _post_stt(api_base, samples, language):
    payload = samples.astype("<f4").tobytes()
    started = time.perf_counter()
    response = requests.post(
        f"{api_base}/api/stt",
        params={"language": language},
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        timeout=REQUEST_TIMEOUT,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json().get("text", ""), elapsed_ms, len(payload)
```

`base64` kan tas bort ur importerna.

- [ ] **Step 6: Verifiera att båda formaten fungerar**

Run:
```bash
lsof -ti:3100 | xargs kill 2>/dev/null
PORT=3100 venv/bin/python3 backend/server.py > /tmp/bench-backend.log 2>&1 &
sleep 25
venv/bin/python3 -c "
import base64, numpy as np, requests
from bench.fixtures import ensure_wav, load_pcm_16k, load_fixtures
samples = load_pcm_16k(ensure_wav('sv-short', load_fixtures()['sv-short']))
raw = samples.astype('<f4').tobytes()

binary = requests.post('http://localhost:3100/api/stt', params={'language': 'sv'},
                       data=raw, headers={'Content-Type': 'application/octet-stream'}, timeout=120)
legacy = requests.post('http://localhost:3100/api/stt',
                       json={'audio_base64': base64.b64encode(raw).decode(), 'language': 'sv'}, timeout=120)
print('binary:', binary.status_code, repr(binary.json()['text']))
print('legacy:', legacy.status_code, repr(legacy.json()['text']))
assert binary.json()['text'] == legacy.json()['text'], 'formaten gav olika transkription'
print('OK: båda formaten ger samma resultat')
"
lsof -ti:3100 | xargs kill 2>/dev/null
```
Expected: `OK: båda formaten ger samma resultat`

- [ ] **Step 7: Verifiera i browsern**

Run: `npm --prefix frontend run dev -- --port 5273`

Håll Z, säg en mening, släpp. Expected: transkriptionen fungerar som förut,
och `[mic]`-raden i konsolen visar en rimlig längd. Om ljudet blir tomt eller
förvrängt har webbläsaren avvisat 16 kHz-begäran — kontrollera
`audioContextRef.current.sampleRate` i konsolen och låt fallbacken i
`samplesToPayload` ta hand om det.

- [ ] **Step 8: Mät**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 07-binary-pcm --compare 06-tts-prefetch`
Expected: något lägre `STT` (mindre payload att ta emot och avkoda). Vinsten
är liten på Mac och större på Pi.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/hooks/useAudioRecorder.js frontend/src/utils/api.js frontend/src/TranslatorApp.jsx backend/server.py bench/runner.py bench/results/07-binary-pcm.json
git commit -m "Send raw 16 kHz PCM to STT instead of base64"
```

---

## Task 15: Optimering 8 — strömma Gemma till TTS

**Files:**
- Modify: `backend/server.py` (`handle_proxy`)
- Modify: `frontend/src/utils/api.js` (`parseTranslation` bryts ut, `translateTextStreaming` läggs till)
- Modify: `frontend/src/TranslatorApp.jsx` (TTS-kö, `processTranslation`)
- Modify: `bench/runner.py` (`_post_llm_streaming`, `RunResult.llm_first_sentence_ms`)
- Modify: `bench/report.py` (`METRICS`)
- Modify: `bench/bench.py` (`--stream`-flagga)

Den enda ändringen som rör arkitekturen. `handle_proxy` läser idag hela
LLM-svaret med `response.read()` innan något skickas vidare, vilket gör
strömning omöjlig. Målet: så snart Gemma skrivit klart första meningen börjar
den syntetiseras och spelas, medan resten fortfarande genereras.

Den här taskens TTS-kö ersätter prefetch-logiken från Task 13 — kön skapar
`Audio`-elementen när meningarna köas, vilket ger samma överlappning.

- [ ] **Step 1: Låt proxyn strömma vidare**

I `handle_proxy`, ersätt `try`-blockets `urlopen`-del:

```python
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                self.send_response(response.status)
                # Content-Length droppas: vi vet inte längden i förväg när vi
                # strömmar, och HTTP/1.0-svar avslutas ändå av connection close.
                for key, val in response.headers.items():
                    if key.lower() not in ['content-length', 'connection', 'transfer-encoding']:
                        self.send_header(key, val)
                self.end_headers()
                # Skriv vidare chunk för chunk i stället för att buffra hela
                # svaret — annars kan klienten inte se en SSE-delta förrän
                # genereringen är klar, och strömningen är meningslös.
                while True:
                    chunk = response.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
```

- [ ] **Step 2: Verifiera att proxyn strömmar**

Run:
```bash
lsof -ti:3100 | xargs kill 2>/dev/null
PORT=3100 venv/bin/python3 backend/server.py > /tmp/bench-backend.log 2>&1 &
sleep 25
venv/bin/python3 -c "
import time, urllib.parse, requests
url = urllib.parse.quote('http://localhost:9379/v1/chat/completions', safe='')
started = time.perf_counter()
first = None
with requests.post(f'http://localhost:3100/proxy?url={url}',
                   json={'model': 'gemma4-e2b', 'stream': True,
                         'messages': [{'role': 'user', 'content': 'Count from one to twenty in words.'}]},
                   stream=True, timeout=300) as response:
    for line in response.iter_lines():
        if line and first is None:
            first = time.perf_counter() - started
total = time.perf_counter() - started
print(f'första chunk efter {first*1000:.0f} ms, hela svaret efter {total*1000:.0f} ms')
assert first < total * 0.6, 'proxyn buffrar fortfarande — första chunken kom för sent'
print('OK: proxyn strömmar')
"
lsof -ti:3100 | xargs kill 2>/dev/null
```
Expected: `OK: proxyn strömmar`

- [ ] **Step 3: Bryt ut JSON-uttolkningen i api.js**

Lägg till ovanför `translateText` i `frontend/src/utils/api.js`:

```js
// Tolerates ``` fences and falls back to the raw reply when the model didn't
// return the JSON envelope the prompt asked for.
function parseTranslation(modelResponse) {
  let cleanJson = modelResponse.trim()
  if (cleanJson.startsWith("```json")) cleanJson = cleanJson.slice(7)
  if (cleanJson.startsWith("```")) cleanJson = cleanJson.slice(3)
  if (cleanJson.endsWith("```")) cleanJson = cleanJson.slice(0, -3)
  try {
    return JSON.parse(cleanJson.trim()).translation || ""
  } catch (e) {
    return modelResponse
  }
}
```

Ersätt sedan `try { ... } catch (e) { translationVal = modelResponse }`-blocket
i `translateText` med `const translationVal = parseTranslation(modelResponse)`.

- [ ] **Step 4: Lägg till den strömmande varianten**

Efter `translateText` i `frontend/src/utils/api.js`:

```js
// Streaming chat-completions. Calls `onText(fullTextSoFar)` after every delta
// so the caller can start speaking sentence one while the rest generates.
export async function translateTextStreaming(transcribedText, config, onText) {
  const { endpointUrl, useProxy, apiKey, modelName, systemPrompt } = config
  const targetUrl = `${getNormalizedBaseUrl(endpointUrl)}/chat/completions`
  const payload = JSON.parse(
    generatePayloadJSON(transcribedText, modelName, systemPrompt),
  )
  payload.stream = true

  const headers = { "Content-Type": "application/json" }
  if (apiKey && apiKey.trim() !== "") {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`
  }
  const fetchUrl = useProxy
    ? `/proxy?url=${encodeURIComponent(targetUrl)}`
    : targetUrl

  const startRequestTime = Date.now()
  const response = await fetch(fetchUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`API ${response.status}: ${errorText || response.statusText}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let text = ""

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // Keep the last, possibly incomplete, line in the buffer.
    const lines = buffer.split("\n")
    buffer = lines.pop()
    for (const line of lines) {
      if (!line.startsWith("data:")) continue
      const data = line.slice(5).trim()
      if (!data || data === "[DONE]") continue
      try {
        const delta = JSON.parse(data).choices?.[0]?.delta?.content
        if (delta) {
          text += delta
          onText(text)
        }
      } catch (e) {
        // A malformed line is not worth aborting the stream over.
      }
    }
  }

  return {
    translation: parseTranslation(text),
    duration: ((Date.now() - startRequestTime) / 1000).toFixed(2),
  }
}
```

- [ ] **Step 5: Bygg om TTS till en kö**

I `frontend/src/TranslatorApp.jsx`, ersätt `stopSpeaking` och `playTTS`:

```jsx
  // Sentences waiting to be spoken. Each entry is an already-constructed Audio
  // element: creating it starts the fetch, so queued chunks download while the
  // current one plays.
  const ttsQueueRef = useRef({ pending: [], playing: false })

  const markFirstAudio = useCallback(() => {
    const marks = timingRef.current
    if (!marks || marks.logged) return
    marks.logged = true
    const since = (mark) => `${(mark - marks.keyup) | 0}ms`
    console.log(
      `[latency] keyup→stt ${since(marks.stt)} | →llm ${since(marks.llm)} ` +
        `| →first audio ${since(performance.now())}`,
    )
    setMetaText(
      `STT ${since(marks.stt)} · LLM ${since(marks.llm)} · ljud ${since(performance.now())}`,
    )
  }, [])

  const stopSpeaking = useCallback(() => {
    ttsQueueRef.current.pending = []
    ttsQueueRef.current.playing = false
    if (onlineAudioPlayerRef.current) {
      onlineAudioPlayerRef.current.pause()
      onlineAudioPlayerRef.current = null
    }
  }, [])

  const pumpTTSQueue = useCallback(() => {
    const queue = ttsQueueRef.current
    if (queue.playing) return
    const player = queue.pending.shift()
    if (!player) return
    queue.playing = true
    onlineAudioPlayerRef.current = player
    player.onplaying = markFirstAudio
    player.onended = () => {
      queue.playing = false
      pumpTTSQueue()
    }
    player.onerror = () => {
      queue.playing = false
      stopSpeaking()
      alert("TTS playback failed. Backend server may be offline.")
    }
    player.play().catch((e) => {
      console.error("Audio play error:", e)
      queue.playing = false
      stopSpeaking()
    })
  }, [markFirstAudio, stopSpeaking])

  // Queue text for playback. Safe to call repeatedly as sentences arrive.
  const enqueueTTS = useCallback(
    (text, targetLang) => {
      if (!text) return
      for (const chunk of splitTextIntoSpeechChunks(text)) {
        const url = `/api/tts?text=${encodeURIComponent(chunk)}&lang=${encodeURIComponent(targetLang)}`
        const player = new Audio(url)
        player.preload = "auto"
        player.volume = 1.0
        ttsQueueRef.current.pending.push(player)
      }
      pumpTTSQueue()
    },
    [pumpTTSQueue],
  )
```

Byt importen från `translateText` till `translateTextStreaming` överst i filen.

- [ ] **Step 6: Tala meningarna medan de kommer**

I `processTranslation`, ersätt hela steg 2-blocket (från `const result = await translateText(` till och med `playTTS(...)`):

```jsx
      // 2. Translation, streamed — speak each sentence as soon as it lands.
      let spokenChars = 0
      const speakCompleteSentences = (full, isFinal) => {
        let upto = full.length
        if (!isFinal) {
          const lastEnd = Math.max(
            full.lastIndexOf("."),
            full.lastIndexOf("!"),
            full.lastIndexOf("?"),
            full.lastIndexOf("…"),
          )
          if (lastEnd < spokenChars) return
          upto = lastEnd + 1
        }
        const ready = full.slice(spokenChars, upto).trim()
        if (!ready) return
        spokenChars = upto
        if (config.enableTts) enqueueTTS(ready, dst.ttsLang)
      }

      const result = await translateTextStreaming(
        transcribedText,
        {
          ...config,
          modelName: config.modelName,
          systemPrompt: `Translate the user's text from ${src.name.split(" ")[0]} into ${dst.name.split(" ")[0]}. Reply with the translation only — no explanations, no alternatives, no quotes, no preamble.`,
        },
        (partial) => {
          setTranslationData((prev) => ({ ...prev, text: partial }))
          speakCompleteSentences(partial, false)
        },
      )

      if (timingRef.current) timingRef.current.llm = performance.now()
      setTranslationData((prev) => ({ ...prev, text: result.translation }))
      speakCompleteSentences(result.translation, true)
```

`timingRef.current.llm` stämplas nu när strömmen är slut, medan
`first audio` kan inträffa långt tidigare. Att `→llm` blir *större* än
`→first audio` i loggen är alltså det förväntade beviset på att strömningen
fungerar.

- [ ] **Step 7: Verifiera i browsern**

Run: `npm --prefix frontend run dev -- --port 5273`

Säg en lång mening som ger flera meningar i översättningen.
Expected: uppspelningen börjar innan hela texten skrivits ut i drawern, och
`[latency]`-raden visar ett `→first audio` som är mindre än `→llm`.

- [ ] **Step 8: Lägg till strömmande mätning i bench**

I `bench/runner.py`, lägg till fältet i `RunResult` (efter `llm_ms`):

```python
    llm_first_sentence_ms: float = 0.0
```

och ändra `time_to_first_audio_ms` samt lägg till strömfunktionen:

```python
    @property
    def time_to_first_audio_ms(self):
        # Vid strömning börjar TTS på första meningen, inte på hela svaret.
        llm_part = self.llm_first_sentence_ms or self.llm_ms
        return self.stt_ms + llm_part + self.tts_first_ms
```

```python
_SENTENCE_END = (".", "!", "?", "…")


def _post_llm_streaming(api_base, llm_url, model, text, src_lang, dst_lang):
    """Som _post_llm men mäter också när första hela meningen är klar."""
    prompt = system_prompt(LANGUAGE_NAMES[src_lang], LANGUAGE_NAMES[dst_lang])
    payload = build_llm_payload(text, model, prompt)
    payload["stream"] = True
    proxied = f"{api_base}/proxy?url={urllib.parse.quote(llm_url, safe='')}"

    started = time.perf_counter()
    first_sentence_ms = 0.0
    accumulated = ""
    with requests.post(proxied, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if not delta:
                continue
            accumulated += delta
            if not first_sentence_ms and accumulated.rstrip().endswith(_SENTENCE_END):
                first_sentence_ms = (time.perf_counter() - started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    if not first_sentence_ms:
        # Svar utan skiljetecken: hela svaret är "första meningen".
        first_sentence_ms = total_ms
    return parse_translation(accumulated), total_ms, first_sentence_ms
```

Lägg till `import json` överst, och en `stream=False`-parameter på
`run_fixture` som väljer väg:

```python
def run_fixture(api_base, llm_url, model, fixture_id, spec, stream=False):
    ...
        if stream:
            result.translation, result.llm_ms, result.llm_first_sentence_ms = _post_llm_streaming(
                api_base, llm_url, model, result.transcript, spec["lang"], spec["target"]
            )
        else:
            result.translation, result.llm_ms = _post_llm(
                api_base, llm_url, model, result.transcript, spec["lang"], spec["target"]
            )
```

`warmup` får samma parameter och skickar den vidare.

I `bench/report.py`, lägg `"llm_first_sentence_ms"` i `METRICS` efter
`"llm_ms"`. I `bench/bench.py`, lägg till flaggan och skicka den vidare:

```python
    parser.add_argument("--stream", action="store_true", help="Mät den strömmande LLM-vägen")
```

```python
        warmup(api_base, llm_url, args.model, fixtures, stream=args.stream)
```

```python
                result = run_fixture(api_base, llm_url, args.model, fixture_id, spec, stream=args.stream)
```

- [ ] **Step 9: Kör enhetstesterna**

Run: `venv/bin/python3 -m unittest discover -s bench/tests -t . -v`
Expected: PASS. `summarize` hämtar metrics med `getattr`, så det nya fältet
följer med automatiskt.

- [ ] **Step 10: Mät**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 08-streaming --stream --compare 07-binary-pcm`
Expected: `Till första ljud` klart lägre, medan `wall_total` är ungefär
oförändrad. Det är precis den vinst huvudmåttet är valt för att fånga.

- [ ] **Step 11: Commit**

```bash
git add backend/server.py frontend/src/utils/api.js frontend/src/TranslatorApp.jsx bench/runner.py bench/report.py bench/bench.py bench/results/08-streaming.json
git commit -m "Stream Gemma output and speak each sentence as it lands"
```

---

## Task 16: Optimering 9 — GPU-varianten av modellen

**Files:**
- Modify: `frontend/src/components/SettingsOverlay.jsx` (modellfältets hjälptext)
- Modify: `README.md`

litert-lm exponerar två modell-id: `gemma4-e2b` och `gemma4-e2b,gpu`. Uppmätt
på denna Mac med kort systemprompt: 1126–1660 ms på CPU mot 579 ms varm på
GPU. Första anropet kostar däremot 7,3 s medan vikterna laddas.

Detta ska **inte** bli default i repot: GPU-varianten är plattformsberoende
och Pi 5 är målhårdvaran. Vi mäter den, dokumenterar den, och låter den vara
ett val.

- [ ] **Step 1: Mät GPU-varianten**

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label 09-gpu-model --stream --model "gemma4-e2b,gpu" --compare 08-streaming`

Uppvärmningen i bench körs före mätningen, så 7-sekunderskostnaden för
modelladdning hamnar inte i siffrorna.
Expected: lägre `LLM`-kolumn, oförändrad WER (samma modellvikter).

- [ ] **Step 2: Dokumentera valet i README**

I `README.md`, i benchmark-avsnittet från Task 6, lägg till:

```markdown
On machines with a supported GPU, litert-lm also serves the model as
`gemma4-e2b,gpu`. Point the model name in Settings at it, or measure it with
`--model "gemma4-e2b,gpu"`. It is roughly twice as fast per translation on an
M1 Mac but costs several seconds on the first request while weights load, and
it is not the default because the Raspberry Pi 5 target has no such variant.
```

- [ ] **Step 3: Nämn varianten i inställningarna**

I `frontend/src/components/SettingsOverlay.jsx`, i hjälptexten under
modellnamnsfältet, lägg till meningen:

```
Try "gemma4-e2b,gpu" if your machine has a supported GPU.
```

- [ ] **Step 4: Commit**

```bash
git add README.md frontend/src/components/SettingsOverlay.jsx bench/results/09-gpu-model.json
git commit -m "Measure and document the GPU model variant"
```

---

## Task 17: Sammanställ kampanjen

**Files:**
- Modify: `README.md` (`## Latency notes & ideas`)

- [ ] **Step 1: Kontrollera kvalitetsdrift mot baseline**

Varje steg har jämförts med sin föregångare, vilket betyder att WER kan ha
krupit uppåt några hundradelar i taget utan att någon enskild grind fällde.

Run: `STT_CHUNK_MARGIN_S=5 venv/bin/python3 -m bench.bench --label final --stream --compare baseline`
Expected: exitkod 0. Fäller den här körningen har kvaliteten drivit, och det
måste redas ut innan kampanjen kallas färdig — läs `REGRESSION`-raderna och
identifiera vilket steg som bär ansvaret genom att jämföra `wer`-fälten i
`bench/results/*.json`.

- [ ] **Step 2: Kör browser-milstolpen**

Run: `npm --prefix frontend run dev -- --port 5273`

Håll Z, säg samma mening som vid baseline-mätningen i Task 7, släpp. Anteckna
`[latency]`-raden och jämför mot den du sparade då.

- [ ] **Step 3: Skriv om README:s latensavsnitt**

Ersätt hela `## Latency notes & ideas`-avsnittet. Idélistan är genomförd, så
den ska bli en resultattabell. Hämta siffrorna ur `bench/results/`:

```markdown
## Latency

Measured with `bench/` on a MacBook Air M1 (8 cores), Whisper `small` int8,
`gemma4-e2b` via LiteRT-LM. Median time to first audio across nine fixtures:

| Change | Time to first audio | Δ |
| :--- | ---: | ---: |
| Baseline | _fyll i från bench/results/baseline.json_ | — |
| + Whisper VAD filter | | |
| + short mel padding (`STT_CHUNK_MARGIN_S=5`) | | |
| + decode flags and `cpu_threads` | | |
| + sentence-level TTS chunking | | |
| + plain-text prompt instead of JSON | | |
| + TTS prefetch | | |
| + raw PCM upload | | |
| + streamed Gemma → TTS | | |
| + `gemma4-e2b,gpu` (Mac only) | | |

Re-measure with `venv/bin/python3 -m bench.bench --label <name> --compare baseline`.

Still on the table: streaming mic audio to Whisper *during* recording, which
would take STT off the critical path almost entirely. It needs a chunked
upload protocol and incremental decoding, so it was left out of this round.
```

Fyll i varje rad med `median_time_to_first_audio_ms` ur motsvarande
resultatfil. Har någon optimering visat sig vara en försämring och därför inte
committats, ta bort dess rad och notera i stycket under tabellen att den
prövades och förkastades, med siffran som motiverar det.

- [ ] **Step 4: Commit**

```bash
git add README.md bench/results/final.json
git commit -m "Replace the latency idea list with measured results"
```

---

## Verifiering av hela planen

Kampanjen är klar när:

- `venv/bin/python3 -m unittest discover -s bench/tests -t .` är grön
- `venv/bin/python3 -m bench.bench --label final --stream --compare baseline` ger exitkod 0
- `bench/results/` innehåller en fil per steg, alla committade
- README:s latenstabell är ifylld med riktiga siffror
- Browser-milstolpen visar `→first audio` mindre än `→llm`, vilket bevisar att strömningen ger effekt i den riktiga appen

---

## Task 18: Parvis A/B-mätning

**Files:**
- Modify: `backend/server.py` (VAD bakom env-flagga)
- Modify: `bench/bench.py` (`--ab`, två backends, varvade repetitioner)
- Modify: `bench/runner.py` (oförändrat gränssnitt, men anropas per arm)
- Modify: `bench/report.py` (parvis sammanställning)
- Create: `bench/tests/test_report_ab.py`

### Varför

Mätningen av optimering 1 gav −20 % till första ljud. Kontrollen visade att
siffran inte gick att tillskriva ändringen: samtliga nio transkriptioner och
samtliga nio översättningar var bytemässigt identiska mellan baseline och
`01-vad` — LLM-steget gjorde alltså exakt samma arbete — men dess summerade
tid föll ändå från 25 996 ms till 21 035 ms. VAD kan inte göra Gemma
snabbare. Skillnaden var maskinbelastning.

Två körningar tagna vid olika tidpunkter är inte jämförbara på den här
maskinen. Effekterna vi jagar (enstaka procent till tiotals procent) är
mindre än driften mellan körningar. Lösningen är att mäta båda
konfigurationerna i samma körning, varvade, så att drift träffar båda armarna
lika.

### Designbeslut

**1. Optimeringar env-gatas, med produktens ursprungsbeteende som default.**
En optimering som är inbakad i koden går inte att A/B-testa utan att bygga om.
Varje optimering får därför en env-flagga som default är av, alltså
oförändrat beteende. Defaultarna flippas till på i Task 17, efter att de
mätts. Det gör också att produkten inte tar in någon ändring innan den är
bevisad.

VAD flyttas som första exempel: `STT_VAD` (default `0`).

**2. Bänken startar två backends och varvar per repetition.**
`--ab "KEY=VAL[,KEY2=VAL2]"` startar arm B på `--api-port + 1` med de
angivna env-variablerna ovanpå arm A:s miljö. Arm A är den oförändrade
miljön.

**3. Ordningen alterneras (ABBA).** Varvar man alltid A före B lägger sig en
eventuell systematisk ordningseffekt — cache-uppvärmning, termisk drift —
konsekvent på den ena armen. Repetition 1 kör A,B; repetition 2 kör B,A; och
så vidare.

**4. Statistiken är parvis.** Rapportera medianen av *kvoterna per par*
(`B/A` för varje repetition), inte kvoten mellan medianerna. Ett par mätt
inom några sekunder av varandra delar belastningstillstånd; att jämföra
aggregat över hela körningen gör inte det.

**5. Drift redovisas per steg.** Rapporten skriver ut den parvisa kvoten för
varje steg — `stt_ms`, `llm_ms`, `tts_first_ms`. Den som läser tabellen kan
då själv se om ett steg som ändringen omöjligt kan påverka ändå rört sig,
vilket är signalen att mätningen är otillförlitlig. Det var precis den
kontrollen som avslöjade problemet med optimering 1.

- [ ] **Step 1: Env-gata VAD**

I `backend/server.py`, bredvid `WHISPER_MODEL_SIZE`:

```python
# Optimeringar hålls bakom flaggor så att bench kan A/B-testa dem i samma
# körning, och så att produkten behåller sitt ursprungsbeteende tills en
# ändring är uppmätt. Defaultarna flippas när kampanjen är klar.
STT_VAD = os.environ.get("STT_VAD", "0") == "1"
```

och i `transcribe`, ersätt `vad_filter=True` med `vad_filter=STT_VAD`.

- [ ] **Step 2: Verifiera att båda lägena fungerar**

Run:
```bash
lsof -ti:3100,3101 | xargs kill 2>/dev/null
for v in 0 1; do
  STT_VAD=$v PORT=3100 venv/bin/python3 backend/server.py > /tmp/vad-$v.log 2>&1 &
  sleep 25
  venv/bin/python3 -c "
import numpy as np, requests, base64
from bench.fixtures import ensure_wav, load_pcm_16k, load_fixtures
s = load_pcm_16k(ensure_wav('sv-medium', load_fixtures()['sv-medium']))
r = requests.post('http://localhost:3100/api/stt',
                  json={'audio_base64': base64.b64encode(s.astype('<f4').tobytes()).decode(), 'language':'sv'},
                  timeout=120)
print('STT_VAD=$v ->', repr(r.json()['text']))
"
  lsof -ti:3100 | xargs kill 2>/dev/null; sleep 1
done
```
Expected: båda ger en rimlig svensk transkription. De behöver inte vara
identiska — VAD ändrar vad modellen ser.

- [ ] **Step 3: Lägg till `--ab` i bench**

`start_backend(port, extra_env=None)` tar nu extra miljövariabler. `main`
parsar `--ab` till en dict, startar arm B på `args.api_port + 1` när flaggan
är satt, och kör mätslingan så här:

```python
        for fixture_id, spec in fixtures.items():
            runs_a, runs_b = [], []
            for attempt in range(args.repeats):
                # ABBA: varannan repetition kör B först, så att en systematisk
                # ordningseffekt inte lägger sig på samma arm varje gång.
                order = ["a", "b"] if attempt % 2 == 0 else ["b", "a"]
                for arm in order:
                    base = api_base_a if arm == "a" else api_base_b
                    result = run_fixture(base, llm_url, args.model, fixture_id, spec)
                    (runs_a if arm == "a" else runs_b).append(result)
                    print(
                        f"[bench] {fixture_id} {attempt + 1}/{args.repeats} arm {arm.upper()}: "
                        f"{result.time_to_first_audio_ms:.0f} ms, "
                        f"{'ok' if result.ok else 'FEL ' + result.error}",
                        flush=True,
                    )
            per_fixture_a[fixture_id] = summarize(runs_a[1:] or runs_a, spec["text"])
            per_fixture_b[fixture_id] = summarize(runs_b[1:] or runs_b, spec["text"])
            paired[fixture_id] = paired_ratios(runs_a[1:] or runs_a, runs_b[1:] or runs_b)
```

Båda armarna värms upp innan mätningen börjar. Resultatfilen får både armarnas
fulla rapporter och den parvisa sammanställningen.

- [ ] **Step 4: Parvis statistik i `bench/report.py`**

```python
def paired_ratios(runs_a, runs_b):
    """Median av kvoten B/A per repetitionspar, per mätpunkt.

    Paren mäts inom sekunder från varandra och delar därför
    belastningstillstånd. Att i stället dela medianerna mot varandra jämför
    aggregat som kan ha tagits under helt olika förhållanden — vilket är
    exakt det felet den här funktionen finns för att undvika.
    """
    ratios = {}
    for metric in METRICS:
        values = []
        for run_a, run_b in zip(runs_a, runs_b):
            if not (run_a.ok and run_b.ok):
                continue
            a = getattr(run_a, metric)
            if a > 0:
                values.append(getattr(run_b, metric) / a)
        ratios[metric] = median(values) if values else None
    return ratios
```

Och en renderare som skriver en rad per fixtur med den parvisa kvoten för
`time_to_first_audio_ms`, plus en driftrad per steg över alla fixturer.

- [ ] **Step 5: Tester**

Skapa `bench/tests/test_report_ab.py` med minst:
- att `paired_ratios` parar ihop repetitionerna i ordning, inte aggregat
- att ett par där någon arm misslyckades hoppas över utan att skeva resten
- att en konstruerad 20-procentig förbättring i arm B ger kvot 0,8, även när
  de absoluta talen driver kraftigt mellan paren (bygg testdatan så att
  kvoten mellan medianerna skulle ge ett annat svar — det är hela poängen)

- [ ] **Step 6: Mät om VAD parvis**

Run: `venv/bin/python3 -m bench.bench --label 01-vad-ab --ab "STT_VAD=1" --repeats 5`

Expected: `llm_ms`-driftkvoten ligger nära 1,00 — LLM:et gör identiskt arbete
i båda armarna, så allt annat än ~1,00 betyder att mätningen fortfarande är
otillförlitlig och att `--repeats` behöver höjas. `stt_ms`-kvoten är VAD:s
verkliga effekt.

Rapportera den siffran som optimering 1:s resultat och notera i rapporten hur
den skiljer sig från de −20 % den okontrollerade jämförelsen gav.

- [ ] **Step 7: Commit**

```bash
git add backend/server.py bench/bench.py bench/report.py bench/tests/test_report_ab.py bench/results/01-vad-ab.json
git commit -m "Measure optimizations as paired A/B runs to cancel machine load drift"
```

### Följd för optimering 2-9

Varje kvarvarande optimering env-gatas på samma sätt och mäts med `--ab` i
stället för `--compare`. `--compare` finns kvar för att jämföra körningar över
tid, men är inte längre hur en optimering bedöms.
