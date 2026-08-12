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
  frontend/src/utils/api.js:154-168  splitTextIntoSpeechChunks
  frontend/src/utils/api.js:72-83    generatePayloadJSON
  frontend/src/utils/api.js:126-144  JSON-uttolkningen i translateText
  frontend/src/TranslatorApp.jsx:229 systemprompten i processTranslation

Ändras någon av dem måste den här filen ändras i samma commit.
"""

import json
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
    """Speglar prompten som byggs i TranslatorApp.jsx:229 (processTranslation)."""
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
