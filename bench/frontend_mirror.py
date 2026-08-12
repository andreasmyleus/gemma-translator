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
  frontend/src/TranslatorApp.jsx:229 systemprompten i handleRecordStop

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
    """Speglar prompten som byggs i TranslatorApp.jsx:229 (handleRecordStop)."""
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


def parse_translation(model_response):
    """Speglar JSON-uttolkningen i translateText (api.js:126-144).

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
