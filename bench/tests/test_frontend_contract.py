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

"""Låser kontraktet mellan frontend och bench:s spegling av den.

Två av kampanjens dyraste fel var av den här sorten och båda är osynliga för
vanliga tester: bench mätte en systemprompt produkten inte längre skickade
(fyra veckors mätningar på fel sida kontextklippan), och en omskrivning av
uppspelningsvägen tog tyst bort en tidigare bugfix. Testerna nedan läser
JSX/JS-källan som text — trubbigt, men det är det enda som fäller när någon
ändrar den ena sidan och glömmer den andra.
"""

import pathlib
import re
import unittest

from bench.frontend_mirror import system_prompt

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
TRANSLATOR_APP = REPO_DIR / "frontend" / "src" / "TranslatorApp.jsx"
API_JS = REPO_DIR / "frontend" / "src" / "utils" / "api.js"


def read(path):
    return path.read_text(encoding="utf-8")


class TestSystemPromptIsInSync(unittest.TestCase):
    def test_bench_default_prompt_is_byte_identical_to_the_products(self):
        source = read(TRANSLATOR_APP)
        match = re.search(r"systemPrompt: `([^`]*)`", source)
        self.assertIsNotNone(
            match, "hittar ingen systemPrompt-template i TranslatorApp.jsx"
        )
        template = match.group(1)
        # Två platshållare i ordning: källspråk, målspråk.
        names = iter(["Swedish", "English"])
        rendered = re.sub(r"\$\{[^}]*\}", lambda _: next(names), template)
        self.assertEqual(
            rendered,
            system_prompt("Swedish", "English"),
            "frontend_mirror.system_prompt måste vara exakt prompten produkten "
            "skickar — annars mäter varje enarmskörning och varje arm A något "
            "appen aldrig gör.",
        )


class TestPlaybackPathContract(unittest.TestCase):
    def setUp(self):
        self.source = read(TRANSLATOR_APP)

    def test_playback_binds_its_own_marks_object(self):
        # Commit 08c6df0. Läses timingRef.current om inuti onplaying kan en sent
        # startad chunk från yttrande 1 stämpla `logged` på yttrande 2:s marks.
        self.assertIn("reportLatency(marks, true)", self.source)
        pump = self.source.split("const pumpTTSQueue")[1].split("const enqueueTTS")[0]
        self.assertNotIn(
            "timingRef.current",
            pump,
            "pumpTTSQueue får inte läsa timingRef.current — marks ska följa med "
            "från kön (commit 08c6df0).",
        )

    def test_latency_line_never_prints_a_fabricated_zero(self):
        # `(undefined - keyup) | 0` är 0, vilket såg ut som "LLM 0ms".
        self.assertIn(
            'typeof mark === "number" ? `${(mark - marks.keyup) | 0}ms` : "—"',
            self.source,
        )

    def test_llm_mark_is_stamped_at_the_first_token(self):
        self.assertIn("if (marks && !marks.llm) marks.llm = performance.now()", self.source)

    def test_a_superseded_translation_stops_speaking_and_writing(self):
        self.assertIn("generationRef.current += 1", self.source)
        self.assertIn("const isCurrent = () => generationRef.current === generation", self.source)
        speak = self.source.split("const speakCompleteSentences")[1].split("\n      }")[0]
        self.assertIn(
            "if (!isCurrent()) return",
            speak,
            "speakCompleteSentences måste vägra tala för ett överkört yttrande.",
        )

    def test_meta_line_is_written_even_with_tts_disabled(self):
        self.assertIn("if (!cfg.enableTts) reportLatency(marks, false)", self.source)

    def test_enter_mid_capture_finishes_the_utterance_instead_of_cancelling(self):
        # Enter byter person mitt i ett yttrande: klippet ska STT:as och spelas
        # upp för den som talade, inte kastas (cancelCapture + abandonActiveTurn).
        enter = self.source.split('if (e.key === "Enter")')[1].split(
            "if (config.keyboardMode"
        )[0]
        self.assertIn("finishCapture()", enter)
        self.assertNotIn("cancelCapture()", enter)
        self.assertNotIn("abandonActiveTurn", enter)
        self.assertNotIn(
            "generationRef.current += 1",
            enter,
            "Enter får inte bumpa generation — då tystnar STT/TTS för klippet.",
        )

    def test_speech_end_uses_the_capture_snapshot_not_live_refs(self):
        # Encode är async: nästa person kan öppna en tur innan onSpeechEnd
        # kör. Utan snapshoten från speech-start skulle klippet STT:as med
        # fel språkpar.
        end = self.source.split("const endUtterance")[1].split(
            "const handleInterim"
        )[0]
        self.assertIn("utteranceFromCapture", end)
        self.assertIn(
            "const utterance = utteranceFromCapture || activeUtteranceRef.current",
            end,
        )

    def test_other_person_ducks_tts_instead_of_cutting_it(self):
        begin = self.source.split("const beginUtterance")[1].split(
            "const endUtterance"
        )[0]
        self.assertIn("setTtsDuckRef.current?.(true)", begin)
        self.assertIn("ttsQueueRef.current.lane === lane", begin)

    def test_same_speaker_continuation_reuses_the_open_turn(self):
        begin = self.source.split("const beginUtterance")[1].split(
            "const endUtterance"
        )[0]
        self.assertIn("CONTINUE_WINDOW_MS", begin)
        self.assertIn('merge: "continue"', begin)
        self.assertIn("REPAIR_WINDOW_MS", begin)

    def test_stt_auto_language_is_on_for_final_captures(self):
        self.assertIn("autoLanguage: true", self.source)
        self.assertIn("isBackchannel", read(API_JS))
        self.assertIn("endsWithContinuationCue", read(API_JS))
        self.assertIn("stripRepairCue", read(API_JS))

    def test_whisper_nospeech_tokens_are_stripped_before_translate(self):
        self.assertIn("normalizeSttText", read(API_JS))
        self.assertIn("normalizeSttText(stt.text", self.source)
        self.assertIn("dropTurn(turnId)", self.source)


class TestStreamingEnvelopeContract(unittest.TestCase):
    def test_partials_are_suppressed_for_a_legacy_json_envelope(self):
        source = read(API_JS)
        self.assertIn('head.startsWith("{") || head.startsWith("```")', source)
        self.assertIn("if (isEnvelope === false && onText) onText(text)", source)

    def test_caller_resets_offsets_when_the_parsed_text_differs(self):
        self.assertIn(
            "if (result.translation !== result.raw) spokenChars = 0",
            read(TRANSLATOR_APP),
        )


if __name__ == "__main__":
    unittest.main()
