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
VAD_JS = REPO_DIR / "frontend" / "src" / "hooks" / "useVoiceActivity.js"
SERVER_PY = REPO_DIR / "backend" / "server.py"


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
        self.source_vad = read(VAD_JS)

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
        # ttsSession är null exakt när cfg.enableTts är av; utan uppspelning
        # finns ingen onFinished som kan rapportera latensen.
        self.assertIn("if (!ttsSession) reportLatency(marks, false)", self.source)

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
        self.assertIn("speakingLane() === lane", begin)

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

    def test_silence_is_not_sent_to_whisper_as_radio_copy(self):
        vad = read(VAD_JS)
        self.assertIn("AudioContext resumed", vad)
        self.assertIn("mute.gain.value = 0", vad)
        server = read(SERVER_PY)
        self.assertIn("condition_on_previous_text=False", server)
        self.assertIn("keep_stt_segment", server)

    def test_spoken_language_reroutes_the_turn_without_enter(self):
        self.assertIn("routeSpokenTurn", read(API_JS))
        self.assertIn("routeSpokenTurn(", self.source)
        self.assertIn("flipped", self.source)

    def test_continuation_only_glues_while_waiting_for_a_cue(self):
        # Gluing every burst within 1.5s concatenates the other person's
        # speech onto the open turn — fatal for free two-language talk.
        begin = self.source.split("const beginUtterance")[1].split(
            "const endUtterance"
        )[0]
        self.assertIn("pendingTranslateRef.current", begin)

    def test_tts_chunks_are_enqueued_in_order(self):
        enqueue = self.source.split("const enqueueTTS")[1].split(
            "const sealTTSQueue"
        )[0]
        self.assertIn("session.chain = session.chain.then(run, run)", enqueue)

    def test_each_turn_speaks_through_its_own_session(self):
        # En delad kö lät en överlappande tur lägga sina meningar i den tur som
        # just spelade, och dess seal avslutade *den* sessionen — de två gled
        # permanent ur fas. Sessionshandtaget måste följa med hela vägen.
        self.assertIn("const ttsSessionsRef = useRef([])", self.source)
        enqueue = self.source.split("const enqueueTTS")[1].split(
            "const sealTTSQueue"
        )[0]
        self.assertIn("session.pending.push({ buffer, marks })", enqueue)
        self.assertIn(
            "enqueueTTS(ttsSession, ready, spokenDst.ttsLang, marks)",
            self.source,
            "speakCompleteSentences måste tala in i sin egen turs session.",
        )

    def test_seal_waits_for_the_fetches_it_sealed_over(self):
        # sealTTSQueue() kördes synkront efter ett `void enqueueTTS(...)`, så en
        # kort översättning settlade sessionen innan sin egen ljudchunk fanns:
        # metaraden skrev "ljud —" och nästa tur startade för tidigt.
        pump = self.source.split("const pumpTTSQueue")[1].split(
            "const beginTTSSession"
        )[0]
        self.assertIn("session.outstanding > 0", pump)
        enqueue = self.source.split("const enqueueTTS")[1].split(
            "const sealTTSQueue"
        )[0]
        self.assertIn("session.outstanding += 1", enqueue)
        self.assertIn("session.outstanding -= 1", enqueue)

    def test_a_stale_onended_cannot_advance_the_next_session(self):
        pump = self.source.split("const pumpTTSQueue")[1].split(
            "const beginTTSSession"
        )[0]
        self.assertIn("if (onlineAudioPlayerRef.current !== source) return", pump)

    def test_continuation_hold_is_released_when_nobody_continues(self):
        # Annars betalar nästa yttrande — från vem som helst — 1400 ms extra
        # tystnad innan VAD stänger det.
        arm = self.source.split("const armContinuationWait")[1].split(
            "const abortTurnPipeline"
        )[0]
        self.assertIn("setSilenceHoldRef.current?.(0)", arm)

    def test_stt_inflight_is_balanced_around_the_await(self):
        # Den yttre catchen dekrementerade också, så fel *efter* STT (och
        # knownText-vägen, som aldrig inkrementerar) räknade ner för mycket.
        self.assertNotIn(
            "sttInflightRef.current = Math.max(0, sttInflightRef.current - 1)",
            self.source,
        )
        self.assertIn("} finally {\n          sttInflightRef.current -= 1", self.source)

    def test_continuation_cues_exclude_ordinary_endings(self):
        cues = read(API_JS).split("const CONTINUATION_CUES")[1].split(
            "export function endsWithContinuationCue"
        )[0]
        self.assertNotIn('"that"', cues)
        self.assertNotIn('"så"', cues)
        self.assertNotIn('"att"', cues)
        self.assertIn('"och"', cues)
        self.assertIn('"and"', cues)

    def test_aec_converts_mic_index_into_tts_sample_time(self):
        helpers = read(REPO_DIR / "frontend" / "src" / "utils" / "audioHelpers.js")
        self.assertIn("export function farEndSampleIndex", helpers)
        vad = read(REPO_DIR / "frontend" / "src" / "hooks" / "useVoiceActivity.js")
        self.assertIn("farEndSampleIndex(", vad)
        self.assertNotIn(
            "Math.floor(t * tts.sampleRate) + i",
            vad,
            "AEC must not treat mic samples as TTS samples.",
        )

    def test_vad_timing_constants_match_the_bench_mirror(self):
        # Bench segmenterar samtalen med sin egen kopia av VAD:en. Glider en
        # konstant isär mäter bench ett annat samtal än produkten hör, och
        # segmenteringsgrindarna i test_conversation.py blir meningslösa.
        from bench.frontend_mirror import (
            MIN_SPEECH_MS,
            PRE_ROLL_CHUNKS,
            SILENCE_MS,
            SPECULATIVE_STT_MS,
            SPEECH_RMS,
            VAD_FRAME,
        )

        vad = read(VAD_JS)
        for name, value in [
            ("SPEECH_RMS", SPEECH_RMS),
            ("SILENCE_MS", SILENCE_MS),
            ("MIN_SPEECH_MS", MIN_SPEECH_MS),
            ("SPECULATIVE_STT_MS", SPECULATIVE_STT_MS),
            ("PRE_ROLL_CHUNKS", PRE_ROLL_CHUNKS),
        ]:
            match = re.search(rf"^const {name} = ([0-9.]+)$", vad, re.MULTILINE)
            self.assertIsNotNone(match, f"hittar inte {name} i useVoiceActivity.js")
            self.assertEqual(
                float(match.group(1)), float(value), f"{name} skiljer sig mot spegeln"
            )
        self.assertIn(f"createScriptProcessor(\n        {VAD_FRAME},", vad)

    def test_the_two_tier_silence_rule_is_gone(self):
        # lastLoudRms mätte utklingningen, alltså yttrandets tystaste tal, så
        # kortvägen fyrade slumpmässigt och resten föll på 1250 ms — längre än
        # en vanlig paus mellan två talare.
        vad = read(VAD_JS)
        loop = vad.split("const handleAudioProcess")[1]
        self.assertNotIn("SILENCE_MS_LONG", loop)
        self.assertNotIn("abrupt", loop)
        self.assertIn("Math.max(SILENCE_MS, silenceHoldMsRef.current)", loop)

    def test_short_utterance_guard_measures_speech_not_hangover(self):
        # `now - startedAt` innehöll alltid hela hangovern, som är längre än
        # MIN_SPEECH_MS — så grinden kunde aldrig falla ut och varje hostning
        # kostade ett STT-anrop.
        finish = self.source_vad.split("const finishCapture")[1].split(
            "const cancelCapture"
        )[0]
        self.assertIn("Math.max(0, lastLoudAt - startedAt)", finish)

    def test_speculative_stt_result_is_only_used_when_no_speech_followed(self):
        vad = read(VAD_JS)
        self.assertIn("earlySpeechAfterRef.current = true", vad)
        self.assertIn(
            "const earlyUsable = earlyFiredRef.current && !earlySpeechAfterRef.current",
            vad,
        )
        self.assertIn("payload?.earlyUsable", self.source)
        self.assertIn("extra.knownTextPromise", self.source)
        # Måste hämtas före varje tidig return, annars fortsätter en kastad
        # spekulativ förfrågan hålla backendens STT-lås mot nästa tur.
        end = self.source.split("const endUtterance")[1].split("const handleInterim")[0]
        claim = end.index("const early = earlySttRef.current")
        self.assertLess(claim, end.index("abandonActiveTurn(turnId)"))

    def test_aec_tap_loops_stay_off_the_modulo_path(self):
        # onaudioprocess kör på huvudtråden. 1024 tappar × 48 kHz med en modulo
        # per tapp, plus en 1024-termers effektsumma per sample, är för mycket
        # för en Pi. Fönstret ska vara sammanhängande och effekten löpande.
        vad = read(VAD_JS)
        cancel = vad.split("const cancelEcho")[1].split("const handleAudioProcess")[0]
        self.assertNotIn("% AEC_FILTER_LEN", cancel)
        self.assertIn("for (let k = 0; k < N; k++) y += w[k] * xHist[p + k]", cancel)
        self.assertIn("pow += v * v - dropped * dropped", cancel)
        self.assertIn("aecZeroRunRef.current >= N", cancel)


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
