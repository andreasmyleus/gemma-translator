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

"""Grindar för live-översättning av ett *samtal*, inte ett ensamt yttrande.

Enarmsfixturerna i test_bench_cli/test_runner säger ingenting om det som
faktiskt gick sönder i produkten: att VAD:en klistrade ihop två talare till ett
yttrande så fort pausen mellan dem var kortare än hangovern. Det syns bara när
båda talarna ligger i samma mikrofonström.

Tre lager, snabbast först:
  * TestSegmentation      — syntetiska pulser, ingen modell, millisekunder.
  * TestConversationBaseline — den incheckade körningen av hela kedjan.
  * TestLiveConversation  — kör om kedjan på riktigt, hoppas över utan backend.
"""

import json
import os
import pathlib
import unittest

import numpy as np

from bench.frontend_mirror import (
    BROWSER_SAMPLE_RATE,
    MIN_SPEECH_MS,
    SILENCE_MS,
    SPECULATIVE_STT_MS,
    SPEECH_RMS,
    segment_utterances,
)

REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_DIR / "bench" / "results"
BASELINES = (
    "conv-sv-en-cafe",
    "conv-sv-en-directions",
    "conv-sv-en-tight",
    "conv-sv-en-same-speaker",
    "conv-sv-fi-pharmacy",
)


def burst(seconds, level=0.15, rate=BROWSER_SAMPLE_RATE, seed=0):
    """Talliknande brus på en känd nivå (RMS ≈ level)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * rate)) * level).astype(np.float32)


def room_tone(seconds, rate=BROWSER_SAMPLE_RATE, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * rate)) * 0.0015).astype(np.float32)


def stream(pattern):
    """pattern: lista av ('tal', s) / ('tyst', s)."""
    parts = [room_tone(0.5, seed=99)]
    for kind, seconds in pattern:
        parts.append(
            burst(seconds, seed=len(parts)) if kind == "tal" else room_tone(seconds, seed=len(parts))
        )
    parts.append(room_tone(1.5, seed=98))
    return np.concatenate(parts)


class TestSegmentation(unittest.TestCase):
    """VAD:ens segmentering — det enda som avgör om två talare hålls isär."""

    def test_a_one_second_gap_splits_two_speakers(self):
        # Regressionen: med den gamla tvånivåregeln föll de flesta avslut
        # tillbaka på 1250 ms hangover, längre än en helt vanlig paus mellan
        # två personer, så båda replikerna hamnade i samma yttrande och
        # översattes som en enda text.
        segments = segment_utterances(
            stream([("tal", 2.0), ("tyst", 1.0), ("tal", 2.0)]), BROWSER_SAMPLE_RATE
        )
        self.assertEqual(
            len(segments),
            2,
            "en sekunds paus mellan talare måste ge två yttranden, inte ett",
        )

    def test_five_turns_at_conversational_pace_stay_apart(self):
        pattern = []
        for _ in range(5):
            pattern += [("tal", 2.0), ("tyst", 1.05)]
        segments = segment_utterances(stream(pattern), BROWSER_SAMPLE_RATE)
        self.assertEqual(len(segments), 5)

    def test_a_pause_inside_a_sentence_does_not_chop(self):
        # Uppmätt i bench-samtalen: den längsta pausen *inuti* ett yttrande är
        # 256 ms. 400 ms här ger marginal och måste hålla ihop.
        segments = segment_utterances(
            stream([("tal", 1.5), ("tyst", 0.4), ("tal", 1.5)]), BROWSER_SAMPLE_RATE
        )
        self.assertEqual(len(segments), 1)

    def test_hangover_sits_between_the_two_measured_populations(self):
        # Grinden som gör de två testerna ovan meningsfulla: värdet måste ligga
        # med marginal över den längsta pausen inuti ett yttrande (256 ms) och
        # under den kortaste pausen mellan talare (1109 ms).
        self.assertGreater(SILENCE_MS, 256 * 1.3)
        self.assertLess(SILENCE_MS, 1109 * 0.95)

    def test_a_blip_shorter_than_min_speech_is_dropped(self):
        segments = segment_utterances(
            stream([("tal", MIN_SPEECH_MS / 1000 / 3)]), BROWSER_SAMPLE_RATE
        )
        self.assertEqual(segments, [])

    def test_room_tone_alone_never_opens_a_capture(self):
        self.assertEqual(segment_utterances(stream([("tyst", 8.0)]), BROWSER_SAMPLE_RATE), [])

    def test_capture_covers_the_speech_it_was_opened_for(self):
        segments = segment_utterances(
            stream([("tal", 2.0), ("tyst", 1.2), ("tal", 2.0)]), BROWSER_SAMPLE_RATE
        )
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertGreaterEqual(
                float(np.sqrt(np.mean(segment["samples"] ** 2))),
                SPEECH_RMS,
                "ett fångat yttrande som mest består av tystnad slösar STT-tid",
            )


class TestConversationBaseline(unittest.TestCase):
    """Den incheckade körningen av hela kedjan (bench/results/conv-*.json).

    Uppdateras med `venv/bin/python -m bench.conversation --id <id> --save conv-<id>`.
    Grinderna är beteende, inte exakta siffror — modellen är inte deterministisk.
    """

    def baselines(self):
        for name in BASELINES:
            path = RESULTS_DIR / f"{name}.json"
            self.assertTrue(path.exists(), f"saknar baslinje {path}")
            with open(path, encoding="utf-8") as handle:
                yield name, json.load(handle)

    def test_the_baseline_was_measured_on_a_quiet_machine(self):
        # Latenssiffrorna i baslinjen grindas inte (se nedan), men de läses av
        # människor och citeras i README. En körning på en belastad maskin
        # blåste upp STT, LLM och TTS med ~2x samtidigt; utan den här
        # kontrollen hade den checkats in och citerats som om den betydde något.
        from bench.conversation import MAX_LOAD_PER_CORE

        for name, report in self.baselines():
            with self.subTest(name):
                self.assertIn("load_per_core", report, "gammal baslinje utan lastkontext")
                self.assertLessEqual(report["load_per_core"], MAX_LOAD_PER_CORE, name)

    def test_every_spoken_turn_becomes_exactly_one_utterance(self):
        for name, report in self.baselines():
            with self.subTest(name):
                self.assertEqual(
                    report["detected_segments"],
                    report["spoken_turns"],
                    "VAD:en delade eller slog ihop turer",
                )

    def test_no_utterance_spans_more_than_one_spoken_turn(self):
        for name, report in self.baselines():
            with self.subTest(name):
                merged = [t["position"] for t in report["turns"] if t.get("merged_turns")]
                self.assertEqual(merged, [], "yttranden med två talare i")

    def test_every_turn_reaches_audio(self):
        for name, report in self.baselines():
            with self.subTest(name):
                outcomes = [t.get("outcome") for t in report["turns"]]
                self.assertEqual(set(outcomes), {"ok"}, f"{name}: {outcomes}")

    def test_the_spoken_language_decides_the_lane_without_enter(self):
        # Hela poängen med commit #5: två personer ska bara kunna prata.
        # Regressionen som fällde det här testet först: på 0,85 s pauser
        # felklassade detektorn korta svenska klipp, argmax-vägen föll tillbaka
        # på banans prior — den som talade förra turen — och två av fem turer
        # transkriberades som engelska. Se resolve_lane_language.
        for name, report in self.baselines():
            with self.subTest(name):
                for turn in report["turns"]:
                    self.assertTrue(
                        turn.get("source_lang_correct"),
                        f"{name} tur {turn['position']}: "
                        f"{turn.get('routed_src')} != {turn.get('expected_lang')}",
                    )

    def test_the_translation_comes_out_in_the_other_lane_language(self):
        # Språkgissningen är stoppordsbaserad och bara tillförlitlig där båda
        # listorna är välfyllda, alltså sv/en. Den rapporteras för alla samtal
        # men grindar bara där den betyder något.
        for name, report in self.baselines():
            if "fi" in (report["lane1"], report["lane2"]):
                continue
            with self.subTest(name):
                for turn in report["turns"]:
                    self.assertTrue(
                        turn.get("translation_lang_correct"),
                        f"{name} tur {turn['position']}: översattes till "
                        f"{turn.get('translation_lang')}, väntade {turn.get('routed_dst')}",
                    )

    def test_a_tight_conversational_pace_still_separates_the_speakers(self):
        # 0,85 s pauser är det tätaste samtalet i korpusen och ligger närmast
        # hangovern. Faller det här är 700 ms för långt för riktigt tempo.
        path = RESULTS_DIR / "conv-sv-en-tight.json"
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["detected_segments"], report["spoken_turns"])
        self.assertEqual([t for t in report["turns"] if t.get("merged_turns")], [])

    def test_two_specialised_checkpoints_still_auto_route(self):
        # sv/fi har en specialiserad checkpoint per bana. Innan språkdetektorn
        # flyttades ut ur LRU:n gick paret antingen helt utan autoroutning
        # eller laddade om en modell per tur.
        path = RESULTS_DIR / "conv-sv-fi-pharmacy.json"
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        ok = [t for t in report["turns"] if t.get("outcome") == "ok"]
        self.assertEqual(len(ok), report["spoken_turns"])
        self.assertTrue(all(t.get("source_lang_correct") for t in ok))
        self.assertIn("fi", {t.get("routed_src") for t in ok})
        self.assertIn("sv", {t.get("routed_src") for t in ok})

    def test_the_same_speaker_twice_does_not_flip_the_lane(self):
        path = RESULTS_DIR / "conv-sv-en-same-speaker.json"
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        first_three = [t for t in report["turns"] if t["position"] < 3]
        self.assertEqual(
            {t.get("routed_src") for t in first_three},
            {"sv"},
            "tre svenska turer i rad ska alla stanna på svenska banan",
        )

    def test_both_latency_numbers_are_recorded_for_every_turn(self):
        """Latensen *registreras*, den grindas inte — och det är avsiktligt.

        En incheckad baslinje jämförd mot en framtida körning är precis den
        korsjämförelse mellan separat körda mätningar som README:ns metodnot
        säger inte går att lita på: uppmätt här steg STT, LLM och TTS med ~2x
        samtidigt när maskinens bakgrundslast ändrades, och LLM-steget rör
        ingen av ändringarna i den här grenen. Ett absolut tak hade antingen
        fällt på främmande last eller släppt igenom en verklig regression,
        beroende på var det råkade sitta.

        Tidsjämförelser hör hemma i `bench.bench --ab-*`, som kör båda armarna
        i samma process under samma last. Här grindas det som *är*
        lastoberoende: att siffrorna finns och hänger ihop.
        """
        for name, report in self.baselines():
            with self.subTest(name):
                for turn in report["turns"]:
                    if turn.get("outcome") != "ok":
                        continue
                    self.assertIn("time_to_first_audio_ms", turn)
                    self.assertIn("perceived_ms", turn)
                    # Upplevd latens innehåller väntan som time_to_first_audio
                    # per definition inte ser, så den kan aldrig vara mindre.
                    self.assertGreaterEqual(
                        turn["perceived_ms"],
                        turn["time_to_first_audio_ms"],
                        f"{name} tur {turn['position']}",
                    )

    def test_perceived_latency_accounts_for_the_hangover(self):
        # Appens egen mätare startar när VAD:en stängt yttrandet och kan alltså
        # inte se tystnadsmarginalen. Skillnaden mellan de två talen är just den
        # väntan, och den får inte tyst försvinna ur rapporten.
        for name, report in self.baselines():
            with self.subTest(name):
                ok = [t for t in report["turns"] if t.get("outcome") == "ok"]
                self.assertTrue(ok)
                for turn in ok:
                    expected = (
                        max(SILENCE_MS, SPECULATIVE_STT_MS + turn["stt_ms"])
                        + turn["llm_first_sentence_ms"]
                        + turn["tts_first_ms"]
                    )
                    self.assertAlmostEqual(
                        turn["perceived_ms"], expected, delta=1.0,
                        msg=f"{name} tur {turn['position']}",
                    )

    def test_speculative_stt_actually_hides_behind_the_hangover(self):
        # Vinsten finns bara om STT hinner starta före stängningen. Går
        # SPECULATIVE_STT_MS över SILENCE_MS är hela mekanismen en no-op.
        self.assertLess(SPECULATIVE_STT_MS, SILENCE_MS)
        self.assertGreater(
            SPECULATIVE_STT_MS,
            256,
            "måste ligga över den längsta uppmätta pausen inuti ett yttrande, "
            "annars fyrar den på vanliga ordmellanrum",
        )


@unittest.skipUnless(
    os.environ.get("BENCH_LIVE"), "sätt BENCH_LIVE=1 med backend+litert-lm igång"
)
class TestLiveConversation(unittest.TestCase):
    """Kör om samtalet mot en riktig backend och jämför med baslinjen."""

    def test_live_run_matches_the_baseline_shape(self):
        from bench.conversation import (
            build_stream,
            load_conversations,
            load_stream,
            run_conversation,
            summarize_conversation,
        )

        api = os.environ.get("BENCH_API_BASE", "http://localhost:3100")
        llm = os.environ.get("BENCH_LLM_URL", "http://localhost:9379/v1/chat/completions")
        conversations = load_conversations()
        for conv_id in ("sv-en-cafe", "sv-en-directions"):
            with self.subTest(conv_id):
                spec = conversations[conv_id]
                wav, meta = build_stream(conv_id, spec)
                report = run_conversation(
                    api, llm, "gemma4-e2b", conv_id, spec, meta, load_stream(wav)
                )
                summary = summarize_conversation(report)
                self.assertEqual(summary["detected_segments"], summary["spoken_turns"])
                self.assertEqual(summary["errors"], 0)
                self.assertEqual(summary["merged_segments"], 0)
                self.assertEqual(summary["source_lang_correct"], summary["translated"])


if __name__ == "__main__":
    unittest.main()
