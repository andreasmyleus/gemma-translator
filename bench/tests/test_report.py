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

import unittest

from bench.report import (
    build_report,
    config_mismatch,
    gate,
    median,
    render_markdown,
    summarize,
)
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

    def test_corpus_wer_ignores_fixtures_without_reference_words(self):
        # En fixtur vars facittext normaliseras till noll ord går inte att
        # poängsätta: `edits` blir längden på transkriptionen och nämnaren är
        # noll. Den får varken lyfta täljaren eller nämnaren i korpus-WER.
        unscoreable = self._fixture(0, 0)
        unscoreable["edits"] = 3  # word_edit_distance([], hyp) == len(hyp)
        unscoreable["wer"] = 1.0
        report = build_report("x", {
            "ok-one": self._fixture(1, 10),
            "no-reference": unscoreable,
        })
        self.assertAlmostEqual(report["corpus_wer"], 0.1)
        self.assertEqual(report["corpus_edits"], 1)
        self.assertEqual(report["corpus_ref_words"], 10)

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


class TestProvenance(unittest.TestCase):
    """Utan provenance går två resultatfiler inte att skilja åt i konfiguration
    — `baseline.json` och `final.json` mättes med olika systemprompt och såg
    ändå identiska ut."""

    def _report(self, label, config=None):
        return build_report(label, {"sv-short": summarize([make_result()], "x")}, config)

    def test_config_is_recorded_when_given(self):
        report = self._report("final", {"model": "gemma4-e2b", "prompt": "plain", "stream": True})
        self.assertEqual(report["config"]["prompt"], "plain")

    def test_matching_configs_allow_a_delta(self):
        config = {"model": "m", "prompt": "plain", "stream": False}
        self.assertIsNone(config_mismatch(self._report("b", config), self._report("a", config)))

    def test_differing_stream_flag_is_reported(self):
        current = self._report("b", {"model": "m", "prompt": "plain", "stream": True})
        baseline = self._report("a", {"model": "m", "prompt": "plain", "stream": False})
        self.assertIn("stream", config_mismatch(current, baseline))

    def test_missing_config_counts_as_mismatch(self):
        current = self._report("b", {"model": "m", "prompt": "plain", "stream": False})
        self.assertIn("saknar config", config_mismatch(current, self._report("a")))

    def test_delta_column_is_dropped_when_the_runs_differ(self):
        current = self._report("b", {"model": "m", "prompt": "plain", "stream": True})
        baseline = self._report("a", {"model": "m", "prompt": "json", "stream": False})
        rendered = render_markdown(current, baseline)
        self.assertIn("Δ utelämnad", rendered)
        # Ingen procentsiffra i tabellen: Δ byggs av olika komponenter.
        self.assertNotIn("%", rendered)


if __name__ == "__main__":
    unittest.main()
