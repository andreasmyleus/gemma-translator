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

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "backend"))

from server import (  # noqa: E402
    LID_MIN_MASS,
    LID_MIN_RATIO,
    keep_stt_segment,
    resolve_lane_language,
    resolve_spoken_language,
)


class TestResolveSpokenLanguage(unittest.TestCase):
    def test_expected_language_wins_when_detect_agrees(self):
        self.assertEqual(resolve_spoken_language("sv", 0.9, "sv", "en"), "sv")

    def test_other_lane_wins_when_confident(self):
        self.assertEqual(resolve_spoken_language("en", 0.8, "sv", "en"), "en")

    def test_uncertain_detect_keeps_the_lane_prior(self):
        self.assertEqual(resolve_spoken_language("en", 0.3, "sv", "en"), "sv")

    def test_unsupported_detect_keeps_the_lane_prior(self):
        self.assertEqual(resolve_spoken_language("de", 0.99, "sv", "en"), "sv")

    def test_strips_region_suffix(self):
        self.assertEqual(resolve_spoken_language("en-US", 0.9, "sv", "en"), "en")

    def test_other_lane_at_the_confidence_floor(self):
        self.assertEqual(resolve_spoken_language("en", 0.5, "sv", "en"), "en")
        self.assertEqual(resolve_spoken_language("en", 0.49, "sv", "en"), "sv")

    def test_a_low_confidence_detect_never_steals_the_other_lane(self):
        # Fallback-regeln, som bara används när tvåvägsvalet avstår.
        self.assertEqual(resolve_spoken_language("sv", 0.46, "en", "sv"), "en")
        self.assertEqual(resolve_spoken_language("sv", 0.36, "sv", "en"), "sv")


class TestResolveLaneLanguage(unittest.TestCase):
    """Tvåvägsvalet mellan banornas språk — det som faktiskt routar turen."""

    def test_the_larger_lane_probability_wins(self):
        probs = [("sv", 0.80), ("en", 0.05), ("de", 0.10)]
        self.assertEqual(resolve_lane_language(probs, "en", "sv"), "sv")
        self.assertEqual(resolve_lane_language(probs, "sv", "en"), "sv")

    def test_a_confident_third_language_cannot_decide_the_lane(self):
        # Den riktiga regressionen: detektorn svarade `zh` p=0.94 på 1,4 s
        # svenska. Argmax-vägen såg ett språk utanför SUPPORTED_STT_LANGS och
        # föll tillbaka på banans prior — alltså på den som talade *förra*
        # turen, vilket i ett växlande samtal är fel varje gång.
        probs = [("zh", 0.94), ("sv", 0.030), ("en", 0.003)]
        self.assertEqual(resolve_lane_language(probs, "en", "sv"), "sv")

    def test_a_coin_flip_refuses_to_guess(self):
        # Uppmätt på sv-en-tight#2: 0.0568 mot 0.0676, kvot 1.19.
        self.assertIsNone(
            resolve_lane_language([("sv", 0.0568), ("en", 0.0676)], "en", "sv")
        )

    def test_a_winner_below_the_mass_floor_refuses_to_guess(self):
        probs = [("sv", LID_MIN_MASS / 2), ("en", LID_MIN_MASS / 100)]
        self.assertIsNone(resolve_lane_language(probs, "en", "sv"))

    def test_the_ratio_floor_is_exactly_the_boundary(self):
        low = LID_MIN_MASS * 2
        self.assertEqual(
            resolve_lane_language([("sv", low * LID_MIN_RATIO), ("en", low)], "en", "sv"),
            "sv",
        )
        self.assertIsNone(
            resolve_lane_language(
                [("sv", low * LID_MIN_RATIO * 0.99), ("en", low)], "en", "sv"
            )
        )

    def test_the_measured_base_margins_are_accepted(self):
        # Smalaste korrekta beslutet `base` gjorde över de 27 klippen: massa
        # 0.0478, kvot 7.2. Golven måste ligga med marginal under det.
        self.assertEqual(
            resolve_lane_language([("sv", 0.0478), ("en", 0.0478 / 7.2)], "en", "sv"),
            "sv",
        )

    def test_missing_or_degenerate_input_never_guesses(self):
        self.assertIsNone(resolve_lane_language(None, "sv", "en"))
        self.assertIsNone(resolve_lane_language([], "sv", "en"))
        self.assertIsNone(resolve_lane_language([("sv", 0.9)], "sv", None))
        self.assertIsNone(resolve_lane_language([("sv", 0.9)], "sv", "sv"))

    def test_a_lane_absent_from_the_distribution_counts_as_zero(self):
        self.assertEqual(resolve_lane_language([("sv", 0.9)], "en", "sv"), "sv")


class TestKeepSttSegment(unittest.TestCase):
    def test_keeps_real_speech(self):
        seg = SimpleNamespace(text="Var ligger stationen?", no_speech_prob=0.05)
        self.assertTrue(keep_stt_segment(seg))

    def test_drops_confident_silence_hallucination(self):
        # High logprob radio copy still has high no_speech_prob.
        seg = SimpleNamespace(
            text="Juniormusikens sändning av Melodifestivalen visas idag.",
            no_speech_prob=0.72,
        )
        self.assertFalse(keep_stt_segment(seg))

    def test_drops_empty_text(self):
        self.assertFalse(keep_stt_segment(SimpleNamespace(text="  ", no_speech_prob=0.0)))


if __name__ == "__main__":
    unittest.main()
