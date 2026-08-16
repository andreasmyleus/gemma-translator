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
    DEFAULT_STT_MODEL,
    can_auto_detect_without_eviction,
    keep_stt_segment,
    resolve_spoken_language,
    _whisper_models,
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

    def test_sv_en_can_detect_without_a_third_model(self):
        self.assertTrue(can_auto_detect_without_eviction("sv", "en"))

    def test_two_specialised_models_skip_detect_unless_multilingual_is_warm(self):
        _whisper_models.clear()
        self.assertFalse(can_auto_detect_without_eviction("sv", "fi"))
        _whisper_models[DEFAULT_STT_MODEL] = object()
        try:
            self.assertTrue(can_auto_detect_without_eviction("sv", "fi"))
        finally:
            _whisper_models.clear()


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
