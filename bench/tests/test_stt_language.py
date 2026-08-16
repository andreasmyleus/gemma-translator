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

from server import keep_stt_segment, resolve_spoken_language  # noqa: E402


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
