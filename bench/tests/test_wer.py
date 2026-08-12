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
