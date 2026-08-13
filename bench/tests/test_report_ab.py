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

"""Tester för den parvisa A/B-statistiken.

Se bench/report.paired_ratios: poängen är att jämföra par mätta inom
sekunder av varandra, inte aggregat (medianer) tagna över hela körningen.
Testet `test_paired_median_disagrees_with_ratio_of_medians` nedan är den
viktigaste — det bygger data där de två metoderna ger olika svar, och
verifierar att funktionen ger det rätta.
"""

import unittest

from bench.report import median, paired_ratios
from bench.runner import RunResult


def make_result(stt=0.0, ok=True):
    return RunResult(fixture_id="sv-short", stt_ms=stt, ok=ok)


class TestPairedRatiosOrdering(unittest.TestCase):
    def test_pairs_repetitions_in_order_not_as_sorted_aggregates(self):
        # Given i den ordning de kördes: par 1 av A hör ihop med par 1 av B,
        # osv. Sorterar man i stället A och B var för sig innan man delar
        # (aggregatmetoden) får man ett annat facit — se kommentaren nedan.
        runs_a = [make_result(stt=30), make_result(stt=10), make_result(stt=20)]
        runs_b = [make_result(stt=300), make_result(stt=100), make_result(stt=10)]

        ratios = paired_ratios(runs_a, runs_b)

        # Parvis, i given ordning: 300/30=10, 100/10=10, 10/20=0.5 -> median 10.
        self.assertAlmostEqual(ratios["stt_ms"], 10.0)
        # Hade man i stället sorterat båda listorna var för sig och parat ihop
        # dem (aggregatmetoden) hade man fått: sorted A=[10,20,30],
        # sorted B=[10,100,300] -> 10/10=1, 100/20=5, 300/30=10 -> median 5.
        # Det är ett annat svar, vilket är precis varför ordningen måste bevaras.
        wrong_aggregate_ratios = sorted(b / a for a, b in zip(sorted([30, 10, 20]), sorted([300, 100, 10])))
        self.assertNotAlmostEqual(median(wrong_aggregate_ratios), ratios["stt_ms"])


class TestPairedRatiosSkipFailures(unittest.TestCase):
    def test_pair_with_a_failed_arm_is_skipped_without_skewing_the_rest(self):
        runs_a = [make_result(stt=100.0, ok=True), make_result(stt=100.0, ok=False)]
        runs_b = [make_result(stt=80.0, ok=True), make_result(stt=5.0, ok=True)]

        ratios = paired_ratios(runs_a, runs_b)

        # Om det trasiga paret räknades med skulle medianen av [0.8, 0.05]
        # bli 0.425. Att den i stället är 0.8 visar att paret hoppades över.
        self.assertAlmostEqual(ratios["stt_ms"], 0.8)

    def test_pair_with_a_failed_b_arm_is_also_skipped(self):
        runs_a = [make_result(stt=100.0, ok=True), make_result(stt=100.0, ok=True)]
        runs_b = [make_result(stt=80.0, ok=True), make_result(stt=5.0, ok=False)]

        ratios = paired_ratios(runs_a, runs_b)

        self.assertAlmostEqual(ratios["stt_ms"], 0.8)


class TestPairedMedianVersusRatioOfMedians(unittest.TestCase):
    def test_paired_median_disagrees_with_ratio_of_medians(self):
        # Fem par vars absoluta nivåer driver kraftigt (maskinbelastning),
        # men varje par för sig visar B mot A inom samma belastningstillstånd.
        # Kvoterna per par är [1.5, 1.0, 0.5, 0.6, 0.8] -> median 0,8, dvs en
        # konstruerad 20-procentig förbättring i arm B.
        pairs = [
            (10, 15),  # ratio 1.5
            (20, 20),  # ratio 1.0
            (30, 15),  # ratio 0.5
            (40, 24),  # ratio 0.6
            (50, 40),  # ratio 0.8
        ]
        runs_a = [make_result(stt=a) for a, _ in pairs]
        runs_b = [make_result(stt=b) for _, b in pairs]

        ratios = paired_ratios(runs_a, runs_b)
        self.assertAlmostEqual(ratios["stt_ms"], 0.8)

        # Kvoten mellan medianerna (fel metod) ger ett annat svar: medianen av
        # A är 30 (från paret med kvot 0.5) och medianen av B är 20 (från
        # paret med kvot 1.0) — två olika par. 20/30 ≈ 0.667, inte 0.8. Det är
        # hela anledningen till att paired_ratios existerar i stället för att
        # bara jämföra de två armarnas medianer.
        median_a = median([a for a, _ in pairs])
        median_b = median([b for _, b in pairs])
        ratio_of_medians = median_b / median_a

        self.assertAlmostEqual(median_a, 30.0)
        self.assertAlmostEqual(median_b, 20.0)
        self.assertAlmostEqual(ratio_of_medians, 2 / 3)
        self.assertNotAlmostEqual(ratio_of_medians, ratios["stt_ms"], places=2)


if __name__ == "__main__":
    unittest.main()
