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

"""Ordvis WER utan externa beroenden.

Kvalitetsgrinden behöver bara ett tal per fixtur, och en klassisk
Levenshtein över ordlistor räcker. Normaliseringen är medvetet grov: vi
jämför Whispers utdata mot texten vi syntetiserade, så skiljetecken och
versaler är brus.
"""

import re

# Behåller bokstäver (inklusive å ä ö ü) och siffror; allt annat är skiljetecken.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text):
    """Gemener, skiljetecken bort, whitespace kollapsad."""
    stripped = _PUNCTUATION.sub("", text.lower())
    return stripped.split()


def word_edit_distance(ref, hyp):
    """Levenshtein över två ordlistor."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,      # deletion
                    current[j - 1] + 1,   # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def wer(reference, hypothesis):
    """Word error rate i intervallet 0.0 (perfekt) och uppåt."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        # Ingen referens att dela med: tyst utdata är rätt, allt annat är fel.
        return 0.0 if not hyp else 1.0
    return word_edit_distance(ref, hyp) / len(ref)
