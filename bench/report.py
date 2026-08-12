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

"""Sammanfattar körningar, jämför mot baseline och fäller på regression."""

from bench.wer import normalize, word_edit_distance

METRICS = (
    "stt_ms",
    "llm_ms",
    "tts_first_ms",
    "tts_rest_ms",
    "time_to_first_audio_ms",
    "wall_total_ms",
)

# Absolut ökning i korpus-WER som får passera. Med ~121 facitord i sviten
# motsvarar ett enda ändrat ord ca 0,008, så tröskeln rymmer ungefär två
# ords variation innan den fäller.
DEFAULT_WER_THRESHOLD = 0.02


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize(results, spec_text):
    """Slår ihop repetitionerna för en fixtur till en rad.

    Både `wer` och råtalen `edits`/`ref_words` sparas: per-fixtur-WER visas i
    tabellen, medan korpus-WER måste summeras ur täljare och nämnare var för
    sig — ett medelvärde av per-fixtur-WER hade gett treordsfixturerna lika
    stor vikt som trettioordsfixturerna.
    """
    reference = normalize(spec_text)
    hypothesis = normalize(results[-1].transcript)
    edits = word_edit_distance(reference, hypothesis)
    ref_words = len(reference)

    summary = {
        "ok": all(result.ok for result in results),
        "errors": [result.error for result in results if not result.ok],
        "translation": results[-1].translation,
        "transcript": results[-1].transcript,
        "chunk_count": results[-1].chunk_count,
        "upload_bytes": results[-1].upload_bytes,
        "edits": edits,
        "ref_words": ref_words,
        "wer": (edits / ref_words) if ref_words else (0.0 if not hypothesis else 1.0),
    }
    for metric in METRICS:
        values = [getattr(result, metric) for result in results]
        summary[metric] = {
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def build_report(label, per_fixture):
    ok_fixtures = [data for data in per_fixture.values() if data["ok"]]
    total_edits = sum(data["edits"] for data in ok_fixtures)
    total_words = sum(data["ref_words"] for data in ok_fixtures)
    overall = [data["time_to_first_audio_ms"]["median"] for data in ok_fixtures]
    return {
        "label": label,
        "fixtures": per_fixture,
        "median_time_to_first_audio_ms": median(overall),
        "corpus_wer": (total_edits / total_words) if total_words else 0.0,
        "corpus_edits": total_edits,
        "corpus_ref_words": total_words,
    }


def _delta(current_value, baseline_value):
    if baseline_value in (None, 0):
        return ""
    change = (current_value - baseline_value) / baseline_value * 100
    return f"{change:+.0f}%"


def render_markdown(current, baseline=None):
    baseline_fixtures = (baseline or {}).get("fixtures", {})
    lines = [
        f"### {current['label']}",
        "",
        "| Fixtur | STT | LLM | TTS #1 | Till första ljud | Δ | WER |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fixture_id, data in current["fixtures"].items():
        if not data["ok"]:
            lines.append(f"| {fixture_id} | — | — | — | **FEL** | | {data['errors'][0][:40]} |")
            continue
        first_audio = data["time_to_first_audio_ms"]["median"]
        reference = baseline_fixtures.get(fixture_id, {}).get("time_to_first_audio_ms", {}).get("median")
        lines.append(
            f"| {fixture_id} "
            f"| {data['stt_ms']['median']:.0f} ms "
            f"| {data['llm_ms']['median']:.0f} ms "
            f"| {data['tts_first_ms']['median']:.0f} ms "
            f"| {first_audio:.0f} ms "
            f"| {_delta(first_audio, reference)} "
            f"| {data['wer']:.2f} |"
        )

    overall = current["median_time_to_first_audio_ms"]
    baseline_overall = (baseline or {}).get("median_time_to_first_audio_ms")
    lines.append("")
    lines.append(
        f"**Median till första ljud: {overall:.0f} ms** {_delta(overall, baseline_overall)}"
    )
    lines.append(
        f"**Korpus-WER: {current['corpus_wer']:.3f}** "
        f"({current['corpus_edits']} fel på {current['corpus_ref_words']} ord)"
    )
    return "\n".join(lines)


def gate(current, baseline=None, wer_threshold=DEFAULT_WER_THRESHOLD):
    """Returnerar (passerade, meddelanden).

    Fäller på misslyckade fixturer och på korpus-WER som stigit mer än
    tröskeln. Ändrad översättning noteras men fäller inte — Gemma får
    formulera om sig så länge transkriptionen håller.
    """
    messages = []
    passed = True

    for fixture_id, data in current["fixtures"].items():
        if not data["ok"]:
            passed = False
            messages.append(f"FEL {fixture_id}: {data['errors'][0]}")

    if not baseline:
        return passed, messages

    if "corpus_wer" not in baseline:
        messages.append(
            "VARNING: jämförelsekörningen saknar corpus_wer — WER-grinden hoppades över."
        )
    else:
        rise = current["corpus_wer"] - baseline["corpus_wer"]
        if rise > wer_threshold:
            passed = False
            worse = [
                f"{fixture_id} {baseline['fixtures'][fixture_id]['wer']:.2f}→{data['wer']:.2f}"
                for fixture_id, data in current["fixtures"].items()
                if data["ok"]
                and fixture_id in baseline["fixtures"]
                and data["wer"] > baseline["fixtures"][fixture_id]["wer"]
            ]
            messages.append(
                f"REGRESSION korpus-WER {baseline['corpus_wer']:.3f} → "
                f"{current['corpus_wer']:.3f} (+{rise:.3f}, tröskel {wer_threshold:.3f}). "
                f"Försämrade fixturer: {', '.join(worse) or 'inga enskilda'}"
            )

    for fixture_id, data in current["fixtures"].items():
        reference = baseline["fixtures"].get(fixture_id)
        if not reference or not data["ok"]:
            continue
        if data["translation"] != reference["translation"]:
            messages.append(
                f"NOTERA {fixture_id}: översättning ändrad\n"
                f"  förr: {reference['translation']!r}\n"
                f"  nu:   {data['translation']!r}"
            )

    return passed, messages
