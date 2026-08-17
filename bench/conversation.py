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

"""Spelar upp ett helt samtal genom kedjan, som om det talats i webbläsaren.

Skillnaden mot bench/runner.py: den mäter ett yttrande i taget med känt språk
och känd bana. Den här modulen matar in *en sammanhängande mikrofonström* med
båda talarna i, låter VAD:en hitta yttrandena själv, och låter språkdetektionen
avgöra vilken bana varje tur hamnar på. Det är där de intressanta felen bor —
segmentering, routning och turordning syns inte i en enarmskörning.

Kedjan är den riktiga: samma HTTP-endpoints, samma payloads, samma
frontend-logik (via frontend_mirror). Ingen del av produkten stubbas.
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
import wave

import numpy as np

from bench.fixtures import _import_backend, _synthesis_config
from bench.frontend_mirror import (
    BROWSER_SAMPLE_RATE,
    SILENCE_MS,
    SPECULATIVE_STT_MS,
    STT_SAMPLE_RATE,
    is_backchannel,
    is_speakable,
    normalize_stt_text,
    resample,
    route_spoken_turn,
    segment_utterances,
)
from bench.runner import _get_tts, _post_llm_streaming, _post_stt
from bench.wer import wer

BENCH_DIR = pathlib.Path(__file__).resolve().parent
CONVERSATION_SPEC = BENCH_DIR / "conversations.json"
FIXTURE_DIR = BENCH_DIR / "fixtures"
RESULTS_DIR = BENCH_DIR / "results"

# AVAILABLE_LANGUAGES i TranslatorApp.jsx.
LANGUAGES = {
    "sv": {"code": "sv", "name": "Swedish", "ttsLang": "sv"},
    "en": {"code": "en", "name": "English", "ttsLang": "en"},
    "fi": {"code": "fi", "name": "Finnish", "ttsLang": "fi"},
    "es": {"code": "es", "name": "Spanish", "ttsLang": "es"},
    "fr": {"code": "fr", "name": "French", "ttsLang": "fr"},
}

# Grov språkgissning för rapporten: räcker gott för att avgöra om en
# sv→en-översättning faktiskt blev engelsk. Bara diagnostik, aldrig en grind.
_STOPWORDS = {
    "sv": {"och", "att", "det", "är", "jag", "en", "ett", "på", "för", "med",
           "kan", "har", "vill", "inte", "här", "där", "tack", "ni", "vi", "då"},
    "en": {"and", "the", "is", "it", "i", "a", "an", "on", "for", "with",
           "can", "have", "would", "not", "here", "there", "thanks", "you", "we", "your"},
    "fi": {"ja", "on", "se", "minä", "en", "että", "kanssa", "voi", "ei", "kiitos"},
    "es": {"y", "que", "el", "la", "es", "un", "una", "con", "para", "gracias"},
    "fr": {"et", "que", "le", "la", "est", "un", "une", "avec", "pour", "merci"},
}


def guess_language(text):
    words = {w.strip(".,!?…:;\"'").lower() for w in (text or "").split()}
    scores = {code: len(words & stop) for code, stop in _STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# Över den här 1-minuterslasten per kärna är tidmätningen inte jämförbar.
# Uppmätt: en körning med load 4.7 och en främmande VM på 41 % CPU blåste upp
# STT, LLM *och* TTS med ungefär 2x samtidigt — tre stadier utan något
# gemensamt utom processorn. Korrektheten (segmentering, routning) påverkas
# inte, men latenssiffrorna blir obrukbara som baslinje.
MAX_LOAD_PER_CORE = 0.7


def load_per_core():
    return os.getloadavg()[0] / (os.cpu_count() or 1)


def wait_for_quiet(timeout_s=300, poll_s=10):
    """Vänta tills maskinen är tyst nog att mäta på, och returnera lasten.

    Måste anropas *före* körningen. Mätningen skapar sin egen last, och
    getloadavg är ett 1-minutersmedel — läses den efteråt rapporterar den
    harnessets eget arbete, inte förhållandena mätningen faktiskt gjordes
    under. Det gäller även två samtal i rad: nummer två ärver nummer etts last.
    """
    waited = 0
    load = load_per_core()
    while load > MAX_LOAD_PER_CORE and waited < timeout_s:
        print(
            f"[conv] last {load:.2f}/kärna > {MAX_LOAD_PER_CORE}, väntar {poll_s}s "
            f"({waited}/{timeout_s}s)",
            flush=True,
        )
        time.sleep(poll_s)
        waited += poll_s
        load = load_per_core()
    return load


def load_conversations():
    with open(CONVERSATION_SPEC, encoding="utf-8") as handle:
        return json.load(handle)


def build_stream(conv_id, spec, force=False):
    """Syntetiserar samtalet till en enda 16 kHz-ström och cachar den.

    Varje tur renderas med produktens Piper-röst för sitt språk och läggs efter
    varandra med spec:ens pauser emellan. Resultatet är exakt vad mikrofonen
    hade hört om två personer suttit vid kiosken — inklusive att båda talarna
    ligger i samma ström, vilket är själva poängen.
    """
    wav_path = FIXTURE_DIR / f"conv-{conv_id}.wav"
    meta_path = FIXTURE_DIR / f"conv-{conv_id}.json"
    if wav_path.exists() and meta_path.exists() and not force:
        with open(meta_path, encoding="utf-8") as handle:
            return wav_path, json.load(handle)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    server = _import_backend()
    syn = _synthesis_config()

    pieces = []
    truth = []
    cursor = 0
    # Lite rumston i pauserna. Helt digital tystnad är orealistisk och låter
    # dessutom VAD:ens brusgolv krypa mot noll, vilket gör tröskeln känsligare
    # än den någonsin är i verkligheten.
    rng = np.random.default_rng(20260816)

    lead_in = (rng.standard_normal(int(0.6 * STT_SAMPLE_RATE)) * 0.0015).astype(np.float32)
    pieces.append(lead_in)
    cursor += len(lead_in)

    for index, turn in enumerate(spec["turns"]):
        print(f"[conv] syntetiserar tur {index + 1}/{len(spec['turns'])} ({turn['lang']})", flush=True)
        samples, rate = server.synthesize(turn["text"], turn["lang"], syn)
        samples = resample(np.asarray(samples, dtype=np.float32), rate, STT_SAMPLE_RATE)
        pieces.append(samples)
        truth.append(
            {
                "index": index,
                "speaker": turn["speaker"],
                "lang": turn["lang"],
                "text": turn["text"],
                "start_s": cursor / STT_SAMPLE_RATE,
                "end_s": (cursor + len(samples)) / STT_SAMPLE_RATE,
            }
        )
        cursor += len(samples)
        gap = int(turn.get("gap_after_s", 1.2) * STT_SAMPLE_RATE)
        room_tone = (rng.standard_normal(gap) * 0.0015).astype(np.float32)
        pieces.append(room_tone)
        cursor += gap

    stream = np.concatenate(pieces)
    pcm16 = (np.clip(stream, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(STT_SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())

    meta = {"id": conv_id, "duration_s": len(stream) / STT_SAMPLE_RATE, "turns": truth}
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return wav_path, meta


def load_stream(path):
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def match_truth(segment, truth):
    """Vilken/vilka spec-turer överlappar det VAD:en fångade."""
    hits = []
    for turn in truth:
        overlap = min(segment["end_s"], turn["end_s"]) - max(
            segment["start_s"], turn["start_s"]
        )
        if overlap > 0.15:
            hits.append(turn)
    return hits


def run_conversation(api_base, llm_url, model, conv_id, spec, meta, stream_16k, load=None):
    """Kör hela samtalet och returnerar en tur-för-tur-rapport."""
    lane_lang = {1: LANGUAGES[spec["lane1"]], 2: LANGUAGES[spec["lane2"]]}

    # Webbläsaren VAD:ar i AudioContext-takt och resamplar först det den
    # fångat ner till 16 kHz. Samma ordning här, annars mäter vi en
    # segmentering som produkten aldrig gör.
    stream_48k = resample(stream_16k, STT_SAMPLE_RATE, BROWSER_SAMPLE_RATE)
    segments = segment_utterances(stream_48k, BROWSER_SAMPLE_RATE)

    # En omätt runda per språkpar först. Appen förvärmer modellerna vid start
    # (PREWARM_LANGS + /api/prewarm), så utan det här mäter tur 0 en kallstart
    # som ingen användare någonsin ser — och den dominerar medianen på ett
    # samtal med fem turer.
    if segments:
        warm = resample(segments[0]["samples"], BROWSER_SAMPLE_RATE, STT_SAMPLE_RATE)
        for lane in (1, 2):
            src, dst = lane_lang[lane], lane_lang[2 if lane == 1 else 1]
            try:
                _post_stt(api_base, warm, src["code"], other_language=dst["code"], auto_language=True)
                _post_llm_streaming(api_base, llm_url, model, "Hello.", src["code"], dst["code"])
                _get_tts(api_base, "Hello.", dst["ttsLang"])
            except Exception as err:  # noqa: BLE001
                print(f"[conv] uppvärmning misslyckades ({err}) — fortsätter", flush=True)

    active_lane = 1
    turns = []
    for position, segment in enumerate(segments):
        samples = resample(segment["samples"], BROWSER_SAMPLE_RATE, STT_SAMPLE_RATE)
        src = lane_lang[active_lane]
        dst = lane_lang[2 if active_lane == 1 else 1]
        expected = match_truth(segment, meta["turns"])

        record = {
            "position": position,
            "vad_start_s": round(segment["start_s"], 2),
            "vad_end_s": round(segment["end_s"], 2),
            "closed_at_s": round(segment["closed_at_s"], 2),
            "captured_s": round(len(samples) / STT_SAMPLE_RATE, 2),
            "lane_before": active_lane,
            "assumed_src": src["code"],
            "expected_turns": [t["index"] for t in expected],
            "expected_lang": expected[0]["lang"] if len(expected) == 1 else None,
            "expected_text": expected[0]["text"] if len(expected) == 1 else None,
            "merged_turns": len(expected) > 1,
        }

        try:
            raw_text, stt_language, stt_ms, _ = _post_stt(
                api_base,
                samples,
                src["code"],
                other_language=dst["code"],
                auto_language=True,
            )
            text = normalize_stt_text(raw_text)
            record["stt_ms"] = round(stt_ms, 1)
            record["stt_language"] = stt_language
            record["transcript"] = text

            if not text or not is_speakable(text) or is_backchannel(text):
                record["outcome"] = "dropped"
                turns.append(record)
                continue

            routed = route_spoken_turn(stt_language, active_lane, src, dst, None, None)
            spoken_src, spoken_dst = routed["src"], routed["dst"]
            active_lane = routed["lane"]
            record["routed_src"] = spoken_src["code"]
            record["routed_dst"] = spoken_dst["code"]
            record["flipped"] = routed["flipped"]
            record["lane_after"] = active_lane

            translation, llm_ms, llm_first_ms = _post_llm_streaming(
                api_base,
                llm_url,
                model,
                text,
                spoken_src["code"],
                spoken_dst["code"],
            )
            record["llm_ms"] = round(llm_ms, 1)
            record["llm_first_sentence_ms"] = round(llm_first_ms, 1)
            record["translation"] = translation

            from bench.frontend_mirror import split_text_into_speech_chunks

            chunks = split_text_into_speech_chunks(translation)
            record["chunk_count"] = len(chunks)
            if not chunks:
                record["outcome"] = "no-tts"
                turns.append(record)
                continue
            record["tts_first_ms"] = round(
                _get_tts(api_base, chunks[0], spoken_dst["ttsLang"]), 1
            )
            record["tts_rest_ms"] = round(
                sum(_get_tts(api_base, c, spoken_dst["ttsLang"]) for c in chunks[1:]), 1
            )
            record["time_to_first_audio_ms"] = round(
                record["stt_ms"] + record["llm_first_sentence_ms"] + record["tts_first_ms"], 1
            )
            # Det tal användaren faktiskt känner: från sista stavelsen till
            # första ljudet. time_to_first_audio startar först när VAD:en
            # stängt yttrandet, alltså efter hangovern — appens egen metrik
            # (marks.keyup) gör samma sak och döljer därmed den vänteti den.
            #
            # STT startar spekulativt vid SPECULATIVE_STT_MS in i tystnaden, så
            # den överlappar resten av hangovern i stället för att köa efter.
            record["perceived_ms"] = round(
                max(SILENCE_MS, SPECULATIVE_STT_MS + record["stt_ms"])
                + record["llm_first_sentence_ms"]
                + record["tts_first_ms"],
                1,
            )
            record["outcome"] = "ok"

            # Kvalitet
            if record["expected_text"]:
                record["stt_wer"] = round(wer(record["expected_text"], text), 3)
            record["source_lang_correct"] = (
                record["expected_lang"] is None
                or spoken_src["code"] == record["expected_lang"]
            )
            record["translation_lang"] = guess_language(translation)
            record["translation_lang_correct"] = (
                record["translation_lang"] is None
                or record["translation_lang"] == spoken_dst["code"]
            )
        except Exception as err:  # noqa: BLE001 — en trasig tur får inte stoppa samtalet
            record["outcome"] = "error"
            record["error"] = f"{type(err).__name__}: {err}"
        turns.append(record)

    return {
        "id": conv_id,
        "load_per_core": round(load if load is not None else load_per_core(), 2),
        "description": spec.get("description", ""),
        "lane1": spec["lane1"],
        "lane2": spec["lane2"],
        "spoken_turns": len(meta["turns"]),
        "detected_segments": len(segments),
        "turns": turns,
    }


def summarize_conversation(report):
    ok = [t for t in report["turns"] if t.get("outcome") == "ok"]
    firsts = [t["time_to_first_audio_ms"] for t in ok if "time_to_first_audio_ms" in t]
    perceived = [t["perceived_ms"] for t in ok if "perceived_ms" in t]
    wers = [t["stt_wer"] for t in ok if "stt_wer" in t]
    return {
        "spoken_turns": report["spoken_turns"],
        "detected_segments": report["detected_segments"],
        "translated": len(ok),
        "dropped": sum(1 for t in report["turns"] if t.get("outcome") == "dropped"),
        "errors": sum(1 for t in report["turns"] if t.get("outcome") == "error"),
        "merged_segments": sum(1 for t in report["turns"] if t.get("merged_turns")),
        "source_lang_correct": sum(1 for t in ok if t.get("source_lang_correct")),
        "translation_lang_correct": sum(1 for t in ok if t.get("translation_lang_correct")),
        "median_time_to_first_audio_ms": round(statistics.median(firsts), 1) if firsts else None,
        "median_perceived_ms": round(statistics.median(perceived), 1) if perceived else None,
        "max_perceived_ms": round(max(perceived), 1) if perceived else None,
        "max_time_to_first_audio_ms": round(max(firsts), 1) if firsts else None,
        "median_stt_wer": round(statistics.median(wers), 3) if wers else None,
    }


def render(report):
    summary = summarize_conversation(report)
    lines = [f"# Samtal: {report['id']}", "", report["description"], ""]
    lines.append(
        f"Bana 1 = {report['lane1']}, bana 2 = {report['lane2']}. "
        f"{summary['spoken_turns']} talade turer → {summary['detected_segments']} yttranden ur VAD."
    )
    lines.append("")
    lines.append("| # | VAD s | fångat | STT-språk | Transkription | Översättning | →ljud |")
    lines.append("|---|-------|--------|-----------|---------------|--------------|-------|")
    for turn in report["turns"]:
        first = turn.get("time_to_first_audio_ms")
        lines.append(
            f"| {turn['position']} "
            f"| {turn['vad_start_s']:.1f}–{turn['vad_end_s']:.1f} "
            f"| {turn['captured_s']:.1f}s "
            f"| {turn.get('stt_language', '—')}"
            f"{'→' + turn['routed_dst'] if turn.get('routed_dst') else ''} "
            f"| {turn.get('transcript', turn.get('error', '—'))[:60]} "
            f"| {turn.get('translation', '—')[:60]} "
            f"| {f'{first:.0f} ms' if first else '—'} |"
        )
    lines.append("")
    lines.append("## Sammanfattning")
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="sv-en-cafe", help="Samtals-id ur conversations.json")
    parser.add_argument("--api-base", default="http://localhost:3100")
    parser.add_argument("--llm-url", default="http://localhost:9379/v1/chat/completions")
    parser.add_argument("--model", default="gemma4-e2b")
    parser.add_argument("--rebuild", action="store_true", help="Bygg om ljudet även om det är cachat")
    parser.add_argument("--save", help="Skriv JSON-rapport till bench/results/<namn>.json")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Spara även om maskinen är belastad (siffrorna blir inte jämförbara)",
    )
    args = parser.parse_args()

    conversations = load_conversations()
    if args.id not in conversations:
        parser.error(f"okänt samtal {args.id!r}; finns: {', '.join(conversations)}")
    spec = conversations[args.id]

    wav_path, meta = build_stream(args.id, spec, force=args.rebuild)
    stream = load_stream(wav_path)
    print(f"[conv] {wav_path.name}: {meta['duration_s']:.1f}s, {len(meta['turns'])} turer", flush=True)

    # Före allt arbete: annars mäter vi vår egen belastning.
    load = wait_for_quiet() if args.save and not args.force else load_per_core()

    started = time.perf_counter()
    report = run_conversation(
        args.api_base, args.llm_url, args.model, args.id, spec, meta, stream, load=load
    )
    report["wall_s"] = round(time.perf_counter() - started, 1)
    print()
    print(render(report))

    if args.save and report["load_per_core"] > MAX_LOAD_PER_CORE and not args.force:
        print(
            f"\n[conv] VÄGRAR spara: last {report['load_per_core']:.2f}/kärna över "
            f"{MAX_LOAD_PER_CORE}. Under last blåses alla stadier upp samtidigt och "
            f"baslinjen blir inte jämförbar. Kör om på en tyst maskin, eller --force "
            f"om du bara vill åt korrektheten.",
            file=sys.stderr,
        )
        return 2

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{args.save}.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\n[conv] skrev {out}")

    summary = summarize_conversation(report)
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
