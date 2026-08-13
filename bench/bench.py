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

"""CLI för latensmätningen.

Startar en egen backend på en fri port, värmer upp modellerna, kör varje
fixtur N gånger, och skriver både en markdown-tabell till terminalen och en
JSON-fil till bench/results/. Exitkoden är nollskild om kvalitetsgrinden
fäller, så en körning kan användas som grind i ett skript.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import requests

from bench.fixtures import load_fixtures
from bench.report import (
    aggregate_paired_ratios,
    build_report,
    gate,
    paired_ratios,
    render_markdown,
    render_markdown_ab,
    summarize,
)
from bench.runner import run_fixture, warmup

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"

BACKEND_STARTUP_TIMEOUT = 180


def start_backend(port, extra_env=None):
    """Startar backend/server.py på `port` och väntar tills den svarar.

    `extra_env` läggs ovanpå den ärvda miljön, t.ex. `{"STT_VAD": "1"}` för
    arm B i en A/B-körning. Arm A får den oförändrade miljön.
    """
    env = dict(os.environ, PORT=str(port), PYTHONUNBUFFERED="1")
    if extra_env:
        env.update(extra_env)
    log_path = pathlib.Path(f"/tmp/bench-backend-{port}.log")
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(REPO_DIR / "venv" / "bin" / "python3"), str(REPO_DIR / "backend" / "server.py")],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_DIR),
    )
    print(f"[bench] Startade backend på {port} (pid {process.pid}), logg: {log_path}", flush=True)

    deadline = time.time() + BACKEND_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Backend dog vid start, se {log_path}")
        try:
            # /api/tts utan text ger 400 — vilket räcker som bevis på att den lyssnar.
            requests.get(f"http://localhost:{port}/api/tts", timeout=2)
            return process
        except requests.RequestException:
            time.sleep(1)
    process.terminate()
    raise RuntimeError(f"Backend svarade inte inom {BACKEND_STARTUP_TIMEOUT}s, se {log_path}")


def parse_env_overrides(spec):
    """Parsar "KEY=VAL[,KEY2=VAL2]" till en dict. Tom sträng/None ger {}."""
    overrides = {}
    if not spec:
        return overrides
    for pair in spec.split(","):
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"Ogiltig --ab-post: {pair!r} (väntade KEY=VAL)")
        overrides[key.strip()] = value.strip()
    return overrides


def load_result(label):
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        raise SystemExit(f"Hittar ingen tidigare körning med label {label!r} ({path})")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_ab(args, fixtures, ab_env):
    """Kör arm A (oförändrad miljö) och arm B (`ab_env` ovanpå) varvat, ABBA.

    Varvningen är poängen: ett par mätt inom sekunder av varandra delar
    belastningstillstånd på den här maskinen, så kvoten mellan dem är
    jämförbar på ett sätt två körningar tagna minuter isär inte är. Se
    `bench/report.paired_ratios` och docstringen på `--ab` ovan.
    """
    api_base_a = f"http://localhost:{args.api_port}"
    api_base_b = f"http://localhost:{args.api_port + 1}"
    llm_url = f"http://localhost:{args.llm_port}/v1/chat/completions"

    backend_a = start_backend(args.api_port)
    backend_b = start_backend(args.api_port + 1, extra_env=ab_env)
    try:
        warmup(api_base_a, llm_url, args.model, fixtures)
        warmup(api_base_b, llm_url, args.model, fixtures)

        per_fixture_a, per_fixture_b, paired = {}, {}, {}
        for fixture_id, spec in fixtures.items():
            runs_a, runs_b = [], []
            for attempt in range(args.repeats):
                # ABBA: varannan repetition kör B först, så att en systematisk
                # ordningseffekt inte lägger sig på samma arm varje gång.
                order = ["a", "b"] if attempt % 2 == 0 else ["b", "a"]
                for arm in order:
                    base = api_base_a if arm == "a" else api_base_b
                    result = run_fixture(base, llm_url, args.model, fixture_id, spec)
                    (runs_a if arm == "a" else runs_b).append(result)
                    print(
                        f"[bench] {fixture_id} {attempt + 1}/{args.repeats} arm {arm.upper()}: "
                        f"{result.time_to_first_audio_ms:.0f} ms, "
                        f"{'ok' if result.ok else 'FEL ' + result.error}",
                        flush=True,
                    )
            # Första paret kastas: det bär cache- och JIT-kostnader, liksom i
            # enarmsläget.
            measured_a = runs_a[1:] or runs_a
            measured_b = runs_b[1:] or runs_b
            per_fixture_a[fixture_id] = summarize(measured_a, spec["text"])
            per_fixture_b[fixture_id] = summarize(measured_b, spec["text"])
            paired[fixture_id] = paired_ratios(measured_a, measured_b)
    finally:
        backend_a.terminate()
        backend_a.wait(timeout=30)
        backend_b.terminate()
        backend_b.wait(timeout=30)
        print("[bench] Backends stoppade.", flush=True)

    report_a = build_report(f"{args.label}-a", per_fixture_a)
    report_b = build_report(f"{args.label}-b", per_fixture_b)
    drift = aggregate_paired_ratios(paired)
    report = {
        "label": args.label,
        "mode": "ab",
        "ab_env": ab_env,
        "arm_a": report_a,
        "arm_b": report_b,
        "paired": paired,
        "drift": drift,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{args.label}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print(render_markdown_ab(args.label, paired, drift, ab_env))
    print()

    # Arm A är facit i grinden: en optimering som knäcker WER mot sitt eget
    # oförändrade jämförelseläge ska fälla precis som mot en historisk baseline.
    passed, messages = gate(report_b, report_a)
    for message in messages:
        print(message)
    print(f"\nSkrev {output_path}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="Mät latens i översättningskedjan.")
    parser.add_argument("--label", required=True, help="Namn på körningen, blir filnamn i bench/results/")
    parser.add_argument("--compare", help="Label att jämföra mot, t.ex. baseline")
    parser.add_argument(
        "--ab",
        help=(
            "Kör en parvis A/B-mätning i samma process. Arm B startas på "
            "--api-port + 1 med \"KEY=VAL[,KEY2=VAL2]\" ovanpå arm A:s miljö, "
            "t.ex. \"STT_VAD=1\". Arm A är den oförändrade miljön. Kan inte "
            "kombineras med --compare — driften mellan körningar är precis "
            "det --ab finns för att kringgå."
        ),
    )
    parser.add_argument("--api-port", type=int, default=3100)
    parser.add_argument("--llm-port", type=int, default=9379)
    parser.add_argument("--model", default="gemma4-e2b")
    # Uppmätt spridning mellan repetitioner vid 3 var för bred på en tredjedel
    # av fixturerna. baseline.json är inspelad med 5 repetitioner, så
    # defaultvärdet måste matcha annars jämförs körningar mot fel urval.
    parser.add_argument("--repeats", type=int, default=5, help="Körningar per fixtur; den första kastas")
    parser.add_argument("--fixtures", help="Kommaseparerad lista med fixtur-id, default alla")
    args = parser.parse_args()

    if args.ab and args.compare:
        raise SystemExit(
            "--ab och --compare kan inte kombineras: --ab jämför inom samma körning "
            "för att kringgå precis den drift --compare inte skyddar mot."
        )

    fixtures = load_fixtures()
    if args.fixtures:
        wanted = set(args.fixtures.split(","))
        fixtures = {key: value for key, value in fixtures.items() if key in wanted}
        if not fixtures:
            raise SystemExit(f"Inga fixturer matchade {args.fixtures!r}")

    if args.ab:
        return run_ab(args, fixtures, parse_env_overrides(args.ab))

    api_base = f"http://localhost:{args.api_port}"
    llm_url = f"http://localhost:{args.llm_port}/v1/chat/completions"

    baseline = load_result(args.compare) if args.compare else None

    backend = start_backend(args.api_port)
    try:
        warmup(api_base, llm_url, args.model, fixtures)

        per_fixture = {}
        for fixture_id, spec in fixtures.items():
            runs = []
            for attempt in range(args.repeats):
                result = run_fixture(api_base, llm_url, args.model, fixture_id, spec)
                marker = " (uppvärmning, kastas)" if attempt == 0 else ""
                status = "ok" if result.ok else f"FEL {result.error}"
                print(
                    f"[bench] {fixture_id} {attempt + 1}/{args.repeats}{marker}: "
                    f"{result.time_to_first_audio_ms:.0f} ms till första ljud, {status}",
                    flush=True,
                )
                runs.append(result)
            # Första körningen kastas: den bär cache- och JIT-kostnader.
            measured = runs[1:] if len(runs) > 1 else runs
            per_fixture[fixture_id] = summarize(measured, spec["text"])
    finally:
        backend.terminate()
        backend.wait(timeout=30)
        print("[bench] Backend stoppad.", flush=True)

    report = build_report(args.label, per_fixture)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{args.label}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print()
    print(render_markdown(report, baseline))
    print()

    passed, messages = gate(report, baseline)
    for message in messages:
        print(message)
    print(f"\nSkrev {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
