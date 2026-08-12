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
from bench.report import build_report, gate, render_markdown, summarize
from bench.runner import run_fixture, warmup

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"

BACKEND_STARTUP_TIMEOUT = 180


def start_backend(port):
    """Startar backend/server.py på `port` och väntar tills den svarar."""
    env = dict(os.environ, PORT=str(port), PYTHONUNBUFFERED="1")
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


def load_result(label):
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        raise SystemExit(f"Hittar ingen tidigare körning med label {label!r} ({path})")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Mät latens i översättningskedjan.")
    parser.add_argument("--label", required=True, help="Namn på körningen, blir filnamn i bench/results/")
    parser.add_argument("--compare", help="Label att jämföra mot, t.ex. baseline")
    parser.add_argument("--api-port", type=int, default=3100)
    parser.add_argument("--llm-port", type=int, default=9379)
    parser.add_argument("--model", default="gemma4-e2b")
    parser.add_argument("--repeats", type=int, default=3, help="Körningar per fixtur; den första kastas")
    parser.add_argument("--fixtures", help="Kommaseparerad lista med fixtur-id, default alla")
    args = parser.parse_args()

    api_base = f"http://localhost:{args.api_port}"
    llm_url = f"http://localhost:{args.llm_port}/v1/chat/completions"

    fixtures = load_fixtures()
    if args.fixtures:
        wanted = set(args.fixtures.split(","))
        fixtures = {key: value for key, value in fixtures.items() if key in wanted}
        if not fixtures:
            raise SystemExit(f"Inga fixturer matchade {args.fixtures!r}")

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
