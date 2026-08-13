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
import errno
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import requests

from bench.fixtures import load_fixtures
from bench.frontend_mirror import DEFAULT_PROMPT, PROMPT_VARIANTS
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


def assert_port_free(port):
    """Fäller om något redan lyssnar på `port`.

    Utan den här kontrollen är beredskapstestet i `start_backend` nöjt med
    *vem som helst* som svarar — inklusive en kvarglömd backend från en tidigare
    körning — medan den nystartade barnprocessen tyst dör på "Address already in
    use". Konsekvensen är värre än ett krasch: arm B skulle då köras mot arm A:s
    process, med arm A:s miljö, och varje parvis kvot blir ~1,000. Det är exakt
    samma siffra som "optimeringen har ingen effekt", vilket är den vanligaste
    slutsatsen i den här kampanjen — alltså ett fel som ser ut som ett resultat.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Samma SO_REUSEADDR som backendens TCPServer sätter. Utan den fäller
    # kontrollen på TIME_WAIT-rester från nyss stängda anslutningar, alltså på
    # en port barnet mycket väl hade kunnat binda. En aktiv lyssnare stoppar
    # bindningen ändå — SO_REUSEADDR tillåter inte det, bara SO_REUSEPORT gör.
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        # Wildcard, inte 127.0.0.1: backendens TCPServer binder ("", PORT), och
        # på macOS/BSD tillåts en adress-specifik bindning ovanpå en wildcard-
        # bindning. En 127.0.0.1-probe rapporterade därför "ledig" mot en
        # körande backend, vilket är precis det fall kontrollen finns för.
        probe.bind(("", port))
    except OSError as err:
        if err.errno in (errno.EADDRINUSE, errno.EACCES):
            raise SystemExit(
                f"Porten {port} är upptagen. bench startar en egen backend där och "
                f"kan inte mäta mot någon annans process — kvoterna skulle bli ~1,000 "
                f"och se ut som 'ingen effekt'. Stoppa det som lyssnar "
                f"(lsof -nP -iTCP:{port} -sTCP:LISTEN) eller välj en annan --api-port."
            ) from err
        raise
    finally:
        probe.close()


def start_backend(port, extra_env=None):
    """Startar backend/server.py på `port` och väntar tills den svarar.

    `extra_env` läggs ovanpå den ärvda miljön, t.ex. `{"STT_VAD": "1"}` för
    arm B i en A/B-körning. Arm A får den oförändrade miljön.
    """
    assert_port_free(port)
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
        except requests.RequestException:
            time.sleep(1)
            continue
        # Porten var fri innan vi startade, så den som svarar nu är vårt barn —
        # om barnet fortfarande lever. Dog det i stället under uppstarten har
        # någon annan hunnit ta porten emellan, och då mäter vi fel process.
        if process.poll() is not None:
            raise RuntimeError(
                f"Backend på {port} dog under uppstarten men porten svarar — "
                f"någon annan äger socketen. Se {log_path}"
            )
        return process
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


def run_ab(args, fixtures, ab_env, model_b, prompt_variant=None, stream_b=False):
    """Kör arm A (oförändrad miljö) och arm B (`ab_env` ovanpå) varvat, ABBA.

    Varvningen är poängen: ett par mätt inom sekunder av varandra delar
    belastningstillstånd på den här maskinen, så kvoten mellan dem är
    jämförbar på ett sätt två körningar tagna minuter isär inte är. Se
    `bench/report.paired_ratios` och docstringen på `--ab` ovan.

    `model_b` är modell-id:t arm B anropar litert-lm med (`--ab-model`,
    default samma som arm A). Det är skilt från `ab_env` eftersom modellen
    väljs per anrop via `--model`-argumentet till `run_fixture`, inte av
    backend-processens miljö — `--ab`s KEY=VAL-mekanism kan inte uttrycka
    det.

    `prompt_variant` väljer arm B:s systemprompt (`--ab-prompt`, en nyckel i
    PROMPT_VARIANTS, t.ex. "json"). Skild av samma skäl som `model_b`:
    prompten byggs i bench:s egen process av frontend_mirror.system_prompt,
    inte av backend-processens miljö, så `--ab` kan inte uttrycka det. Arm A
    kör `--prompt` (default "plain", alltså produktens egen prompt).

    `stream_b` väljer om arm B strömmar LLM-svaret (`--ab-stream`). Skild av
    samma skäl som `model_b`/`prompt_variant`: strömning är en runner-nivå-
    ändring i vilken anropsfunktion `run_fixture` använder
    (`_post_llm_streaming` mot `_post_llm`), inte något backend-processens
    miljö kan uttrycka. Proxyändringen i handle_proxy är transparent för
    icke-strömmande anrop, så arm A och arm B kan dela samma backend-kod —
    bara vilken funktion bench:s egen process anropar skiljer dem åt.
    """
    api_base_a = f"http://localhost:{args.api_port}"
    api_base_b = f"http://localhost:{args.api_port + 1}"
    llm_url = f"http://localhost:{args.llm_port}/v1/chat/completions"
    prompt_fn_a = PROMPT_VARIANTS[args.prompt]
    prompt_fn_b = PROMPT_VARIANTS[prompt_variant] if prompt_variant else prompt_fn_a

    backend_a = start_backend(args.api_port)
    backend_b = start_backend(args.api_port + 1, extra_env=ab_env)
    try:
        warmup(api_base_a, llm_url, args.model, fixtures, prompt_fn=prompt_fn_a)
        # Arm B värms med model_b, inte args.model — annars laddar litert-lm
        # fortfarande CPU-modellen i uppvärmningen och arm B:s första mätta
        # repetition bär GPU-buildens ~7 s vikt-laddningskostnad i stället.
        # Samma resonemang gäller prompt_fn_b: värm med den prompt som
        # faktiskt mäts, inte standardprompten. stream=stream_b av samma
        # skäl: värm den kodväg som faktiskt mäts.
        warmup(api_base_b, llm_url, model_b, fixtures, prompt_fn=prompt_fn_b, stream=stream_b)

        per_fixture_a, per_fixture_b, paired = {}, {}, {}
        for fixture_id, spec in fixtures.items():
            runs_a, runs_b = [], []
            for attempt in range(args.repeats):
                # ABBA: varannan repetition kör B först, så att en systematisk
                # ordningseffekt inte lägger sig på samma arm varje gång.
                order = ["a", "b"] if attempt % 2 == 0 else ["b", "a"]
                for arm in order:
                    base = api_base_a if arm == "a" else api_base_b
                    model = args.model if arm == "a" else model_b
                    prompt_fn = prompt_fn_a if arm == "a" else prompt_fn_b
                    stream = False if arm == "a" else stream_b
                    result = run_fixture(base, llm_url, model, fixture_id, spec, prompt_fn, stream=stream)
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

    report_a = build_report(
        f"{args.label}-a",
        per_fixture_a,
        {"model": args.model, "prompt": args.prompt, "stream": False, "repeats": args.repeats},
    )
    report_b = build_report(
        f"{args.label}-b",
        per_fixture_b,
        {
            "model": model_b,
            "prompt": prompt_variant or args.prompt,
            "stream": stream_b,
            "repeats": args.repeats,
            "env": dict(ab_env or {}),
        },
    )
    drift = aggregate_paired_ratios(paired)
    report = {
        "label": args.label,
        "mode": "ab",
        "ab_env": ab_env,
        "model_a": args.model,
        "model_b": model_b,
        "prompt_a": args.prompt,
        "prompt_b": prompt_variant or args.prompt,
        "stream_a": False,
        "stream_b": stream_b,
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
    ab_note = dict(ab_env or {})
    if model_b != args.model:
        ab_note["model"] = f"{args.model} -> {model_b}"
    if prompt_variant:
        ab_note["prompt"] = f"{args.prompt} -> {prompt_variant}"
    if stream_b:
        ab_note["stream"] = "off -> on"
    print(render_markdown_ab(args.label, paired, drift, ab_note))
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
    parser.add_argument(
        "--ab-model",
        help=(
            "Kör en parvis A/B-mätning där arm B anropar litert-lm med ett "
            "annat modell-id än arm A, t.ex. \"gemma4-e2b,gpu\". Skild från "
            "--ab: modellen väljs per anrop via --model, inte av backend-"
            "processens miljö, så --ab kan inte uttrycka den här ändringen. "
            "Måste vara ett eget flagg snarare än gå via --ab också av en "
            "annan anledning — \"gemma4-e2b,gpu\" innehåller ett kommatecken, "
            "som är --ab:s egen separator mellan KEY=VAL-par, så det skulle "
            "delas upp fel. Kan kombineras med --ab (byt miljövariabel och "
            "modell samtidigt) men inte med --compare, av samma skäl som --ab."
        ),
    )
    parser.add_argument(
        "--prompt",
        choices=sorted(PROMPT_VARIANTS),
        default=DEFAULT_PROMPT,
        help=(
            "Systemprompt att mäta med (arm A i A/B-läge). Default \"plain\", "
            "vilket är den prompt produkten faktiskt skickar. \"json\" är den "
            "gamla wrapper-prompten, kvar för att kunna reproducera Task 12."
        ),
    )
    parser.add_argument(
        "--ab-prompt",
        choices=sorted(PROMPT_VARIANTS),
        help=(
            "Kör en parvis A/B-mätning där arm B använder en annan systemprompt "
            "än arm A, t.ex. \"json\". Skild från --ab: prompten byggs i "
            "bench:s egen process (frontend_mirror.system_prompt), inte av "
            "backend-processens miljö eller modell-id:t, så varken --ab eller "
            "--ab-model kan uttrycka den här ändringen — bara den startas i sitt "
            "eget flagg. Kan kombineras med --ab och/eller --ab-model men inte "
            "med --compare, av samma skäl som de."
        ),
    )
    parser.add_argument(
        "--ab-stream",
        action="store_true",
        help=(
            "Kör en parvis A/B-mätning där arm B strömmar LLM-svaret och mäter "
            "tid-till-första-mening (arm A svarar icke-strömmande som idag). "
            "Skild av samma skäl som --ab-model/--ab-prompt: strömning väljs "
            "per anrop i run_fixture (vilken _post_llm-funktion som anropas), "
            "inte av backend-processens miljö, så --ab kan inte uttrycka det. "
            "Proxyändringen är transparent för icke-strömmande anrop, så arm A "
            "och arm B kör samma backend-kod. Kan kombineras med --ab/--ab-model/"
            "--ab-prompt men inte med --compare, av samma skäl som de."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Mät den strömmande LLM-vägen i enarmsläget (utan --ab/--ab-model/--ab-prompt/--ab-stream)",
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
    if args.ab_model and args.compare:
        raise SystemExit(
            "--ab-model och --compare kan inte kombineras, av samma skäl som --ab."
        )
    if args.ab_prompt and args.compare:
        raise SystemExit(
            "--ab-prompt och --compare kan inte kombineras, av samma skäl som --ab."
        )
    if args.ab_stream and args.compare:
        raise SystemExit(
            "--ab-stream och --compare kan inte kombineras, av samma skäl som --ab."
        )
    if args.stream and (args.ab or args.ab_model or args.ab_prompt or args.ab_stream):
        # run_ab låser arm A till icke-strömmande (det är jämförelseläget), så
        # ett --stream här hade tyst ignorerats för arm A och bara råkat
        # sammanfalla med --ab-stream för arm B.
        raise SystemExit(
            "--stream gäller bara enarmsläget. I A/B-läge är arm A per definition "
            "det oförändrade läget (icke-strömmande) — använd --ab-stream för att "
            "låta arm B strömma."
        )
    if args.repeats < 1:
        raise SystemExit("--repeats måste vara minst 1.")
    if args.repeats % 2 == 0:
        # ABBA-varvningen börjar med A i repetition 0, och just den kastas som
        # uppvärmning. Med ett jämnt antal blir de mätta repetitionerna udda
        # till antalet och den ena armen får en ordningsposition mer än den
        # andra, alltså en systematisk ordningseffekt på en av armarna.
        raise SystemExit(
            f"--repeats måste vara udda (fick {args.repeats}): den första "
            f"repetitionen kastas, och ett jämnt antal ger obalanserad ABBA-varvning."
        )

    fixtures = load_fixtures()
    if args.fixtures:
        wanted = set(args.fixtures.split(","))
        fixtures = {key: value for key, value in fixtures.items() if key in wanted}
        if not fixtures:
            raise SystemExit(f"Inga fixturer matchade {args.fixtures!r}")

    if args.ab or args.ab_model or args.ab_prompt or args.ab_stream:
        return run_ab(
            args,
            fixtures,
            parse_env_overrides(args.ab),
            args.ab_model or args.model,
            args.ab_prompt,
            args.ab_stream,
        )

    api_base = f"http://localhost:{args.api_port}"
    llm_url = f"http://localhost:{args.llm_port}/v1/chat/completions"

    baseline = load_result(args.compare) if args.compare else None

    prompt_fn = PROMPT_VARIANTS[args.prompt]

    backend = start_backend(args.api_port)
    try:
        warmup(api_base, llm_url, args.model, fixtures, prompt_fn=prompt_fn, stream=args.stream)

        per_fixture = {}
        for fixture_id, spec in fixtures.items():
            runs = []
            for attempt in range(args.repeats):
                result = run_fixture(
                    api_base, llm_url, args.model, fixture_id, spec, prompt_fn, stream=args.stream
                )
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

    report = build_report(
        args.label,
        per_fixture,
        {
            "model": args.model,
            "prompt": args.prompt,
            "stream": args.stream,
            "repeats": args.repeats,
            "fixtures": sorted(fixtures),
        },
    )
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
