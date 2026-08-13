<img width="960" height="540" src="https://storage.googleapis.com/experiments-uploads/gemma-translator/gemma-translator.gif" />

# Gemma Translator

This repo was built with the assistance of [Google Antigravity](https://antigravity.google/) and includes code to run an on-device, fully offline voice translator powered by [Gemma 4](https://ai.google.dev/gemma/docs/core) and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-lm). This project features a web frontend optimized for small handheld displays (e.g., 480x320) and a Python API server (`http.server`) that communicates with Gemma. Speech recognition is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and speech synthesis by [Piper](https://github.com/OHF-Voice/piper1-gpl).

https://github.com/user-attachments/assets/343072ce-dc78-44a7-a783-99312845cabe

## Features

- **On-Device Inference**: Uses LiteRT-LM to run the `gemma4-e2b` model entirely locally. No internet required after setup.
- **Nordic Languages**: Swedish and Finnish are supported alongside the original six (see [Languages](#languages)).
- **Voice Interface**: Captures microphone audio, processes it, and sends it to the local model.
- **Optimized UI**: Retro-terminal styling custom-built for small hardware screens (like Raspberry Pi displays).
- **Unified Startup**: One script to launch the LLM server, the Python API, and the React frontend.

## Prerequisites

- Python 3.10+. `setup.sh` picks the newest `python3.1x` on `PATH` that qualifies; override with `PYTHON=/path/to/python3 ./setup.sh`
- Node.js 18+ (20 LTS recommended) & npm — installed automatically by `deploy-pi.sh` on Raspberry Pi OS / Debian
- Linux or macOS

## Required Hardware

- **Compute**: Raspberry Pi 5 with 8GB RAM
- **Audio Input**: Microphone or USB audio capture interface
- **Audio Output**: Speaker or headphone output device
- **Display**: Display monitor or touchscreen (e.g., 480x320 kiosk display)

<img width="3024" height="1672" src="https://storage.googleapis.com/experiments-uploads/gemma-translator/gemma-translator-cad.gif" />

## Setup Instructions

1. **Make Scripts Executable**
   Ensure the setup, download, start, and deployment scripts have execute permissions:
   ```bash
   chmod +x setup.sh download_model.sh start.sh deploy-pi.sh
   ```

2. **Install Dependencies**
   Run the setup script to create a Python virtual environment (`venv`) and install all required packages:
   ```bash
   ./setup.sh
   ```

3. **Download the Model**
   Run the model downloader script to fetch the `gemma4-e2b` model from Hugging Face and import it into LiteRT-LM:
   ```bash
   ./download_model.sh
   ```

## Running the Application

Start all services (LiteRT-LM, the Python API server, and the Vite Web UI) in development mode:
```bash
./start.sh
```

To run in production mode (skipping Vite dev server and serving compiled UI assets from `frontend/dist/` via `backend/server.py` on port 3000):
```bash
./start.sh --prod
```

The application will be accessible at:
- **Web UI (Dev)**: `http://localhost:5173`
- **Web UI (Prod) / API server**: `http://localhost:3000`
- **LiteRT-LM**: `http://localhost:9379`

## Languages

The lanes default to **Swedish** and **English**; rotate to Finnish, Arabic, Spanish, Japanese, Chinese, or Korean with the arrow keys. Each leg of the pipeline runs one engine:

| Step | Engine |
| :--- | :--- |
| Speech recognition (all languages) | faster-whisper (`small`, int8 CPU) |
| Translation (all languages) | Gemma 4 E2B via LiteRT-LM |
| Speech synthesis (all but `ja`) | Piper |
| Speech synthesis (`ja`) | moonshine-voice (Kokoro) |

The backend pre-warms the models for both default lanes at startup, so the first server start downloads the Whisper and Piper models (~500MB combined) into `~/.cache/huggingface` and `~/.local/share/piper-voices`. Everything runs offline afterwards. Two environment variables tune this:

- `WHISPER_MODEL_SIZE`: Whisper checkpoint (default `small`). `medium` is noticeably more accurate on Swedish and Finnish but roughly 3x slower. Any faster-whisper-compatible model ID works, including Swedish-tuned ones such as `KBLab/kb-whisper-medium`.
- `PIPER_VOICE_DIR`: where Piper voices are stored (default `~/.local/share/piper-voices`).

### Adding a language

1. Add it to `AVAILABLE_LANGUAGES` in `frontend/src/TranslatorApp.jsx`.
2. In `backend/server.py`, add its code to `SUPPORTED_STT_LANGS` (Whisper covers ~99 languages; an unlisted code is auto-detected instead) and to `PIPER_VOICE_MAP`, picking a voice ID from `piper.download_voices.list_voices()`.

Piper covers 54 languages but has no Japanese voice, which is the one case still routed to moonshine-voice via `TTS_LANG_MAP`. Any other language Piper lacks needs the same treatment, or a different engine.

## Benchmarking

`bench/` measures the full push-to-talk chain over HTTP and gates on quality
regression. It synthesizes its own fixtures with Piper, so no recordings are
needed:

```bash
# Requires litert-lm already running on :9379 (./start.sh)
venv/bin/python3 -m bench.bench --label baseline
venv/bin/python3 -m bench.bench --label my-change --compare baseline
```

Each run starts its own backend on port 3100, warms the models, and reports
median time-to-first-audio per fixture alongside word error rate. A run exits
non-zero if *corpus* WER — total word errors over total reference words across
all fixtures — rises more than 2 points against the comparison run. Per-fixture
WER is reported too but doesn't gate: with 3-33 word references, one changed
word moves a single fixture's WER by 3 to 33 points, so a per-fixture threshold
would be zero tolerance wearing a percentage sign. Results are committed to
`bench/results/` as a history of the optimization campaign.

Things worth knowing before you run it:

- **The port must be free.** bench refuses to start if anything already listens
  on `--api-port` (or, in A/B mode, `--api-port + 1`). It cannot measure against
  someone else's process, and the failure mode it prevents is silent: arm B
  would run arm A's configuration and every ratio would read ≈1.000, which is
  indistinguishable from "the optimization had no effect".
- **`--repeats` must be odd.** The first repetition is discarded, and the ABBA
  interleave starts A-first, so an even count gives one arm an extra order
  position. Default 5.
- **`--prompt {plain,json}`** picks the system prompt. `plain` is the default
  and is the prompt the product actually sends; `json` is the retired wrapper
  prompt, kept so the measurement behind it can be reproduced. `bench` asserts
  that its `plain` prompt is byte-identical to the one in `TranslatorApp.jsx`.
- **`--stream`** measures the streaming LLM path in single-arm mode; use
  `--ab-stream` to compare it against non-streaming in one paired run. The two
  cannot be combined — in A/B mode arm A is the unchanged, non-streaming arm by
  definition.
- **The Δ column is dropped when two runs measured different configurations**
  (model, prompt or streaming), including when the older run predates
  provenance recording. WER is still compared: it has no time component.

### The GPU model variant (Mac only, opt-in)

On machines with a supported GPU, litert-lm also serves the same weights as
`gemma4-e2b,gpu`. Point the model name in Settings at it, or measure it with
`--ab-model "gemma4-e2b,gpu"`. It is **not the default**: the deployment target
is a Raspberry Pi 5, which has no such build.

Measured as a paired A/B run against the CPU build
(`bench/results/09-gpu-model.json`, 5 repeats, ABBA-interleaved). **The win is
concentrated in long utterances**, which is what a token-throughput win looks
like — the GPU speeds up generation, so the more tokens the answer has, the more
it saves, while short answers stay dominated by fixed per-request overhead:

| Fixtures | `llm_ms` GPU/CPU | Reading |
| :--- | ---: | :--- |
| long (3 fixtures) | 0.366 / 0.372 / 0.399 | ~2.7× faster, unambiguous |
| short and medium (6) | 0.75–1.05 | no clear effect at this run's noise floor |

Two caveats before switching:

- **The first request costs about 7 seconds** while the GPU weights load. Every
  request after that is warm.
- **The two builds do not produce identical translations.** Same weights, but a
  different numerical path: 4 of 9 fixtures came out worded differently (e.g.
  "how I get to the station" vs "how to get to the station"). The differences
  look benign, and corpus WER is unchanged at 0.091 — but WER scores
  *transcription*, not translation, so it cannot catch this. Switching to the
  GPU build changes translation behaviour as well as speed.

## Latency

Measured with `bench/` (see [Benchmarking](#benchmarking)) on a MacBook Air M1
(8 cores), Whisper `small` int8, `gemma4-e2b` via LiteRT-LM. Baseline: median
time-to-first-audio 3225 ms across the original nine fixtures, corpus WER
0.091 (11 errors / 121 reference words) — `bench/results/baseline.json`.

**A methodology note that shapes every number below.** The first optimization
measured in this campaign (the Whisper VAD filter) initially looked like a
20% win under `--label X --compare baseline`. It wasn't: `llm_ms`, a stage the
VAD filter cannot touch, moved by almost exactly the same amount between the
two runs, because the two `bench` invocations simply ran under different
machine load. **Comparing medians from two separately-run bench labels is
confounded and cannot be trusted for timing.** Every ratio below therefore
comes from a **paired A/B run** — one `bench` process, one ABBA-interleaved
sequence of repeats, both arms sharing the same load — not from a diff
against `baseline.json`. Word error rate has no time component and isn't
subject to this confound, so the WER comparison below *is* a valid cross-run
diff.

### Quality: corpus WER against baseline

Current (`bench/results/final.json`): **0.089** (13 errors / 146 reference
words). Baseline (`bench/results/baseline.json`): **0.091** (11 errors / 121
reference words).

The denominators aren't directly comparable: `sv-multi` was added later (for
the streaming measurement, see below), so the current corpus has ten fixtures
and 146 reference words against baseline's nine and 121. Read this as "no
quality drift is visible across the campaign", not as a precise before/after
percentage.

### Results

| Optimization | Result | Aggregate ratio | Long-fixture ratio | Evidence |
| :--- | :--- | ---: | ---: | :--- |
| Whisper VAD filter | Rejected — no measurable effect | STT 1.011 | — | `bench/results/01-vad-ab.json` |
| Short mel padding (`STT_CHUNK_MARGIN_S`) | Rejected — no measurable effect; code reverted | STT 0.999 | — | `bench/results/02-chunklen-ab.json` |
| Whisper decode flags (`cpu_threads`, etc.) | Never built — rejected by extension of the two STT nulls above | — | — | — |
| Sentence-level TTS chunking | Never built — targets a stage worth ~5% of time-to-first-audio, below this harness's resolution | — | — | — |
| TTS prefetch | Never built — same reason | — | — | — |
| Client audio path (e.g. binary PCM upload) | Never built — same reason | — | — | — |
| Shorter system prompt (drop the JSON wrapper) | **Real win — now the default** | `llm_ms` 0.620 | 0.341–0.395 | `bench/results/05-plain-prompt.json` |
| GPU model build (`gemma4-e2b,gpu`) | **Real win — opt-in, not the default** (see [above](#the-gpu-model-variant-mac-only-opt-in)) | `llm_ms` 0.771 | 0.366–0.399 | `bench/results/09-gpu-model.json` |
| Streaming Gemma → TTS | **Accepted, narrow — now the default** | time-to-first-audio 0.996 (a null) | `sv-multi` 0.760, `sv-long` 0.822 | `bench/results/15-streaming.json` |

"Aggregate ratio" and "long-fixture ratio" are B/A medians from each paired
A/B run (optimization on ÷ optimization off); below 1.0 is faster. Rejected
and never-built rows have no ratio: either the paired run showed no
distinguishable effect against the ~1–3% noise floor set by two control
stages the optimization cannot touch, or the optimization was never
implemented, so there is nothing to cite. Two of nine optimizations produced
real wins, one is narrow, three were rejected (two measured directly, one by
extension of that measurement), and three were never built because they
targeted a combined ~5% of the metric — below what this harness can resolve.

### Whisper's cost is a fixed-size encoder pass

Two independent optimizations tried to shrink Whisper's cost by shrinking the
audio fed to it. Both measured as nulls:

- **VAD filter** (trim silence before decoding, `STT_VAD=1`): STT ratio
  **1.011** — no improvement, if anything slightly slower.
  `bench/results/01-vad-ab.json`. Left env-gated off by default.
- **Short mel padding** (`STT_CHUNK_MARGIN_S=5`, pad only to audio length +
  ~5s instead of the fixed 30s): STT ratio **0.999**.
  `bench/results/02-chunklen-ab.json`. Code reverted.

Both are explained by the same mechanism. `faster_whisper`'s `chunk_length`
parameter does shorten the mel spectrogram as expected (`n_samples` and
`nb_max_frames` drop), but CTranslate2's Whisper encoder takes a **fixed-size
input** — `pad_or_trim` pads the mel back out to the full window before the
encoder runs. Shortening the audio therefore saves only the mel computation,
which is negligible next to the encoder pass itself. Moonshine is ~4–7×
faster than Whisper on short clips because its *encoder* is variable-length,
not because of any padding arithmetic — no padding or silence-trimming trick
will reproduce that win on Whisper.

**Consequence for anyone chasing STT latency next: don't shorten the audio.**
Change the encoder itself, or the number of encoder passes. This is also why
Whisper decode flags (`without_timestamps=True`,
`condition_on_previous_text=False`, `cpu_threads`) were never built — after
two STT nulls with the mechanism understood, they were judged the least
promising lever left, and the remaining effort went to streaming instead.

### The system prompt is a latency cost — there is a context cliff

**If you edit the system prompt in `frontend/src/TranslatorApp.jsx`, you are editing latency.** Keep it short, and keep `bench/frontend_mirror.py` in sync.

Replacing the old ~430-character JSON-wrapper prompt with a ~144-character plain-text one cut `llm_ms` by 38% overall and made long utterances **2.6× faster** — see `bench/results/05-plain-prompt.json`. The reason is *not* the ~10 output tokens of `{"translation": "..."}` saved. Padding the short prompt back out to 430 characters with neutral filler, holding the instruction and the generated output byte-identical, reproduces the full slowdown (2061 ms → 5066 ms). The cost is the prompt's **length**, not the wrapper.

But it is a threshold, not a per-character rate. Sweeping prompt length at a fixed long input and 155-character output:

| System prompt | Approx. total context | Median |
| ---: | ---: | ---: |
| 144–415 chars | 451–722 chars | ~1700–1780 ms |
| 425 chars | 728 chars | 4772 ms |
| 432 chars | 739 chars | 4648 ms |

Flat across 271 extra characters, then a 2.6× cliff between ~722 and ~728 characters of **total context — system prompt + user input + generated output combined**. Below the cliff, prompt length is genuinely free: tripling the prompt while generating a 29- or 60-character reply costs nothing measurable (−29 ms and −24 ms, i.e. noise). Above it, the same 288 characters cost ~2900 ms.

So a long system prompt is harmless on short utterances and brutal on long ones, because only long utterances push the total past the boundary. That is exactly why the win concentrated on the long fixtures. `litert-lm serve` runs with no explicit context or cache flag (`start.sh:44`), so this is a default; the exact token boundary behind the ~725-character figure has not been identified.

Practical consequence: further shortening the prompt does not speed up requests that already sit below the cliff — it raises the utterance length at which you fall off it.

### Streaming Gemma → TTS: helps only on multi-sentence translations

The LLM response is streamed, and each sentence is synthesized and played as soon as it is complete, while the rest is still generating. The proxy in `backend/server.py` forwards chunk-by-chunk rather than buffering the whole response; `translateTextStreaming` reads the SSE deltas; `TranslatorApp` keeps a TTS queue whose `Audio` elements start fetching at enqueue time, so queued sentences download while the current one plays.

**The benefit is confined to translations that run to more than one sentence.** With a single-sentence answer the first sentence *is* the whole response, so there is nothing to overlap and streaming saves exactly nothing. Measured as a paired A/B run (`bench/results/15-streaming.json`, 5 repeats, ABBA-interleaved):

| Fixture | Time-to-first-audio B/A | Why |
| :--- | ---: | :--- |
| `sv-multi` (three sentences out) | **0.760** | audio starts 1083 ms before generation ends |
| `sv-long` (internal sentence break) | **0.822** | 697 ms of overlap |
| the other 8 (single sentence) | 0.954–1.061 | nothing to overlap; neutral |

The suite-wide aggregate is 0.996 — a null. **Do not quote that figure without the qualification**: on its own it reads as "streaming does nothing", which is wrong in the other direction. Eight of the ten fixtures are structurally incapable of showing the effect because they translate to one sentence, so the aggregate describes the fixture set rather than the optimization. On the two that can show it, the first sentence lands at 47% and 62% of the full response time, and that whole remainder is spent talking rather than waiting.

Both stage controls that streaming cannot touch came in clean — `stt_ms` 1.004 and `llm_ms` 1.009 — and all ten translations were byte-identical across arms with identical corpus WER (0.089 in both), so the null on single-sentence output is a real measurement rather than an effect lost in noise. `tts_first_ms` drifted 1.075 in this run, wider than the other two: it is a 38–340 ms quantity, so a few tens of milliseconds of jitter is several percent. Cite the two tight controls, not that one.

These figures replace an earlier, more pessimistic set (aggregate 0.985, `sv-multi` 0.851, `sv-long` 0.936). Three defects understated the effect and were fixed together: the proxy read the SSE stream in 1024-byte blocks and so withheld 4–5 tokens per flush; bench detected the first sentence with `endswith`, which only fires when a delta *ends* on punctuation, where the product uses `lastIndexOf`; and bench's default system prompt was still the retired 432-character JSON one, which put the long fixtures on the far side of the context cliff and inflated both arms' LLM times about threefold.

**A trap for anyone adding another streaming measurement:** `requests.iter_lines(decode_unicode=True)` falls back to **ISO-8859-1** for a `text/*` response with no charset, and litert-lm's `text/event-stream` has none. That turns `är` into `Ã¤r` and, because mojibake is longer than the text it replaces, inflates the measured TTS time on non-ASCII output — it corrupts timings, not just strings. Set `response.encoding = "utf-8"` first; see `bench/runner._post_llm_streaming`. The browser was never affected, since `TextDecoder` defaults to UTF-8.

### The fixture set is biased toward short utterances

Every bench fixture is a short, single-utterance prompt: one to three sentences, 1.1–10.1s of audio, 17–171 characters of output. **Any optimization whose benefit scales with output length is systematically under-measured here.**

That is not hypothetical — it has now happened three times. The GPU build (0.771 aggregate `llm_ms`, but 0.37 on the long fixtures), the shortened system prompt (same shape, same reason), and streaming (a null aggregate against 0.760 on the one multi-sentence fixture) all have their real effect concentrated in the fixtures the suite has fewest of.

Read every aggregate in `bench/results/` with that in mind, and add longer-output fixtures before concluding that a length-scaling optimization is worthless.

### Summary

Nine optimizations were tried. Two produced real, shipped wins (the shorter
system prompt, the opt-in GPU build). One produced a narrow, real win visible
only on multi-sentence output (streaming). Two were measured directly and
rejected (the VAD filter, short mel padding), and a third (Whisper decode
flags) was rejected by extension of that same measurement without being
built. The remaining three (sentence-level TTS chunking, TTS prefetch, the
client audio path) were never built because they targeted a combined ~5% of
time-to-first-audio — below what a ~1–3% noise floor lets this harness
resolve.

That is the honest shape of the result: most ideas on the original list
turned out not to matter on this fixture set, and finding that out with a
paired measurement before building all of them — rather than shipping each
one on faith — was the point of the campaign.

## Raspberry Pi Appliance Deployment

To deploy as a permanent systemd kiosk service on a Raspberry Pi 5 (8GB):
```bash
./deploy-pi.sh
```
This automated script installs Debian audio/venv packages, sets up the Python environment, builds production UI assets, downloads the LiteRT model, registers the systemd unit from `deploy/gemma-translator.service`, and configures LXDE GUI autostart (`~/.config/lxsession/rpd-x/autostart`) to launch Chromium in kiosk mode pointing to `http://localhost:3000`.

## Project Structure

- `frontend/` - React (Vite) web frontend (`index.html`, `src/`, styles, and Vite configuration).
- `backend/` - Python API server (`server.py` and `requirements.txt`) for Whisper STT, Piper TTS, and model proxying.
- `deploy/` - Parameterizable systemd service unit template (`gemma-translator.service`).
- `stl/` - STL files for 3D printing the hardware case.
- `setup.sh` - Automates Python virtual environment creation and dependency installation.
- `download_model.sh` - Fetches the required LiteRT model.
- `start.sh` - Multi-process launcher supporting `--prod` and development modes.
- `deploy-pi.sh` - One-command Raspberry Pi automated deployment script.

## Keyboard Shortcuts

The Gemma Translator supports **two keyboard modes**. Switch between them anytime from the **Settings panel → "Keyboard Mode"** dropdown. The choice is remembered across restarts (stored in the browser's `localStorage` under the key `keyboardMode`).

The app has two lanes (two people facing each other on the kiosk):
- **Lane 1 / Person 1** — the left/top lane.
- **Lane 2 / Person 2** — the right/bottom lane.

Each lane has a rotating language "revolver" and records speech, which is transcribed (Whisper STT), translated (Gemma), and spoken back in the other lane's language (Piper TTS).

### Landscape Mode (default) — "active person"
One lane is the **active person** at a time. The active lane is framed with **corner brackets on all four corners**. You drive everything from a single set of keys and switch focus with Space.

| Key | Action | Description |
| :--- | :--- | :--- |
| **Spacebar** | Switch active person | Toggles the active lane (Person 1 ⇄ Person 2). Disabled while recording. |
| **Z** | Record (push-to-talk) | Hold to record the **active** person; release to transcribe & translate. |
| **← Left Arrow** | Previous language | Rotates the **active** person's language backward. |
| **→ Right Arrow** | Next language | Rotates the **active** person's language forward. |

Notes:
- The active lane shows four-corner brackets; while it is recording, the brackets invert to black along with the lane's color reversal.
- Best for one-handed / single-operator use.

### Vertical Mode — "two-hand" (original mapping)
Each lane has its **own dedicated keys** — there is no active-person concept and **no bracket highlight**. Both people can be controlled independently.

| Key | Action | Description |
| :--- | :--- | :--- |
| **Z** | Record — Person 1 (push-to-talk) | Hold to record Lane 1; release to transcribe & translate. |
| **X** | Record — Person 2 (push-to-talk) | Hold to record Lane 2; release to transcribe & translate. |
| **← Left Arrow** | Previous language — Person 1 | Rotates Lane 1's language backward. |
| **→ Right Arrow** | Next language — Person 1 | Rotates Lane 1's language forward. |
| **− Minus** (`_`) | Previous language — Person 2 | Rotates Lane 2's language backward. |
| **+ Plus** (`=`) | Next language — Person 2 | Rotates Lane 2's language forward. |

Notes:
- No corner-bracket selection highlight in this mode.
- Best for two operators, each handling their own side.

### Common behavior (both modes)
- **Input focus guard:** all shortcuts are ignored while focus is on a configuration field (`<input>`, `<textarea>`, or `<select>`) — e.g. when editing the API endpoint or settings.
- **Recording lock:** language rotation is blocked while a recording is in progress.
- **Keyboard-driven:** recording and language rotation are keyboard-only in the current build; on-screen touch controls are not enabled.

### Switching modes
Open **Settings (⚙)** → **Keyboard Mode** → choose **Landscape** or **Vertical**. The change takes effect immediately and persists on the device.

| Setting value | Mode |
| :--- | :--- |
| `landscape` | Active-person scheme (Space / Z / ← →) — default |
| `vertical` | Two-hand scheme (Z / X / ← → / − +) |

### Credits
Made by a small team at [Google Creative Lab](https://github.com/googlecreativelab):
- [Alan Yam](https://github.com/alanvww)
- [Shashwath Santosh](https://x.com/shashwth)
- [Dan Motzenbecker](https://github.com/dmotz)

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
