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

## Latency notes & ideas

A typical push-to-talk round trip is ~7s today. Rough split on this Mac (CPU, Whisper `small` int8):

| Step | Time | Notes |
| :--- | :--- | :--- |
| STT (faster-whisper) | ~1s | Dominated by Whisper's fixed 30s mel padding |
| Translation (Gemma) | 1–6s | The main remaining cost |
| TTS (Piper) | ~0.1s | Already fast |

Moonshine was ~4–7× faster than stock Whisper on 1–3s clips for the same reason (no 30s pad), but it has no Swedish/Finnish. Benchmarks showed that padding only to *audio length + ~5s silence* (instead of 30s), plus `cpu_threads=8`, brings Whisper roughly to Moonshine parity with no WER loss on the clips tested. That patch is **not** in the tree yet — it needs a `pad_or_trim` monkeypatch and a safety margin (zero padding loops).

### Ideas worth trying next (not implemented)

Ordered by expected impact on perceived latency:

1. **Whisper short padding + `cpu_threads`** — env-gated (+5s margin, keep 30s as default). Closes most of the Moonshine gap (~0.7s/utterance).
2. **Drop the JSON wrapper on Gemma** — ask for plain translation text instead of `{"translation":"..."}`. Fewer output tokens; the UI already falls back if JSON parse fails.
3. **Stream Gemma → TTS** — synthesize/play the first sentence while the rest generates. Cuts perceived wait by 1–3s even if wall time is unchanged (needs LiteRT streaming + sentence buffering).
4. **Prefetch TTS chunk N+1 while chunk N plays** — `playTTS` today fetches the next chunk only after `onended`.
5. **Whisper decode flags** — `without_timestamps=True`, `condition_on_previous_text=False` alongside `cpu_threads` (~0.2s, stabler short clips).
6. **Trim leading/trailing silence** before STT — shorter audio helps short-pad Whisper and sends less noise to Gemma.
7. **Binary PCM upload** instead of base64 Float32 — tiny on 2–3s clips; more relevant on a Pi.
8. **Lighter Piper voices** (`…-low` / `…-x_low`) — TTS is already ~0.1s; mainly a Pi memory/CPU win at some quality cost.

Suggested build order: (1)+(5) together, then (2), then (3) if it still feels slow.

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
One lane is the **active person** at a time. The active lane is framed with **corner brackets on all four corners**. You drive everything from a single set of keys and switch focus with Enter.

| Key | Action | Description |
| :--- | :--- | :--- |
| **Enter** | Switch active person | Toggles the active lane (Person 1 ⇄ Person 2). Disabled while recording. |
| **Spacebar** | Record (push-to-talk) | Hold to record the **active** person; release to transcribe & translate. |
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
| `landscape` | Active-person scheme (Enter / Space / ← →) — default |
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
