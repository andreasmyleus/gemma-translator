# Edge STT fine-tunes for fi / es / fr

Note: one-off local script numbers — not from the paired A/B harness, and not a committed gate.

Direct faster-whisper comparison on Piper fixtures, 2026-08-13.
Same audio, `language=fi`, `beam_size=1`, int8 CPU.

## Finnish (`fi-*` fixtures)

| Fixture | stock `small` WER | `mpasila/faster-whisper-medium-finnish` WER |
|---|---:|---:|
| fi-short | 0.667 | 0.000 |
| fi-medium | 0.000 | 0.000 |
| fi-long | 0.222 | 0.056 |
| **corpus** | **0.214** | **0.036** |

Load OK for the CT2 medium fine-tune. Finnish-NLP ships CT2 only for large-v3 (skipped as default on Pi 5 / 8GB); no credible Hub small CT2 Finnish fine-tune found. Default: `STT_MODEL_FI=mpasila/faster-whisper-medium-finnish`.

## Spanish / French

No `es-*` / `fr-*` entries in `bench/fixtures.json`, so no WER comparison.

Candidates that **load** with `WhisperModel(..., device="cpu", compute_type="int8")` but were **not** wired as defaults:

- es: `EtMmohammedHafsati/es__HiTZ__whisper-small-es__int8`, `Jarbas/faster-whisper-small-es-cv13`
- fr: `songzewu/bofenghuang-whisper-small-cv11-french-ct2`, `artkeep/faster-whisper-small-cv11-french`

Upstream transformers fine-tunes exist (HiTZ/zuazo Spanish small, bofenghuang French small) but converting them ourselves was out of scope. Leave es/fr on multilingual `small`; override via `STT_MODEL_ES` / `STT_MODEL_FR` if needed.
