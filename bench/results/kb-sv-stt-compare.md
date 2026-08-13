# KB-Whisper vs stock Whisper (Swedish STT only)

Note: one-off local script numbers — not from the paired A/B harness, and not a committed gate.

Direct faster-whisper comparison on Piper fixtures, 2026-08-13.
Same audio, `language=sv`, `beam_size=1`, int8 CPU.

| Fixture | stock WER | KB WER |
|---|---:|---:|
| sv-short | 0.667 | 0.667 |
| sv-medium | 0.091 | 0.091 |
| sv-long | 0.071 | 0.036 |
| sv-multi | 0.080 | 0.040 |
| **corpus** | **0.104** | **0.075** |

KB wins on long/multi; short still fails "stationen" on both (Piper voice + small capacity).
Latency similar (~1.2–2.0 s). Default: `STT_MODEL_SV=KBLab/kb-whisper-small`.
