# Walkthrough script (2 minutes)

Record your screen with the repo open; talk through this. Aim for ~120 seconds.

## 0:00–0:20 — The dubs
- Play 10 s of the MIT lecture source, then 10 s of our dubbed output.
- State the scoreboard: 43-min video dubbed in ~13 min
  (see `work/eval_30m/timing.json`), 2-hour CS50 in ~41 min
  (`work/eval_2h/timing.json`). Same video, English voice.

## 0:20–0:50 — Pipeline (show README diagram)
- `yt-dlp` download → `faster-whisper` translate → `edge-tts` → ffmpeg mux.
- One key decision: Whisper's `translate` task does transcription + translation
  in a single pass, so there is no separate MT model, no torch, no transformers.

## 0:50–1:20 — Same voice, same energy (show `voice.py`)
- No cloning: we measure median pitch and loudness per segment with plain numpy.
- Gender from the 165 Hz pitch split, speakers split on pitch jumps, then the
  closest Edge neural voice plus rate/pitch/volume nudges.
- Why not XTTS/pyannote: torch + GB weights, fights Python 3.13, too slow on CPU
  for 2-hour videos. Same *kind* of voice instead of a clone.

## 1:20–1:50 — Timing that survives long videos
- Clips that fit sit at the window start; long ones speed up (≤1.35×); overflow
  truncates at the next segment — one voice never overlaps itself.
- Video stream copied (`-c:v copy`), never re-encoded; resume + `timing.json`
  so a 2-hour run survives interruptions.

## 1:50–2:00 — Close
- 35 real tests, no mocks on the stages; `uv run pytest`.
- Honest limits: pitch clustering found 6 "speakers" on CS50 (applause counts);
  overlapping speech is serialized. Depth lives in `ARCHITECTURE.md`.
