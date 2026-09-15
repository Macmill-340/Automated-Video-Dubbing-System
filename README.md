# Automated Video Dubbing System

Turn a YouTube video in any language into an English-dubbed MP4: same video,
new English voice-over. CLI prints progress per stage and logs timing.

```powershell
uv sync
uv run dub-video "https://www.youtube.com/watch?v=..."
```

No URL? Just run `uv run dub-video` and it prompts. Output defaults to
`dubbed_<timestamp>.mp4`; intermediates stay in `work/dub_<timestamp>/`.

## How it works

```text
YouTube URL → yt-dlp download → ffmpeg 16 kHz WAV → faster-whisper
(task=translate → English segments) → voice-match (F0/RMS → Edge
voice/rate/pitch/volume per segment) → edge-tts per-segment MP3s →
fit to windows (atempo) → assemble dub track → ffmpeg mux (-c:v copy)
```

Why these tools (researched, not just the assignment hints):

| Stage | Choice | Why |
|---|---|---|
| Download | `yt-dlp` | Still the reliable standard; best video+audio merged to MP4. |
| Transcribe + translate | `faster-whisper` (`small`, CPU int8, `task="translate"`) | One pass gives timestamped **English** directly — no separate MT model. CTranslate2 backend is ~4× faster than `openai-whisper` with far less RAM, and pulls **no torch**. |
| No IndicTrans2 | Deliberate | Would add torch + transformers + large MT weights for a gain this pipeline doesn't need; Whisper's built-in X→English covers the dub-to-English requirement. Revisit if translation quality becomes the bottleneck. |
| Voice | `edge-tts` (default `en-US-AriaNeural`) | Free, natural neural voices, zero local weights/GPU. Needs internet. |
| Voice match | `voice.py` (numpy F0 + RMS, no ML) | Gender from median pitch (165 Hz split), speakers split on pitch jumps, then the closest Edge voice + rate/pitch/volume nudge per segment. Same *kind* of voice, not a clone — cloning needs torch. |
| Timing | `atempo` + pure-Python `wave` assembly | Clips that fit sit at the window start; long clips speed up (≤1.35×); leftover overflow truncates at the next segment so one voice never overlaps itself. No numpy, no giant filter graphs. |
| Remix | system `ffmpeg` | `-c:v copy` (no video re-encode, fast on 2-hour inputs) + AAC audio. |

## Setup

- Python ≥ 3.13 via `uv` (`uv sync` installs everything, incl. dev tools).
- System `ffmpeg` **and** `ffprobe` on PATH (`ffmpeg -version`).
- Internet: YouTube, edge-tts service, and a one-time faster-whisper model
  download from Hugging Face (~500 MB for `small`).

## Usage

```powershell
uv run dub-video "<YouTube URL>" [--out dubbed.mp4] [--workdir work/my-dub] `
  [--model-size small] [--voice auto] [--rate +0%] [--concurrency 4] `
  [--batched] [--no-resume]
```

- `--model-size`: `tiny` (fastest) → `large-v3` (best). `small` is the sweet spot.
- `--voice`: `auto` matches the speaker (default); or name an `edge-tts --list-voices` voice, e.g. `en-US-ChristopherNeural`.
- `--rate`: e.g. `+10%` if dubs consistently overflow their windows.
- `--batched`: faster Whisper decoding on long videos (same model).
- `--format`: yt-dlp format selector. Best quality by default; for videos over
  ~30 min prefer `'bv*[height<=720]+ba/b'` — 4K lecture rips are tens of GB
  and get throttled.
- Resume is on by default: re-running with the same `--workdir` skips finished
  stages (`source.*`, `segments.json`, existing TTS clips). Each run writes
  `timing.json` (per-stage seconds) into the workdir — that is the "how long
  it took" record.
- `python -m video_dubber` works as a `dub-video` alias.

## Layout

```text
src/video_dubber/  cli.py  download.py  transcribe.py  synthesize.py
                   audio.py  remix.py  models.py  voice.py
tests/             real integrations (19 s YouTube video, tiny whisper model,
                   live edge-tts) + local ffmpeg checks + CLI wiring
```

## Evaluation dubs

- ~40 min: MIT 6.0001 Lecture 1, `work/eval_30m/` (`nykOeWgQcHM`, female lecturer)
- ~2 hr: CS50 2024 Lecture 0, `work/eval_2h/` (`3LPJfIKxwWc`, male lecturer)

Each folder keeps `source.mp4`, `dubbed.mp4`, and `timing.json`.
See `WALKTHROUGH.md` for the 2-minute video script.

## Tests

```powershell
uv run pytest                 # full suite, ~35 s, needs internet
uv run pytest -m "not e2e"    # offline subset (ffmpeg + logic only)
```

23 tests, none faked. The `e2e`-marked tests run the real thing once per
session against "Me at the zoo" (19 s, stable since 2005): real yt-dlp
download, real faster-whisper `tiny` transcription, real edge-tts voices,
and a full `run_pipeline` dub asserting the output keeps the source video
codec (copied, e.g. AV1) with AAC audio. First run also downloads the tiny
model weights (~75 MB) to the Hugging Face cache. Run the suite after every
stage change.

## Limits & next steps

- Single narrator voice; overlapping speech is serialized by window order.
- Stretch goals (not attempted): speaker diarization (`pyannote.audio`) and
  voice cloning (Coqui XTTS) — both heavy; do them only if dub quality demands it.
- Possible upgrades: `--task transcribe` + dedicated MT flag, word-level
  timestamps for tighter fits, per-speaker `--voice` mapping.
