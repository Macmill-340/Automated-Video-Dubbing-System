# AGENTS.md

Pipeline CLI (`dub-video`): YouTube → English-dubbed MP4. Spec is `automated-video-dubbing-assignment.md`.

## Stack
- Python `>=3.13` via `uv`; run everything with `uv run` (venv already synced).
- Deps in `pyproject.toml`: `yt-dlp`, `faster-whisper` (CTranslate2, **no torch**), `edge-tts`, `numpy` (F0/RMS in `voice.py`). Do NOT add transformers/IndicTrans2/torch TTS — Whisper `task="translate"` does X→English in one pass (see README).
- `ffmpeg` + `ffprobe` are system binaries, not pip packages. Fail loud if missing.
- Layout: `src/video_dubber/` (`cli download transcribe synthesize audio remix models voice`), `tests/` (one file per stage).

## Commands
- `uv run dub-video "<url>"` — end-to-end; `uv run pytest` — full suite (35 tests, ~40 s, needs internet; run after every stage change).
- `uv run pytest -m "not e2e"` for the offline subset (ffmpeg + logic only).
- `uv sync` after touching deps. `uv run pytest tests/test_<stage>.py` for a focused check.

## Gotchas
- Windows + PowerShell 5.1 here: quote paths with spaces, use `; if ($?) { ... }` for dependent commands, never `&&`/`cd` — use `workdir` param.
- Long media runs (30-min and 2-hr evaluation videos): intermediates stay in `work/dub_<ts>/` (gitignored), video stream copied (`-c:v copy`), timing logged per stage.
- Timing policy: clip fits → window start; too long → `atempo` speedup (cap 1.35×); overflow → truncate at next segment start, never overlap.
- Tests are real (no stage mocks): `e2e` tests share one session download of the 19 s "Me at the zoo" video + one `tiny`-model transcript; source video codec varies (AV1) so e2e asserts copy by comparing to source, not `h264`.
