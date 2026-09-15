# AGENTS.md

Greenfield Python project — no source, tests, lint, or CI yet. Spec is `automated-video-dubbing-assignment.md` (source of truth for requirements).

## Stack
- Python `>=3.13` only (`pyproject.toml`). Env manager is `uv`; `.venv/` already exists (CPython 3.13.13).
- `pyproject.toml` has zero `dependencies` — add each pipeline dep there when you introduce it.
- `ffmpeg` is a system binary, not a pip package. Verify with `ffmpeg -version` before touching remix code; fail loud if missing.

## Target pipeline (per assignment spec)
`yt-dlp` download → Whisper transcribe → translate (e.g. IndicTrans2 for Indian languages) → `edge-tts` synthesize → `ffmpeg` mux new audio without re-encoding video.
- CLI: accept YouTube URL as argv or prompt; print progress to terminal; save dubbed video to disk.
- Stretch only (diarization via `pyannote.audio`, cloning via Coqui XTTS): attempt after core pipeline works.
- Judged on dub accuracy (translation, natural voice, timing) + code clarity.

## Gotchas
- Windows + PowerShell 5.1 here: quote paths with spaces, use `; if ($?) { ... }` for dependent commands, never `&&`/`cd` — use `workdir` param.
- Long media runs (30-min and 2-hr evaluation videos): keep intermediates on disk, stream/copy video (`-c:v copy`) instead of re-encoding, and log timing per stage.
- No test/lint config exists — do not invent `pytest`/`ruff` commands; verify with a short real video run instead.
