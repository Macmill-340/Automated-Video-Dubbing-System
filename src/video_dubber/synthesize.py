"""Synthesis stage: natural English voice-over per segment via edge-tts.

edge-tts drives Microsoft Edge's free online neural voices, so there are no
local model weights and no GPU requirement. Segments are synthesized
concurrently (bounded) and saved as one MP3 per segment for the align stage.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import edge_tts

from .models import Segment

ProgressFn = Callable[[str], None]

DEFAULT_VOICE = "en-US-AriaNeural"


async def _synthesize_one(text: str, voice: str, rate: str, volume: str, pitch: str, out_path: Path) -> None:
    await edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch).save(str(out_path))


async def _synthesize_all(
    jobs: list[tuple[Segment, Path]],
    voice: str,
    rate: str,
    concurrency: int,
    notify: ProgressFn,
    skip_existing: bool,
) -> list[Path | None]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[Path | None] = [None] * len(jobs)

    async def worker(index: int, segment: Segment, path: Path) -> None:
        async with semaphore:
            if not segment.text.strip():
                return
            if skip_existing and path.is_file() and path.stat().st_size > 0:
                notify(f"keeping {index + 1}/{len(jobs)} (already synthesized)")
                results[index] = path
                return
            notify(f"synthesizing {index + 1}/{len(jobs)}")
            await _synthesize_one(
                segment.text,
                segment.voice or voice,
                segment.rate or rate,
                segment.volume or "+0%",
                segment.pitch or "+0Hz",
                path,
            )
            results[index] = path

    await asyncio.gather(*(worker(i, seg, path) for i, (seg, path) in enumerate(jobs)))
    return results


def synthesize(
    segments: list[Segment],
    out_dir: str | Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    concurrency: int = 4,
    on_progress: ProgressFn | None = None,
    skip_existing: bool = False,
) -> list[Path | None]:
    """Synthesize each segment to MP3; empty texts yield None (skipped downstream)."""
    notify = on_progress or (lambda _msg: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(seg, out_dir / f"seg_{i:04d}.mp3") for i, seg in enumerate(segments)]
    notify(f"synthesizing {len(jobs)} clips with voice '{voice}'")
    clips = asyncio.run(_synthesize_all(jobs, voice, rate, concurrency, notify, skip_existing))
    done = sum(1 for c in clips if c is not None)
    notify(f"synthesized {done}/{len(jobs)} clips")
    return clips
