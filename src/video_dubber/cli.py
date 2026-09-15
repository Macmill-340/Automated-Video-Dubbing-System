"""Command-line interface: ``dub-video <YouTube URL>`` end-to-end pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio import Placement, assemble_timeline, fit_clip, probe_duration
from .download import download
from .remix import remix
from .synthesize import DEFAULT_VOICE, synthesize
from .transcribe import transcribe


@dataclass
class PipelineResult:
    final_video: Path
    language: str
    segments: int
    elapsed: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dub a YouTube video into English with the original video kept intact.",
    )
    parser.add_argument("url", nargs="?", help="YouTube URL (prompted if omitted).")
    parser.add_argument("--out", default=None, help="Output MP4 path (default: dubbed_<timestamp>.mp4).")
    parser.add_argument("--workdir", default=None, help="Directory for intermediates (default: work/dub_<timestamp>).")
    parser.add_argument("--model-size", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="faster-whisper model (default: small).")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"edge-tts voice (default: {DEFAULT_VOICE}).")
    parser.add_argument("--rate", default="+0%", help="edge-tts speech rate (default: +0%%).")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel TTS requests (default: 4).")
    return parser


def _progress(stage: str):
    return lambda msg: print(f"[{stage}] {msg}", flush=True)


def _timed(label: str, func, *args, **kwargs):
    started = time.perf_counter()
    result = func(*args, **kwargs)
    print(f"[{label}] done in {time.perf_counter() - started:.1f}s", flush=True)
    return result


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_pipeline(
    url: str,
    workdir: str | Path,
    out: str | Path,
    model_size: str = "small",
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    concurrency: int = 4,
) -> PipelineResult:
    """Run download → transcribe → synthesize → align → remix; return the result."""
    started_total = time.perf_counter()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    downloaded = _timed("download", download, url, workdir, _progress("download"))
    print(f"[download] title: {downloaded.title}", flush=True)

    segments, language = _timed(
        "transcribe", transcribe, downloaded.audio_path,
        model_size=model_size, on_progress=_progress("transcribe"),
    )
    print(f"[transcribe] {len(segments)} English segments (source: {language})", flush=True)

    clips = _timed(
        "synthesize", synthesize, segments, workdir / "tts",
        voice=voice, rate=rate, concurrency=concurrency, on_progress=_progress("synthesize"),
    )

    total_duration = _timed("probe", probe_duration, downloaded.video_path)
    fit_dir = workdir / "fit"
    fit_dir.mkdir(exist_ok=True)
    placements: list[Placement] = []
    skipped = 0
    for segment, clip in zip(segments, clips):
        if clip is None or segment.duration <= 0:
            skipped += 1
            continue
        fitted = fit_dir / f"{Path(clip).stem}.wav"
        _timed("fit", fit_clip, clip, fitted, segment.duration)
        placements.append(Placement(start=segment.start, clip=fitted))
    print(f"[align] placed {len(placements)} clips, skipped {skipped}", flush=True)

    dub_wav = _timed("align", assemble_timeline, placements, total_duration, workdir / "dub.wav")
    final = _timed("remix", remix, downloaded.video_path, dub_wav, out)

    elapsed = time.perf_counter() - started_total
    return PipelineResult(final_video=Path(final), language=language,
                          segments=len(segments), elapsed=elapsed)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    url = args.url or input("YouTube URL: ").strip()
    if not url:
        print("error: no URL provided", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else Path(f"dubbed_{_stamp()}.mp4")
    workdir = Path(args.workdir) if args.workdir else Path("work") / f"dub_{_stamp()}"
    try:
        result = run_pipeline(url, workdir, out, args.model_size, args.voice, args.rate, args.concurrency)
    except Exception as exc:  # fail loud but cleanly, with a usable message
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"dubbed video: {result.final_video} "
          f"({result.segments} segments, source lang {result.language}, "
          f"{result.elapsed:.1f}s total)", flush=True)
    return 0
