"""Command-line interface: ``dub-video <YouTube URL>`` end-to-end pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio import Placement, assemble_timeline, fit_clip, probe_duration
from .download import DownloadResult, download
from .models import Segment
from .remix import remix
from .synthesize import DEFAULT_VOICE, synthesize
from .transcribe import transcribe
from .voice import assign_voices


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
    parser.add_argument("--voice", default="auto",
                        help=f"'auto' matches the speaker, or an edge-tts voice (default: auto).")
    parser.add_argument("--rate", default="+0%", help="edge-tts speech rate (default: +0%%).")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel TTS requests (default: 4).")
    parser.add_argument("--format", default="bv*+ba/b",
                        help="yt-dlp format (default: best; e.g. 'bv*[height<=720]+ba/b' for long videos).")
    parser.add_argument("--batched", action="store_true", help="batched Whisper decoding (faster on long videos).")
    parser.add_argument("--no-resume", action="store_true", help="redo every stage even if intermediates exist.")
    return parser


def _progress(stage: str):
    return lambda msg: print(f"[{stage}] {msg}", flush=True)


def _timed(timings: dict, label: str, func, *args, **kwargs):
    started = time.perf_counter()
    result = func(*args, **kwargs)
    took = time.perf_counter() - started
    timings[label] = round(took, 1)
    print(f"[{label}] done in {took:.1f}s", flush=True)
    return result


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _save_segments(path: Path, segments: list) -> None:
    path.write_text(json.dumps([dataclasses.asdict(s) for s in segments], indent=1), encoding="utf-8")


def _load_segments(path: Path) -> list[Segment]:
    return [Segment(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def run_pipeline(
    url: str,
    workdir: str | Path,
    out: str | Path,
    model_size: str = "small",
    voice: str = "auto",
    rate: str = "+0%",
    concurrency: int = 4,
    batched: bool = False,
    resume: bool = True,
    format: str = "bv*+ba/b",
) -> PipelineResult:
    """Run download → transcribe → synthesize → align → remix; return the result."""
    started_total = time.perf_counter()
    timings: dict[str, float] = {}
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    video_path, audio_path = workdir / "source.mp4", workdir / "source.wav"
    if resume and video_path.is_file() and audio_path.is_file():
        title = url
        meta = workdir / "meta.json"
        if meta.is_file():
            title = json.loads(meta.read_text(encoding="utf-8")).get("title", url)
        print("[download] reusing existing source files (resume)", flush=True)
        timings["download"] = 0.0
        downloaded = DownloadResult(video_path=video_path, audio_path=audio_path, title=title)
    else:
        downloaded = _timed(timings, "download", download, url, workdir, _progress("download"), format)
        (workdir / "meta.json").write_text(
            json.dumps({"title": downloaded.title, "url": url}), encoding="utf-8")
    print(f"[download] title: {downloaded.title}", flush=True)

    seg_file = workdir / "segments.json"
    if resume and seg_file.is_file():
        segments = _load_segments(seg_file)
        print(f"[transcribe] reusing {len(segments)} saved segments (resume)", flush=True)
        timings["transcribe"] = 0.0
        timings["voice-match"] = 0.0
        language = "resumed"
    else:
        segments, language = _timed(
            timings, "transcribe", transcribe, downloaded.audio_path,
            model_size=model_size, batched=batched, on_progress=_progress("transcribe"),
        )
        if voice == "auto":
            segments = _timed(timings, "voice-match", assign_voices,
                              segments, downloaded.audio_path, _progress("voice-match"))
        else:
            for seg in segments:
                seg.voice, seg.rate = voice, rate
        _save_segments(seg_file, segments)
    print(f"[transcribe] {len(segments)} English segments (source: {language})", flush=True)

    clips = _timed(
        timings, "synthesize", synthesize, segments, workdir / "tts",
        voice=voice if voice != "auto" else DEFAULT_VOICE, rate=rate,
        concurrency=concurrency, on_progress=_progress("synthesize"), skip_existing=resume,
    )

    total_duration = _timed(timings, "probe", probe_duration, downloaded.video_path)
    fit_dir = workdir / "fit"
    fit_dir.mkdir(exist_ok=True)
    placements: list[Placement] = []
    skipped = 0
    fit_started = time.perf_counter()
    for segment, clip in zip(segments, clips):
        if clip is None or segment.duration <= 0:
            skipped += 1
            continue
        fitted = fit_dir / f"{Path(clip).stem}.wav"
        if resume and fitted.is_file() and fitted.stat().st_mtime >= Path(clip).stat().st_mtime:
            placements.append(Placement(start=segment.start, clip=fitted))
            continue
        fit_clip(clip, fitted, segment.duration)
        placements.append(Placement(start=segment.start, clip=fitted))
    timings["fit"] = round(time.perf_counter() - fit_started, 1)
    print(f"[align] placed {len(placements)} clips, skipped {skipped}", flush=True)

    dub_wav = _timed(timings, "align", assemble_timeline, placements, total_duration, workdir / "dub.wav")
    final = _timed(timings, "remix", remix, downloaded.video_path, dub_wav, out)

    elapsed = time.perf_counter() - started_total
    timing_file = workdir / "timing.json"
    timing_file.write_text(json.dumps({
        "title": downloaded.title,
        "language": language,
        "segments": len(segments),
        "stages": timings,
        "total_seconds": round(elapsed, 1),
    }, indent=1), encoding="utf-8")
    print(f"[timing] wrote {timing_file}", flush=True)
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
        result = run_pipeline(url, workdir, out, args.model_size, args.voice,
                              args.rate, args.concurrency, args.batched,
                              not args.no_resume, args.format)
    except Exception as exc:  # fail loud but cleanly, with a usable message
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"dubbed video: {result.final_video} "
          f"({result.segments} segments, source lang {result.language}, "
          f"{result.elapsed:.1f}s total)", flush=True)
    return 0
