"""Download stage: fetch the source video and extract a transcription-ready WAV.

Uses yt-dlp for the download (best video+audio, MP4 container) and the
system ffmpeg binary to decode a 16 kHz mono PCM WAV for transcription.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from yt_dlp import YoutubeDL

ProgressFn = Callable[[str], None]


@dataclass
class DownloadResult:
    video_path: Path
    audio_path: Path
    title: str


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to use the download stage.")
    return ffmpeg


def _progress_hook(notify: ProgressFn):
    def hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        downloaded = status.get("downloaded_bytes") or 0
        if total:
            notify(f"download {downloaded / total:.0%} ({downloaded}/{total} bytes)")

    return hook


def download(url: str, workdir: str | Path, on_progress: ProgressFn | None = None) -> DownloadResult:
    """Download *url* into *workdir*; return video path, extracted WAV path, title."""
    notify = on_progress or (lambda _msg: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _require_ffmpeg()

    notify("fetching video info")
    with YoutubeDL({"quiet": True, "no_warnings": True}) as info_dl:
        info = info_dl.extract_info(url, download=False)
    title = info.get("title") or "video"

    notify("downloading video")
    with YoutubeDL(
        {
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": str(workdir / "source.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [_progress_hook(notify)],
        }
    ) as video_dl:
        video_dl.download([url])

    video_path = next(
        (p for p in sorted(workdir.glob("source.*")) if p.suffix.lower() != ".wav"),
        None,
    )
    if video_path is None:
        raise RuntimeError(f"yt-dlp finished but no video file appeared in {workdir}")

    audio_path = workdir / "source.wav"
    notify("extracting 16 kHz mono audio")
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(video_path), "-vn",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg audio extraction failed: {exc.stderr}") from exc

    notify(f"ready: {video_path.name} + {audio_path.name}")
    return DownloadResult(video_path=video_path, audio_path=audio_path, title=title)
