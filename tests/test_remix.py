"""Remix stage tests with real ffmpeg/ffprobe on synthetic media."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from video_dubber import remix as remix_mod
from video_dubber.remix import remix
from conftest import make_wav, media_streams

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(ffmpeg is None or ffprobe is None, reason="needs ffmpeg + ffprobe")


@needs_ffmpeg
def test_remix_copies_video_and_encodes_audio(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        [ffmpeg, "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=128x128:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
        check=True, capture_output=True,
    )
    dub = make_wav(tmp_path / "dub.wav", seconds=2.0)
    out = tmp_path / "dubbed.mp4"

    assert remix(src, dub, out) == out
    assert out.stat().st_size > 0

    streams = media_streams(out)
    assert streams["video"]["codec_name"] == "h264", "video stream must be copied, not re-encoded"
    assert streams["audio"]["codec_name"] == "aac"


def test_missing_ffmpeg_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(remix_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        remix(tmp_path / "v.mp4", tmp_path / "d.wav", tmp_path / "o.mp4")


def test_ffmpeg_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(remix_mod.shutil, "which", lambda _name: "ffmpeg")

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="mux exploded")

    monkeypatch.setattr(remix_mod.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="mux exploded"):
        remix(tmp_path / "v.mp4", tmp_path / "d.wav", tmp_path / "o.mp4")
