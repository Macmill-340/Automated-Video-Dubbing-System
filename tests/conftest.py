"""Shared test helpers and real end-to-end fixtures.

The ``zoo_*`` fixtures hit the real network once per session: a 19-second
YouTube video ("Me at the zoo", stable since 2005), real faster-whisper
``tiny`` transcription, real edge-tts synthesis. Everything else uses the
local ffmpeg/ffprobe binaries and synthetic media.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

ZOO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def make_wav(path: Path, seconds: float, freq: float = 440.0, rate: int = 44100) -> Path:
    """Write a mono 16-bit PCM sine WAV; returns *path*."""
    frames = int(seconds * rate)
    data = struct.pack(
        f"<{frames}h",
        *(int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(frames)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data)
    return path


def media_streams(path: Path) -> dict[str, dict]:
    """ffprobe stream map (``{"video": ..., "audio": ...}``) for a media file."""
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None, "ffprobe required"
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return {s["codec_type"]: s for s in json.loads(completed.stdout)["streams"]}


@pytest.fixture(scope="session")
def zoo_download(tmp_path_factory):
    """Real yt-dlp download of the 19 s test video + extracted WAV."""
    from video_dubber.download import download

    work = tmp_path_factory.mktemp("zoo") / "dl"
    messages: list[str] = []
    result = download(ZOO_URL, work, on_progress=messages.append)
    return result, messages


@pytest.fixture(scope="session")
def zoo_transcript(zoo_download):
    """Real faster-whisper ``tiny`` transcription of the test video audio."""
    from video_dubber.transcribe import transcribe

    result, _ = zoo_download
    return transcribe(result.audio_path, model_size="tiny")
