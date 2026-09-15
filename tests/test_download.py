"""Download stage tests against a real 19 s YouTube video (one fetch per session)."""

from __future__ import annotations

import wave

import pytest

from video_dubber import download as download_mod
from video_dubber.download import DownloadResult


@pytest.mark.e2e
def test_real_download_produces_video_and_wav(zoo_download):
    result, messages = zoo_download

    assert isinstance(result, DownloadResult)
    assert result.video_path.is_file() and result.video_path.stat().st_size > 0
    assert result.audio_path.is_file() and result.audio_path.stat().st_size > 0
    assert "zoo" in result.title.lower(), f"unexpected video: {result.title!r}"
    assert messages, "expected progress callbacks during the real download"


@pytest.mark.e2e
def test_extracted_audio_is_16khz_mono_pcm(zoo_download):
    result, _ = zoo_download
    with wave.open(str(result.audio_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() > 16_000 * 10, "expected roughly the full 19 s clip"


def test_missing_ffmpeg_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(download_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        download_mod.download("https://www.youtube.com/watch?v=jNQXAC9IVRw", tmp_path)
