"""Audio timing tests: real ffprobe/ffmpeg probing and fitting, real timeline assembly."""

from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from video_dubber.audio import (
    MAX_SPEED,
    SAMPLE_RATE,
    Placement,
    assemble_timeline,
    fit_clip,
    probe_duration,
)
from conftest import make_wav


def _make_mp3(path: Path, seconds: float) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "ffmpeg required"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "libmp3lame", str(path)],
        check=True, capture_output=True,
    )
    return path


def test_probe_measures_real_mp3_duration(tmp_path):
    mp3 = _make_mp3(tmp_path / "tone.mp3", 4.0)
    assert probe_duration(mp3) == pytest.approx(4.0, abs=0.15)


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        probe_duration(tmp_path / "nope.mp3")


def _wav_props(path: Path) -> tuple:
    with wave.open(str(path), "rb") as wav:
        return wav.getnchannels(), wav.getsampwidth(), wav.getframerate()


def test_fit_clip_passthrough_keeps_duration(tmp_path):
    mp3 = _make_mp3(tmp_path / "s.mp3", 2.0)
    fitted = tmp_path / "f.wav"

    assert fit_clip(mp3, fitted, target_duration=5.0) == pytest.approx(2.0, abs=0.15)
    assert _wav_props(fitted) == (1, 2, SAMPLE_RATE)


def test_fit_clip_speeds_up_and_caps(tmp_path):
    mp3 = _make_mp3(tmp_path / "s.mp3", 4.0)
    fitted = tmp_path / "f.wav"

    duration = fit_clip(mp3, fitted, target_duration=2.0)

    # 4 s squeezed into a 2 s window would need 2x; the cap holds it at 1.35x.
    assert duration == pytest.approx(4.0 / MAX_SPEED, abs=0.2)
    assert _wav_props(fitted) == (1, 2, SAMPLE_RATE)


def test_fit_clip_failure_raises(tmp_path, monkeypatch):
    import video_dubber.audio as audio_mod

    monkeypatch.setattr(audio_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        fit_clip(tmp_path / "s.mp3", tmp_path / "f.wav", 1.0)


def _read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 44100)
        return wav.readframes(wav.getnframes())


def test_assemble_places_clips_with_silence(tmp_path):
    clip_a = make_wav(tmp_path / "a.wav", seconds=1.0, freq=440.0)
    clip_b = make_wav(tmp_path / "b.wav", seconds=0.5, freq=880.0)
    out = tmp_path / "dub.wav"

    # Reversed input order proves placements are sorted by start.
    assemble_timeline([Placement(2.0, clip_b), Placement(0.0, clip_a)], 3.0, out)

    raw = _read_frames(out)
    assert len(raw) == 3 * 44100 * 2
    assert raw[: 44100 * 2] == _read_frames(clip_a)
    assert raw[44100 * 2 : 2 * 44100 * 2] == b"\x00" * 44100 * 2
    assert raw[2 * 44100 * 2 : int(2.5 * 44100) * 2] == _read_frames(clip_b)
    assert raw[int(2.5 * 44100) * 2 :] == b"\x00" * int(0.5 * 44100) * 2


def test_assemble_truncates_at_next_start(tmp_path):
    long_clip = make_wav(tmp_path / "long.wav", seconds=2.0, freq=440.0)
    short_clip = make_wav(tmp_path / "short.wav", seconds=0.25, freq=880.0)
    out = tmp_path / "dub.wav"

    assemble_timeline([Placement(0.0, long_clip), Placement(1.0, short_clip)], 2.0, out)

    raw = _read_frames(out)
    assert raw[: 44100 * 2] == _read_frames(long_clip)[: 44100 * 2]
    assert raw[44100 * 2 : int(1.25 * 44100) * 2] == _read_frames(short_clip)
    assert raw[int(1.25 * 44100) * 2 :] == b"\x00" * (len(raw) - int(1.25 * 44100) * 2)


def test_assemble_rejects_non_mono_s16(tmp_path):
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(struct.pack("<4h", 0, 0, 0, 0))
    with pytest.raises(RuntimeError, match="mono 16-bit"):
        assemble_timeline([Placement(0.0, stereo)], 1.0, tmp_path / "dub.wav")
