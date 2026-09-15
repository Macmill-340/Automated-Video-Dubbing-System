"""Voice matching tests on tones with known pitch (no ML, no network)."""

from __future__ import annotations

import pytest

from video_dubber import voice as voice_mod
from video_dubber.models import Segment
from video_dubber.voice import (
    assign_voices,
    cluster_speakers,
    edge_pitch,
    edge_rate,
    edge_volume,
    gender_of,
    loudness,
    median_f0,
    read_slice,
    voice_for,
)
from conftest import make_wav


def test_low_tone_reads_male(tmp_path):
    samples, rate = read_slice(make_wav(tmp_path / "low.wav", 1.0, freq=110.0), 0.0, 1.0)
    pitch = median_f0(samples, rate)
    assert pitch == pytest.approx(110.0, rel=0.05)
    assert gender_of(pitch) == "male"


def test_high_tone_reads_female(tmp_path):
    samples, rate = read_slice(make_wav(tmp_path / "high.wav", 1.0, freq=220.0), 0.0, 1.0)
    pitch = median_f0(samples, rate)
    assert pitch == pytest.approx(220.0, rel=0.05)
    assert gender_of(pitch) == "female"


def test_silence_has_no_pitch(tmp_path):
    samples, rate = read_slice(make_wav(tmp_path / "s.wav", 0.5, freq=440.0), 5.0, 6.0)
    assert samples.size == 0
    assert loudness(samples) == 0.0
    assert median_f0(samples, rate) is None
    assert gender_of(None) == "female"


def test_pitch_jump_starts_new_speaker():
    assert cluster_speakers([100.0, 105.0, 220.0, 225.0, 102.0]) == [0, 0, 1, 1, 2]
    assert cluster_speakers([None, 110.0]) == [0, 0]


def test_voices_come_from_the_pools():
    assert voice_for("female", 0) == "en-US-AriaNeural"
    assert voice_for("male", 0) == "en-US-ChristopherNeural"
    assert voice_for("female", 1) != voice_for("female", 0)
    assert voice_for("male", 1) != voice_for("male", 0)


def test_edge_param_shapes():
    assert edge_rate("hi", 5.0) == "+0%"  # fits already
    assert edge_rate("x" * 150, 5.0) == "+35%"  # 10 s of text in 5 s, capped
    assert edge_pitch(None, "male") == "+0Hz"
    assert edge_pitch(150.0, "male") == "+30Hz"
    assert edge_pitch(400.0, "female") == "+40Hz"  # capped
    assert edge_volume(0.0, 0.1) == "+0%"
    assert edge_volume(0.2, 0.1) == "+12%"  # twice as loud as the median


def test_assign_voices_end_to_end(tmp_path):
    import wave

    low = make_wav(tmp_path / "low.wav", 1.0, freq=110.0)
    high = make_wav(tmp_path / "high.wav", 1.0, freq=220.0)
    both = tmp_path / "both.wav"
    with wave.open(str(low), "rb") as a, wave.open(str(high), "rb") as b:
        params = a.getparams()
        frames = a.readframes(a.getnframes()) + b.readframes(b.getnframes())
    with wave.open(str(both), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)

    segments = [Segment(0.0, 1.0, "low part"), Segment(1.0, 2.0, "high part")]
    assign_voices(segments, both)

    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert segments[0].voice == "en-US-ChristopherNeural"
    assert segments[1].voice == "en-US-JennyNeural"  # second speaker gets the next female voice
    assert all(s.rate and s.pitch and s.volume for s in segments)


def test_assign_voices_imports_cleanly():
    assert voice_mod.GENDER_CUTOFF_HZ == 165.0
