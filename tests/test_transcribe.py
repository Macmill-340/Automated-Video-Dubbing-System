"""Transcribe stage tests with the real faster-whisper ``tiny`` model (once per session)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def test_real_transcribe_yields_english_segments(zoo_transcript):
    segments, language = zoo_transcript

    assert language == "en"
    assert len(segments) >= 1, "the 19 s clip has speech; expected segments"
    for segment in segments:
        assert segment.text.strip(), "blank segments must be dropped"
        assert segment.end > segment.start


def test_segments_cover_speech_not_silence(zoo_transcript):
    segments, _ = zoo_transcript
    spoken = sum(s.duration for s in segments)
    assert spoken > 5.0, f"expected most of the 19 s clip to be speech, got {spoken:.1f}s"


def test_batched_path_runs(tmp_path):
    from conftest import make_wav
    from video_dubber.transcribe import transcribe

    wav = make_wav(tmp_path / "tone.wav", 2.0, rate=16000)
    segments, language = transcribe(wav, model_size="tiny", batched=True)
    assert isinstance(segments, list)
    assert isinstance(language, str)
