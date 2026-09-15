"""Synthesis stage tests with the real edge-tts service."""

from __future__ import annotations

import pytest

from video_dubber.models import Segment
from video_dubber.synthesize import DEFAULT_VOICE, synthesize
from video_dubber.audio import probe_duration

pytestmark = pytest.mark.e2e


def test_real_synthesize_clips_in_order(tmp_path):
    segments = [
        Segment(0.0, 3.0, "Hello world, this is a dubbing test."),
        Segment(3.0, 4.0, "   "),  # blank -> skipped without a network call
        Segment(4.0, 7.0, "Second line for the timeline."),
    ]
    seen: list[str] = []
    clips = synthesize(segments, tmp_path, voice="en-US-AriaNeural", rate="+0%",
                       concurrency=2, on_progress=seen.append)

    assert clips[0] is not None and clips[0].stat().st_size > 0
    assert clips[1] is None
    assert clips[2] is not None and clips[2].stat().st_size > 0
    assert clips[0].name == "seg_0000.mp3" and clips[2].name == "seg_0002.mp3"
    assert probe_duration(clips[0]) > 0.5, "TTS clip should hold real speech audio"
    assert seen, "expected progress callbacks"


def test_default_voice_is_explicit(tmp_path):
    assert DEFAULT_VOICE == "en-US-AriaNeural"
    clips = synthesize([Segment(0.0, 2.0, "Hi")], tmp_path)
    assert clips[0] is not None and clips[0].stat().st_size > 0
