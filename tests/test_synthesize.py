"""Synthesis stage tests with the real edge-tts service."""

from __future__ import annotations

import pytest

from video_dubber.models import Segment
from video_dubber.synthesize import DEFAULT_VOICE, synthesize
from video_dubber.audio import probe_duration
from video_dubber import synthesize as synth_mod

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


@pytest.mark.e2e
def test_per_segment_voice_beats_default(tmp_path):
    segments = [
        Segment(0.0, 2.0, "Hello", voice="en-US-ChristopherNeural"),
        Segment(2.0, 4.0, "Hi there", voice="en-US-AriaNeural"),
    ]
    clips = synthesize(segments, tmp_path, voice="en-US-JennyNeural")
    assert all(c is not None and c.stat().st_size > 0 for c in clips)


def test_skip_existing_never_hits_network(tmp_path, monkeypatch):
    target = tmp_path / "seg_0000.mp3"
    target.write_bytes(b"marker")

    def _boom(*args, **kwargs):
        raise AssertionError("cached clip must not be re-synthesized")

    monkeypatch.setattr(synth_mod.edge_tts, "Communicate", _boom)
    clips = synthesize([Segment(0.0, 1.0, "Hello")], tmp_path, skip_existing=True)
    assert clips == [target]
    assert target.read_bytes() == b"marker"
