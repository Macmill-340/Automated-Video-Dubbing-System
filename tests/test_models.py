"""Tests for shared data models."""

from video_dubber.models import Segment


def test_duration_is_end_minus_start():
    assert Segment(1.0, 3.5, "hi").duration == 3.5 - 1.0


def test_duration_never_negative():
    assert Segment(5.0, 2.0, "backwards").duration == 0.0
    assert Segment(2.0, 2.0, "empty").duration == 0.0
