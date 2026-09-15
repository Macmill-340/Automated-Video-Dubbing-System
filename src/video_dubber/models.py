"""Shared data models for the dubbing pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """One timestamped English utterance.

    Produced by the transcribe stage (in English already, via Whisper's
    ``translate`` task) and consumed by synthesis (text) and alignment
    (start/end window).
    """

    start: float  # window start in seconds
    end: float  # window end in seconds
    text: str  # English text to speak

    @property
    def duration(self) -> float:
        """Window length in seconds; never negative."""
        return max(0.0, self.end - self.start)
