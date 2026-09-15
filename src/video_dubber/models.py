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
    speaker: str = ""  # e.g. "SPEAKER_00" (empty until clustered)
    voice: str = ""  # edge-tts voice for this segment (empty = CLI default)
    rate: str = ""  # edge-tts rate like "+10%" (empty = CLI default)
    pitch: str = ""  # edge-tts pitch like "-5Hz" (empty = CLI default)
    volume: str = ""  # edge-tts volume like "+8%" (empty = CLI default)

    @property
    def duration(self) -> float:
        """Window length in seconds; never negative."""
        return max(0.0, self.end - self.start)
