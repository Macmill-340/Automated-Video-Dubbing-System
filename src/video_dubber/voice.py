"""Voice matching: pick an Edge voice per segment from the source audio itself.

No ML models here — just pitch (F0) and loudness (RMS) measured on the
16 kHz transcription WAV with numpy, which is already installed:

- gender from median F0 (below 165 Hz reads male, above reads female)
- speakers split when F0 jumps between consecutive segments
- Edge voice/rate/pitch/volume nudged toward the original speaker

Same *kind* of voice (gender + pitch + energy), not a clone. Cloning would
need torch and GB-scale weights; this needs nothing new.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable

import numpy as np

ProgressFn = Callable[[str], None]

GENDER_CUTOFF_HZ = 165.0  # median F0 below this reads male
SPEAKER_JUMP_HZ = 40.0  # bigger F0 jump between segments starts a new speaker
MAX_SPEAKERS = 6
MALE_BASELINE_HZ = 120.0
FEMALE_BASELINE_HZ = 210.0
NOMINAL_CHARS_PER_SEC = 15.0  # typical Edge English pace, for the rate guess

FEMALE_VOICES = ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-MichelleNeural"]
MALE_VOICES = ["en-US-ChristopherNeural", "en-US-GuyNeural", "en-US-EricNeural"]


def read_slice(wav_path: str | Path, start: float, end: float) -> tuple[np.ndarray, int]:
    """Read WAV seconds [start, end) as float samples in [-1, 1]; return samples + rate."""
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        first = int(max(0.0, start) * rate)
        count = max(0, int(end * rate) - first)
        wav.setpos(min(first, wav.getnframes()))
        raw = wav.readframes(count)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def loudness(samples: np.ndarray) -> float:
    """RMS energy of samples; 0 for silence."""
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def median_f0(samples: np.ndarray, rate: int) -> float | None:
    """Median pitch in Hz via autocorrelation; None when nothing voiced is found."""
    frame = int(0.03 * rate)  # 30 ms frames
    hop = int(0.01 * rate)  # 10 ms apart
    if samples.size < frame:
        return None
    lowest_lag = max(1, rate // 500)  # ignore pitches above 500 Hz
    highest_lag = rate // 50  # ignore pitches below 50 Hz
    slice_loud = loudness(samples)
    found: list[float] = []
    for at in range(0, samples.size - frame, hop):
        piece = samples[at : at + frame].copy()
        piece -= piece.mean()
        if loudness(piece) < max(0.02, 0.25 * slice_loud):
            continue  # quiet/unvoiced frame
        corr = np.correlate(piece, piece, mode="full")[frame - 1 :]
        corr[:lowest_lag] = 0.0
        lag = int(np.argmax(corr[lowest_lag:highest_lag])) + lowest_lag
        if corr[lag] > 0:
            found.append(rate / lag)
    return float(np.median(found)) if found else None


def gender_of(pitch_hz: float | None) -> str:
    """'male' below the cutoff, else 'female' (unknown pitch keeps the default voice)."""
    return "male" if pitch_hz is not None and pitch_hz < GENDER_CUTOFF_HZ else "female"


def cluster_speakers(pitches: list[float | None]) -> list[int]:
    """Group consecutive segments into speakers; a big F0 jump starts a new one."""
    labels: list[int] = []
    typical: list[list[float]] = []  # median-ish F0 history per speaker
    for pitch in pitches:
        if not labels:
            labels.append(0)
            typical.append([pitch] if pitch else [])
            continue
        current = labels[-1]
        ref = typical[current]
        center = float(np.median(ref)) if ref else None
        new_speaker = (
            pitch is not None
            and center is not None
            and abs(pitch - center) > SPEAKER_JUMP_HZ
            and current < MAX_SPEAKERS - 1
        )
        if new_speaker:
            labels.append(len(typical))
            typical.append([pitch])
        else:
            labels.append(current)
            if pitch:
                typical[current].append(pitch)
    return labels


def voice_for(gender: str, speaker_index: int) -> str:
    """Edge voice for a gender, rotating the pool so speakers sound distinct."""
    pool = MALE_VOICES if gender == "male" else FEMALE_VOICES
    return pool[speaker_index % len(pool)]


def edge_rate(text: str, window: float) -> str:
    """Edge rate guess so the clip lands near its window; '+0%' when it already fits."""
    expected = len(text) / NOMINAL_CHARS_PER_SEC
    if window > 0 and expected > window:
        return f"+{min(100.0 * (expected / window - 1.0), 35.0):.0f}%"
    return "+0%"


def edge_pitch(pitch_hz: float | None, gender: str) -> str:
    """Edge pitch shift toward the speaker's F0, capped so it stays natural."""
    if pitch_hz is None:
        return "+0Hz"
    baseline = MALE_BASELINE_HZ if gender == "male" else FEMALE_BASELINE_HZ
    shift = max(-40.0, min(40.0, pitch_hz - baseline))
    return f"{shift:+.0f}Hz"


def edge_volume(seg_loud: float, median_loud: float) -> str:
    """Edge volume from segment loudness vs the video median, capped at ±20%."""
    if seg_loud <= 0 or median_loud <= 0:
        return "+0%"
    change = max(-20.0, min(20.0, 2.0 * 20.0 * np.log10(seg_loud / median_loud)))
    return f"{change:+.0f}%"


def assign_voices(segments, wav_path: str | Path, on_progress: ProgressFn | None = None):
    """Fill speaker/voice/rate/pitch/volume on each segment, in place."""
    notify = on_progress or (lambda _msg: None)
    pitches: list[float | None] = []
    levels: list[float] = []
    for i, seg in enumerate(segments):
        samples, rate = read_slice(wav_path, seg.start, seg.end)
        pitches.append(median_f0(samples, rate))
        levels.append(loudness(samples))
        notify(f"voice-match {i + 1}/{len(segments)}")
    mid_level = float(np.median(levels)) if levels else 0.0
    labels = cluster_speakers(pitches)
    for seg, pitch, level, label in zip(segments, pitches, levels, labels):
        gender = gender_of(pitch)
        seg.speaker = f"SPEAKER_{label:02d}"
        seg.voice = voice_for(gender, label)
        seg.rate = edge_rate(seg.text, seg.duration)
        seg.pitch = edge_pitch(pitch, gender)
        seg.volume = edge_volume(level, mid_level)
    speakers = sorted({s.speaker for s in segments})
    notify(f"matched {len(speakers)} speaker(s): {', '.join(speakers)}")
    return segments
