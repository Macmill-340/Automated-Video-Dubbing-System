"""Audio timing: fit TTS clips into segment windows and build the dub track.

Policy per segment window ``[start, end)``:

- Clip fits: placed at the window start, the rest stays silent.
- Clip too long: sped up with ffmpeg ``atempo`` up to ``MAX_SPEED``; any
  remaining overflow is hard-truncated at the next segment's start, so the
  single narrator voice never overlaps itself.

The timeline is assembled sample-accurately in pure Python (``wave`` module):
no giant ffmpeg filter graph and no numpy needed.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 44100
MAX_SPEED = 1.35


@dataclass
class Placement:
    """A fitted WAV clip pasted at *start* seconds into the dub timeline."""

    start: float
    clip: Path  # 44.1 kHz mono s16 WAV, as produced by fit_clip


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to process audio.")
    return ffmpeg


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds (ffprobe; WAV header fallback)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(completed.stdout.strip())
    if Path(path).suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    raise RuntimeError("ffprobe not found and duration fallback only supports WAV files.")


def fit_clip(
    src_mp3: str | Path,
    dst_wav: str | Path,
    target_duration: float,
    max_speed: float = MAX_SPEED,
) -> float:
    """Transcode *src_mp3* to 44.1 kHz mono WAV, speeding up to fit the window.

    Returns the fitted clip's duration in seconds.
    """
    ffmpeg = _require_ffmpeg()
    actual = probe_duration(src_mp3)
    speed = 1.0
    if target_duration > 0 and actual > target_duration:
        speed = min(actual / target_duration, max_speed)

    command = [ffmpeg, "-y", "-i", str(src_mp3)]
    if speed > 1.0:
        command += ["-filter:a", f"atempo={speed:.3f}"]
    command += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst_wav)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg clip fit failed: {exc.stderr}") from exc
    return probe_duration(dst_wav)


def _read_mono_s16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError(f"{path} must be mono 16-bit PCM (run it through fit_clip).")
        if wav.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"{path} must be {SAMPLE_RATE} Hz (run it through fit_clip).")
        return wav.readframes(wav.getnframes())


def assemble_timeline(
    placements: list[Placement],
    total_duration: float,
    out_path: str | Path,
) -> Path:
    """Paste clips into a silent PCM timeline; truncate each at the next start."""
    out_path = Path(out_path)
    total_frames = max(0, int(total_duration * SAMPLE_RATE))
    timeline = bytearray(total_frames * 2)

    ordered = sorted(placements, key=lambda p: p.start)
    limits = [p.start for p in ordered[1:]] + [total_duration]
    for place, limit in zip(ordered, limits):
        offset = int(max(0.0, place.start) * SAMPLE_RATE)
        if offset >= total_frames:
            continue
        room = int(max(0.0, min(limit, total_duration) - max(0.0, place.start)) * SAMPLE_RATE)
        raw = _read_mono_s16(place.clip)
        take = min(len(raw) // 2, room, total_frames - offset)
        if take <= 0:
            continue
        timeline[offset * 2 : (offset + take) * 2] = raw[: take * 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(timeline))
    return out_path
