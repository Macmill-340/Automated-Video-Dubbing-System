"""Transcribe + translate stage.

A single faster-whisper pass with ``task="translate"`` turns speech in any
supported language straight into timestamped English text. That removes the
need for a separate machine-translation model (no torch, no transformers,
no IndicTrans2) while keeping every utterance aligned to its audio window.

Trade-off: Whisper's built-in translation is solid but less polished than a
dedicated MT model; a ``transcribe`` + external-MT path can be added later
behind a flag if dub accuracy needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from faster_whisper import BatchedInferencePipeline, WhisperModel

from .models import Segment

ProgressFn = Callable[[str], None]


def transcribe(
    audio_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    vad_filter: bool = True,
    batched: bool = False,
    on_progress: ProgressFn | None = None,
) -> tuple[list[Segment], str]:
    """Transcribe *audio_path* and translate to English; return segments + language code."""
    notify = on_progress or (lambda _msg: None)

    notify(f"loading faster-whisper model '{model_size}' ({device}/{compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    notify("transcribing + translating to English")
    options = {
        "task": "translate",
        "vad_filter": vad_filter,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if batched:  # faster on long audio: same model, segments decoded in parallel
        pipeline = BatchedInferencePipeline(model=model)
        raw_segments, info = pipeline.transcribe(str(audio_path), batch_size=8, **options)
    else:
        raw_segments, info = model.transcribe(str(audio_path), **options)
    segments = [
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in raw_segments
        if s.text and s.text.strip()
    ]
    language = getattr(info, "language", None) or "unknown"
    notify(f"detected language: {language}; {len(segments)} segments")
    return segments, language
