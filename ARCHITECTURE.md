# Architecture

YouTube URL in any language → English-dubbed MP4. Same pictures, new English
speech timed to the original windows.

## Why the stack is tiny

Four pip packages: `yt-dlp`, `faster-whisper`, `edge-tts`, `numpy`. System
`ffmpeg`/`ffprobe`. **No torch, transformers, IndicTrans2, pyannote, or XTTS.**
Those fight Python 3.13, add gigabytes of weights, and would make a 2-hour CPU
run impractical. Whisper already translates to English; Edge already ships
natural voices; pitch and loudness need only numpy, which faster-whisper
pulls in anyway.

## Pipeline

```text
URL
 → download.py     source.mp4 + source.wav (16 kHz mono)
 → transcribe.py   list[Segment]: English text + start/end
 → voice.py        fill speaker, voice, rate, pitch, volume
 → synthesize.py   tts/seg_NNNN.mp3
 → audio.py        fit with atempo → dub.wav
 → remix.py        dubbed.mp4 (video copied, AAC audio)
```

`cli.py` is the only orchestrator: progress lines, per-stage seconds, resume,
and `timing.json`. One module per stage, one test file per stage.

## The data object

`Segment` (`models.py`) carries an utterance through the pipeline:

- `start`, `end`, `text` — filled by transcribe, already English.
- `speaker`, `voice`, `rate`, `pitch`, `volume` — filled by voice-match,
  consumed by synthesize. Empty means "use the CLI default".

## Download

`yt-dlp` with `bv*+ba/b` merged to MP4: best video plus best audio. ffmpeg
then peels a **16 kHz mono PCM WAV** — Whisper's native rate, so nothing is
wasted on stereo or 48 kHz. For videos over ~30 minutes use
`--format 'bv*[height<=720]+ba/b'`: a 2-hour lecture at best quality was
~18 GB and YouTube throttled it mid-download (HTTP 503).

## Transcribe + translate

faster-whisper (CTranslate2 backend, CPU int8) with `task="translate"`: one
pass turns speech in any supported language into timestamped **English**.
No separate translation model. VAD drops silence longer than half a second.
`--batched` switches to `BatchedInferencePipeline` — the same model decoding
segments in parallel — for the 30-minute and 2-hour runs.

## Voice match

No ML. Each Whisper window is sliced out of `source.wav` and measured:

- **Pitch (F0):** autocorrelation over 30 ms frames, median in Hz.
- **Gender:** below 165 Hz reads male, above reads female.
- **Speakers:** a pitch jump over 40 Hz between consecutive segments starts
  `SPEAKER_01`, and so on, capped at 6. Distinct speakers get distinct voices:
  Aria/Jenny/Michelle for female, Christopher/Guy/Eric for male.
- **Rate:** if `len(text) / 15` chars per second would overflow the window,
  Edge gets `+N%` up to 35%.
- **Pitch:** speaker F0 versus 120 Hz (male) / 210 Hz (female), clamped ±40 Hz.
- **Volume:** segment RMS versus the video median, clamped ±20%.

This is the same *kind* of voice — gender, pitch, energy — not a clone.
Cloning would mean torch and a voice model per speaker.

## Synthesize

Edge's free online neural TTS, up to 4 requests in flight, one MP3 per
segment. Blank text is skipped. Transient service errors retry with backoff;
a clip that still fails is skipped instead of killing a multi-hour run.
With resume on, existing MP3s are kept.

## Fit + assemble

English rarely fits the original window, so each clip is fitted:

1. Fits → placed at the window start, the rest stays silent.
2. Too long → sped up with ffmpeg `atempo`, capped at **1.35×**.
3. Still overflowing → hard-cut at the next segment's start, so the voice
   never overlaps itself.

The dub track is a silent 44.1 kHz mono `bytearray` with the fitted WAVs
pasted in sample-accurately — plain `wave`, no giant ffmpeg filter graph.

## Remix

`ffmpeg -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest`: the video codec
stays whatever YouTube served (AV1 on the zoo clip, h264 on the 720p evals),
audio becomes AAC. Copying instead of re-encoding is why a 2-hour mux takes
minutes, and why the picture is bit-identical to the source.

## Resume + timing

Re-running with the same `--workdir` skips finished work: existing
`source.mp4`/`source.wav`, saved `segments.json`, existing TTS clips, and fit
WAVs newer than their MP3. Every run writes `timing.json` (per-stage seconds
plus total) — that file is the submission's "how long it took" record.

Measured on this machine (CPU, `small` model, batched):

| Video | Dub time | Segments | Speakers |
|---|---|---|---|
| CEC Hindi Diwas lecture (30 min, Hindi) | ~8 min | 68 | 2 |
| NPTEL Regional Language Workshop (1 h 46, Hindi) | ~24 min, 1 run | 241 | 1 |

Artifacts live in `work/eval_30m/` and `work/eval_2h/`: `source.mp4`,
`dubbed.mp4`, `timing.json`.

## Tests

38 tests, none faked. The `e2e`-marked ones share one session download of the
19-second "Me at the zoo" video plus one `tiny`-model transcript: real yt-dlp,
real transcription, real edge-tts voices, and a full `run_pipeline` dub that
asserts the output video codec matches the source (copy, not re-encode).
Offline subset: `uv run pytest -m "not e2e"`. The long videos are not in CI.

## Honest limits

- Pitch clustering is not pyannote; applause or laughter can mint extra
  "speakers" (crowd noise on a noisier recording would).
- Whisper's built-in translation is good, not literary MT.
- The 1.35× cap plus truncation can clip a rushed last word.
- Edge TTS needs internet; there is no offline voice.
- Overlapping original speech is serialized into windows.
