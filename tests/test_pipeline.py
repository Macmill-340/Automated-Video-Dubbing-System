"""Pipeline tests: a real end-to-end dub of the 19 s test video plus CLI wiring."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from video_dubber import cli as cli_mod
from video_dubber.cli import _say, main, run_pipeline
from video_dubber.models import Segment
from conftest import ZOO_URL, make_wav, media_streams

ffmpeg = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="needs ffmpeg")


@pytest.mark.e2e
def test_end_to_end_dub(tmp_path):
    result = run_pipeline(ZOO_URL, tmp_path / "work", tmp_path / "dubbed.mp4",
                          model_size="tiny", concurrency=2)

    assert result.final_video.is_file() and result.final_video.stat().st_size > 0
    assert result.language == "en"
    assert result.segments >= 1
    assert result.elapsed > 0

    streams = media_streams(result.final_video)
    source_streams = media_streams(tmp_path / "work" / "source.mp4")
    assert streams["video"]["codec_name"] == source_streams["video"]["codec_name"], \
        "video must be copied, not re-encoded"
    assert streams["audio"]["codec_name"] == "aac"

    # Intermediates stay on disk for inspection.
    work = tmp_path / "work"
    assert (work / "source.mp4").is_file()
    assert (work / "source.wav").is_file()
    assert (work / "dub.wav").is_file()
    assert list((work / "tts").glob("seg_*.mp3")), "expected per-segment TTS clips"


def test_main_prompts_for_missing_url(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "   ")
    assert main([]) == 2


def test_main_failure_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_mod, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert main(["https://youtu.be/fake", "--workdir", str(tmp_path)]) == 1
    assert "boom" in capsys.readouterr().err


def test_say_falls_back_to_ascii(monkeypatch, capsys):
    real_print = print

    def flaky(*args, **kwargs):
        if any("हिंदी" in str(a) for a in args):
            raise UnicodeEncodeError("ascii", "x", 0, 1, "x")
        return real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", flaky)
    _say("title: हिंदी lecture")
    assert "title:" in capsys.readouterr().out


@needs_ffmpeg
def test_resume_skips_finished_stages(tmp_path):
    """A workdir with source + segments + one TTS clip finishes with no network."""
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=128x128:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(work / "source.mp4")],
        check=True, capture_output=True,
    )
    make_wav(work / "source.wav", 2.0, rate=16000)
    (work / "meta.json").write_text(json.dumps({"title": "Resume Test", "url": "x"}))
    seg = Segment(0.0, 2.0, "Hello world test", speaker="SPEAKER_00",
                  voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz", volume="+0%")
    (work / "segments.json").write_text(json.dumps([seg.__dict__]))
    tts_dir = work / "tts"
    tts_dir.mkdir()
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:a", "libmp3lame", str(tts_dir / "seg_0000.mp3")],
        check=True, capture_output=True,
    )

    result = run_pipeline("https://youtu.be/unused", work, tmp_path / "dubbed.mp4", resume=True)

    assert result.final_video.is_file()
    assert result.segments == 1
    timing = json.loads((work / "timing.json").read_text())
    assert timing["stages"]["download"] == 0.0
    assert timing["stages"]["transcribe"] == 0.0
    assert timing["total_seconds"] > 0
    assert timing["runs"] == 1

    again = run_pipeline("https://youtu.be/unused", work, tmp_path / "dubbed.mp4", resume=True)
    merged = json.loads((work / "timing.json").read_text())
    assert merged["runs"] == 2
    assert merged["total_seconds"] >= timing["total_seconds"]
    assert again.final_video.is_file()
