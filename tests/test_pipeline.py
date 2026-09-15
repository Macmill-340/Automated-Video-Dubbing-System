"""Pipeline tests: a real end-to-end dub of the 19 s test video plus CLI wiring."""

from __future__ import annotations

import pytest

from video_dubber import cli as cli_mod
from video_dubber.cli import main, run_pipeline
from conftest import ZOO_URL, media_streams


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
