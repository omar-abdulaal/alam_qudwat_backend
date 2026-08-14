"""Unit tests for app/services/tts.py's sentence-segmenting and streaming
synthesis — SageMaker itself isn't incremental, so this is what actually
makes audio start early for a long answer. No AWS calls: SageMaker
invocation itself is monkeypatched out.

No pytest-asyncio in this project (see requirements.txt) — async cases
are driven with plain asyncio.run() inside an otherwise-sync test
function, matching how the rest of the suite avoids that dependency.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from app.services import tts as tts_module
from app.services.tts import SilmaSageMakerTTS, TTSUpstreamAudioError, split_into_speech_segments


def test_splits_on_sentence_terminators_and_newlines():
    text = "الجملة الأولى. الجملة الثانية؟ الجملة الثالثة!\nفقرة جديدة."
    segments = split_into_speech_segments(text, max_chars=1000)
    assert segments == [
        "الجملة الأولى.",
        "الجملة الثانية؟",
        "الجملة الثالثة!",
        "فقرة جديدة.",
    ]


def test_each_sentence_is_its_own_segment_even_when_short():
    # "generate speech for each sentence instead of all the text
    # together" -- short sentences are NOT merged into one call; each
    # terminator ends a segment. max_chars only bounds a single run-on
    # sentence that has no terminator at all (see the test below).
    text = "أ. ب. ج."
    segments = split_into_speech_segments(text, max_chars=1000)
    assert segments == ["أ.", "ب.", "ج."]


def test_a_single_run_on_sentence_is_force_split_at_a_word_boundary():
    text = "كلمة " * 200  # ~1000 chars, no punctuation anywhere
    segments = split_into_speech_segments(text.strip(), max_chars=100)
    assert len(segments) > 1
    for segment in segments:
        assert len(segment) <= 100
        assert not segment.startswith(" ")


def test_no_text_yields_no_segments():
    assert split_into_speech_segments("") == []
    assert split_into_speech_segments("   \n  ") == []


def test_trailing_text_with_no_terminator_is_still_included():
    segments = split_into_speech_segments("جملة كاملة. بقية بلا نقطة نهاية")
    assert segments[-1] == "بقية بلا نقطة نهاية"


async def _stream_from_list(pieces: list[str]) -> AsyncIterator[str]:
    for piece in pieces:
        yield piece


class _FakeSettings:
    def __init__(self, endpoint_name: str | None = "fake-endpoint", debug_save_audio_dir: str | None = None):
        self.silma_sagemaker_endpoint_name = endpoint_name
        self.tts_default_voice_id = "pixel"
        self.tts_debug_save_audio_dir = debug_save_audio_dir


def _patch_sagemaker(monkeypatch, calls: list[str]):
    def fake_invoke(text: str, voice_id: str, endpoint_name: str):
        calls.append(text)
        return (b"\x00\x01" * 10, 32000, 1)

    monkeypatch.setattr(tts_module, "_invoke_silma_sagemaker", fake_invoke)
    monkeypatch.setattr(tts_module, "get_app_settings", lambda: _FakeSettings())


def test_speak_synthesizes_one_segment_per_sentence(monkeypatch):
    calls: list[str] = []
    _patch_sagemaker(monkeypatch, calls)

    async def run():
        tts = SilmaSageMakerTTS()
        audio = await tts.speak(_stream_from_list(["جملة أولى. جملة ثانية. جملة ثالثة."]))
        return [chunk async for chunk in audio.chunks]

    chunks = asyncio.run(run())
    assert len(chunks) > 0
    assert calls == ["جملة أولى.", "جملة ثانية.", "جملة ثالثة."]


def test_speak_returns_after_only_the_first_segment_is_synthesized(monkeypatch):
    """The whole point: speak() must not wait for every segment before
    returning — only the first one, so headers/first audio are available
    fast even for a long, multi-sentence answer."""
    synthesized: list[str] = []
    _patch_sagemaker(monkeypatch, synthesized)

    async def run():
        tts = SilmaSageMakerTTS()
        audio = await tts.speak(_stream_from_list(["الأولى. الثانية. الثالثة."]))

        # speak() has returned, but only the first segment should be
        # synthesized so far -- the rest happen lazily as `chunks` is
        # pulled.
        after_return = list(synthesized)

        chunks_iter = audio.chunks
        await chunks_iter.__anext__()
        after_first_chunk = list(synthesized)

        remaining = [chunk async for chunk in chunks_iter]
        return after_return, after_first_chunk, remaining

    after_return, after_first_chunk, remaining = asyncio.run(run())
    assert after_return == ["الأولى."]
    # Still only the first segment's audio pulled -- second segment
    # synthesis happens on-demand as iteration continues, not eagerly.
    assert after_first_chunk == ["الأولى."]
    assert isinstance(remaining, list)
    assert synthesized == ["الأولى.", "الثانية.", "الثالثة."]


def test_speak_raises_when_stream_has_no_text(monkeypatch):
    monkeypatch.setattr(tts_module, "get_app_settings", lambda: _FakeSettings())

    async def run():
        tts = SilmaSageMakerTTS()
        await tts.speak(_stream_from_list(["   ", "\n"]))

    with pytest.raises(TTSUpstreamAudioError):
        asyncio.run(run())


def test_speak_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(tts_module, "get_app_settings", lambda: _FakeSettings(endpoint_name=None))

    async def run():
        tts = SilmaSageMakerTTS()
        await tts.speak(_stream_from_list(["نص"]))

    with pytest.raises(RuntimeError, match="SILMA_SAGEMAKER_ENDPOINT_NAME"):
        asyncio.run(run())


def test_speak_rejects_a_later_segment_with_a_different_sample_rate(monkeypatch):
    """A later segment silently returning a different sample rate than the
    first (already sent to the client in the response headers) is exactly
    what would make that one segment play back at the wrong speed/pitch --
    this must fail loudly instead of streaming the mismatched audio."""
    calls: list[str] = []

    def fake_invoke(text: str, voice_id: str, endpoint_name: str):
        calls.append(text)
        # First segment: 32000 Hz. Second: a different rate, as if SILMA
        # had switched formats mid-answer.
        rate = 32000 if len(calls) == 1 else 22050
        return (b"\x00\x01" * 10, rate, 1)

    monkeypatch.setattr(tts_module, "_invoke_silma_sagemaker", fake_invoke)
    monkeypatch.setattr(tts_module, "get_app_settings", lambda: _FakeSettings())

    async def run():
        tts = SilmaSageMakerTTS()
        audio = await tts.speak(_stream_from_list(["الأولى. الثانية."]))
        return [chunk async for chunk in audio.chunks]

    with pytest.raises(TTSUpstreamAudioError, match="inconsistent audio format"):
        asyncio.run(run())


def test_speak_accepts_later_segments_with_a_matching_sample_rate(monkeypatch):
    _patch_sagemaker(monkeypatch, [])  # every segment returns 32000 Hz/mono

    async def run():
        tts = SilmaSageMakerTTS()
        audio = await tts.speak(_stream_from_list(["الأولى. الثانية. الثالثة."]))
        return [chunk async for chunk in audio.chunks]

    chunks = asyncio.run(run())
    assert len(chunks) > 0


def test_debug_save_audio_dir_writes_a_wav_and_text_file_per_segment(monkeypatch, tmp_path):
    debug_dir = tmp_path / "tts_debug"
    monkeypatch.setattr(tts_module, "get_app_settings", lambda: _FakeSettings(debug_save_audio_dir=str(debug_dir)))
    monkeypatch.setattr(
        tts_module, "_invoke_silma_sagemaker", lambda text, voice_id, endpoint_name: (b"\x00\x01" * 10, 32000, 1)
    )

    async def run():
        tts = SilmaSageMakerTTS()
        audio = await tts.speak(_stream_from_list(["الأولى. الثانية."]))
        return [chunk async for chunk in audio.chunks]

    asyncio.run(run())

    wav_files = sorted(debug_dir.glob("*.wav"))
    txt_files = sorted(debug_dir.glob("*.txt"))
    assert len(wav_files) == 2
    assert len(txt_files) == 2
    assert txt_files[0].read_text(encoding="utf-8") == "الأولى."
    assert txt_files[1].read_text(encoding="utf-8") == "الثانية."

    import wave as wave_module

    with wave_module.open(str(wav_files[0]), "rb") as wf:
        assert wf.getframerate() == 32000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_debug_save_audio_dir_unset_by_default_writes_nothing(monkeypatch, tmp_path):
    _patch_sagemaker(monkeypatch, [])  # _FakeSettings() defaults tts_debug_save_audio_dir to None

    async def run():
        tts = SilmaSageMakerTTS()
        audio = await tts.speak(_stream_from_list(["نص واحد."]))
        return [chunk async for chunk in audio.chunks]

    asyncio.run(run())

    # Nothing should have been written anywhere -- there's no directory to
    # even check, which is the point: the feature must be fully inert
    # unless explicitly configured.
    assert not any(tmp_path.iterdir())
