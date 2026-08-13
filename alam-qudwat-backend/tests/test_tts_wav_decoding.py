"""Unit tests for app/services/tts.py's WAV-decoding step — the fix for
SILMA/SageMaker now returning {"audio_base64": <complete WAV file>,
"sample_rate": ...} instead of headerless raw PCM. No AWS calls."""
import io
import wave

import pytest

from app.services.tts import TTSUpstreamAudioError, _extract_pcm_from_wav


def _make_wav_bytes(*, frames: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return buf.getvalue()


def test_extract_pcm_strips_header_and_reports_actual_sample_rate():
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"  # 4 int16 samples
    wav_bytes = _make_wav_bytes(frames=pcm, sample_rate=24000)

    data, sample_rate, channels, sample_width = _extract_pcm_from_wav(wav_bytes)

    assert data == pcm
    assert sample_rate == 24000
    assert channels == 1
    assert sample_width == 2


def test_extract_pcm_uses_the_wav_headers_own_rate_not_a_guess():
    # A different rate than the module's old fixed 32000 Hz assumption —
    # the whole point of the fix is that this must come from the file.
    wav_bytes = _make_wav_bytes(frames=b"\x00\x00" * 10, sample_rate=44100)

    _data, sample_rate, _channels, _width = _extract_pcm_from_wav(wav_bytes)

    assert sample_rate == 44100


def test_extract_pcm_rejects_garbage_bytes():
    with pytest.raises(TTSUpstreamAudioError):
        _extract_pcm_from_wav(b"this is not a wav file")


def test_extract_pcm_does_not_leak_the_wav_container_into_the_pcm_data():
    """The RIFF/fmt header bytes must never end up mixed into the audio
    samples handed to the client — that's what produced static/noise."""
    pcm = b"\xAA\xBB" * 50
    wav_bytes = _make_wav_bytes(frames=pcm, sample_rate=32000)

    data, *_rest = _extract_pcm_from_wav(wav_bytes)

    assert data == pcm
    assert b"RIFF" not in data
    assert b"WAVE" not in data
