import asyncio
import uuid

from app.api.deps import get_tts
from app.db.models import Conversation, Message
from app.main import app
from app.services.live_generation import get_live_generation_registry
from tests.fake_tts import FakeTTS


def _install_fake_tts(tts: FakeTTS) -> None:
    app.dependency_overrides[get_tts] = lambda: tts


def test_speak_with_raw_text_streams_audio_with_headers(client):
    tts = FakeTTS(chunks=[b"abc", b"def"])
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا"})

    assert resp.status_code == 200
    assert resp.content == b"abcdef"
    assert resp.headers["X-Audio-Sample-Rate"] == "32000"
    assert resp.headers["X-Audio-Channels"] == "1"
    assert resp.headers["X-Audio-Sample-Format"] == "pcm_s16le"
    assert tts.calls == [("مرحبا", None)]


def test_speak_strips_citation_markers_before_synthesis(client):
    tts = FakeTTS()
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "هذا رد يعتمد على المصادر [1] و[٢]."})

    assert resp.status_code == 200
    spoken_text = tts.calls[0][0]
    assert "[1]" not in spoken_text
    assert "[٢]" not in spoken_text
    assert "هذا رد يعتمد على المصادر" in spoken_text


def test_speak_with_raw_text_is_never_diacritized(client):
    """There is no LLM-based diacritizer at all, not even as a fallback —
    text with no diacritics is simply spoken without them."""
    tts = FakeTTS()
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا بالجميع"})

    assert resp.status_code == 200
    assert tts.calls[0][0] == "مرحبا بالجميع"


def test_speak_with_message_id_uses_stored_content_when_no_diacritized_version(client, db_session):
    tts = FakeTTS()
    _install_fake_tts(tts)

    conv = Conversation(id=uuid.uuid4(), character_slug="abu_bakr", narrator_mode="adults")
    db_session.add(conv)
    db_session.flush()
    msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="assistant", content="نص الرسالة المخزنة")
    db_session.add(msg)
    db_session.commit()

    resp = client.post("/api/v1/tts/speak", json={"message_id": str(msg.id)})

    assert resp.status_code == 200
    assert tts.calls[0][0] == "نص الرسالة المخزنة"


def test_speak_uses_precomputed_diacritized_content_when_present(client, db_session):
    """When the chat flow already produced diacritized text for this
    answer (app/api/routes/chat.py), TTS must use it verbatim — never
    generated or edited here, and never via an LLM call."""
    tts = FakeTTS()
    _install_fake_tts(tts)

    conv = Conversation(id=uuid.uuid4(), character_slug="abu_bakr", narrator_mode="adults")
    db_session.add(conv)
    db_session.flush()
    msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="assistant",
        content="هذا رد مخزن.",
        extra={"diacritized_content": "هَذَا رَدٌّ مُخَزَّنٌ."},
    )
    db_session.add(msg)
    db_session.commit()

    resp = client.post("/api/v1/tts/speak", json={"message_id": str(msg.id)})

    assert resp.status_code == 200
    assert tts.calls[0][0] == "هَذَا رَدٌّ مُخَزَّنٌ."  # the precomputed diacritized text, untouched


def test_speak_rejects_unknown_message_id(client):
    tts = FakeTTS()
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"message_id": str(uuid.uuid4())})
    assert resp.status_code == 404


def test_speak_rejects_both_text_and_message_id(client):
    resp = client.post(
        "/api/v1/tts/speak", json={"text": "x", "message_id": str(uuid.uuid4())}
    )
    assert resp.status_code == 422


def test_speak_rejects_neither_text_nor_message_id(client):
    resp = client.post("/api/v1/tts/speak", json={})
    assert resp.status_code == 422


def test_speak_rejects_text_too_long(client):
    tts = FakeTTS()
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "a" * 5000})
    assert resp.status_code == 400


def test_speak_returns_503_when_backend_not_configured(client):
    tts = FakeTTS(raise_error=RuntimeError("SILMA_SAGEMAKER_ENDPOINT_NAME is not set."))
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا"})
    assert resp.status_code == 503


def test_speak_uses_the_sample_rate_the_backend_actually_returns(client):
    """SILMA's native output rate isn't guaranteed to be 32000 Hz — the
    header must reflect whatever this call's audio actually is, not a
    fixed assumption."""
    tts = FakeTTS(chunks=[b"abc"], sample_rate=24000)
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا"})

    assert resp.status_code == 200
    assert resp.headers["X-Audio-Sample-Rate"] == "24000"


def test_speak_returns_502_when_upstream_audio_is_unusable(client):
    from app.services.tts import TTSUpstreamAudioError

    tts = FakeTTS(raise_error=TTSUpstreamAudioError("SageMaker returned empty audio."))
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا"})
    assert resp.status_code == 502


def test_speak_live_synthesizes_a_generation_streamed_via_the_registry(client):
    """The Flutter app is meant to call this right after receiving the
    generation_id from chat/stream's "conversation" event, without
    waiting for `done` -- exercised here against a generation that's
    already finished by the time /speak/live is called, which is the
    part reachable through a synchronous test client; genuine
    mid-generation interleaving is covered at the unit level in
    tests/test_live_generation.py and tests/test_tts_segmenting.py."""
    tts = FakeTTS()
    _install_fake_tts(tts)

    registry = get_live_generation_registry()
    generation_id, broadcast = registry.create()

    async def produce():
        await broadcast.publish("جملة أولى. ")
        await broadcast.publish("جملة ثانية.")
        await broadcast.finish()

    asyncio.run(produce())

    resp = client.post("/api/v1/tts/speak/live", json={"generation_id": str(generation_id)})

    assert resp.status_code == 200
    assert tts.calls[0][0] == "جملة أولى. جملة ثانية."


def test_speak_live_rejects_unknown_generation_id(client):
    tts = FakeTTS()
    _install_fake_tts(tts)

    resp = client.post("/api/v1/tts/speak/live", json={"generation_id": str(uuid.uuid4())})
    assert resp.status_code == 404


def test_speak_live_maps_a_failed_generation_to_503(client):
    tts = FakeTTS()
    _install_fake_tts(tts)

    registry = get_live_generation_registry()
    generation_id, broadcast = registry.create()

    async def fail():
        await broadcast.publish("جزء غير مكتمل.")
        await broadcast.finish(RuntimeError("chat generation failed"))

    asyncio.run(fail())

    resp = client.post("/api/v1/tts/speak/live", json={"generation_id": str(generation_id)})
    assert resp.status_code == 503
