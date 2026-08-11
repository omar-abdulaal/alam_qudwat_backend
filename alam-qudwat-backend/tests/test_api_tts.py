import uuid

from app.api.deps import get_tts
from app.db.models import Conversation, Message
from app.main import app
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


def test_speak_with_message_id_uses_stored_content(client, db_session):
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
