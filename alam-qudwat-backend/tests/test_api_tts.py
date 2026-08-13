import uuid

import pytest

from app.api.deps import get_diacritizer, get_tts
from app.db.models import Conversation, Message
from app.main import app
from tests.fake_diacritizer import FakeDiacritizer
from tests.fake_tts import FakeTTS


def _install_fake_tts(tts: FakeTTS) -> None:
    app.dependency_overrides[get_tts] = lambda: tts


def _install_fake_diacritizer(diacritizer: FakeDiacritizer) -> None:
    app.dependency_overrides[get_diacritizer] = lambda: diacritizer


@pytest.fixture(autouse=True)
def _default_diacritizer():
    """Identity transform by default, so tests unrelated to diacritization
    (message lookup, citation stripping, error codes, ...) don't need to
    know or care about it. Dedicated tests below override this to verify
    the diacritization wiring itself."""
    diacritizer = FakeDiacritizer(transform=lambda text: text)
    _install_fake_diacritizer(diacritizer)
    return diacritizer


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


def test_speak_uses_precomputed_diacritized_content_without_calling_diacritizer(client, db_session):
    """When the chat flow already produced diacritized text for this
    answer (app/api/routes/chat.py), TTS must use it verbatim -- no
    second LLM call just to re-diacritize the same content."""
    tts = FakeTTS()
    _install_fake_tts(tts)
    diacritizer = FakeDiacritizer()
    _install_fake_diacritizer(diacritizer)

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
    assert diacritizer.calls == []  # never called -- that's the whole point


def test_speak_falls_back_to_diacritizer_when_no_precomputed_content(client, db_session):
    """Older/ungrounded-fallback messages have no `diacritized_content` —
    must still fall back to the on-demand diacritizer, unchanged."""
    tts = FakeTTS()
    _install_fake_tts(tts)
    diacritizer = FakeDiacritizer(transform=lambda text: f"[diacritized]{text}")
    _install_fake_diacritizer(diacritizer)

    conv = Conversation(id=uuid.uuid4(), character_slug="abu_bakr", narrator_mode="adults")
    db_session.add(conv)
    db_session.flush()
    msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="assistant", content="نص بلا تشكيل مخزن.")
    db_session.add(msg)
    db_session.commit()

    resp = client.post("/api/v1/tts/speak", json={"message_id": str(msg.id)})

    assert resp.status_code == 200
    assert diacritizer.calls == ["نص بلا تشكيل مخزن."]
    assert tts.calls[0][0] == "[diacritized]نص بلا تشكيل مخزن."


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


def test_speak_sends_diacritized_text_to_tts_not_plain_text(client):
    """The whole point of the feature: SILMA must receive the diacritized
    version, not the plain text — while the request/response contract
    (what the client sends and gets back) is untouched by this."""
    tts = FakeTTS()
    _install_fake_tts(tts)
    diacritizer = FakeDiacritizer(transform=lambda text: text.replace("مرحبا", "مَرْحَبًا"))
    _install_fake_diacritizer(diacritizer)

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا بالجميع"})

    assert resp.status_code == 200
    assert diacritizer.calls == ["مرحبا بالجميع"]  # diacritizer sees the plain, citation-stripped text
    assert tts.calls[0][0] == "مَرْحَبًا بالجميع"  # TTS receives the diacritized text


def test_speak_falls_back_to_plain_text_if_diacritization_fails(client):
    """Diacritization is a pronunciation enhancement, not a correctness
    requirement — a failure there must not block voice output."""
    tts = FakeTTS()
    _install_fake_tts(tts)
    _install_fake_diacritizer(FakeDiacritizer(raise_error=RuntimeError("openai boom")))

    resp = client.post("/api/v1/tts/speak", json={"text": "مرحبا"})

    assert resp.status_code == 200
    assert tts.calls[0][0] == "مرحبا"


def test_speak_diacritizes_message_id_content_too(client, db_session):
    tts = FakeTTS()
    _install_fake_tts(tts)
    diacritizer = FakeDiacritizer(transform=lambda text: f"[TASHKEEL]{text}")
    _install_fake_diacritizer(diacritizer)

    conv = Conversation(id=uuid.uuid4(), character_slug="abu_bakr", narrator_mode="adults")
    db_session.add(conv)
    db_session.flush()
    msg = Message(id=uuid.uuid4(), conversation_id=conv.id, role="assistant", content="نص الرسالة المخزنة")
    db_session.add(msg)
    db_session.commit()

    resp = client.post("/api/v1/tts/speak", json={"message_id": str(msg.id)})

    assert resp.status_code == 200
    assert tts.calls[0][0] == "[TASHKEEL]نص الرسالة المخزنة"
