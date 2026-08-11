from app.api.deps import get_stt
from app.main import app
from tests.fake_stt import FakeSTT


def _install_fake_stt(stt: FakeSTT) -> None:
    app.dependency_overrides[get_stt] = lambda: stt


def test_transcribe_returns_text_without_creating_a_conversation(client, db_session):
    from app.db.models import Conversation

    stt = FakeSTT("مرحبا هذا اختبار")
    _install_fake_stt(stt)

    resp = client.post(
        "/api/v1/stt/transcribe",
        files={"audio": ("test.wav", b"\x00\x01" * 1000, "audio/wav")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "مرحبا هذا اختبار"}
    assert len(stt.calls) == 1
    assert stt.calls[0][1] == "test.wav"
    assert db_session.query(Conversation).count() == 0


def test_transcribe_rejects_unsupported_extension(client):
    stt = FakeSTT()
    _install_fake_stt(stt)

    resp = client.post(
        "/api/v1/stt/transcribe",
        files={"audio": ("test.exe", b"not audio", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_transcribe_rejects_empty_file(client):
    stt = FakeSTT()
    _install_fake_stt(stt)

    resp = client.post(
        "/api/v1/stt/transcribe",
        files={"audio": ("test.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400
