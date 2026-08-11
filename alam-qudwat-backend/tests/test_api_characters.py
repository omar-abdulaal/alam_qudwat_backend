SEEDED_SLUGS = {"abu_bakr", "umar", "uthman", "ali"}


def test_list_characters_returns_seeded_data(client):
    resp = client.get("/api/v1/characters")
    assert resp.status_code == 200
    slugs = {c["slug"] for c in resp.json()}
    assert SEEDED_SLUGS <= slugs


def test_list_characters_filters_by_era(client):
    resp = client.get("/api/v1/characters", params={"era": "الخلافة الراشدة"})
    assert resp.status_code == 200
    assert SEEDED_SLUGS <= {c["slug"] for c in resp.json()}

    resp = client.get("/api/v1/characters", params={"era": "no-such-era"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_character_detail(client):
    resp = client.get("/api/v1/characters/abu_bakr")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "abu_bakr"
    assert body["era"]
    assert body["short_description"]


def test_get_unknown_character_returns_404(client):
    resp = client.get("/api/v1/characters/does-not-exist")
    assert resp.status_code == 404


def test_story_of_day_returns_an_active_character(client):
    resp = client.get("/api/v1/story-of-day")
    assert resp.status_code == 200
    body = resp.json()
    assert body["character"]["slug"] in SEEDED_SLUGS
    assert "is_curated" in body


def test_story_of_day_is_deterministic_for_a_given_date(client):
    resp1 = client.get("/api/v1/story-of-day", params={"date": "2026-01-01"})
    resp2 = client.get("/api/v1/story-of-day", params={"date": "2026-01-01"})
    assert resp1.json()["character"]["slug"] == resp2.json()["character"]["slug"]
