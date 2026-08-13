"""Unit tests for app/services/live_generation.py -- the in-memory fan-out
that lets POST /api/v1/tts/speak/live consume the same text a concurrent
POST /api/v1/chat/stream call is still producing.

No pytest-asyncio in this project -- async cases are driven with plain
asyncio.run() inside an otherwise-sync test function, matching the rest
of the suite (see tests/test_tts_segmenting.py).
"""
from __future__ import annotations

import asyncio

from app.services.live_generation import LiveGenerationRegistry, LiveTextBroadcast


def test_subscriber_receives_buffered_then_live_pieces():
    # asyncio.Queue.get() resolves synchronously (no event-loop suspension)
    # when the queue is already non-empty, so awaiting the first item from
    # subscribe() deterministically proves the subscriber is registered
    # for live pieces too -- no sleep(0)/task-race needed to prove order.
    async def run():
        broadcast = LiveTextBroadcast()
        await broadcast.publish("أولاً. ")

        agen = broadcast.subscribe()
        first = await agen.__anext__()

        await broadcast.publish("ثانياً.")
        await broadcast.finish()

        rest = [piece async for piece in agen]
        return [first, *rest]

    assert asyncio.run(run()) == ["أولاً. ", "ثانياً."]


def test_subscriber_joining_after_finish_gets_full_buffer_then_stops():
    async def run():
        broadcast = LiveTextBroadcast()
        await broadcast.publish("جملة.")
        await broadcast.finish()

        return [piece async for piece in broadcast.subscribe()]

    assert asyncio.run(run()) == ["جملة."]


def test_finish_with_error_is_raised_after_the_buffer_is_exhausted():
    async def run():
        broadcast = LiveTextBroadcast()
        await broadcast.publish("جزء أول.")
        await broadcast.finish(RuntimeError("generation failed"))

        received = []
        try:
            async for piece in broadcast.subscribe():
                received.append(piece)
        except RuntimeError as exc:
            return received, str(exc)
        return received, None

    received, error = asyncio.run(run())
    assert received == ["جزء أول."]
    assert error == "generation failed"


def test_publish_after_finish_is_ignored_not_an_error():
    async def run():
        broadcast = LiveTextBroadcast()
        await broadcast.finish()
        await broadcast.publish("متأخر جداً")  # must not raise
        return [piece async for piece in broadcast.subscribe()]

    assert asyncio.run(run()) == []


def test_registry_create_then_get_returns_the_same_broadcast():
    registry = LiveGenerationRegistry()
    generation_id, broadcast = registry.create()
    assert registry.get(generation_id) is broadcast


def test_registry_get_unknown_id_returns_none():
    import uuid

    registry = LiveGenerationRegistry()
    assert registry.get(uuid.uuid4()) is None


def test_registry_evicts_entries_well_past_their_ttl(monkeypatch):
    from app.services import live_generation as live_generation_module

    fake_now = [1000.0]
    monkeypatch.setattr(live_generation_module.time, "monotonic", lambda: fake_now[0])

    registry = LiveGenerationRegistry()
    generation_id, broadcast = registry.create()

    async def finish():
        await broadcast.finish()

    asyncio.run(finish())

    fake_now[0] += live_generation_module._TTL_AFTER_FINISH_SECONDS + 1
    assert registry.get(generation_id) is None
