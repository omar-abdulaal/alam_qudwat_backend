"""In-memory pairing between an in-flight chat generation and a concurrent
TTS request, so POST /api/v1/tts/speak/live can start synthesizing audio
while POST /api/v1/chat/stream is still producing text for the same turn.

app/api/routes/chat.py creates one LiveTextBroadcast per turn and publishes
each diacritized LLM token to it as it streams (the same "raw_parts" text
already kept for `extra.diacritized_content`); app/api/routes/tts.py's
/speak/live subscribes to it by generation_id and feeds the resulting
async iterator straight into TextToSpeech.speak(), which already accepts a
live stream (see app/services/tts.py) — no changes needed there.

A LiveTextBroadcast buffers every piece published so far and fans it out
to any number of subscribers, including ones that join after some pieces
were already published: a subscriber replays the buffer first, then keeps
receiving new pieces live. This is what lets the Flutter app fire the
/speak/live request as soon as it has the generation_id from the
"conversation" SSE event, without having to race the LLM's first token.

Both requests must land on the same process (this is plain in-memory
state, the same simplifying assumption the module-level SageMaker
client/lock in app/services/tts.py already makes) — it will not work
across multiple worker processes/instances behind a load balancer without
sticky routing or a shared broker; acceptable for the current
single-instance deployment, revisit if/when this API scales out.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator, Optional

# How long a finished generation's buffer stays available for a
# late-joining TTS request before being evicted (generous enough for "the
# app calls /speak/live a couple seconds after chat/stream's `done`").
_TTL_AFTER_FINISH_SECONDS = 5 * 60

# Safety net only: evicts a generation that somehow never had finish()
# called on it (a bug, not a normal path — chat.py always calls it from a
# finally block), so a stuck entry can't leak forever.
_MAX_AGE_SECONDS = 15 * 60


class LiveTextBroadcast:
    def __init__(self) -> None:
        self.created_at = time.monotonic()
        self.finished_at: Optional[float] = None
        self._pieces: list[str] = []
        self._subscribers: list["asyncio.Queue[Optional[str]]"] = []
        self._done = False
        self._error: Optional[BaseException] = None
        self._lock = asyncio.Lock()

    async def publish(self, piece: str) -> None:
        if not piece:
            return
        async with self._lock:
            if self._done:
                return
            self._pieces.append(piece)
            for queue in self._subscribers:
                queue.put_nowait(piece)

    async def finish(self, error: Optional[BaseException] = None) -> None:
        async with self._lock:
            if self._done:
                return
            self._done = True
            self._error = error
            self.finished_at = time.monotonic()
            for queue in self._subscribers:
                queue.put_nowait(None)

    async def subscribe(self) -> AsyncIterator[str]:
        """Yields every piece published so far, then live pieces as they
        arrive, until finish() is called — raising its error, if any,
        once the stream ends."""
        queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        async with self._lock:
            for piece in self._pieces:
                queue.put_nowait(piece)
            if self._done:
                queue.put_nowait(None)
            else:
                self._subscribers.append(queue)
        try:
            while True:
                piece = await queue.get()
                if piece is None:
                    break
                yield piece
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
        if self._error is not None:
            raise self._error


class LiveGenerationRegistry:
    def __init__(self) -> None:
        self._entries: dict[uuid.UUID, LiveTextBroadcast] = {}

    def create(self) -> tuple[uuid.UUID, LiveTextBroadcast]:
        self._evict_expired()
        generation_id = uuid.uuid4()
        broadcast = LiveTextBroadcast()
        self._entries[generation_id] = broadcast
        return generation_id, broadcast

    def get(self, generation_id: uuid.UUID) -> Optional[LiveTextBroadcast]:
        self._evict_expired()
        return self._entries.get(generation_id)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            gid
            for gid, broadcast in self._entries.items()
            if (broadcast.finished_at is not None and now - broadcast.finished_at > _TTL_AFTER_FINISH_SECONDS)
            or (now - broadcast.created_at > _MAX_AGE_SECONDS)
        ]
        for gid in expired:
            self._entries.pop(gid, None)


_registry = LiveGenerationRegistry()


def get_live_generation_registry() -> LiveGenerationRegistry:
    return _registry
