from __future__ import annotations

from pydantic import BaseModel


class TranscriptionOut(BaseModel):
    text: str
