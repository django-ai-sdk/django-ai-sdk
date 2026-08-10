from __future__ import annotations

from pydantic import BaseModel, Field


class MessagePart(BaseModel):
    type: str
    text: str | None = None


class Message(BaseModel):
    role: str
    parts: list[MessagePart]
    id: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message]
    agent_id: str | None = None
    id: str | None = None
    trigger: str | None = None


class RateMessagePayload(BaseModel):
    rating: int | None = Field(
        None, description="Rating value: 1 for good, -1 for bad, or None to unrate (optional)"
    )
    feedback: str = Field(default="", description="Optional explanation for the rating")
