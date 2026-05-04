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
    assistant_id: str | None = None
    id: str | None = None
    trigger: str | None = None


class RateMessagePayload(BaseModel):
    rating: int = Field(..., description="Rating value: 1 for good, -1 for bad")
