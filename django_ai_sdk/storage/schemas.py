from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ThreadInfo(BaseModel):
    """
    Thread metadata returned by storage adapters.
    """

    id: str
    title: str = ""
    assistant_id: str = ""
    model: str = ""
    user_id: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)
    message_count: int = 0
    file_memory_id: str | None = None


class ThreadDetail(BaseModel):
    """Full thread information including messages for API responses.

    Returned by Assistant.history() for complete thread views.
    Contains both thread metadata and the conversation messages
    formatted according to the protocol handler.
    """

    thread: ThreadInfo
    messages: list[dict[str, Any]]
