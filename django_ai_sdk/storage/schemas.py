from __future__ import annotations

from datetime import datetime  # noqa: TC003 — pydantic needs runtime access for field types
from typing import Any

from pydantic import BaseModel, Field


class ThreadInfo(BaseModel):
    """
    Thread metadata returned by storage adapters.
    """

    id: str
    title: str = ""
    agent_id: str = ""
    model: str = ""
    user_id: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)
    message_count: int = 0
    file_memory_id: str | None = None


class ThreadDetail(BaseModel):
    thread: ThreadInfo
    messages: list[dict[str, Any]]
