from __future__ import annotations

import uuid  # noqa: TC003
from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceOut(BaseModel):
    """A single span"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    operation_name: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    parent_id: uuid.UUID | None = None
    thread_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    agent_id: uuid.UUID | None = None
    agent_name: str = ""
    model_name: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token totals across a set of spans."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    agent_name: str = ""
    by_subagent: dict[str, TokenUsage] = Field(default_factory=dict)
