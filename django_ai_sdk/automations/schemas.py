"""Output schemas for automations. Plain pydantic: the SDK does not pick the framework."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AutomationRunOut(BaseModel):
    """One execution, for a history list or a "last run" summary."""

    id: str
    status: str
    trigger: str
    scheduled_for: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # The workflow's outputs, keyed by each step's output_key.
    output: Any = None
    skip_reason: str = ""
    error: str = ""
    workflow_run_id: str | None = None
    dispatch_id: str


class AutomationOut(BaseModel):
    """One automation's current state, for a generic GET /automations list."""

    name: str
    label: str
    description: str = ""

    schedule: str
    next_run_at: datetime | None = None

    enabled: bool
    # Which layer decided `enabled`: "db" | "settings" | "code" | "kill-switch".
    enabled_source: str

    workflow: str
    requires: list[str] = []
    audience: str
    # Always False when `audience` is "app".
    subscribed: bool = False

    last_run: AutomationRunOut | None = None

    detail: str | None = None


__all__ = ["AutomationOut", "AutomationRunOut"]
