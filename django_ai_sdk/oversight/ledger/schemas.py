"""Request and response schemas for the ledger routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from django_ai_sdk.oversight.services.access import Reason

# Shortest reason accepted for a reveal.
MIN_REASON_CHARS = 10
# Most problems and unreadable entries a verify lists.
MAX_LISTED = 50


class CheckpointOut(BaseModel):
    """A checkpoint over the first `tree_size` entries."""

    id: UUID
    tree_size: int
    root_hash: str
    note: str
    signed: bool
    created_at: datetime


class VerifyOut(BaseModel):
    """The outcome of walking the chain (`chain.verify()`), plus the newest checkpoint check.

    `state` is "ok", "unreadable" or "failed". "unreadable" means some content is
    under a master key not configured here and was not checked. On "failed", see
    `problems`.
    """

    ok: bool = True
    checked: int = 0
    # Rows deleted under a "prune" entry.
    pruned: int = 0
    # Content erased under an "erase" entry, or shredded under a "forget" entry.
    forgotten: int = 0
    # Content under a master key not configured here. Its digest was not checked.
    unreadable: int = 0
    head: str = ""
    problems: list[dict[str, Any]] = Field(default_factory=list)
    # The first unreadable content entries, as {"seq", "ref", "key_id"}.
    unread: list[dict[str, Any]] = Field(default_factory=list)
    # Whether the newest checkpoint still matches the chain. None if there is none.
    checkpoint_ok: bool | None = None
    checkpoint_size: int | None = None

    @computed_field
    @property
    def state(self) -> str:
        if not self.ok:
            return "failed"
        return "unreadable" if self.unreadable else "ok"

    def fail(self, seq: int, problem: str, **detail: Any) -> None:
        self.ok = False
        if len(self.problems) < MAX_LISTED:
            self.problems.append({"seq": seq, "problem": problem, **detail})

    def cannot_read(self, seq: int, ref: str, key_id: str) -> None:
        self.unreadable += 1
        if len(self.unread) < MAX_LISTED:
            self.unread.append({"seq": seq, "ref": ref, "key_id": key_id})


class RevealIn(BaseModel):
    """The reason for a reveal. It is stored on the access record."""

    reason: Reason


class RevealOut(BaseModel):
    """Who asked a turn. `user_id` is "" when there was no user or it was deleted."""

    turn_id: str
    actor_id: str
    user_id: str = ""
    label: str = ""
