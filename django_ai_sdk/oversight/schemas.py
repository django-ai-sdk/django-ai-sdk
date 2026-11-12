"""Request and response models for the review API.

Plain pydantic, so the SDK does not tie you to a web framework. Fields marked
"Content" are filled only for readers allowed to see the turn's content
(`VIEW_OVERSIGHT_CONTENT`, within `scope_turns`). For others they stay empty.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from django_ai_sdk.oversight.types import Outcome

# Fixed vocabularies, so a typo is refused instead of matching nothing. "" does not filter.
TurnStatus = Literal["", "open", "closed", "interrupted"]
ReviewStatus = Literal["", "open", "done", "dismissed"]
# "open" (waiting for a reviewer), "reviewed" (a person closed one) or "unreviewed" (neither).
ReviewState = Literal["", "open", "reviewed", "unreviewed"]
# A turn's capture level, or why it shows no content (`TurnDetail.content_state`).
ContentState = Literal["metadata", "masked", "full", "withheld", "none", "expired"]


class TurnFilter(BaseModel):
    """Filters for the turns feed. Empty fields do not filter."""

    outcome: Outcome | Literal[""] = ""
    # The outcome a rule in report mode would have caused.
    report_outcome: Outcome | Literal[""] = ""
    status: TurnStatus = ""
    agent_id: str = ""
    # Matched as a prefix, so "webhook" matches every webhook channel.
    channel: str = ""
    policy: str = ""
    severity: str = ""
    # A lens that has an answer on the turn. With `value`, that answer must match.
    label: str = ""
    value: str = ""
    rule: str = ""
    review: ReviewState = ""
    # Content. Filtering by thread needs VIEW_OVERSIGHT_CONTENT.
    thread_id: str = ""
    since: datetime | None = None
    until: datetime | None = None


class RuleMatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Annotated[str, BeforeValidator(str)]
    rule: str
    stage: str
    mode: str
    acted: bool
    actions: list[dict[str, Any]] = Field(default_factory=list)
    condition: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class TurnOut(BaseModel):
    """One line of the turns feed."""

    id: str
    created_at: datetime
    status: str
    channel: str
    agent_id: str
    agent_name: str
    policy: str
    outcome: str
    report_outcome: str
    severity: str = ""
    labels: dict[str, Any] = Field(default_factory=dict)
    matches: list[RuleMatchOut] = Field(default_factory=list)
    model: str = ""
    tokens: int | None = None
    cost: float | None = None
    latency_ms: float | None = None
    reviews_open: int = 0
    reflected: bool = False
    # Content. The start of the question ("" when not shown), and the asker's user id
    # ("" if anonymous).
    preview: str = ""
    actor_id: str = ""


class LensAnswerOut(BaseModel):
    """One answer to one lens.

    The rationale, spans and per-item scores are content. Otherwise `value` is
    shown as the record stores it.
    """

    id: str
    lens: str
    label: str = ""
    kind: str = ""
    stage: str
    answerer: str
    model: str = ""
    value: Any = None
    confidence: float | None = None
    error: str = ""
    cost: float | None = None
    latency_ms: float | None = None
    created_at: datetime
    trail: list[dict[str, Any]] = Field(default_factory=list)
    # Content.
    rationale: str = ""
    spans: list[dict[str, Any]] = Field(default_factory=list)
    items: dict[str, float] = Field(default_factory=dict)


class ReviewOut(BaseModel):
    """A review item. `note` is content: empty when the reader may not see it."""

    id: str
    turn_id: str
    queue: str
    rule: str = ""
    status: Literal["open", "done", "dismissed"]
    created_at: datetime
    resolved_at: datetime | None = None
    # The reviewer's user id.
    resolved_by: str = ""
    # Whether the reviewer agreed with the rule. None when open or dismissed.
    agree: bool | None = None
    note: str = ""
    turn: TurnOut | None = None


class TurnDetail(TurnOut):
    # The stored content. None when the reader may not see it or nothing was kept.
    content: dict[str, Any] | None = None
    # The capture level ("metadata", "masked" or "full"), or why there is no content:
    # "withheld" (not for this reader), "none" (never kept or forgotten) or "expired".
    content_state: ContentState = "withheld"
    answers: list[LensAnswerOut] = Field(default_factory=list)
    reviews: list[ReviewOut] = Field(default_factory=list)
    # What went wrong, such as a failed stage or a disconnect. Never content.
    error: str = ""
    # The outer turn, when this one ran nested under another policy, as a subagent does.
    parent_turn_id: str = ""
    agent_fingerprint: str = ""
    reflect_state: str = ""
    # Content.
    thread_id: str = ""
    message_id: str = ""


class Page[T](BaseModel):
    """A page of a list. Pass `next_cursor` as `cursor` to get the next page."""

    items: list[T]
    limit: int
    # Empty on the last page.
    next_cursor: str = ""


class ResolveIn(BaseModel):
    """Close a review item: say whether the rule was right (`agree`), or `dismiss` it."""

    agree: bool | None = None
    note: str = Field("", max_length=5_000)
    dismiss: bool = False


class RuleCountOut(BaseModel):
    """How often one rule in one mode matched, and how often it acted."""

    rule: str
    mode: str
    matches: int
    acted: int


class MetricsOut(BaseModel):
    """Counts over a window of turns, from SQL aggregates."""

    since: datetime
    turns: int
    by_outcome: dict[str, int] = Field(default_factory=dict)
    # Outcomes that rules in report mode would have caused.
    by_report_outcome: dict[str, int] = Field(default_factory=dict)
    by_channel: dict[str, int] = Field(default_factory=dict)
    # Keyed by agent id, like the `agent_id` filters, so a rename does not split counts.
    by_agent: dict[str, int] = Field(default_factory=dict)
    rules: list[RuleCountOut] = Field(default_factory=list)
    # Open review items per queue, regardless of age.
    queues: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "ContentState",
    "LensAnswerOut",
    "MetricsOut",
    "Page",
    "ResolveIn",
    "ReviewOut",
    "ReviewState",
    "ReviewStatus",
    "RuleCountOut",
    "RuleMatchOut",
    "TurnDetail",
    "TurnFilter",
    "TurnOut",
    "TurnStatus",
]
