from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from django_ai_sdk.oversight.types import Outcome

TurnStatus = Literal["", "open", "closed", "interrupted"]
ReviewStatus = Literal["", "open", "done", "dismissed"]
ReviewState = Literal["", "open", "reviewed", "unreviewed"]
ContentState = Literal["metadata", "masked", "full", "withheld", "none", "expired"]


class TurnFilter(BaseModel):
    """Filters for the turns feed. Empty fields do not filter."""

    outcome: Outcome | Literal[""] = ""
    report_outcome: Outcome | Literal[""] = ""
    status: TurnStatus = ""
    agent_id: str = ""
    channel: str = ""
    policy: str = ""
    severity: str = ""
    label: str = ""
    value: str = ""
    rule: str = ""
    review: ReviewState = ""
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
    resolved_by: str = ""
    agree: bool | None = None
    note: str = ""
    turn: TurnOut | None = None


class TurnDetail(TurnOut):
    content: dict[str, Any] | None = None
    content_state: ContentState = "withheld"
    answers: list[LensAnswerOut] = Field(default_factory=list)
    reviews: list[ReviewOut] = Field(default_factory=list)
    error: str = ""
    parent_turn_id: str = ""
    agent_fingerprint: str = ""
    reflect_state: str = ""
    thread_id: str = ""
    message_id: str = ""


class Page[T](BaseModel):
    """A page of a list. Pass `next_cursor` as `cursor` to get the next page."""

    items: list[T]
    limit: int
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
    by_report_outcome: dict[str, int] = Field(default_factory=dict)
    by_channel: dict[str, int] = Field(default_factory=dict)
    by_agent: dict[str, int] = Field(default_factory=dict)
    rules: list[RuleCountOut] = Field(default_factory=list)
    queues: dict[str, int] = Field(default_factory=dict)
