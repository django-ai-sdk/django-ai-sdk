from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from django_ai_sdk.utils import map_strings

if TYPE_CHECKING:
    from collections.abc import Sequence


class Subject(StrEnum):
    QUESTION = "input"
    ANSWER = "output"
    EXCHANGE = "exchange"
    SOURCES = "sources"
    TOOL_CALL = "tool_call"


class Stage(StrEnum):
    """Only gate, tool and finish can change the turn."""

    GATE = "gate"
    TOOL = "tool"
    FINISH = "finish"
    REFLECT = "reflect"


INLINE_STAGES = (Stage.GATE, Stage.TOOL, Stage.FINISH)


class Mode(StrEnum):
    OFF = "off"
    REPORT = "report"
    ENFORCE = "enforce"


class Capture(StrEnum):
    NONE = "none"
    METADATA = "metadata"
    MASKED = "masked"
    FULL = "full"


class Severity(StrEnum):
    """How urgent an alert or notification is, least urgent first."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER = ("", *Severity)


def rank(order: Sequence[Any], value: Any, *, missing: int | None = None) -> int:
    if value in order:
        return order.index(value)
    return len(order) if missing is None else missing


def join_error(current: str, error: str) -> str:
    if not error or error in current.split("; "):
        return current
    return f"{current}; {error}" if current else error


def most_severe(*levels: str) -> str:
    return max(levels, key=lambda level: rank(SEVERITY_ORDER, level, missing=0), default="")


class Outcome(StrEnum):
    """What happened to a turn, worst first."""

    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    MASKED = "masked"
    FLAGGED = "flagged"
    ALLOWED = "allowed"


def worst_outcome(*outcomes: Outcome) -> Outcome:
    return min(outcomes, key=lambda o: rank(tuple(Outcome), o), default=Outcome.ALLOWED)


class TextSpan(BaseModel):
    start: int
    end: int
    label: str = ""


class Answer(BaseModel):
    """One reply to one lens. `value`: bool (Flag), label (Choice), 0..1 (Score), labels."""

    value: Any = None
    confidence: float = 1.0
    rationale: str = ""
    spans: list[TextSpan] = Field(default_factory=list)
    items: dict[str, float] = Field(default_factory=dict)

    answerer: str = ""
    model: str = ""
    latency_ms: float | None = None
    tokens: int | None = None
    cost: float | None = None
    trail: list[Answer] = Field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.value is not None

    @property
    def disputed(self) -> bool:
        """Whether a model asked earlier for this lens gave another answer than this one."""
        return bool(self.model) and any(
            a.ok and a.model and a.value != self.value for a in self.trail
        )


class Source(BaseModel):
    key: str = ""
    title: str = ""
    url: str = ""
    content: str = ""
    kind: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    memory_id: str = ""
    seeded: bool = False


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: Any = None
    result: Any = None
    error: bool = False
    agent: str = ""


class HistoryItem(BaseModel):
    role: str
    content: str


SOURCE_HEAD_CHARS = 120


class TurnContext(BaseModel):
    turn_id: str = ""
    thread_id: str = ""
    channel: str = ""
    agent_id: str = ""
    agent_name: str = ""
    input: str = ""
    output: str = ""
    history: list[HistoryItem] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call: ToolCall | None = None
    answers: dict[str, Answer] = Field(default_factory=dict)

    def text(self, subject: Subject) -> str:
        if subject is Subject.QUESTION:
            return self.input
        if subject is Subject.ANSWER:
            return self.output
        if subject is Subject.TOOL_CALL:
            call = self.tool_call
            return f"{call.name} {_compact(call.arguments)}" if call else ""
        if subject is Subject.SOURCES:
            return "\n\n".join(s.content for s in self.grounding())
        return f"{self.input}\n\n{self.output}"

    def fresh_sources(self) -> list[Source]:
        return [s for s in self.sources if not s.seeded]

    def grounding(self, *, earlier: bool = False) -> list[Source]:
        """What the answer may be based on: this turn's sources, then tool results.

        A tool result repeating a listed source is skipped.
        """
        sources = self.fresh_sources()
        heads = [s.content.strip()[:SOURCE_HEAD_CHARS] for s in sources if s.content.strip()]
        results = [
            Source(title=f"{call.name} result", content=_compact(call.result), kind="tool")
            for call in self.tool_calls
            if not call.error
            and call.result not in (None, "", [], {})
            and not _echoes(call.result, heads)
        ]
        carried = [s for s in self.sources if s.seeded and s.content.strip()] if earlier else []
        return [*sources, *results, *carried]


def _echoes(result: Any, heads: list[str]) -> bool:
    if not heads:
        return False
    found: list[str] = []

    def collect(text: str) -> str:
        found.append(text)
        return text

    map_strings(result, collect)
    return any(head in "\n".join(found) for head in heads)


def _compact(value: Any) -> str:
    import json

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
