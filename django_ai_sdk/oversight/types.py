"""Shared types for oversight: what is checked, when, and the result."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from django_ai_sdk.utils import map_strings

if TYPE_CHECKING:
    from collections.abc import Sequence


class Subject(StrEnum):
    """What a lens looks at."""

    INPUT = "input"
    OUTPUT = "output"
    # The question and the answer together.
    EXCHANGE = "exchange"
    # Retrieved sources, read together with the question.
    SOURCES = "sources"
    # One tool call's name and arguments, before it runs.
    TOOL_CALL = "tool_call"


class Stage(StrEnum):
    """When a lens runs. Only gate, tool and finish can change the turn."""

    # Before the question is stored or reaches the model.
    GATE = "gate"
    # Before each tool call runs, and on each tool result before the model reads it.
    TOOL = "tool"
    # Once the answer is complete, before it is stored.
    FINISH = "finish"
    # In the background, after the turn is recorded.
    REFLECT = "reflect"


# Stages that run while the user waits, in order.
INLINE_STAGES = (Stage.GATE, Stage.TOOL, Stage.FINISH)


class Mode(StrEnum):
    """What a rule does when its condition matches."""

    OFF = "off"
    # Record what the rule would have done, but leave the turn alone.
    REPORT = "report"
    ENFORCE = "enforce"


class Capture(StrEnum):
    """How much of a turn's content the record keeps."""

    # Rows only, no text.
    NONE = "none"
    # Lengths, labels, source kinds and tool names. No text, titles or URLs.
    METADATA = "metadata"
    # Text with every mask a rule found applied.
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
    """Return where `value` sits in `order`.

    A value not in `order` gets `missing`, by default a place after all of them,
    so an unknown kind sorts last instead of raising.
    """
    if value in order:
        return order.index(value)
    return len(order) if missing is None else missing


# A reviewer's verdict on a rule is recorded as an answer to "rule:<name>".
RULE_VERDICT_PREFIX = "rule:"


def join_error(current: str, error: str) -> str:
    """Add `error` to the "; "-joined errors in `current`, once."""
    if not error or error in current.split("; "):
        return current
    return f"{current}; {error}" if current else error


def most_severe(*levels: str) -> str:
    """Return the most urgent of `levels`, or "" when there are none."""
    return max(levels, key=lambda level: rank(SEVERITY_ORDER, level, missing=0), default="")


class Outcome(StrEnum):
    """What happened to a turn, worst first."""

    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    MASKED = "masked"
    FLAGGED = "flagged"
    ALLOWED = "allowed"


OUTCOME_ORDER = tuple(Outcome)


def worst_outcome(*outcomes: Outcome) -> Outcome:
    """Return the worst of `outcomes`, or ALLOWED when there are none."""
    return min(outcomes, key=lambda o: rank(OUTCOME_ORDER, o), default=Outcome.ALLOWED)


class TextSpan(BaseModel):
    """A span of the subject's text that an answerer pointed at."""

    start: int
    end: int
    label: str = ""


class Answer(BaseModel):
    """One reply to one lens: from its patterns, its check or one of its judges.

    The type of `value` depends on the lens. A Flag gives a bool, a Choice a label,
    a Score 0..1 and Entities a list of labels.
    """

    value: Any = None
    # How sure the answerer is, 0..1. A pattern lens is always 1; a Patterns judge's miss is not.
    confidence: float = 1.0
    rationale: str = ""
    spans: list[TextSpan] = Field(default_factory=list)
    # Per-item probabilities, by key, when the answerer judges item by item.
    # For a `per_source` lens the keys are source keys.
    items: dict[str, float] = Field(default_factory=dict)

    answerer: str = ""
    model: str = ""
    latency_ms: float | None = None
    tokens: int | None = None
    cost: float | None = None
    # Answers from the lens's earlier judges that led to this one.
    trail: list[Answer] = Field(default_factory=list)
    # Set when the answerer failed or timed out.
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.value is not None


class Source(BaseModel):
    """One retrieved source, as the model saw it."""

    key: str = ""
    title: str = ""
    url: str = ""
    content: str = ""
    kind: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    memory_id: str = ""
    # Carried over from an earlier turn instead of found for this one.
    seeded: bool = False


class ToolCall(BaseModel):
    """One tool call of the turn."""

    id: str = ""
    name: str
    arguments: Any = None
    result: Any = None
    error: bool = False
    agent: str = ""


class HistoryItem(BaseModel):
    role: str
    content: str


# A tool result that repeats this many leading characters of a source counts as that source.
SOURCE_HEAD_CHARS = 120


class TurnContext(BaseModel):
    """Everything a lens may read when it runs."""

    turn_id: str = ""
    channel: str = ""
    agent_id: str = ""
    agent_name: str = ""
    input: str = ""
    output: str = ""
    history: list[HistoryItem] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set only while a TOOL_CALL lens judges one call.
    tool_call: ToolCall | None = None
    # Answers so far by lens name, so one lens can read another's.
    answers: dict[str, Answer] = Field(default_factory=dict)

    def text(self, subject: Subject) -> str:
        """Return the plain text a text-only answerer reads for `subject`."""
        if subject is Subject.INPUT:
            return self.input
        if subject is Subject.OUTPUT:
            return self.output
        if subject is Subject.TOOL_CALL:
            call = self.tool_call
            return f"{call.name} {_compact(call.arguments)}" if call else ""
        if subject is Subject.SOURCES:
            return "\n\n".join(s.content for s in self.fresh_sources())
        return f"{self.input}\n\n{self.output}"

    def fresh_sources(self) -> list[Source]:
        """Sources found for this turn, excluding ones carried over."""
        return [s for s in self.sources if not s.seeded]

    def grounding(self, *, earlier: bool = False) -> list[Source]:
        """Return what the answer may be based on: this turn's sources, then tool results.

        Tool results ground an answer just like retrieved chunks, e.g. a weather tool
        saying "12°C and rain". A tool result that repeats a listed source (like a
        search tool's) is skipped so it is not read twice. `earlier` also appends
        sources from earlier turns, since a follow-up may answer from them.
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
    """Whether a tool result contains the start of one of these sources."""
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


__all__ = [
    "INLINE_STAGES",
    "Answer",
    "Capture",
    "HistoryItem",
    "Mode",
    "Outcome",
    "RULE_VERDICT_PREFIX",
    "SEVERITY_ORDER",
    "Severity",
    "Source",
    "Stage",
    "Subject",
    "TextSpan",
    "ToolCall",
    "TurnContext",
    "join_error",
    "most_severe",
    "rank",
    "worst_outcome",
]
