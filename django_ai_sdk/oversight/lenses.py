"""Lenses. A lens is one typed question about a turn, declared by how it is answered.

    class Competitors(PatternLens):       # code finds spans, which a Mask rule can mask
        name = "competitors"
        keywords = {"acme": ["Acme"], "globex": ["Globex"]}

    class EvenAmount(CheckLens):          # plain Python, like a Django validator
        name = "even_amount"
        validators = [validate_even]      # raise Violation("...") to say yes

    class PromptInjection(JudgeLens):     # judges, cheapest first
        name = "prompt_injection"
        judges = [Patterns(INJECTION_PATTERNS), "classifier?", "judge?"]

The rules:

1. A lens is a PatternLens, a CheckLens or a JudgeLens, and only answers. What to
   do about the answer is a `Rule`, declared on the policy.
2. An attribute lives on the class whose behaviour it changes: `patterns`,
   `keywords` and `verify` on PatternLens; `validators` on CheckLens; `judges`,
   `min_confidence` and what a judge reads (`question`, `guidance`,
   `reads_question`) on JudgeLens. An answer option (`yes`, `labels`, `low`, ...)
   belongs to the answer types that read it. Declaring one that does not apply is
   ImproperlyConfigured when the lens is checked.
3. `answer` picks the answer type: `Flag` (yes or no), `Choice` (one of
   `labels`), `Score` (0 to 1) or `Entities` (which `labels` occur, and where).
4. A PatternLens answers Entities: what it found, with spans. `L(name).is_true`
   means it found something. What patterns find is a fact: hit and miss are sure (1).
5. A CheckLens answers with `check(text, view)`. By default that runs its
   `validators`, each of which raises `Violation` when the text breaks it, and
   answers a Flag. Override `check` for another answer: it returns the value (or
   an Answer to add a reason or confidence, or None to abstain).
6. A JudgeLens asks its judges in order. An answer at or above `min_confidence`
   ends the chain; otherwise the next judge is asked, and the last usable answer
   counts. A pattern pre-filter is just the first judge: it answers the lens's
   question, so a `Patterns` hit is sure (1) and a miss is not (0.5).
7. A judge alias ending in "?" is optional: the lens skips it when the deployment
   does not define it, and system check W003 names every lens left without a
   model judge.

Inline, pass the same attributes as keyword arguments::

    JudgeLens(name="topic", answer=Choice, labels=["billing", "hr", "other"], judges=["judge"])
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import re
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.oversight.actions import build_from_kind
from django_ai_sdk.oversight.judges.base import (
    DEFAULT_TIMEOUT,
    Judge,
    alias_defined,
    alias_of,
    is_optional,
    resolve,
)
from django_ai_sdk.oversight.judges.patterns import Patterns
from django_ai_sdk.oversight.types import Answer, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from django.utils.functional import Promise

    from django_ai_sdk.oversight.patterns import PatternTable, Verifier
    from django_ai_sdk.oversight.types import TextSpan, TurnContext

# The subjects each stage reads. A lens never runs at a stage that skips its subject.
STAGE_SUBJECTS: dict[Stage, frozenset[Subject]] = {
    Stage.GATE: frozenset({Subject.INPUT}),
    Stage.TOOL: frozenset({Subject.TOOL_CALL}),
    Stage.FINISH: frozenset({Subject.OUTPUT, Subject.EXCHANGE, Subject.SOURCES}),
    Stage.REFLECT: frozenset({Subject.INPUT, Subject.OUTPUT, Subject.EXCHANGE, Subject.SOURCES}),
}
# Default stages per subject: the first stage that reads it.
DEFAULT_STAGES: dict[Subject, tuple[Stage, ...]] = {
    subject: (next(stage for stage in Stage if subject in STAGE_SUBJECTS[stage]),)
    for subject in Subject
}


# Answer types. A lens picks one with `answer =`. The type reads its options from
# the lens, which falls back to the type's defaults. They are never instantiated.


class AnswerType:
    """How a lens's answer is shaped, parsed and described to a judge."""

    kind: ClassVar[str] = ""
    # The lens attributes this type reads, with their defaults.
    options: ClassVar[dict[str, Any]] = {}

    @classmethod
    def validate(cls, lens: Lens) -> None:
        """Raise ImproperlyConfigured when the lens's options don't fit this type."""

    @classmethod
    def parse(cls, lens: Lens, value: Any) -> Any:
        """Coerce a raw answer to this type; raise ValueError if it isn't one."""
        raise NotImplementedError

    @classmethod
    def from_detections(cls, lens: Lens, labels: list[str]) -> Any:
        """Turn the labels patterns found into a value."""
        raise NotImplementedError

    @classmethod
    def value_type(cls, lens: Lens) -> Any:
        """The Python type a structured-output judge returns for `value`."""
        raise NotImplementedError

    @classmethod
    def value_description(cls, lens: JudgeLens) -> str:
        return lens.question

    @classmethod
    def describe(cls, lens: Lens) -> dict[str, Any]:
        return {}


def _yes_no(value: Any) -> bool:
    """Read a yes or a no: a bool, a probability of yes (0.5 and up), or a word."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value >= 0.5
    if isinstance(value, str) and value.strip().lower() in {"yes", "true", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"no", "false", "0"}:
        return False
    raise ValueError(f"{value!r} is not a yes or a no")


class Flag(AnswerType):
    """Yes or no. Found labels mean yes. `yes` and `no` say what each means, for judges."""

    kind = "flag"
    options: ClassVar[dict[str, Any]] = {"yes": "Yes.", "no": "No."}

    @classmethod
    def parse(cls, lens: Lens, value: Any) -> bool:
        return _yes_no(value)

    @classmethod
    def from_detections(cls, lens: Lens, labels: list[str]) -> bool:
        return bool(labels)

    @classmethod
    def value_type(cls, lens: Lens) -> Any:
        return bool


class Choice(AnswerType):
    """Exactly one of `labels`. Patterns that find none give `default`, or the last label.

    `criteria` optionally says what a label means; a label without one means itself.
    """

    kind = "choice"
    # `default` is read only on the pattern path, so a judge-only lens can leave it unset.
    options: ClassVar[dict[str, Any]] = {"labels": [], "default": "", "criteria": {}}

    @classmethod
    def validate(cls, lens: Lens) -> None:
        if not lens.labels:
            raise ImproperlyConfigured(f"Choice lens {lens.name!r} declares no `labels`.")
        if lens.default and lens.default not in lens.labels:
            raise ImproperlyConfigured(
                f"Choice lens {lens.name!r}: `default` {lens.default!r} is not one of its "
                f"labels {lens.labels}; add it or set `default` to one."
            )

    @classmethod
    def parse(cls, lens: Lens, value: Any) -> str:
        value = str(value).strip()
        for label in lens.labels:
            if label.lower() == value.lower():
                return label
        raise ValueError(f"{value!r} is not one of {lens.labels}")

    @classmethod
    def from_detections(cls, lens: Lens, labels: list[str]) -> str:
        fallback = lens.default or lens.labels[-1]
        return next((label for label in labels if label in lens.labels), fallback)

    @classmethod
    def value_type(cls, lens: Lens) -> Any:
        return Literal[tuple(lens.labels)]  # ty: ignore[invalid-type-form]

    @classmethod
    def value_description(cls, lens: JudgeLens) -> str:
        return f"{lens.question} One of: {', '.join(lens.labels)}."

    @classmethod
    def describe(cls, lens: Lens) -> dict[str, Any]:
        return {"labels": list(lens.labels), "default": lens.default or lens.labels[-1]}


class Score(AnswerType):
    """A number from 0 to 1, where 1 is the most of what the question asks about.

    `low` and `high` say what 0 and 1 mean; `middle` is the middle level, for a
    classifier that scores in levels.
    """

    kind = "score"
    options: ClassVar[dict[str, Any]] = {
        "low": "not at all",
        "middle": "partly",
        "high": "completely",
    }

    @classmethod
    def parse(cls, lens: Lens, value: Any) -> float:
        # Judges are told 0 to 1 (see `value_description`), so reject anything else.
        number = float(value)
        if not 0 <= number <= 1:
            raise ValueError(f"{value!r} is not a score from 0 to 1")
        return number

    @classmethod
    def from_detections(cls, lens: Lens, labels: list[str]) -> float:
        return 1.0 if labels else 0.0

    @classmethod
    def value_type(cls, lens: Lens) -> Any:
        return float

    @classmethod
    def value_description(cls, lens: JudgeLens) -> str:
        return f"{lens.question} 0 means {lens.low}, 1 means {lens.high}."

    @classmethod
    def describe(cls, lens: Lens) -> dict[str, Any]:
        return {"low": lens.low, "high": lens.high}


class Entities(AnswerType):
    """Which of `labels` occur, and where. Empty means none; no `labels` means any.

    With `per_source`, judges that support it rate each source separately, as the
    quality pack's "taken from" lens does.
    """

    kind = "entities"
    options: ClassVar[dict[str, Any]] = {"labels": [], "per_source": False}

    @classmethod
    def parse(cls, lens: Lens, value: Any) -> list[str]:
        if value is None:
            return []
        items = [value] if isinstance(value, str) else list(value)
        found = [str(item) for item in items]
        if lens.labels:
            found = [label for label in found if label in lens.labels]
        return sorted(set(found))

    @classmethod
    def from_detections(cls, lens: Lens, labels: list[str]) -> list[str]:
        return cls.parse(lens, labels)

    @classmethod
    def value_type(cls, lens: Lens) -> Any:
        return list[str]

    @classmethod
    def value_description(cls, lens: JudgeLens) -> str:
        if lens.labels:
            return f"{lens.question} Any of: {', '.join(lens.labels)}; empty for none."
        return lens.question

    @classmethod
    def describe(cls, lens: Lens) -> dict[str, Any]:
        return {"labels": list(lens.labels)}


ANSWER_TYPES: dict[str, type[AnswerType]] = {
    cls.kind: cls for cls in (Flag, Choice, Score, Entities)
}
# Every answer option, whichever type reads it.
OPTIONS = frozenset(name for cls in ANSWER_TYPES.values() for name in cls.options)


class Violation(Exception):  # noqa: N818 - read like Django's ValidationError
    """Raised by a check lens's validator: the text breaks the rule, so the answer is yes.

    `message` becomes the answer's rationale.
    """

    def __init__(self, message: str | Promise) -> None:
        super().__init__(message)
        self.message = message


class Lens:
    """Base lens. Subclass PatternLens, CheckLens or JudgeLens."""

    # How the lens is answered: "pattern", "check" or "judge".
    kind: ClassVar[str] = ""
    answer: type[AnswerType] = Flag

    # Unique within a policy. Answers are stored under it, so renaming starts a new series.
    name: str = ""
    # Shown for the lens. Empty means its name.
    label: str = ""
    subject: Subject = Subject.INPUT
    # Empty means the default for the subject (see DEFAULT_STAGES).
    stages: tuple[Stage, ...] = ()
    # Fraction of turns the reflect stage judges, 0 to 1. Inline stages always run.
    sample: float = 1.0
    enabled: bool = True
    # Only ask when there are sources or tool results to judge against, from this
    # turn or an earlier one. Groundedness means nothing without them.
    needs_grounding: bool = False

    # Answer options. Each is set only when the answer type reads it (see
    # `AnswerType.options`), from the lens or else the type's default.
    yes: str
    no: str
    labels: list[str]
    default: str
    criteria: dict[str, str]
    low: str
    middle: str
    high: str
    per_source: bool

    def __init__(self, **overrides: Any) -> None:
        for key, value in overrides.items():
            if not hasattr(type(self), key) and key not in OPTIONS:
                raise ImproperlyConfigured(f"{type(self).__name__} has no attribute {key!r}")
            setattr(self, key, value)
        if isinstance(self.answer, str):
            if self.answer not in ANSWER_TYPES:
                raise ImproperlyConfigured(
                    f"Unknown answer {self.answer!r}; one of {sorted(ANSWER_TYPES)}"
                )
            self.answer = ANSWER_TYPES[self.answer]
        self.subject = Subject(self.subject)
        self.stages = tuple(Stage(s) for s in self.stages) or DEFAULT_STAGES[self.subject]
        # An answer option the lens leaves out takes its answer type's default.
        for option, default in self.answer.options.items():
            if not hasattr(self, option):
                setattr(self, option, copy.copy(default))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"

    @property
    def display_label(self) -> str:
        return self.label or self.name.replace("_", " ")

    def applies(self, view: TurnContext) -> bool:
        """Whether this turn has what the lens judges."""
        if self.subject is Subject.SOURCES:
            return bool(view.grounding())
        if self.needs_grounding:
            return bool(view.grounding(earlier=True))
        return True

    def validate(self) -> None:
        """Raise ImproperlyConfigured when the lens cannot be asked as declared."""
        if not self.name:
            raise ImproperlyConfigured(f"{type(self).__name__} lens without a `name`.")
        if not (isinstance(self.answer, type) and issubclass(self.answer, AnswerType)):
            raise ImproperlyConfigured(f"Lens {self.name!r}: {self.answer!r} is not an answer.")
        if misplaced := self._misplaced():
            raise ImproperlyConfigured(
                f"Lens {self.name!r} declares {misplaced}, which a {self.kind} lens answering "
                f"{self.answer.kind} does not read."
            )
        self.answer.validate(self)

    def _misplaced(self) -> list[str]:
        """Lens attributes declared here that this class or its answer type does not read."""
        mechanism = LENS_KINDS.get(self.kind, Lens)
        own = {*vars(Lens), *vars(mechanism), *self.answer.options}
        declared = set(self.__dict__)
        for cls in type(self).__mro__:
            if cls not in (Lens, mechanism, object):
                declared.update(vars(cls))
        return sorted((declared & LENS_ATTRIBUTES) - own)

    def steps(self) -> list[Any]:
        """What answers the lens, in order: judges, or specs for `judges.base.ask`."""
        raise NotImplementedError

    def settles(self, answer: Answer) -> bool:
        """Whether `answer` ends the lens's chain of steps."""
        return answer.ok

    def reports_kinds(self, answer: Answer) -> bool:
        """Whether `answer` names kinds patterns found, which the record may keep as they are.

        Other names may quote the turn, so the record keeps only declared labels. A
        person's answer on the lens names what they typed, not a kind.
        """
        for step in self.steps():
            try:
                judge = resolve(step)
            except Exception:  # noqa: BLE001 - a judge that does not build answered nothing
                continue
            if isinstance(judge, Patterns) and judge.label() == answer.answerer:
                return True
        return False

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary of the lens, for checks and logs."""
        return {
            "name": self.name,
            "kind": self.kind,
            "answer": self.answer.kind,
            "label": self.display_label,
            "subject": str(self.subject),
            "stages": [str(s) for s in self.stages],
            "sample": self.sample,
            "enabled": self.enabled,
            **self.answer.describe(self),
        }


class PatternLens(Lens):
    """Answered by regexes and keywords. It answers Entities, with spans a Mask can mask.

    `patterns` maps a label to its regexes; a plain list labels every hit with the
    lens's name. `keywords` maps a label to whole words. `verify` maps a label to a
    check its matches must pass, like a checksum. `flags` applies to patterns and
    keywords alike. Hit or miss, the answer is sure.
    """

    kind = "pattern"
    answer = Entities
    patterns: PatternTable = {}  # noqa: RUF012
    keywords: PatternTable = {}  # noqa: RUF012
    verify: Mapping[str, Verifier] = {}  # noqa: RUF012
    flags: int = re.IGNORECASE
    # Give each stretch of text one label: the first pattern to match it keeps it.
    exclusive: bool = False

    @cached_property
    def finder(self) -> Patterns:
        return Patterns(
            self.pattern_table(),
            keywords=self.keywords,
            verify=self.verify,
            flags=self.flags,
            exclusive=self.exclusive,
            miss_confidence=1.0,
        )

    def pattern_table(self) -> PatternTable:
        """The patterns to compile. A preset adds its own here."""
        return self.patterns

    def find(self, text: str) -> list[TextSpan]:
        """Every span the lens finds in `text`."""
        return self.finder.find(text, self.name)

    def validate(self) -> None:
        super().validate()
        if self.answer is not Entities:
            raise ImproperlyConfigured(f"Pattern lens {self.name!r} answers Entities only.")
        if not self.pattern_table() and not self.keywords:
            raise ImproperlyConfigured(
                f"Pattern lens {self.name!r} has no `patterns` or `keywords`."
            )
        try:
            self.finder  # noqa: B018 - compiling is the check
        except re.error as exc:
            raise ImproperlyConfigured(f"Pattern lens {self.name!r}: bad regex: {exc}") from None

    def steps(self) -> list[Any]:
        return [self.finder]


class CheckLens(Lens):
    """Answered by plain Python, in `check(text, view)`.

    By default `check` runs `validators` and answers a Flag: yes when any raises
    `Violation`, with their messages as the reason. A validator is `fn(text)` or
    `fn(text, view)`, or the name of one registered with `register_validator`.
    Override `check` for another answer, as `ToolErrorsLens` does. It returns the
    value, an Answer to give a reason or confidence too, or None to abstain. A
    sync `check` runs in a thread, so a slow one cannot block the event loop.
    """

    kind = "check"
    validators: list[Callable[..., Any] | str] = []  # noqa: RUF012
    timeout: float = DEFAULT_TIMEOUT

    def check(self, text: str, view: TurnContext) -> Any:
        from django_ai_sdk.oversight.registry import get_validator

        found: list[str] = []
        for validator in self.validators:
            run = get_validator(validator)
            try:
                run(text, view) if _takes_view(run) else run(text)
            except Violation as violation:
                found.append(str(violation.message))
        return Answer(value=bool(found), rationale="; ".join(found))

    def validate(self) -> None:
        from django_ai_sdk.oversight.registry import get_validator

        super().validate()
        overridden = type(self).check is not CheckLens.check
        if not overridden and not self.validators:
            raise ImproperlyConfigured(
                f"Check lens {self.name!r} has no `validators` and does not override `check`."
            )
        if not overridden and self.answer is not Flag:
            raise ImproperlyConfigured(
                f"Check lens {self.name!r}: validators answer a Flag; override `check` "
                f"to answer {self.answer.kind}."
            )
        for validator in self.validators:
            get_validator(validator)

    def steps(self) -> list[Any]:
        return [_Checker(self.timeout)]


class _Checker(Judge):
    """Asks a check lens's `check`, so it gets the same timeout and error handling as a judge."""

    name = "check"

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def answer(self, lens: Lens, view: TurnContext) -> Answer | None:
        assert isinstance(lens, CheckLens)
        text = view.text(lens.subject)
        if inspect.iscoroutinefunction(lens.check):
            result = await lens.check(text, view)
        else:
            result = await asyncio.to_thread(lens.check, text, view)
        if inspect.isawaitable(result):
            result = await result
        if result is None or isinstance(result, Answer):
            return result
        return Answer(value=result)


@cache
def _takes_view(validator: Callable[..., Any]) -> bool:
    """Whether a validator takes the turn as well as the text."""
    try:
        params = inspect.signature(validator).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2 or any(p.kind is p.VAR_POSITIONAL for p in params)


class JudgeLens(Lens):
    """Answered by judges, cheapest first (see rule 6 in the module docstring).

    `judges` are Judge instances, aliases from AI_SDK_OVERSIGHT["JUDGES"] (a
    trailing "?" makes one optional), or {"class": ..., **kwargs} configs.
    """

    kind = "judge"
    judges: list[Judge | str | dict[str, Any]] = []  # noqa: RUF012
    min_confidence: float = 0.75
    # Sent verbatim to a judge. Keep it to one question.
    question: str = ""
    # Extra instructions for a judge, such as scale anchors and edge cases.
    guidance: str = ""
    # Judge a tool call together with the question that led to it. Turn off for a
    # lens that should judge only the call's own content.
    reads_question: bool = True

    def usable_judges(self) -> list[Any]:
        """The judges to ask, minus optional aliases that are not defined.

        Packs mark their model judges optional so they still work without a model.
        """
        return [
            alias_of(spec) if is_optional(spec) else spec
            for spec in self.judges
            if not is_optional(spec) or alias_defined(alias_of(spec))
        ]

    def validate(self) -> None:
        super().validate()
        if not self.judges:
            raise ImproperlyConfigured(f"Judge lens {self.name!r} has no `judges`.")

    def steps(self) -> list[Any]:
        return self.usable_judges()

    def settles(self, answer: Answer) -> bool:
        return answer.ok and answer.confidence >= self.min_confidence

    def answer_model(self) -> type[BaseModel]:
        """The structured output a judge fills in."""
        return create_model(
            f"{type(self).__name__}Answer",
            rationale=(str, Field(description="One or two sentences: why.")),
            value=(
                self.answer.value_type(self),
                Field(description=self.answer.value_description(self)),
            ),
            confidence=(float, Field(ge=0, le=1, description="How sure you are, 0 to 1.")),
        )

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "question": self.question,
            "min_confidence": self.min_confidence,
        }


LENS_KINDS: dict[str, type[Lens]] = {cls.kind: cls for cls in (PatternLens, CheckLens, JudgeLens)}
# Every attribute some lens class reads. One declared on a lens that does not read
# it is refused (rule 2).
LENS_ATTRIBUTES = (
    frozenset(
        name
        for cls in (Lens, *LENS_KINDS.values())
        for name, value in vars(cls).items()
        if not name.startswith("_")
        and not callable(value)
        and not isinstance(value, property | cached_property)
    )
    | OPTIONS
)


def lens_from_dict(node: Mapping[str, Any]) -> Lens:
    """Build a lens from its JSON form, as settings, overlays and authored policies use.

    {"kind": "pattern" | "check" | "judge", "answer": "flag" | "score" | "choice" |
    "entities", ...attributes}. "preset" names a built-in lens instead (see
    `presets.PRESETS`). A check lens names its validators, which must be
    registered: JSON never names code to run.
    """
    from django_ai_sdk.oversight.presets import PRESETS

    node = dict(node)
    preset = node.pop("preset", "")
    if preset:
        if preset not in PRESETS:
            raise ImproperlyConfigured(f"Unknown lens preset {preset!r}; one of {sorted(PRESETS)}")
        cls, kind = PRESETS[preset], node.pop("kind", "")
        if kind and kind != cls.kind:
            raise ImproperlyConfigured(f"Preset {preset!r} is a {cls.kind} lens, not a {kind} one")
    else:
        cls, node = build_from_kind(LENS_KINDS, node, "lens kind")
    lens = cls(**node)
    lens.validate()
    return lens


__all__ = [
    "ANSWER_TYPES",
    "DEFAULT_STAGES",
    "LENS_KINDS",
    "AnswerType",
    "CheckLens",
    "Choice",
    "Entities",
    "Flag",
    "JudgeLens",
    "Lens",
    "PatternLens",
    "Score",
    "Violation",
    "lens_from_dict",
]
