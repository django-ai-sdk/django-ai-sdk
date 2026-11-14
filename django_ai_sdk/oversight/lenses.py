"""Lenses. A lens is one typed question about a turn, declared by how it is answered.

    class Competitors(PatternLens):       # code finds spans, which a Mask rule can mask
        name = "competitors"
        keywords = {"acme": ["Acme"], "globex": ["Globex"]}

    class EvenAmount(CheckLens):          # plain Python, like a Django validator
        name = "even_amount"
        validators = [validate_even]      # raise Violation("...") to say yes

    class PromptInjection(JudgeLens):     # judges, cheapest first
        name = "injection_question"
        judges = [PatternJudge(INJECTION_PATTERNS), Maybe("classifier"), Maybe("judge")]

The rules:

1. A lens is a PatternLens, a CheckLens or a JudgeLens, and only answers. What to
   do about the answer is a `Rule`, declared on the policy.
2. An attribute lives on the class whose behaviour it changes: `patterns`,
   `keywords` and `verify` on PatternLens; `validators` on CheckLens; `judges`,
   `min_confidence` and what a judge reads (`question`, `guidance`,
   `reads_question`) on JudgeLens. An answer option (`yes`, `labels`, `low`, ...)
   belongs to the answer types that read it.
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
   question, so a `PatternJudge` hit is sure (1) and a miss is not (0.5).
7. A judge alias ending in "?" is optional: the lens skips it when the deployment
   does not define it, and system check E007 warns about an enforced rule
   left without a judge.

Inline, pass the same attributes as keyword arguments::

    JudgeLens(name="topic", answer=Choice, labels=["billing", "hr", "other"], judges=["judge"])

The built-in lenses at the end of this module are ready to subclass. Pattern
presets find spans, so a Mask rule can mask what they find::

    class InputPII(PIILens):
        name = "pii_question"
        kinds = [*DEFAULT_PII_KINDS, "bsn"]

Check presets answer structural facts about the turn in plain Python: its length,
its language, empty retrieval and failed tools. None of them calls a model, so
they add no cost and send no data out. `RefusalLens` is a judge lens: the shipped
patterns first, then `Maybe("classifier")` when one is defined.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import re
from collections.abc import Mapping
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.oversight.judges.base import (
    DEFAULT_TIMEOUT,
    Judge,
    Maybe,
    alias_defined,
    judge_from_json,
)
from django_ai_sdk.oversight.judges.patterns import PatternJudge
from django_ai_sdk.oversight.patterns import (
    DEFAULT_PII_KINDS,
    PII_PATTERNS,
    PII_VERIFY,
    REFUSAL_PATTERNS,
    SECRET_PATTERNS,
    WORD,
)
from django_ai_sdk.oversight.rules import build_from_kind
from django_ai_sdk.oversight.types import Answer, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.utils.functional import Promise

    from django_ai_sdk.oversight.patterns import PatternTable, Verifier
    from django_ai_sdk.oversight.types import TurnContext

# The subjects each stage reads. A lens never runs at a stage that skips its subject.
STAGE_SUBJECTS: dict[Stage, frozenset[Subject]] = {
    Stage.GATE: frozenset({Subject.QUESTION}),
    Stage.TOOL: frozenset({Subject.TOOL_CALL}),
    Stage.FINISH: frozenset({Subject.ANSWER, Subject.EXCHANGE, Subject.SOURCES}),
    Stage.REFLECT: frozenset({Subject.QUESTION, Subject.ANSWER, Subject.EXCHANGE, Subject.SOURCES}),
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
    subject: Subject = Subject.QUESTION
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
        if misplaced := sorted(
            key for key in OPTIONS - set(self.answer.options) if hasattr(self, key)
        ):
            raise ValueError(
                f"Lens {self.name!r} answers {self.answer.kind}, which does not read {misplaced}"
            )
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
        if skipped := [str(s) for s in self.stages if self.subject not in STAGE_SUBJECTS[s]]:
            raise ImproperlyConfigured(
                f"Lens {self.name!r} reads the {self.subject} at {', '.join(skipped)}, "
                "which never reads it."
            )
        if not 0 <= self.sample <= 1:
            raise ImproperlyConfigured(f"Lens {self.name!r}: `sample` must be 0 to 1.")
        self.answer.validate(self)

    def steps(self) -> list[Any]:
        """What answers the lens, in order: judges, or specs for `judges.base.ask_judge`."""
        raise NotImplementedError

    def is_final(self, answer: Answer) -> bool:
        """Whether `answer` ends the lens's chain of steps."""
        return answer.ok

    def reports_kinds(self, answer: Answer) -> bool:
        """Whether `answer` names kinds patterns found, which the record may keep as they are.

        Other names may quote the turn, so the record keeps only declared labels. A
        person's answer on the lens names what they typed, not a kind.
        """
        return any(
            isinstance(step, PatternJudge) and step.label() == answer.answerer
            for step in self.steps()
        )

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

    @cached_property
    def finder(self) -> PatternJudge:
        return PatternJudge(
            self.pattern_table(),
            keywords=self.keywords,
            verify=self.verify,
            flags=self.flags,
            miss_confidence=1.0,
        )

    def pattern_table(self) -> PatternTable:
        """The patterns to compile. A preset adds its own here."""
        return self.patterns

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


# Validators a check lens may name, by name.
_validators: dict[str, Callable[..., Any]] = {}


def register_validator(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that lets check lenses, and lenses from JSON, name a validator::

    @register_validator("even_amount")
    def validate_even(text):
        ...
    """

    def register(validator: Callable[..., Any]) -> Callable[..., Any]:
        _validators[name] = validator
        return validator

    return register


def get_validator(validator: Callable[..., Any] | str) -> Callable[..., Any]:
    """The validator itself, or the one registered under a name.

    Raises ImproperlyConfigured for an unknown name. A name is never imported as a
    path, so JSON cannot run code a developer did not register.
    """
    if callable(validator):
        return validator
    if validator not in _validators:
        raise ImproperlyConfigured(
            f"No validator {validator!r}; register it with register_validator."
        )
    return _validators[validator]


def get_validators() -> dict[str, Callable[..., Any]]:
    """Every validator registered in code."""
    return dict(_validators)


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
        found: list[str] = []
        for validator in self.validators:
            run = get_validator(validator)
            try:
                run(text, view) if _takes_view(run) else run(text)
            except Violation as violation:
                found.append(str(violation.message))
        return Answer(value=bool(found), rationale="; ".join(found))

    def validate(self) -> None:
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

    `judges` are Judge instances, aliases from AI_SDK_OVERSIGHT["JUDGES"] (wrap one
    in `Maybe` to skip it when undefined), or {"class": ..., **kwargs} configs.
    """

    kind = "judge"
    judges: list[Judge | str | Maybe | dict[str, Any]] = []  # noqa: RUF012
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
            spec.alias if isinstance(spec, Maybe) else spec
            for spec in self.judges
            if not isinstance(spec, Maybe) or alias_defined(spec.alias)
        ]

    def validate(self) -> None:
        super().validate()
        if not self.judges:
            raise ImproperlyConfigured(f"Judge lens {self.name!r} has no `judges`.")
        if not 0 <= self.min_confidence <= 1:
            raise ImproperlyConfigured(
                f"Judge lens {self.name!r}: `min_confidence` must be 0 to 1."
            )

    def steps(self) -> list[Any]:
        return self.usable_judges()

    def is_final(self, answer: Answer) -> bool:
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


def lens_from_dict(node: Mapping[str, Any]) -> Lens:
    """Build a lens from its JSON form, as settings and authored policies use.

    {"kind": "pattern" | "check" | "judge", "answer": "flag" | "score" | "choice" |
    "entities", ...attributes}. "preset" names a built-in lens instead (see
    `PRESETS`). A check lens names its validators, which must be
    registered: JSON never names code to run.
    """
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
    if isinstance(node.get("judges"), list):
        node["judges"] = [judge_from_json(spec) for spec in node["judges"]]
    lens = cls(**node)
    lens.validate()
    return lens


# Built-in lenses: subclass one, name it, and it is ready to use. `PRESETS` names
# each one for the JSON form ({"kind": "pattern", "preset": "pii"}).

# Don't guess the language of texts shorter than this many words.
MIN_LANGUAGE_WORDS = 3


class PIILens(PatternLens):
    """Personal data by kind (`patterns.PII_PATTERNS`), checked by checksum where it can be.

    `kinds` defaults to the kinds that mean the same in every country
    (DEFAULT_PII_KINDS): email, iban (mod-97 checked), credit_card (Luhn checked),
    phone (international format) and ip_address. Name country kinds to add them
    (COUNTRY_PII_KINDS): bsn (11-check), phone_nl and postcode_nl. Your own
    `patterns` come after the kinds, without a checksum.
    """

    name = "pii"
    kinds: list[str] = list(DEFAULT_PII_KINDS)  # noqa: RUF012
    verify = PII_VERIFY
    flags = 0

    def pattern_table(self) -> PatternTable:
        # No kinds means the default kinds.
        kinds = self.kinds or DEFAULT_PII_KINDS
        unknown = sorted(set(kinds) - set(PII_PATTERNS))
        if unknown:
            raise ImproperlyConfigured(
                f"Unknown personal data kinds {unknown}; known: {sorted(PII_PATTERNS)}"
            )
        own = self.patterns if isinstance(self.patterns, Mapping) else {"": list(self.patterns)}
        return {**{kind: PII_PATTERNS[kind] for kind in kinds}, **own}


class SecretsLens(PatternLens):
    """API keys, tokens and private keys (`patterns.SECRET_PATTERNS`), as in a pasted message."""

    name = "secrets"
    patterns = SECRET_PATTERNS
    flags = 0


class RefusalLens(JudgeLens):
    """Whether the answer declines. Many refusals signal poor quality, not safety."""

    name = "refusal"
    label = "Refusal"
    question = "Does the assistant decline to answer?"
    yes = "Declines, deflects or says it cannot help."
    no = "Answers, even partly."
    subject = Subject.ANSWER
    # The shipped patterns answer a plain refusal, so this works without a classifier.
    judges = [PatternJudge(REFUSAL_PATTERNS), Maybe("classifier")]  # noqa: RUF012


class LengthLens(CheckLens):
    """Whether the subject is longer than `max_chars`."""

    name = "length"
    max_chars: int = 0

    async def check(self, text: str, view: TurnContext) -> Answer:
        return Answer(value=len(text) > self.max_chars, rationale=f"{len(text)} characters")

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.max_chars, int) or self.max_chars < 1:
            raise ImproperlyConfigured(f"Length lens {self.name!r} needs `max_chars` of 1 or more.")


# The most frequent words per language, enough to tell them apart.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and is are was to of in that it for you with on this have be not".split()),
    "nl": frozenset("de het een en is van dat niet je ik op te zijn met voor maar wat".split()),
    "de": frozenset("der die das und ist nicht ich sie es zu mit den von ein eine auf".split()),
    "fr": frozenset("le la les et est un une des que pas je vous il de pour dans sur".split()),
    "es": frozenset("el la los las y es un una que no de en por para con se lo".split()),
}


def guess_language(text: str) -> tuple[str, float]:
    """Guess (language code, confidence) from stopwords. ("", 0) when unsure."""
    words = [word.lower() for word in WORD.findall(text)]
    if len(words) < MIN_LANGUAGE_WORDS:
        return "", 0.0
    counts = {lang: sum(word in stop for word in words) for lang, stop in _STOPWORDS.items()}
    lang, best = max(counts.items(), key=lambda item: item[1])
    if not best:
        return "", 0.0
    return lang, round(best / sum(counts.values()), 3)


class LanguageLens(CheckLens):
    """Whether the answer is in another language than the question. Abstains when unsure.

    For the subject's language as a Choice, write a CheckLens over `guess_language`.
    """

    name = "language"
    subject = Subject.EXCHANGE

    async def check(self, text: str, view: TurnContext) -> Answer | None:
        asked, _ = guess_language(view.input)
        replied, confidence = guess_language(view.output)
        if not asked or not replied:
            return None
        return Answer(
            value=asked != replied,
            confidence=confidence,
            rationale=f"asked in {asked}, answered in {replied}",
        )


class EmptyRetrievalLens(CheckLens):
    """Whether the turn called a tool but found no sources."""

    name = "empty_retrieval"
    label = "Nothing retrieved"
    subject = Subject.SOURCES

    def applies(self, view: TurnContext) -> bool:
        # Runs on any tool call: an empty search leaves no grounding to judge.
        return bool(view.tool_calls)

    async def check(self, text: str, view: TurnContext) -> Answer:
        found = len(view.fresh_sources())
        return Answer(value=bool(view.tool_calls) and not found, rationale=f"{found} sources")


class ToolErrorsLens(CheckLens):
    """The share of the turn's tool calls that failed. `L("tool_errors") > 0`: any did."""

    name = "tool_errors"
    label = "Tool failures"
    answer = Score
    subject = Subject.EXCHANGE

    async def check(self, text: str, view: TurnContext) -> Answer | None:
        if not view.tool_calls:
            return None
        failed = sum(call.error for call in view.tool_calls)
        return Answer(value=failed / len(view.tool_calls), rationale=f"{failed} failed")


PRESETS: dict[str, type[Lens]] = {
    cls.name: cls
    for cls in (
        PIILens,
        SecretsLens,
        RefusalLens,
        LengthLens,
        LanguageLens,
        EmptyRetrievalLens,
        ToolErrorsLens,
    )
}
