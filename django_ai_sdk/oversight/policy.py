"""Policies: an app's lenses and rules, layered over the built-in packs.

    @register_policy
    class Support(Policy):
        name = "support"
        extends = ["security", "quality"]
        lenses = [Topic()]
        rules = [Rule("hr_to_people", when=L("topic") == "hr", then=Queue("hr"))]

The policy class wins over the packs it extends. Every value records where it
came from; nothing changes a lens or rule declared in code at runtime.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import (
    STREAM_MODES,
    setting,
)
from django_ai_sdk.oversight.lenses import Lens
from django_ai_sdk.oversight.rules import Mask, Rule
from django_ai_sdk.oversight.types import Capture, Mode, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = get_logger(__name__)


class Policy:
    """A named bundle of lenses and rules. Subclass it in an app's `oversight.py`."""

    name: str = ""
    label: str = ""
    description: str = ""
    # Packs or policies to inherit lenses and rules from. A lens or rule declared
    # here replaces an inherited one with the same name.
    extends: list[str] = []  # noqa: RUF012
    # A LensPack, RulePack or another policy's list is flattened in place, one level.
    lenses: Sequence[Lens | type[Lens] | Sequence[Any]] = []  # noqa: RUF012
    rules: Sequence[Rule | Sequence[Any]] = []  # noqa: RUF012
    # None inherits. The CAPTURE setting is always the ceiling.
    capture: Capture | None = None
    # "auto" or "live". None inherits.
    stream_mode: str | None = None
    # "closed" or "open"; None inherits FAIL_MODE. "open" suits an observe-only
    # policy: a failed lens never stops a turn.
    fail_mode: str | None = None
    # Days to keep turns, e.g. {"content": 90, "metadata": 730}; missing keys use RETENTION.
    retention: dict[str, int] = {}  # noqa: RUF012
    # "code", "pack:<name>" or "authored".
    origin: str = "code"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.lenses = flatten(cls.lenses)
        cls.rules = flatten(cls.rules)


def policy_lenses(cls: type[Policy]) -> list[Lens]:
    return [lens() if isinstance(lens, type) else lens for lens in flatten(cls.lenses)]


def policy_rules(cls: type[Policy]) -> list[Rule]:
    return flatten(cls.rules)


class LensPack(list[Lens | type[Lens]]):
    """Lenses taken by name from a pack's `lenses`, in the order named; all without names.

    List it in a policy's `lenses`, which takes its lenses in its place::

        lenses = [LensPack(SecurityPack.lenses, "injection_question", "pii_question"), Topic]

    `overrides` set attributes such as `stages` on copies; the pack keeps its own.
    """

    def __init__(
        self, lenses: Iterable[Lens | type[Lens] | Sequence[Any]], *names: str, **overrides: Any
    ) -> None:
        taken = _take(flatten(lenses), names)
        super().__init__(_with_overrides(lens, overrides) for lens in taken)


class RulePack(list[Rule]):
    """Rules taken by name from a pack's `rules`, in the order named; all without names.

    List it in a policy's `rules`, which takes its rules in its place::

        rules = [
            SupportPolicy.rules,
            RulePack(SecurityPack.rules, "pii_in_answer", mode=Mode.ENFORCE),
        ]

    The rules are copies, so setting `mode` or `notice` does not change the pack.
    """

    def __init__(
        self,
        rules: Iterable[Rule | Sequence[Any]],
        *names: str,
        mode: Mode | str | None = None,
        notice: str | None = None,
    ) -> None:
        taken = [copy.copy(rule) for rule in _take(flatten(rules), names)]
        for rule in taken:
            if mode is not None:
                rule.mode = Mode(mode)
            if notice is not None:
                rule.notice = notice
        super().__init__(taken)


def _take[T: (Lens | type[Lens], Rule)](items: Iterable[T], names: tuple[str, ...]) -> list[T]:
    by_name = {item.name: item for item in items}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ImproperlyConfigured(f"Cannot take {missing}: the choices are {sorted(by_name)}")
    return [by_name[name] for name in names] if names else list(by_name.values())


def _with_overrides(lens: Lens | type[Lens], overrides: Mapping[str, Any]) -> Lens | type[Lens]:
    if isinstance(lens, type):
        return lens(**overrides) if overrides else lens
    if not overrides:
        return copy.copy(lens)
    # Rebuilt from what was set on it, so nothing worked out from the old values stays.
    declared = {
        key: value
        for key, value in vars(lens).items()
        if not isinstance(getattr(type(lens), key, None), cached_property)
    }
    return type(lens)(**{**declared, **overrides})


def flatten[T: (Lens | type[Lens], Rule)](items: Iterable[T | Sequence[Any]]) -> list[T]:
    """Replace every list, tuple or pack in `items` by its items, one level deep."""
    flat: list[T] = []
    for item in items:
        if isinstance(item, Sequence):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


# Subjects a Mask can rewrite: the question, the answer, or both.
MASKABLE = frozenset({"input", "output", "exchange"})


def validate_policy(cls: type[Policy]) -> None:
    """Raise ImproperlyConfigured unless every lens and rule is valid.

    Inherited lenses are checked at resolve time (E002): a parent may register later.
    """
    if not cls.name:
        raise ImproperlyConfigured(f"{cls.__name__} must set a non-empty `name`.")
    lenses: dict[str, Lens] = {}
    for lens in policy_lenses(cls):
        lens.validate()
        if lens.name in lenses:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares lens {lens.name!r} twice.")
        lenses[lens.name] = lens
    rules: set[str] = set()
    for rule in policy_rules(cls):
        if rule.name in rules:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares rule {rule.name!r} twice.")
        validate_rule(rule, lenses)
        rules.add(rule.name)


def validate_rule(rule: Rule, lenses: Mapping[str, Lens]) -> None:
    """Raise when `rule` does nothing, or a Mask reads a lens that cannot mask.

    Only an entities lens on the question or answer says where to mask. Lenses
    missing from `lenses` (inherited ones) are checked when the policy resolves.
    """
    if not rule.then:
        raise ImproperlyConfigured(f"Rule {rule.name!r} does nothing: give it `then`.")
    for action in rule.then:
        if not isinstance(action, Mask):
            continue
        for name in sorted(set(action.lenses) or rule.lenses()):
            lens = lenses.get(name)
            if lens is not None and lens.subject not in MASKABLE:
                hint = (
                    "a tool call's arguments are never masked: block the call (Block) or take "
                    "the tool away (RestrictTools). Tool results are masked by the gate's mask "
                    "rules."
                    if lens.subject == "tool_call"
                    else "only the question and the answer are masked."
                )
                raise ImproperlyConfigured(
                    f"Rule {rule.name!r} masks what {lens.subject} lens {name!r} found, but {hint}"
                )
            if lens is not None and lens.answer.kind != "entities":
                raise ImproperlyConfigured(
                    f"Rule {rule.name!r} masks what {lens.answer.kind} lens {name!r} found, but "
                    "only an entities lens says where: name the lenses to mask, "
                    "Mask(lenses=[...])."
                )


@dataclass
class ResolvedPolicy:
    """A policy with packs and settings applied; this is what runs."""

    name: str
    label: str = ""
    description: str = ""
    lenses: dict[str, Lens] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)
    capture: Capture = Capture.FULL
    stream_mode: str = "auto"
    fail_mode: str = "closed"
    retention: dict[str, int] = field(default_factory=dict)
    # Who declared each value, e.g. "lens:pii" -> "security", "policy.capture" -> "settings".
    sources: dict[str, str] = field(default_factory=dict)
    origin: str = "code"
    extends: list[str] = field(default_factory=list)

    def changes_answer_at_finish(self) -> list[str]:
        """Enforced rules that may mask or withhold the answer at finish."""
        finish = self.active_lenses(Stage.FINISH)
        reads = {
            "block": {lens.name for lens in finish},
            "mask": {
                lens.name for lens in finish if lens.subject in {Subject.ANSWER, Subject.EXCHANGE}
            },
        }
        return sorted(
            rule.name
            for rule in self.rules.values()
            if rule.mode is Mode.ENFORCE
            and any(rule.lenses() & reads.get(action.kind, set()) for action in rule.then)
        )

    @property
    def fails_closed(self) -> bool:
        return self.fail_mode != "open"

    def holds_answer(self) -> bool:
        return self.stream_mode == "auto" and bool(self.changes_answer_at_finish())

    def active_lenses(self, stage: Stage, subject: Subject | None = None) -> list[Lens]:
        return [
            lens
            for lens in self.lenses.values()
            if lens.enabled
            and stage in lens.stages
            and (subject is None or lens.subject is subject)
        ]

    def active_rules(self) -> list[Rule]:
        return [rule for rule in self.rules.values() if rule.mode is not Mode.OFF]

    def has_stage(self, stage: Stage) -> bool:
        return any(stage in lens.stages for lens in self.lenses.values() if lens.enabled)


def validate_choice(key: str, value: Any, choices: tuple[str, ...]) -> str:
    """A typo such as "atuo" must not silently stream live."""
    if value not in choices:
        raise ValueError(f"{key} {value!r} is not one of {list(choices)}")
    return str(value)


def default_stream_mode() -> str:
    """Unlike FAIL_MODE, a typo is a configuration error."""
    try:
        return validate_choice("stream_mode", setting("STREAM_MODE"), STREAM_MODES)
    except ValueError as exc:
        raise ImproperlyConfigured(f'AI_SDK_OVERSIGHT["STREAM_MODE"]: {exc}') from None


def default_fail_mode() -> str:
    """Anything but "open", typos included, means "closed"."""
    return "open" if setting("FAIL_MODE") == "open" else "closed"
