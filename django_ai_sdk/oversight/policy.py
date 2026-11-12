"""Policies: an app's lenses and rules, layered over the built-in packs.

    @register_policy
    class Support(Policy):
        name = "support"
        extends = ["security", "quality"]
        lenses = [Topic()]
        rules = [Rule("hr_to_people", when=L("topic") == "hr", then=Queue("hr"))]

Values are layered in this order, later winning: the extended packs, then the
policy class. Every value records where it came from. Nothing changes a lens or
rule declared in code at runtime.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.actions import Mask
from django_ai_sdk.oversight.conf import (
    FAIL_MODES,
    STREAM_MODES,
    setting,
    validate_retention,
)
from django_ai_sdk.oversight.lenses import Lens
from django_ai_sdk.oversight.registry import get_policy_class, validate_rule
from django_ai_sdk.oversight.rules import Rule
from django_ai_sdk.oversight.types import Capture, Mode, Stage, Subject, rank

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


class Policy:
    """A named bundle of lenses and rules. Subclass it in an app's `oversight.py`."""

    name: str = ""
    label: str = ""
    description: str = ""
    # Packs or policies to inherit lenses and rules from. A lens or rule declared
    # here replaces an inherited one with the same name.
    extends: list[str] = []  # noqa: RUF012
    lenses: list[Lens | type[Lens]] = []  # noqa: RUF012
    rules: list[Rule] = []  # noqa: RUF012
    # None inherits. The CAPTURE setting is always the ceiling.
    capture: Capture | None = None
    # "auto" or "live". None inherits.
    stream_mode: str | None = None
    # "closed" or "open". None inherits the FAIL_MODE setting. Use "open" for a
    # policy that only observes, so a failed lens never stops a turn.
    fail_mode: str | None = None
    # Days to keep this policy's turns, e.g. {"content": 90, "metadata": 730}.
    # Missing keys come from the RETENTION setting.
    retention: dict[str, int] = {}  # noqa: RUF012
    # Where this policy's values come from: "code", "pack:<name>" or "authored".
    origin: str = "code"


def policy_lenses(cls: type[Policy]) -> list[Lens]:
    """The lenses `cls` declares, a lens class instantiated."""
    return [lens() if isinstance(lens, type) else lens for lens in cls.lenses]


def pick[T: (Lens | type[Lens], Rule)](
    items: Iterable[T],
    *names: str,
    mode: Mode | str | None = None,
    notice: str | None = None,
) -> list[T]:
    """Pick lenses or rules by name from a pack's `lenses` or `rules`, in that order.

    Use this to take only what you need instead of extending a whole pack::

        lenses = pick(SecurityPack.lenses, "prompt_injection", "pii_input")
        rules = pick(SecurityPack.rules, "block_injection", "mask_pii_in_question")

    Items are copies, so setting `mode` or `notice` does not change the pack.
    """
    by_name = {item.name: item for item in items}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ImproperlyConfigured(f"Cannot pick {missing}: the choices are {sorted(by_name)}")
    picked = [copy.copy(by_name[name]) for name in names]
    if mode is not None or notice is not None:
        for item in picked:
            if not isinstance(item, Rule):
                raise TypeError("pick(mode=..., notice=...) applies to rules only")
            if mode is not None:
                item.mode = Mode(mode)
            if notice is not None:
                item.notice = notice
    return picked


@dataclass
class ResolvedPolicy:
    """A policy with packs and settings applied. This is what runs."""

    name: str
    label: str = ""
    description: str = ""
    lenses: dict[str, Lens] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)
    capture: Capture = Capture.FULL
    stream_mode: str = "auto"
    fail_mode: str = "closed"
    retention: dict[str, int] = field(default_factory=dict)
    # The policy that declared each lens, rule and policy value, e.g. "lens:pii" -> "security",
    # or "policy.capture" -> "settings" when the CAPTURE setting capped it.
    sources: dict[str, str] = field(default_factory=dict)
    # "code", "pack:<name>" or "authored".
    origin: str = "code"
    extends: list[str] = field(default_factory=list)

    def changes_answer_at_finish(self) -> list[str]:
        """Names of enforced rules that may mask or withhold the answer at finish.

        A block withholds the answer whatever its lenses read. A mask changes it
        only through lenses that read the answer.
        """
        finish = self.active_lenses(Stage.FINISH)
        reads = {
            "block": {lens.name for lens in finish},
            "mask": {
                lens.name for lens in finish if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}
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
        """Whether a lens that failed counts as matching this policy's protective rules."""
        return self.fail_mode != "open"

    def holds_answer(self) -> bool:
        """Whether to hold the answer until finish. Only in "auto" mode, and only when needed."""
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


def build(name: str, *, policy_class: type[Policy] | None = None) -> ResolvedPolicy:
    """Resolve `name` from the registry, with its parents and the settings.

    `policy_class` builds that class instead of the registered one, to check a draft
    before registering it. Raises LookupError for an unknown policy and
    ImproperlyConfigured for a cycle.
    """
    resolved = _from_code(name, (), policy_class=policy_class)
    ceiling = Capture(setting("CAPTURE"))
    if rank(tuple(Capture), resolved.capture) > rank(tuple(Capture), ceiling):
        resolved.capture = ceiling
        resolved.sources["policy.capture"] = "settings"
    return resolved


def _inherit(resolved: ResolvedPolicy, parent: ResolvedPolicy) -> None:
    resolved.lenses.update(parent.lenses)
    resolved.rules.update(parent.rules)
    resolved.sources.update(parent.sources)
    # Only carry over values the parent set itself. Values from settings already match.
    for key in ("capture", "stream_mode", "fail_mode"):
        if parent.sources.get(f"policy.{key}"):
            setattr(resolved, key, getattr(parent, key))
    if parent.sources.get("policy.retention"):
        resolved.retention = {**resolved.retention, **parent.retention}


def _from_code(
    name: str, chain: tuple[str, ...], *, policy_class: type[Policy] | None = None
) -> ResolvedPolicy:
    if name in chain:
        raise ImproperlyConfigured(f"Policy {name!r} extends itself: {' -> '.join((*chain, name))}")
    cls = policy_class or get_policy_class(name)
    if cls is None:
        raise LookupError(f"No oversight policy {name!r}")
    policy = cls()
    resolved = ResolvedPolicy(
        name=policy.name,
        label=policy.label,
        description=policy.description,
        capture=Capture(setting("CAPTURE")),
        stream_mode=default_stream_mode(),
        fail_mode=default_fail_mode(),
        retention=validate_retention(setting("RETENTION")),
        origin=cls.origin,
        extends=list(policy.extends),
    )
    for parent_name in policy.extends:
        _inherit(resolved, _from_code(parent_name, (*chain, name)))
    for lens in policy_lenses(cls):
        resolved.lenses[lens.name] = copy.copy(lens)
        resolved.sources[f"lens:{lens.name}"] = name
    for rule in cls.rules:
        resolved.rules[rule.name] = copy.copy(rule)
        resolved.sources[f"rule:{rule.name}"] = name
    for rule in resolved.rules.values():
        # Check again with inherited lenses, which registration could not see.
        masked = {
            lens for action in rule.then if isinstance(action, Mask) for lens in action.lenses
        }
        unknown = sorted((rule.lenses() | masked) - set(resolved.lenses))
        try:
            if unknown:
                raise ImproperlyConfigured(f"Rule {rule.name!r} reads unknown lenses {unknown}.")
            validate_rule(rule, resolved.lenses)
        except ImproperlyConfigured as exc:
            raise ImproperlyConfigured(f"Policy {name!r}: {exc}") from None
    if policy.capture is not None:
        resolved.capture = Capture(policy.capture)
        resolved.sources["policy.capture"] = name
    try:
        if policy.stream_mode is not None:
            resolved.stream_mode = _choice("stream_mode", policy.stream_mode, STREAM_MODES)
            resolved.sources["policy.stream_mode"] = name
        if policy.fail_mode is not None:
            resolved.fail_mode = _choice("fail_mode", policy.fail_mode, FAIL_MODES)
            resolved.sources["policy.fail_mode"] = name
    except ValueError as exc:
        raise ImproperlyConfigured(f"Policy {name!r}: {exc}") from None
    if policy.retention:
        resolved.retention = {**resolved.retention, **validate_retention(policy.retention)}
        resolved.sources["policy.retention"] = name
    return resolved


# The system check reports bad settings (conf.problems), but does not stop a turn.
# So every value that reaches a built policy is checked again here: the settings
# it inherits, and a code or authored policy's own.


def _choice(key: str, value: Any, choices: tuple[str, ...]) -> str:
    """Validate a policy option, so a typo such as "atuo" does not silently stream live."""
    if value not in choices:
        raise ValueError(f"{key} {value!r} is not one of {list(choices)}")
    return str(value)


def default_stream_mode() -> str:
    """The STREAM_MODE setting. Unlike FAIL_MODE, a typo is a configuration error."""
    try:
        return _choice("stream_mode", setting("STREAM_MODE"), STREAM_MODES)
    except ValueError as exc:
        raise ImproperlyConfigured(f'AI_SDK_OVERSIGHT["STREAM_MODE"]: {exc}') from None


def default_fail_mode() -> str:
    """The FAIL_MODE setting. Anything but "open", typos included, means "closed"."""
    return "open" if setting("FAIL_MODE") == "open" else "closed"


__all__ = [
    "Policy",
    "ResolvedPolicy",
    "build",
    "default_fail_mode",
    "pick",
    "policy_lenses",
]
