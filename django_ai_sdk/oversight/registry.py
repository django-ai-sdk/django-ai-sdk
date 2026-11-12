"""Process-wide registry of oversight policies, and of the validators check lenses name.

Each installed app's `oversight` module is imported on startup, so decorating a
Policy subclass there with `register_policy`, or a validator with
`register_validator`, is enough.

An invalid policy is left out and reported by the system check, so one app's
typo does not stop the site from booting. When two classes use the same name, the
first keeps it and the check fails until one is renamed.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.actions import Mask

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.oversight.rules import Rule

logger = get_logger(__name__)

_registry: dict[str, type[Policy]] = {}

# Rejected policies by name, with the reason. Read by the system check.
_invalid: dict[str, str] = {}

# Names used by more than one class, with every class path that uses them.
_duplicates: dict[str, list[str]] = {}

# Unregistered classes that stand in for registered ones in this context only, so a
# check can build a policy against a draft parent that other turns never see.
_drafts: ContextVar[Mapping[str, type[Policy]] | None] = ContextVar("policy_drafts", default=None)

# Validators a check lens may name, by name.
_validators: dict[str, Callable[..., Any]] = {}

# Subjects a Mask can rewrite: the question, the answer, or both.
MASKABLE = frozenset({"input", "output", "exchange"})


def _path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _declaration(cls: type) -> tuple[str, str]:
    """Identify where a class is declared, so a file imported under two module names counts once."""
    try:
        where = inspect.getfile(cls)
    except (TypeError, OSError):
        where = cls.__module__
    return where, cls.__qualname__


def register_policy[T: type[Policy]](cls: T) -> T:
    """Class decorator that registers a Policy, unless it is invalid."""
    try:
        validate_policy(cls)
    except ImproperlyConfigured as exc:
        _invalid[cls.name or cls.__name__] = str(exc)
        logger.warning("Oversight policy not registered: {}", exc)
        return cls

    _invalid.pop(cls.name, None)
    existing = _registry.get(cls.name)
    if existing is not None and _declaration(existing) != _declaration(cls):
        # An agent using this name could run either policy. The first keeps the
        # name, and check ai_sdk.oversight.E010 fails until one is renamed.
        clash = _duplicates.setdefault(cls.name, [_path(existing)])
        if _path(cls) not in clash:
            clash.append(_path(cls))
        logger.error("Oversight policy {!r} is declared by both {} and {}", cls.name, *clash[:2])
        return cls
    # A re-import of the same declaration replaces the old class.
    _registry[cls.name] = cls
    from django_ai_sdk.oversight.resolve import invalidate

    invalidate(cls.name)
    return cls


def unregister_policy(name: str) -> None:
    """Remove a policy registered at runtime.

    Later turns naming it find no policy and are refused, unless FAIL_MODE is "open".
    """
    _registry.pop(name, None)
    _invalid.pop(name, None)
    from django_ai_sdk.oversight.resolve import invalidate

    invalidate(name)


def validate_policy(cls: type[Policy]) -> None:
    """Raise ImproperlyConfigured unless every lens and rule is valid.

    Inherited lenses are checked when the policy resolves (ai_sdk.oversight.E002),
    because a parent may register after its child.
    """
    if not cls.name:
        raise ImproperlyConfigured(f"{cls.__name__} must set a non-empty `name`.")
    from django_ai_sdk.oversight.policy import policy_lenses

    lenses: dict[str, Lens] = {}
    for lens in policy_lenses(cls):
        lens.validate()
        if lens.name in lenses:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares lens {lens.name!r} twice.")
        lenses[lens.name] = lens
    rules: set[str] = set()
    for rule in cls.rules:
        if rule.name in rules:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares rule {rule.name!r} twice.")
        validate_rule(rule, lenses)
        rules.add(rule.name)


def validate_rule(rule: Rule, lenses: Mapping[str, Lens]) -> None:
    """Raise ImproperlyConfigured when `rule` does nothing, or a Mask reads a lens that cannot mask.

    Only an Entities lens says where to mask, and only the question and the answer
    are masked. A Mask over a Flag, Choice or Score lens, or over a lens on sources
    or tool calls, could never mask anything. Lenses missing from `lenses` are
    skipped here: an inherited one is not known until the policy resolves, which
    refuses a rule naming a lens the policy does not have.
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


def get_policy_class(name: str) -> type[Policy] | None:
    return (_drafts.get() or {}).get(name) or _registry.get(name)


@contextmanager
def drafted(classes: Mapping[str, type[Policy]]) -> Generator[None]:
    """Look up `classes` before the registry inside this block, in this context only."""
    token = _drafts.set({**(_drafts.get() or {}), **classes})
    try:
        yield
    finally:
        _drafts.reset(token)


def policy_name(policy: type[Policy] | str) -> str:
    """Return the name of `policy`, a name or a Policy subclass.

    An unregistered subclass is registered here, so `oversight = SupportPolicy` on
    an agent is enough.
    """
    if isinstance(policy, str):
        return policy
    from django_ai_sdk.oversight.policy import Policy

    if not (isinstance(policy, type) and issubclass(policy, Policy)):
        raise TypeError(f"Expected a Policy subclass or a policy name, got {policy!r}")
    known = _registry.get(policy.name)
    if known is None and policy.name not in _invalid:
        register_policy(policy)
    elif known is not None and known is not policy and _declaration(known) == _declaration(policy):
        # Same declaration imported again. The copy in use wins.
        register_policy(policy)
    return policy.name


def get_policies() -> dict[str, type[Policy]]:
    return dict(_registry)


def get_invalid_policies() -> dict[str, str]:
    return dict(_invalid)


def get_duplicate_policies() -> dict[str, list[str]]:
    """Names used by more than one Policy class, with their class paths."""
    return {name: list(paths) for name, paths in _duplicates.items()}


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


def reset_registry() -> None:
    """Clear the registry and every cached resolution. For tests."""
    from django_ai_sdk.oversight.resolve import invalidate

    _registry.clear()
    _invalid.clear()
    _duplicates.clear()
    invalidate()


__all__ = [
    "drafted",
    "get_duplicate_policies",
    "get_invalid_policies",
    "get_policies",
    "get_policy_class",
    "get_validator",
    "get_validators",
    "policy_name",
    "register_policy",
    "register_validator",
    "reset_registry",
    "unregister_policy",
    "validate_policy",
    "validate_rule",
]
