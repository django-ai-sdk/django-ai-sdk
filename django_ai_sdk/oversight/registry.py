"""The registry of Policy classes, building one by name, and the cache.

An invalid policy is left out and reported by the system check, so one app's
typo does not stop the site from booting.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import time
import weakref
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import FAIL_MODES, STREAM_MODES, setting, validate_retention
from django_ai_sdk.oversight.policy import (
    Policy,
    ResolvedPolicy,
    default_fail_mode,
    default_stream_mode,
    policy_lenses,
    policy_rules,
    validate_choice,
    validate_policy,
    validate_rule,
)
from django_ai_sdk.oversight.rules import Mask
from django_ai_sdk.oversight.types import Capture, rank

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

_registry: dict[str, type[Policy]] = {}

# Rejected policies by name, with the reason. Read by the system check.
_invalid: dict[str, str] = {}
_duplicates: dict[str, list[str]] = {}


def _path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _declaration(cls: type) -> tuple[str, str]:
    """So a file imported under two module names counts once."""
    try:
        where = inspect.getfile(cls)
    except (TypeError, OSError):
        where = cls.__module__
    return where, cls.__qualname__


def register_policy[T: type[Policy]](cls: T) -> T:
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
    invalidate_policy(cls.name)
    return cls


def unregister_policy(name: str) -> None:
    """Later turns naming it find no policy and are blocked, unless FAIL_MODE is "open"."""
    _registry.pop(name, None)
    _invalid.pop(name, None)
    invalidate_policy(name)


def get_policy_class(name: str) -> type[Policy] | None:
    return _registry.get(name)


def policy_name(policy: type[Policy] | str) -> str:
    """An unregistered subclass is registered here, so `oversight = SupportPolicy` is enough."""
    if isinstance(policy, str):
        return policy
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
    return {name: list(paths) for name, paths in _duplicates.items()}


def reset_registry() -> None:
    """For tests."""
    _registry.clear()
    _invalid.clear()
    _duplicates.clear()
    invalidate_policy()


def build_policy(name: str, *, policy_class: type[Policy] | None = None) -> ResolvedPolicy:
    """Resolve `name` with its parents and the settings; `policy_class` builds a draft instead.

    Raises LookupError for an unknown policy and ImproperlyConfigured for a cycle.
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
    for rule in policy_rules(cls):
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
            resolved.stream_mode = validate_choice("stream_mode", policy.stream_mode, STREAM_MODES)
            resolved.sources["policy.stream_mode"] = name
        if policy.fail_mode is not None:
            resolved.fail_mode = validate_choice("fail_mode", policy.fail_mode, FAIL_MODES)
            resolved.sources["policy.fail_mode"] = name
    except ValueError as exc:
        raise ImproperlyConfigured(f"Policy {name!r}: {exc}") from None
    if policy.retention:
        resolved.retention = {**resolved.retention, **validate_retention(policy.retention)}
        resolved.sources["policy.retention"] = name
    return resolved


# The system check does not stop a turn, so _from_code checks every value again.

# name -> (ResolvedPolicy or None if it failed, ancestor names)
_cache: dict[str, tuple[ResolvedPolicy | None, frozenset[str]]] = {}
_broken: dict[str, str] = {}
# Each loader registers or unregisters its app's policies when awaited.
_loaders: list[Callable[[], Awaitable[None]]] = []
_loaded_at = -math.inf
# One lock per event loop: async_to_sync runs each call in a loop of its own, and a lock
# is bound to the loop that first waits on it.
_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


def add_policy_loader(loader: Callable[[], Awaitable[None]]) -> None:
    """Await `loader` before policy lookups. Idempotent."""
    global _loaded_at
    if loader not in _loaders:
        _loaders.append(loader)
        _loaded_at = -math.inf


async def run_policy_loaders() -> None:
    """At most once per RUNTIME_CACHE_SECONDS; concurrent lookups wait, so a burst reads once."""
    global _loaded_at
    if not _loaders or time.monotonic() - _loaded_at < float(setting("RUNTIME_CACHE_SECONDS")):
        return
    async with _locks.setdefault(asyncio.get_running_loop(), asyncio.Lock()):
        if time.monotonic() - _loaded_at < float(setting("RUNTIME_CACHE_SECONDS")):
            return
        started = time.monotonic()
        for loader in list(_loaders):
            try:
                await loader()
            except Exception:
                logger.opt(exception=True).warning("An oversight policy loader failed")
        _loaded_at = started


async def get_resolved_policy(name: str) -> ResolvedPolicy | None:
    """The effective policy, cached. None for an unknown or broken one: the caller decides."""
    await run_policy_loaders()
    cached = _cache.get(name)
    if cached:
        return cached[0]
    resolved = _load_policy(name)
    # Cache failures too, so a broken policy logs once, not once per turn.
    _cache[name] = (resolved, frozenset(ancestors(name)))
    return resolved


def _load_policy(name: str) -> ResolvedPolicy | None:
    try:
        resolved = build_policy(name)
    except Exception as exc:
        _broken[name] = str(exc) or type(exc).__name__
        if isinstance(exc, LookupError) and get_policy_class(name) is None:
            # Unknown; a missing parent instead means the policy is broken.
            logger.info("Oversight policy {!r} is unknown", name)
        else:
            logger.exception("Oversight policy {!r} cannot be loaded: {}", name, exc)
        return None
    _broken.pop(name, None)
    return resolved


def broken_reason(name: str) -> str:
    return _broken.get(name, "")


def ancestors(name: str) -> list[str]:
    """Nearest first. build_policy() reports cycles."""
    found: list[str] = []
    todo = list(getattr(get_policy_class(name), "extends", []) or [])
    while todo:
        parent = todo.pop(0)
        if parent in found or parent == name:
            continue
        found.append(parent)
        todo.extend(getattr(get_policy_class(parent), "extends", []) or [])
    return found


def invalidate_policy(name: str | None = None) -> None:
    """Clearing a policy also clears every policy that extends it."""
    global _loaded_at
    if name is None:
        _cache.clear()
        _loaded_at = -math.inf
        return
    for cached, (_, lineage) in list(_cache.items()):
        if cached == name or name in lineage:
            _cache.pop(cached, None)
