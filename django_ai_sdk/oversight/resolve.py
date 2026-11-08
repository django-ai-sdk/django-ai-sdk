"""Resolve the policy a turn runs: code plus the runtime overlay, cached.

`aget_policy(name)` builds the policy with the installed `OverlaySource` and
caches it for the source's `ttl`. Core's source is empty. Apps that switch rule modes
at runtime install their own with `set_overlay_source`. Apps that register whole
policies at runtime add a loader with `add_policy_loader`.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.policy import ResolvedPolicy, build
from django_ai_sdk.oversight.registry import get_policy_class

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

logger = get_logger(__name__)


class OverlaySource:
    """Supplies runtime switches on top of code and settings. The base class supplies none.

    - `rule_modes` returns {policy: {rule: mode}} for the requested policy names.
    - `agent_policies` returns every agent the overlay assigns: agent id -> a policy
      name, or "" for none. An agent missing from it is left to code.

    Policies built with this source, and its agent assignments, are cached for `ttl`
    seconds. When a read fails, turns keep the last good read.
    """

    ttl: float = math.inf

    async def rule_modes(self, names: list[str]) -> dict[str, dict[str, str]]:
        return {}

    async def agent_policies(self) -> dict[str, str]:
        return {}


_source: OverlaySource = OverlaySource()

# Policy loaders let an app register whole policies at runtime (the authoring app
# keeps operator-written ones in the database). Each registers or unregisters its
# policies when awaited.
_loaders: list[Callable[[], Awaitable[None]]] = []
_loaded_at = -math.inf


def add_policy_loader(loader: Callable[[], Awaitable[None]]) -> None:
    """Await `loader` before policy lookups. Adding it twice has no effect."""
    global _loaded_at
    if loader not in _loaders:
        _loaders.append(loader)
        _loaded_at = -math.inf


async def aload_policies() -> None:
    """Run every loader, at most once per RUNTIME_CACHE_SECONDS.

    Failures are logged, and already registered policies stay.
    """
    global _loaded_at
    if not _loaders or time.monotonic() - _loaded_at < float(setting("RUNTIME_CACHE_SECONDS")):
        return
    started = time.monotonic()
    for loader in list(_loaders):
        try:
            await loader()
        except Exception:
            logger.opt(exception=True).warning("An oversight policy loader failed")
    # Marked after the loaders, so a lookup meanwhile loads too instead of resolving
    # against a registry that is not filled yet.
    _loaded_at = started


def set_overlay_source(source: OverlaySource | None) -> None:
    """Install `source`, or the empty default for None, and clear every cache."""
    global _source
    _source = source or OverlaySource()
    reset()


def reset() -> None:
    """Clear every cache, the last good reads too."""
    global _agent_map
    invalidate()
    _last_good.clear()
    _agent_map = None


def overlay_source() -> OverlaySource:
    return _source


async def aassigned_policy(agent_id: str) -> str | None:
    """The policy the overlay chose for an agent: a name, "" for none, or None."""
    if not agent_id:
        return None
    return (await _assignments()).get(agent_id)


def cached_assignment(agent_id: str) -> tuple[bool, str | None]:
    """For sync code, which cannot read the overlay: (known, the overlay's choice).

    Known when there is no overlay, or when its agent assignments were read within its
    `ttl` (every turn reads them as it starts). The choice is as `aassigned_policy`.
    """
    if type(_source) is OverlaySource:
        return True, None
    if _agent_map is None or time.monotonic() - _agents_read_at >= _source.ttl:
        return False, None
    return True, _agent_map.get(agent_id) if agent_id else None


async def _assignments() -> dict[str, str]:
    """The source's agent assignments, read at most once per `ttl`.

    A failed read is not cached, so the next turn tries again. Until then turns use
    the last good read, or code alone before any.
    """
    global _agent_map, _agents_read_at
    if _agent_map is not None and time.monotonic() - _agents_read_at < _source.ttl:
        return _agent_map
    try:
        found = dict(await _source.agent_policies())
    except Exception:
        logger.opt(exception=True).warning(
            "Oversight overlay's agent policies could not be read; agents keep the last read"
        )
        return _agent_map or {}
    _agent_map, _agents_read_at = found, time.monotonic()
    return found


# name -> (resolved at, ResolvedPolicy or None if it failed, ancestor names)
_cache: dict[str, tuple[float, ResolvedPolicy | None, frozenset[str]]] = {}
# name -> why it failed to load
_broken: dict[str, str] = {}
# name -> the last policy built with its overlay. `invalidate` keeps it, so a
# failed overlay read never drops what an operator enforced.
_last_good: dict[str, ResolvedPolicy] = {}
# The last agent assignments and when they were read. `invalidate` keeps the map for
# the same reason.
_agent_map: dict[str, str] | None = None
_agents_read_at = -math.inf


async def aget_policy(name: str) -> ResolvedPolicy | None:
    """Return the effective policy with its overlay, cached for the source's `ttl`.

    Policy loaders run first, so a policy registered at runtime is found.
    Returns None for an unknown or broken policy. The reason is logged once per
    cache period and kept for `broken_reason()`. The caller decides what happens next.
    `TurnGuard.start` refuses the turn unless FAIL_MODE is "open".

    When the overlay cannot be read, this returns the last good policy, or else
    the policy as declared in code. That result is not cached, so the next turn
    tries the overlay again.
    """
    await aload_policies()
    source = _source
    cached = _cache.get(name)
    if cached and time.monotonic() - cached[0] < source.ttl:
        return cached[1]
    lineage = code_ancestors(name)
    try:
        modes = await source.rule_modes([name, *lineage])
    except Exception:
        logger.opt(exception=True).warning("Oversight overlay for {!r} could not be read", name)
        if cached and cached[1] is not None:
            return cached[1]
        if name in _last_good:
            return _last_good[name]
        return _build(name)
    ancestors = {parent: modes.get(parent, {}) for parent in lineage}
    resolved = _build(name, modes=modes.get(name, {}), ancestors=ancestors)
    # Cache failures too, so a broken policy logs once per period, not once per turn.
    _cache[name] = (time.monotonic(), resolved, frozenset(lineage))
    if resolved is not None:
        _last_good[name] = resolved
    return resolved


def _build(name: str, **kwargs: Any) -> ResolvedPolicy | None:
    """Call build(). On failure, record why and return None."""
    try:
        resolved = build(name, **kwargs)
    except Exception as exc:
        _broken[name] = str(exc) or type(exc).__name__
        if isinstance(exc, LookupError) and get_policy_class(name) is None:
            # The policy does not exist. A missing parent is different: then the
            # policy exists but is broken.
            logger.info("Oversight policy {!r} is unknown", name)
        else:
            logger.exception("Oversight policy {!r} cannot be loaded: {}", name, exc)
        return None
    if resolved.rejected:
        logger.warning(
            "Oversight policy {!r}: {} switched modes do not apply", name, len(resolved.rejected)
        )
    _broken.pop(name, None)
    return resolved


def broken_reason(name: str) -> str:
    """Why `name` failed to load last time, or ""."""
    return _broken.get(name, "")


def code_ancestors(name: str, *, parents: Iterable[str] | None = None) -> list[str]:
    """Every policy `name` extends in code, nearest first. build() reports cycles.

    `parents` replaces the `extends` of `name`, for a policy not registered yet.
    """
    found: list[str] = []
    if parents is None:
        parents = getattr(get_policy_class(name), "extends", []) or []
    todo = list(parents)
    while todo:
        parent = todo.pop(0)
        if parent in found or parent == name:
            continue
        found.append(parent)
        todo.extend(getattr(get_policy_class(parent), "extends", []) or [])
    return found


def forget(name: str) -> None:
    """`invalidate(name)`, and drop the last good builds of it and its children.

    For a policy whose declaration changed or went: its last good build is not the
    policy any more, so an unreadable overlay must not bring it back.
    """
    invalidate(name)
    for kept in list(_last_good):
        try:
            lineage = code_ancestors(kept)
        except Exception:
            lineage = []
        if kept == name or name in lineage:
            _last_good.pop(kept, None)


def invalidate(name: str | None = None) -> None:
    """Clear cached policies so the next turn resolves again.

    Clearing a policy also clears every policy that extends it. The last good
    policies are kept, since they are only used when the overlay cannot be read.
    """
    global _loaded_at, _agents_read_at
    if name is None:
        _cache.clear()
        _agents_read_at = _loaded_at = -math.inf
        return
    for cached, (_, _, lineage) in list(_cache.items()):
        if cached == name or name in lineage:
            _cache.pop(cached, None)


__all__ = [
    "OverlaySource",
    "aassigned_policy",
    "add_policy_loader",
    "aget_policy",
    "aload_policies",
    "broken_reason",
    "code_ancestors",
    "forget",
    "invalidate",
    "overlay_source",
    "reset",
    "set_overlay_source",
]
