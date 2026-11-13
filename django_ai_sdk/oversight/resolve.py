"""Resolve the policy a turn runs, cached.

`aget_policy(name)` builds the policy from the registry and caches it until
`invalidate()` clears it. A lens or rule declared in code never changes at runtime.
Apps that register whole policies at runtime (the authoring app) add a loader with
`add_policy_loader`.
"""

from __future__ import annotations

import asyncio
import math
import time
import weakref
from typing import TYPE_CHECKING

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.policy import ResolvedPolicy, build
from django_ai_sdk.oversight.registry import get_policy_class

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

# name -> (ResolvedPolicy or None if it failed, ancestor names)
_cache: dict[str, tuple[ResolvedPolicy | None, frozenset[str]]] = {}
# name -> why it failed to load
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
    """Await `loader` before policy lookups. Adding it twice has no effect."""
    global _loaded_at
    if loader not in _loaders:
        _loaders.append(loader)
        _loaded_at = -math.inf


async def aload_policies() -> None:
    """Run every loader, at most once per RUNTIME_CACHE_SECONDS.

    Failures are logged, and already registered policies stay. Lookups that arrive
    while it runs wait for it, so a burst reads the database once.
    """
    global _loaded_at
    if not _loaders or _fresh():
        return
    async with _locks.setdefault(asyncio.get_running_loop(), asyncio.Lock()):
        if _fresh():
            return
        started = time.monotonic()
        for loader in list(_loaders):
            try:
                await loader()
            except Exception:
                logger.opt(exception=True).warning("An oversight policy loader failed")
        _loaded_at = started


def _fresh() -> bool:
    return time.monotonic() - _loaded_at < float(setting("RUNTIME_CACHE_SECONDS"))


async def aget_policy(name: str) -> ResolvedPolicy | None:
    """Return the effective policy, cached. Policy loaders run first.

    Returns None for an unknown or broken policy. The reason is logged once and
    kept for `broken_reason()`. The caller decides what happens next.
    `TurnGuard.start` refuses the turn unless FAIL_MODE is "open".
    """
    await aload_policies()
    cached = _cache.get(name)
    if cached:
        return cached[0]
    resolved = _build(name)
    # Cache failures too, so a broken policy logs once, not once per turn.
    _cache[name] = (resolved, frozenset(ancestors(name)))
    return resolved


def _build(name: str) -> ResolvedPolicy | None:
    """Call build(). On failure, record why and return None."""
    try:
        resolved = build(name)
    except Exception as exc:
        _broken[name] = str(exc) or type(exc).__name__
        if isinstance(exc, LookupError) and get_policy_class(name) is None:
            # The policy does not exist. A missing parent is different: then the
            # policy exists but is broken.
            logger.info("Oversight policy {!r} is unknown", name)
        else:
            logger.exception("Oversight policy {!r} cannot be loaded: {}", name, exc)
        return None
    _broken.pop(name, None)
    return resolved


def broken_reason(name: str) -> str:
    """Why `name` failed to load last time, or ""."""
    return _broken.get(name, "")


def ancestors(name: str) -> list[str]:
    """Every policy `name` extends, nearest first. build() reports cycles."""
    found: list[str] = []
    todo = list(getattr(get_policy_class(name), "extends", []) or [])
    while todo:
        parent = todo.pop(0)
        if parent in found or parent == name:
            continue
        found.append(parent)
        todo.extend(getattr(get_policy_class(parent), "extends", []) or [])
    return found


def invalidate(name: str | None = None) -> None:
    """Clear cached policies so the next turn resolves again.

    Clearing a policy also clears every policy that extends it.
    """
    global _loaded_at
    if name is None:
        _cache.clear()
        _loaded_at = -math.inf
        return
    for cached, (_, lineage) in list(_cache.items()):
        if cached == name or name in lineage:
            _cache.pop(cached, None)


__all__ = [
    "add_policy_loader",
    "aget_policy",
    "aload_policies",
    "ancestors",
    "broken_reason",
    "invalidate",
]
