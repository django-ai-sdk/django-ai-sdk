"""Resolve the policy a turn runs, cached.

`aget_policy(name)` builds the policy from the registry and caches it until
`invalidate()` clears it. Nothing changes a registered policy at runtime.
"""

from __future__ import annotations

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.policy import ResolvedPolicy, build
from django_ai_sdk.oversight.registry import get_policy_class

logger = get_logger(__name__)

# name -> (ResolvedPolicy or None if it failed, ancestor names)
_cache: dict[str, tuple[ResolvedPolicy | None, frozenset[str]]] = {}
# name -> why it failed to load
_broken: dict[str, str] = {}


async def aget_policy(name: str) -> ResolvedPolicy | None:
    """Return the effective policy, cached.

    Returns None for an unknown or broken policy. The reason is logged once and
    kept for `broken_reason()`. The caller decides what happens next.
    `TurnGuard.start` refuses the turn unless FAIL_MODE is "open".
    """
    cached = _cache.get(name)
    if cached:
        return cached[0]
    resolved = _build(name)
    # Cache failures too, so a broken policy logs once, not once per turn.
    _cache[name] = (resolved, frozenset(code_ancestors(name)))
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


def code_ancestors(name: str) -> list[str]:
    """Every policy `name` extends in code, nearest first. build() reports cycles."""
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
    if name is None:
        _cache.clear()
        return
    for cached, (_, lineage) in list(_cache.items()):
        if cached == name or name in lineage:
            _cache.pop(cached, None)


__all__ = ["aget_policy", "broken_reason", "code_ancestors", "invalidate"]
