"""The policy an agent asks for in code, and how a turn identifies its agent.

An agent names its policy with an `oversight` attribute or an async
`get_oversight(channel=, user=)` hook. An overlay (`resolve.OverlaySource`) can
assign a different one by the agent's key.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.conf import setting

if TYPE_CHECKING:
    from django_ai_sdk.agents.base import Agent

# Length of an agent's fingerprint in hex characters.
FINGERPRINT_CHARS = 16


def agent_key(agent: Agent) -> str:
    """The id that turns and overlays use for an agent.

    A runtime agent built on a registered class uses its own id, not the class's.
    """
    return str(getattr(agent, "agent_id", "") or "")


async def acode_policy(agent: Agent, *, channel: str = "", user: Any = None) -> tuple[str, str]:
    """Return the policy name the agent's code asks for, and the reason.

    The reason is "agent", "off" or "default". Calls `get_oversight()` when the
    agent has it, else reads its `oversight` attribute. Errors from the hook, and
    TypeError for a value that names no policy, propagate so the caller decides.
    """
    getter = getattr(agent, "get_oversight", None)
    if callable(getter):
        chosen = getter(channel=channel, user=user)
        if inspect.isawaitable(chosen):
            chosen = await chosen
    else:
        chosen = getattr(agent, "oversight", None)
    return _code_choice(chosen)


def code_policy(agent: Any) -> tuple[str, str]:
    """Like `acode_policy`, but reads only the `oversight` attribute. Used by checks."""
    return _code_choice(getattr(agent, "oversight", None))


def policy_needs_await(agent: Any) -> bool:
    """Whether only an async run can tell the agent's policy: it overrides
    `get_oversight()`, or an overlay is installed that may assign it one."""
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.oversight.resolve import OverlaySource, overlay_source

    # The base method only returns `oversight`, which sync code reads itself.
    overridden = "get_oversight" in getattr(agent, "__dict__", {}) or (
        getattr(type(agent), "get_oversight", Agent.get_oversight) is not Agent.get_oversight
    )
    return overridden or type(overlay_source()) is not OverlaySource


def _code_choice(chosen: Any) -> tuple[str, str]:
    from django_ai_sdk.oversight.registry import policy_name

    if chosen is False:
        return "", "off"
    if chosen is None:
        return str(setting("DEFAULT_POLICY") or ""), "default"
    if chosen == "":
        # As for the `oversight` attribute, "" is no policy name. Unchecked, it would
        # run the turn unguarded.
        raise TypeError('get_oversight() returned ""; return None for the default, or False')
    return policy_name(chosen), "agent"


def fingerprint(agent: Agent) -> str:
    """A short hash of the agent's model, prompt, tools and integrations."""
    parts: list[str] = []
    for getter in ("get_model", "get_system_prompt"):
        fn = getattr(agent, getter, None)
        try:
            parts.append(str(fn() if callable(fn) else ""))
        except Exception:
            parts.append("")
    tools = getattr(type(agent), "tools", []) or []
    parts.append(",".join(sorted(getattr(t, "__name__", str(t)) for t in tools)))
    parts.append(",".join(sorted(getattr(agent, "integrations", []) or [])))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:FINGERPRINT_CHARS]


__all__ = [
    "FINGERPRINT_CHARS",
    "acode_policy",
    "agent_key",
    "code_policy",
    "fingerprint",
    "policy_needs_await",
]
