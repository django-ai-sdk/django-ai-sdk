from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import policy_name

if TYPE_CHECKING:
    from django_ai_sdk.agents.base import Agent

FINGERPRINT_CHARS = 16
DEFAULT_CHANNEL = "default"


def agent_key(agent: Agent) -> str:
    return str(getattr(agent, "agent_id", "") or "")


def turn_channel(channel: str | None) -> str:
    return DEFAULT_CHANNEL if channel is None else channel


async def declared_policy(
    agent: Agent, *, channel: str | None = None, user: Any = None
) -> tuple[str, str]:
    getter = getattr(agent, "get_oversight", None)
    if callable(getter):
        chosen = getter(channel=turn_channel(channel), user=user)
        if inspect.isawaitable(chosen):
            chosen = await chosen
    else:
        chosen = getattr(agent, "oversight", None)
    return _policy_name_and_reason(chosen)


def attribute_policy(agent: Any) -> tuple[str, str]:
    return _policy_name_and_reason(getattr(agent, "oversight", None))


def overrides_get_oversight(agent: Any) -> bool:
    from django_ai_sdk.agents.base import Agent

    return "get_oversight" in getattr(agent, "__dict__", {}) or (
        getattr(type(agent), "get_oversight", Agent.get_oversight) is not Agent.get_oversight
    )


def _policy_name_and_reason(chosen: Any) -> tuple[str, str]:
    if chosen is False:
        return "", "off"
    if chosen is None:
        return str(setting("DEFAULT_POLICY") or ""), "default"
    if chosen == "":
        raise TypeError('get_oversight() returned ""; return None for the default, or False')
    return policy_name(chosen), "agent"


def validate_oversight(cls: type) -> None:
    value = cls.__dict__.get("oversight")
    if value is None or value is False or (isinstance(value, str) and value):
        return
    if isinstance(value, type):
        if issubclass(value, Policy):
            policy_name(value)
            return
    raise ImproperlyConfigured(
        f"{cls.__qualname__}.oversight must be a Policy subclass, a policy's name, False "
        f"or None, not {value!r}."
    )


def fingerprint(agent: Agent) -> str:
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
