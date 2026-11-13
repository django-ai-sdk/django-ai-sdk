"""Checks on agent classes: the policy each names, and chat adapters that never finish."""

from __future__ import annotations

from typing import Any

from django.core.checks import Error, Warning

from django_ai_sdk.oversight.checks import PREFIX


def policies_in_use() -> set[str]:
    """DEFAULT_POLICY, the policies agent classes name, and all their ancestors."""
    from django_ai_sdk.agents.registry import registered_agent_classes
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.resolve import ancestors

    names = {str(setting("DEFAULT_POLICY") or "")}
    for cls in [*registered_agent_classes(), *agent_runtime_bases()]:
        try:
            names.add(agent_policy_name(cls))
        except Exception:
            continue  # E011 reports it.
    names.discard("")
    for name in list(names):
        names.update(ancestors(name))
    return names


def agent_runtime_bases() -> list[type]:
    """The classes AgentSettings rows build agents on, or none when they do not import."""
    from django_ai_sdk.agents.config import get_runtime_agent_bases

    try:
        return list(get_runtime_agent_bases())
    except Exception:
        return []


def agent_policy_name(cls: type) -> str:
    """The policy an agent class names itself, or "" when it names none or opts out.

    Raises TypeError for a value that names no policy.
    """
    from django_ai_sdk.oversight.agents import code_policy

    name, source = code_policy(cls)
    return name if source == "agent" else ""


def agent_checks() -> list[Any]:
    """Flag agents that name an unregistered policy or an invalid value."""
    from django_ai_sdk.agents.registry import registered_agent_classes
    from django_ai_sdk.oversight.registry import get_policies

    messages: list[Any] = []
    known = get_policies()
    for cls in registered_agent_classes():
        try:
            own = agent_policy_name(cls)
        except Exception as exc:
            messages.append(
                Error(
                    f"Agent {cls.__qualname__}.oversight names no policy: {exc}. Its turns "
                    "cannot be checked.",
                    hint="Set a Policy subclass, a registered policy's name, False or None.",
                    id=f"{PREFIX}.E011",
                )
            )
            continue
        if own and own not in known:
            messages.append(
                Error(
                    f"Agent {cls.__qualname__} names oversight policy {own!r}, which no app "
                    "registers: every turn it runs is refused.",
                    hint="Check the name, and that the app declaring the policy is in "
                    "INSTALLED_APPS with the policy in its oversight.py.",
                    id=f"{PREFIX}.E009",
                )
            )
        messages.extend(_adapter_checks(cls))
    return messages


def _adapter_checks(cls: type) -> list[Any]:
    """Warn when an agent's chat adapter isn't the SDK's Stream, so no hook can check its chats."""
    import typing

    from django_ai_sdk.adapters.base import Stream
    from django_ai_sdk.oversight.agents import code_policy
    from django_ai_sdk.oversight.checks.policies import build_or_none

    policy, _ = code_policy(cls)  # "" when the agent opts out
    method = getattr(cls, "get_pipeline_adapter", None)
    resolved = build_or_none(policy) if policy and method is not None else None
    if resolved is None:
        return []
    try:
        returned = typing.get_type_hints(method).get("return")
    except Exception:
        return []  # Can't tell: the annotation doesn't import.
    if not isinstance(returned, type) or issubclass(returned, Stream):
        return []
    return [
        Warning(
            f"Agent {cls.__qualname__} streams chat through {returned.__qualname__}, which is "
            f"not the SDK's Stream: policy {policy!r} refuses its chats, or lets them through "
            "unchecked if it fails open.",
            hint="Return a Stream (or a subclass) from get_pipeline_adapter().",
            id=f"{PREFIX}.W013",
        )
    ]
