from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django_ai_sdk.agents.runtime import RuntimeAgent


@lru_cache(maxsize=1)
def get_runtime_agent_bases() -> list[type[RuntimeAgent]]:
    """Return allowed RuntimeAgent subclasses from AI_SDK_RUNTIME_AGENT_BASES."""
    paths: list[str] = getattr(settings, "AI_SDK_RUNTIME_AGENT_BASES", [])
    if not paths:
        from django_ai_sdk.agents.runtime import RuntimeAgent as _RuntimeAgent

        return [_RuntimeAgent]
    return [import_string(p) for p in paths]


def get_runtime_agent_class(agent: str | None = None) -> type[RuntimeAgent]:
    """Resolve the correct RuntimeAgent subclass for a given dotted path."""
    bases = get_runtime_agent_bases()
    if not agent:
        return bases[0]
    for cls in bases:
        if f"{cls.__module__}.{cls.__qualname__}" == agent:
            return cls
    return bases[0]


@lru_cache(maxsize=1)
def get_tool_registry() -> dict[str, str]:
    """Return AI_SDK_RUNTIME_AGENT_TOOLS mapping of key → dotted path."""
    return dict(getattr(settings, "AI_SDK_RUNTIME_AGENT_TOOLS", {}))
