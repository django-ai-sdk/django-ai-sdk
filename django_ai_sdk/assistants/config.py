from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django_ai_sdk.assistants.runtime import RuntimeAssistant


@lru_cache(maxsize=1)
def get_runtime_assistant_bases() -> list[type[RuntimeAssistant]]:
    """Return allowed RuntimeAssistant subclasses from AI_SDK_RUNTIME_ASSISTANT_BASES."""
    paths: list[str] = getattr(settings, "AI_SDK_RUNTIME_ASSISTANT_BASES", [])
    if not paths:
        from django_ai_sdk.assistants.runtime import RuntimeAssistant as _RuntimeAssistant

        return [_RuntimeAssistant]
    return [import_string(p) for p in paths]


def get_runtime_assistant_class(assistant: str | None = None) -> type[RuntimeAssistant]:
    """Resolve the correct RuntimeAssistant subclass for a given dotted path."""
    bases = get_runtime_assistant_bases()
    if not assistant:
        return bases[0]
    for cls in bases:
        if f"{cls.__module__}.{cls.__qualname__}" == assistant:
            return cls
    return bases[0]


@lru_cache(maxsize=1)
def get_tool_registry() -> dict[str, str]:
    """Return AI_SDK_RUNTIME_ASSISTANT_TOOLS mapping of key → dotted path."""
    return dict(getattr(settings, "AI_SDK_RUNTIME_ASSISTANT_TOOLS", {}))
