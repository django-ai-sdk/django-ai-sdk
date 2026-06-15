from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from django_ai_sdk.web_assistant.assistant import WebAssistant


@lru_cache(maxsize=1)
def get_web_assistant_bases() -> list[type[WebAssistant]]:
    """Return allowed WebAssistant subclasses from AI_SDK_WEB_ASSISTANT_BASES."""
    paths: list[str] = getattr(settings, "AI_SDK_WEB_ASSISTANT_BASES", [])
    if not paths:
        from django_ai_sdk.web_assistant.assistant import WebAssistant as _WebAssistant

        return [_WebAssistant]
    return [import_string(p) for p in paths]


def get_web_assistant_class(base_class: str | None = None) -> type[WebAssistant]:
    """Resolve the correct WebAssistant subclass for a given dotted path.

    Falls back to first configured base if path is empty or not in the allowlist.
    """
    bases = get_web_assistant_bases()
    if not base_class:
        return bases[0]
    for cls in bases:
        if f"{cls.__module__}.{cls.__qualname__}" == base_class:
            return cls
    return bases[0]


@lru_cache(maxsize=1)
def get_tool_registry() -> dict[str, str]:
    """Return AI_SDK_WEB_ASSISTANT_TOOLS mapping of key → dotted path."""
    return dict(getattr(settings, "AI_SDK_WEB_ASSISTANT_TOOLS", {}))
