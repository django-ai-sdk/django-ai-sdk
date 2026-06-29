from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from django.conf import settings
from django.utils.module_loading import import_string


@runtime_checkable
class BaseAction(Protocol):
    description: str

    async def execute(self, payload: Any) -> None:
        pass


@lru_cache(maxsize=1)
def get_action_registry() -> dict[str, type[BaseAction]]:
    """Actions defined in AI_SDK_WORKFLOW_ACTIONS setting."""
    registry: dict[str, type[BaseAction]] = {}
    extra: dict[str, str] = getattr(settings, "AI_SDK_WORKFLOW_ACTIONS", {})
    for key, path in extra.items():
        registry[key] = import_string(path)
    return registry
