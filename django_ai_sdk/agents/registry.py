from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING, TypeVar

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from .base import Agent


T = TypeVar("T", bound="Agent")

AGENT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class AgentRegistrationError(Exception):
    pass


def auto_register[T: Agent](cls: type[T]) -> type[T]:
    """If also listed in AI_SDK_AGENTS, it will only be loaded once."""
    registry.register(cls)
    return cls


class AgentRegistry:
    _instance: AgentRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> AgentRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._reset()
        return cls._instance

    def _reset(self) -> None:
        self._classes: dict[str, type[Agent]] = {}
        self._instances: dict[str, Agent] = {}
        self._initialized = False

    def register(self, agent_class: type[T]) -> type[T]:
        """Re-registering the same class is a no-op and a redefinition of that path (a reloaded
        module) replaces the previous entry rather than raising.
        """
        if agent_class.__dict__.get("abstract", False):
            return agent_class

        class_path = f"{agent_class.__module__}.{agent_class.__name__}"
        agent_id = str(uuid.uuid5(AGENT_NAMESPACE, class_path))

        if agent_id in self._classes:
            existing = self._classes[agent_id]
            if existing is agent_class:
                return agent_class
        self._classes[agent_id] = agent_class
        agent_class._agent_id = agent_id

        return agent_class

    def setup(self, instantiate: bool = True, load_from_settings: bool = True) -> None:
        """Initialize all registered agents. Call once in AppConfig.ready()."""
        if self._initialized:
            return

        if load_from_settings:
            from django.utils.module_loading import import_string

            agent_paths = resolve_setting("AI_SDK_AGENTS", [])
            for path in agent_paths:
                try:
                    import_string(path)
                except ImportError:
                    pass

        if instantiate:
            for agent_id, agent_class in self._classes.items():
                self._instances[agent_id] = agent_class()

        self._initialized = True

    def get(self, agent_id: str) -> Agent | None:
        """Raises:
        RuntimeError: If setup() has not been called
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call setup() in AppConfig.ready()")

        if agent_id not in self._instances:
            if agent_id in self._classes:
                self._instances[agent_id] = self._classes[agent_id]()
            else:
                return None

        return self._instances.get(agent_id)

    def all(self) -> dict[str, Agent]:
        """Raises:
        RuntimeError: If setup() has not been called
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call setup() in AppConfig.ready()")
        return self._instances.copy()

    def visible(self) -> dict[str, Agent]:
        return {aid: inst for aid, inst in self.all().items() if not getattr(inst, "hidden", False)}

    def ids(self) -> list[str]:
        """List of registered agent IDs (available even before setup())"""
        return list(self._classes.keys())

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._classes


registry = AgentRegistry()


def registered_agent_classes() -> list[type[Agent]]:
    try:
        agents = registry.all().values()
    except RuntimeError:
        return []
    return list(dict.fromkeys(type(agent) for agent in agents))
