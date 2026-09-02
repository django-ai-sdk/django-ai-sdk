"""Agent registry for managing Agent classes and instances."""

from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING, TypeVar

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from ..agent import Agent


T = TypeVar("T", bound="Agent")

# Namespace for deterministic UUID generation
AGENT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class AgentRegistrationError(Exception):
    """Raised when agent registration fails"""

    pass


def auto_register[T: Agent](cls: type[T]) -> type[T]:
    """Decorator to register an Agent class with the registry.

    This is the recommended way to register agents when using the
    decorator-based approach. Can be combined with settings-based loading.

    Usage:
        from django_ai_sdk.agents.registry import auto_register

        @auto_register
        class MyAgent(Agent):
            name = "My Agent"

    The class will be registered immediately when the module is imported.
    If also listed in AI_SDK_AGENTS, it will only be loaded once.

    Args:
        cls: The Agent subclass to register

    Returns:
        The registered class
    """
    registry.register(cls)
    return cls


class AgentRegistry:
    """Singleton registry for managing Agent classes and instances.

    Supports two registration methods:
    1. Settings-based: List paths in AI_SDK_AGENTS setting
    2. Decorator-based: Apply @auto_register to Agent classes

    Both methods can be used together - a class will only be registered once.

    Usage:
        # Method 1: Settings-based (recommended)
        # In your settings.py:
        AI_SDK_AGENTS = [
            "myapp.agents.MyAgent",
        ]

        # Method 2: Decorator-based
        from django_ai_sdk.agents.registry import auto_register

        @auto_register
        class MyAgent(Agent):
            pass

        # In your AppConfig.ready():
        from django_ai_sdk.agents.registry import registry
        from django.utils.module_loading import import_string
        from django.conf import settings

        # Load from settings
        for path in resolve_setting('AI_SDK_AGENTS', []):
            import_string(path)

        # Setup instantiates all registered agents
        registry.setup()

        # Later in views:
        agent = registry.get(agent_id)  # Get by UUID
    """

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
        """Reset registry state."""
        self._classes: dict[str, type[Agent]] = {}
        self._instances: dict[str, Agent] = {}
        self._initialized = False

    def register(self, agent_class: type[T]) -> type[T]:
        """Register an Agent class.

        Skips classes marked abstract = True: a shared base meant only to be
        subclassed, not instantiated or exposed on its own.

        The id is a UUID5 of the class's import path, so re-registering the same
        class is a no-op and a redefinition of that path (a reloaded module) replaces
        the previous entry rather than raising.

        Args:
            agent_class: The Agent subclass to register

        Returns:
            The registered class
        """
        if agent_class.__dict__.get("abstract", False):
            return agent_class

        # Generate deterministic UUID v5 from full class path
        class_path = f"{agent_class.__module__}.{agent_class.__name__}"
        agent_id = str(uuid.uuid5(AGENT_NAMESPACE, class_path))

        # Check if already registered with same class
        if agent_id in self._classes:
            existing = self._classes[agent_id]
            if existing is agent_class:
                # Same class being registered again
                return agent_class
        self._classes[agent_id] = agent_class
        agent_class._agent_id = agent_id

        return agent_class

    def setup(self, instantiate: bool = True, load_from_settings: bool = True) -> None:
        """Initialize all registered agents. Call once in AppConfig.ready().

        This method should be called once when Django starts. It can optionally
        load agents from the AI_SDK_AGENTS setting and instantiates
        all registered agent classes.

        Args:
            instantiate: Whether to create instances immediately. If False,
                        instances will be created on-demand via get().
            load_from_settings: Whether to load agents from AI_SDK_AGENTS
                              setting. Defaults to True.

        Raises:
            RuntimeError: If setup() has already been called.
        """
        if self._initialized:
            return

        # Load agents from settings
        if load_from_settings:
            from django.utils.module_loading import import_string

            agent_paths = resolve_setting("AI_SDK_AGENTS", [])
            for path in agent_paths:
                try:
                    import_string(path)
                except ImportError:
                    # Class may not exist or path is wrong
                    pass

        # Instantiate all registered agents
        if instantiate:
            for agent_id, agent_class in self._classes.items():
                self._instances[agent_id] = agent_class()

        self._initialized = True

    def get(self, agent_id: str) -> Agent | None:
        """Get agent instance by ID.

        Args:
            agent_id: The ID of the agent to retrieve

        Returns:
            The agent instance, or None if not found

        Raises:
            RuntimeError: If setup() has not been called
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call setup() in AppConfig.ready()")

        # Create on-demand if not instantiated yet
        if agent_id not in self._instances:
            if agent_id in self._classes:
                self._instances[agent_id] = self._classes[agent_id]()
            else:
                return None

        return self._instances.get(agent_id)

    def all(self) -> dict[str, Agent]:
        """Get all agent instances.

        Returns:
            Dict mapping agent_id to instance

        Raises:
            RuntimeError: If setup() has not been called
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call setup() in AppConfig.ready()")
        return self._instances.copy()

    def visible(self) -> dict[str, Agent]:
        """Get only non-hidden agent instances.

        Hidden agents (e.g. internal task agents) are excluded.

        Returns:
            Dict mapping agent_id to visible instance
        """
        return {aid: inst for aid, inst in self.all().items() if not getattr(inst, "hidden", False)}

    def ids(self) -> list[str]:
        """Get all registered agent IDs.

        Returns:
            List of registered agent IDs (available even before setup())
        """
        return list(self._classes.keys())

    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent ID is registered.

        Example:
            if "mybot" in registry:
                agent = registry.get("mybot")
        """
        return agent_id in self._classes


# Global singleton instance
registry = AgentRegistry()
