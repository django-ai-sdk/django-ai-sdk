"""Mixins for the Django AI SDK Agent classes."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """Pydantic model for agent metadata.

    Attributes:
        id: Stable UUID v5 identifier for the agent
        name: Display name of the agent
        model: AI model identifier (e.g., "gpt-4o-mini")
        class_name: Python class name of the agent
        description: Optional description of the agent's purpose
        file_upload: Whether this agent supports file uploads in threads
    """

    id: str
    name: str | None = None
    model: str | None = None
    class_name: str
    description: str | None = None
    file_upload: bool = False
    rag: bool = True


# Namespace for deterministic UUID generation (arbitrary but fixed)
AGENT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class AgentInfoMixin:
    """Mixin providing basic agent metadata and introspection."""

    _agent_id: str = ""  # Set by registry

    def info(self) -> AgentInfo:
        """Return agent metadata as a validated Pydantic model."""
        name: str | None = getattr(self, "name", None)
        if name is None:
            get_name: Any = getattr(self, "get_name", None)
            if callable(get_name):
                name_result: Any = get_name()
                name = name_result if isinstance(name_result, str) else None

        return AgentInfo(
            id=self.agent_id,
            name=name,
            model=getattr(self, "model", None),
            class_name=self.__class__.__name__,
            description=getattr(self, "description", None),
            file_upload=getattr(self, "file_upload", False),
            rag=True if getattr(self, "rag_provider", None) else False,
        )

    @property
    def agent_id(self) -> str:
        """Stable UUID v5 ID generated from class path.

        This generates a deterministic UUID based on the full module.class path,
        ensuring stable IDs across restarts and deployments.
        """
        if not self._agent_id:
            # Generate deterministic UUID v5 from full class path
            class_path = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._agent_id = str(uuid.uuid5(AGENT_NAMESPACE, class_path))
        return self._agent_id
