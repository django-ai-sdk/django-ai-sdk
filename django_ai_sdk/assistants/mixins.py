"""Mixins for the Django AI SDK Assistant classes."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


class AssistantInfo(BaseModel):
    """Pydantic model for assistant metadata.

    Attributes:
        id: Stable UUID v5 identifier for the assistant
        name: Display name of the assistant
        model: AI model identifier (e.g., "gpt-4o-mini")
        class_name: Python class name of the assistant
        description: Optional description of the assistant's purpose
        file_upload: Whether this assistant supports file uploads in threads
    """

    id: str
    name: str | None = None
    model: str | None = None
    class_name: str
    description: str | None = None
    file_upload: bool = False


# Namespace for deterministic UUID generation (arbitrary but fixed)
ASSISTANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class AssistantInfoMixin:
    """Mixin providing basic assistant metadata and introspection."""

    _assistant_id: str = ""  # Set by registry

    def info(self) -> AssistantInfo:
        """Return assistant metadata as a validated Pydantic model."""
        name: str | None = getattr(self, "name", None)
        if name is None:
            get_name: Any = getattr(self, "get_name", None)
            if callable(get_name):
                name_result: Any = get_name()
                name = name_result if isinstance(name_result, str) else None

        return AssistantInfo(
            id=self.assistant_id,
            name=name,
            model=getattr(self, "model", None),
            class_name=self.__class__.__name__,
            description=getattr(self, "description", None),
            file_upload=getattr(self, "file_upload", False),
        )

    @property
    def assistant_id(self) -> str:
        """Stable UUID v5 ID generated from class path.

        This generates a deterministic UUID based on the full module.class path,
        ensuring stable IDs across restarts and deployments.
        """
        if not self._assistant_id:
            # Generate deterministic UUID v5 from full class path
            class_path = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._assistant_id = str(uuid.uuid5(ASSISTANT_NAMESPACE, class_path))
        return self._assistant_id
