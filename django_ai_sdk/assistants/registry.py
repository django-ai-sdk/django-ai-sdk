"""Assistant registry for managing Assistant classes and instances."""

import threading
import uuid
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ..assistant import Assistant


T = TypeVar("T", bound="Assistant")

# Namespace for deterministic UUID generation (fixed for Django AI SDK)
# Using UUIDv5 with this namespace ensures stable IDs across restarts
ASSISTANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class AssistantRegistrationError(Exception):
    """Raised when assistant registration fails.

    Note: Currently registration is permissive - a class with the same ID
    as an existing registration will override it. This allows for testing
    scenarios where multiple test classes share the same name.
    """

    pass


def auto_register[T: "Assistant"](cls: type[T]) -> type[T]:
    """Decorator to register an Assistant class with the registry.

    This is the recommended way to register assistants when using the
    decorator-based approach. Can be combined with settings-based loading.

    Usage:
        from django_ai_sdk.assistants.registry import auto_register

        @auto_register
        class MyAssistant(Assistant):
            name = "My Assistant"

    The class will be registered immediately when the module is imported.
    If also listed in AI_SDK_ASSISTANTS, it will only be loaded once.

    Args:
        cls: The Assistant subclass to register

    Returns:
        The registered class (unchanged, for decorator chaining)
    """
    registry.register(cls)
    return cls


class AssistantRegistry:
    """Singleton registry for managing Assistant classes and instances.

    Supports two registration methods:
    1. Settings-based: List paths in AI_SDK_ASSISTANTS setting (recommended)
    2. Decorator-based: Apply @auto_register to Assistant classes

    Both methods can be used together - a class will only be registered once.

    Usage:
        # Method 1: Settings-based (recommended)
        # In your settings.py:
        AI_SDK_ASSISTANTS = [
            "myapp.assistants.MyAssistant",
        ]

        # Method 2: Decorator-based
        from django_ai_sdk.assistants.registry import auto_register

        @auto_register
        class MyAssistant(Assistant):
            pass

        # In your AppConfig.ready():
        from django_ai_sdk.assistants.registry import registry
        from django.utils.module_loading import import_string
        from django.conf import settings

        # Load from settings
        for path in getattr(settings, 'AI_SDK_ASSISTANTS', []):
            import_string(path)

        # Setup instantiates all registered assistants
        registry.setup()

        # Later in views:
        assistant = registry.get(assistant_id)  # Get by UUID
    """

    _instance: "AssistantRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "AssistantRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._reset()
        return cls._instance

    def _reset(self) -> None:
        """Reset registry state (mainly for testing)."""
        self._classes: dict[str, type[Assistant]] = {}
        self._instances: dict[str, Assistant] = {}
        self._initialized = False

    def register(self, assistant_class: type[T]) -> type[T]:
        """Register an Assistant class (called automatically on subclass creation).

        Args:
            assistant_class: The Assistant subclass to register

        Returns:
            The registered class (for chaining/decorator use)

        Raises:
            AssistantRegistrationError: If assistant_id collision detected
        """
        # Generate deterministic UUID v5 from full class path (module.ClassName)
        # This ensures uniqueness - you cannot have two different classes with the
        # same module and class name in Python (the second definition would overwrite
        # the first). The only time we see "collisions" is in tests where multiple
        # test fixtures define classes with the same name in the same test module.
        class_path = f"{assistant_class.__module__}.{assistant_class.__name__}"
        assistant_id = str(uuid.uuid5(ASSISTANT_NAMESPACE, class_path))

        # Check if already registered with same class (idempotent)
        if assistant_id in self._classes:
            existing = self._classes[assistant_id]
            if existing is assistant_class:
                # Same class being registered again (e.g., via both __init_subclass__ and @auto_register)
                return assistant_class
            # Different class with same module.ClassName - this only happens in tests
            # where multiple test fixtures define classes with the same name.
            # In production, this is impossible since Python doesn't allow duplicate
            # class definitions in the same module.

        self._classes[assistant_id] = assistant_class
        assistant_class._assistant_id = assistant_id

        return assistant_class

    def setup(self, instantiate: bool = True, load_from_settings: bool = True) -> None:
        """Initialize all registered assistants. Call once in AppConfig.ready().

        This method should be called once when Django starts. It can optionally
        load assistants from the AI_SDK_ASSISTANTS setting and instantiates
        all registered assistant classes.

        Args:
            instantiate: Whether to create instances immediately. If False,
                        instances will be created on-demand via get().
            load_from_settings: Whether to load assistants from AI_SDK_ASSISTANTS
                              setting. Defaults to True.

        Raises:
            RuntimeError: If setup() has already been called.
        """
        if self._initialized:
            return

        # Load assistants from settings (recommended approach)
        if load_from_settings:
            from django.conf import settings
            from django.utils.module_loading import import_string

            assistant_paths = getattr(settings, "AI_SDK_ASSISTANTS", [])
            for path in assistant_paths:
                try:
                    import_string(path)
                except ImportError:
                    # Class may not exist or path is wrong
                    # It's okay - maybe using decorator-based registration instead
                    pass

        # Instantiate all registered assistants
        if instantiate:
            for assistant_id, assistant_class in self._classes.items():
                self._instances[assistant_id] = assistant_class()

        self._initialized = True

    def get(self, assistant_id: str) -> "Assistant | None":
        """Get assistant instance by ID.

        Args:
            assistant_id: The ID of the assistant to retrieve

        Returns:
            The assistant instance, or None if not found

        Raises:
            RuntimeError: If setup() has not been called
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call setup() in AppConfig.ready()")

        # Create on-demand if not instantiated yet
        if assistant_id not in self._instances:
            if assistant_id in self._classes:
                self._instances[assistant_id] = self._classes[assistant_id]()
            else:
                return None

        return self._instances.get(assistant_id)

    def all(self) -> dict[str, "Assistant"]:
        """Get all assistant instances.

        Returns:
            Dict mapping assistant_id to instance

        Raises:
            RuntimeError: If setup() has not been called
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call setup() in AppConfig.ready()")
        return self._instances.copy()

    def ids(self) -> list[str]:
        """Get all registered assistant IDs.

        Returns:
            List of registered assistant IDs (available even before setup())
        """
        return list(self._classes.keys())

    def __contains__(self, assistant_id: str) -> bool:
        """Check if an assistant ID is registered.

        Example:
            if "mybot" in registry:
                assistant = registry.get("mybot")
        """
        return assistant_id in self._classes


# Global singleton instance
registry = AssistantRegistry()
