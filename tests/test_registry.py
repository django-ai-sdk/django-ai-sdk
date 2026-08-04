"""Tests for the AssistantRegistry."""

import uuid

import pytest
from django_ai_sdk import Assistant
from django_ai_sdk.assistants.registry import (
    AssistantRegistry,
    AssistantRegistrationError,
    auto_register,
    registry,
)


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class TestAssistantRegistry:
    """Test suite for AssistantRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset global registry singleton before each test."""
        registry._reset()
        yield
        registry._reset()

    def test_auto_registration(self):
        """Test that Assistant subclasses are auto-registered with UUID v5 IDs."""

        @auto_register
        class TestBot(Assistant):
            name = "Test Bot"

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        # Check that an ID was registered and it's a valid UUID
        ids = registry.ids()
        assert len(ids) == 1
        assert is_valid_uuid(ids[0])
        assert ids[0] in registry

    def test_id_is_deterministic_uuid_v5(self):
        """Test that the same class always generates the same UUID v5 ID."""
        registry._reset()

        # First registration
        @auto_register
        class DeterministicAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        first_id = DeterministicAssistant._assistant_id

        # Reset and re-register (simulating restart)
        registry._reset()

        # Re-define same class with same decorator
        @auto_register
        class DeterministicAssistant(Assistant):  # noqa: F811
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        second_id = DeterministicAssistant._assistant_id

        # Same class path should generate same UUID
        assert first_id == second_id
        assert is_valid_uuid(first_id)

    def test_different_classes_get_different_ids(self):
        """Test that different classes get unique UUIDs."""

        @auto_register
        class FirstAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        @auto_register
        class SecondAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        ids = registry.ids()
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert all(is_valid_uuid(id_) for id_ in ids)

    def test_id_override_allowed(self):
        """Test that a different class with same ID can override (for testing)."""
        # This allows test classes with same names to work properly

        class OverrideAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        first_id = OverrideAssistant._assistant_id
        registry.register(OverrideAssistant)

        # Create another class with same module.name (simulates test scenario)
        class OverrideAssistant(Assistant):  # noqa: F811
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        second_id = OverrideAssistant._assistant_id
        # Should be same ID (based on module.name)
        assert first_id == second_id

        # Register again - should succeed (replaces previous)
        result = registry.register(OverrideAssistant)
        assert result is OverrideAssistant
        assert first_id in registry.ids()

    def test_base_assistant_not_registered(self):
        """Test that the base Assistant class is not auto-registered."""
        # The base Assistant class should not appear in registry
        assert len(registry.ids()) == 0

    def test_setup_instantiates_assistants(self):
        """Test that setup() creates instances of all assistants."""

        @auto_register
        class SetupAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = SetupAssistant._assistant_id
        registry.setup()

        assistant = registry.get(assistant_id)
        assert assistant is not None
        assert isinstance(assistant, SetupAssistant)
        assert assistant.assistant_id == assistant_id

    def test_setup_is_idempotent(self):
        """Test that calling setup() multiple times doesn't duplicate."""

        @auto_register
        class IdempotentAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = IdempotentAssistant._assistant_id
        registry.setup()
        first_instance = registry.get(assistant_id)

        registry.setup()  # Call again
        second_instance = registry.get(assistant_id)

        assert first_instance is second_instance

    def test_get_before_setup_raises(self):
        """Test that get() before setup() raises RuntimeError."""

        @auto_register
        class EarlyAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = EarlyAssistant._assistant_id

        with pytest.raises(RuntimeError) as exc_info:
            registry.get(assistant_id)

        assert "not initialized" in str(exc_info.value).lower()

    def test_get_unknown_returns_none(self):
        """Test that get() returns None for unknown ID."""

        @auto_register
        class KnownAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        registry.setup()

        # Random valid UUID that doesn't exist
        unknown_id = str(uuid.uuid4())
        assert registry.get(unknown_id) is None

    def test_all_returns_all_instances(self):
        """Test that all() returns dict of all instances."""

        @auto_register
        class FirstAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        @auto_register
        class SecondAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        registry.setup()

        all_assistants = registry.all()
        assert len(all_assistants) == 2
        # All values should be instances, all keys should be valid UUIDs
        for assistant_id, assistant in all_assistants.items():
            assert is_valid_uuid(assistant_id)
            assert isinstance(assistant, Assistant)

    def test_all_before_setup_raises(self):
        """Test that all() before setup() raises RuntimeError."""

        @auto_register
        class AnyAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        with pytest.raises(RuntimeError):
            registry.all()

    def test_ids_available_before_setup(self):
        """Test that ids() works even before setup()."""

        @auto_register
        class PreSetupAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = PreSetupAssistant._assistant_id

        # Can get IDs before setup
        assert assistant_id in registry.ids()
        assert is_valid_uuid(assistant_id)

        # But can't get instances
        with pytest.raises(RuntimeError):
            registry.get(assistant_id)

    def test_in_operator(self):
        """Test the 'in' operator for checking registration."""

        @auto_register
        class InOperatorAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = InOperatorAssistant._assistant_id

        assert assistant_id in registry
        assert "not-a-real-uuid" not in registry

    def test_assistant_id_set_on_class(self):
        """Test that _assistant_id is set on the registered class."""

        @auto_register
        class IdAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assert is_valid_uuid(IdAssistant._assistant_id)

    def test_singleton_instance(self):
        """Test that registry is a singleton."""
        reg1 = AssistantRegistry()
        reg2 = AssistantRegistry()
        assert reg1 is reg2
        assert reg1 is registry


class TestAssistantInfoMixin:
    """Test suite for AssistantInfoMixin."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset global registry singleton before each test."""
        registry._reset()
        yield
        registry._reset()

    def test_info_returns_metadata(self):
        """Test that info() returns correct metadata with UUID id."""

        @auto_register
        class InfoAssistant(Assistant):
            name = "Info Bot"
            model = "gpt-4"
            description = "An info bot"

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = InfoAssistant._assistant_id
        registry.setup()
        assistant = registry.get(assistant_id)

        info = assistant.info()
        assert info.id == assistant_id
        assert is_valid_uuid(info.id)
        assert info.name == "Info Bot"
        assert info.model == "gpt-4"
        assert info.description == "An info bot"
        assert info.class_name == "InfoAssistant"

    def test_assistant_id_property(self):
        """Test the assistant_id property returns UUID."""

        @auto_register
        class PropAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        registry.setup()
        assistant_id = PropAssistant._assistant_id
        assistant = registry.get(assistant_id)

        assert assistant.assistant_id == assistant_id
        assert is_valid_uuid(assistant.assistant_id)

    def test_assistant_id_fallback(self):
        """Test assistant_id fallback generates consistent UUID."""

        # Create instance without going through registry
        @auto_register
        class FallbackAssistant(Assistant):
            async def get_pipeline_adapter(self, thread_id=None):
                pass

        # Manually create instance (won't have _assistant_id cached yet)
        assistant = FallbackAssistant()

        # Should generate UUID from class path
        first_id = assistant.assistant_id
        assert is_valid_uuid(first_id)

        # Second call should return cached value
        second_id = assistant.assistant_id
        assert first_id == second_id

    def test_info_with_get_name_method(self):
        """Test info() uses get_name() when name attribute is None."""

        @auto_register
        class DynamicNameAssistant(Assistant):
            name = None

            def get_name(self):
                return "Dynamic Name"

            async def get_pipeline_adapter(self, thread_id=None):
                pass

        assistant_id = DynamicNameAssistant._assistant_id
        registry.setup()
        assistant = registry.get(assistant_id)

        info = assistant.info()
        assert info.name == "Dynamic Name"
        assert is_valid_uuid(info.id)


class TestAbstractAssistants:
    """`abstract = True` marks a shared base meant only to be subclassed. It exists so
    a project can factor common config (model, permissions, storage) into a base class
    without that base showing up as a usable assistant.
    """

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        registry._reset()
        yield
        registry._reset()

    def test_an_abstract_base_is_not_registered(self):
        class SharedBase(Assistant):
            abstract = True
            name = "Shared Base"

        # register() never ran, so the class-level default is untouched.
        assert SharedBase._assistant_id == ""
        assert SharedBase not in registry._classes.values()

    def test_a_concrete_subclass_of_an_abstract_base_is_registered(self):
        """`abstract` is read off the class's own __dict__, so subclasses don't inherit
        it and don't have to restate `abstract = False`."""

        class SharedBase(Assistant):
            abstract = True
            name = "Shared Base"

        class RealBot(SharedBase):
            name = "Real Bot"

        assert RealBot._assistant_id
        assert registry._classes[RealBot._assistant_id] is RealBot

    def test_abstract_also_blocks_explicit_registration(self):
        """Checked in register() rather than __init_subclass__, so @auto_register and
        AI_SDK_ASSISTANTS skip it too — one rule, every path."""

        @auto_register
        class DecoratedBase(Assistant):
            abstract = True
            name = "Decorated Base"

        assert DecoratedBase not in registry._classes.values()

    def test_setup_does_not_instantiate_an_abstract_base(self):
        class SharedBase(Assistant):
            abstract = True
            name = "Shared Base"

        class RealBot(SharedBase):
            name = "Real Bot"

        registry.setup()

        assert [type(i) for i in registry._instances.values()] == [RealBot]
