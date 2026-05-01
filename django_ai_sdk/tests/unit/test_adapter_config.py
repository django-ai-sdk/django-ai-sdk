"""
Unit tests for adapter configuration.

These tests verify BasePipelineAdapter and adapter configuration
without requiring actual streaming or LLM calls.
"""

import pytest


class TestBasePipelineAdapterConfig:
    """Test BasePipelineAdapter configuration attributes."""

    def test_adapter_has_common_class_attributes(self):
        """Verify BasePipelineAdapter common class attributes exist."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        assert hasattr(BasePipelineAdapter, "model")
        assert hasattr(BasePipelineAdapter, "instructions")
        assert hasattr(BasePipelineAdapter, "query")
        assert hasattr(BasePipelineAdapter, "merge_messages")

    def test_adapter_defaults(self):
        """Verify default attribute values."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        assert BasePipelineAdapter.model is None
        assert BasePipelineAdapter.instructions is None
        assert BasePipelineAdapter.query is None
        assert BasePipelineAdapter.merge_messages is False

    def test_adapter_instance_has_message_result(self):
        """Verify adapter class has message_result attribute."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        # These are instance attributes, not class attributes
        # Check that they're defined in __init__
        import inspect

        init_source = inspect.getsource(BasePipelineAdapter.__init__)
        assert "message_result" in init_source
        assert "_rag_sources" in init_source

    def test_adapter_is_abstract(self):
        """Verify BasePipelineAdapter cannot be instantiated."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        with pytest.raises(TypeError) as exc_info:
            BasePipelineAdapter()

        assert "abstract" in str(exc_info.value).lower()

    def test_adapter_has_abstract_methods(self):
        """Verify abstract methods are defined."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        abstract_methods = getattr(BasePipelineAdapter, "__abstractmethods__", set())
        required = {"get_messages", "stream"}
        assert required <= abstract_methods, f"Missing: {required - abstract_methods}"


class TestAdapterMergeMessagesFlag:
    """Test the merge_messages configuration flag."""

    def test_default_merge_messages_is_false(self):
        """Verify default merge_messages is False."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        assert BasePipelineAdapter.merge_messages is False

    def test_merge_messages_can_be_overridden(self):
        """Test that merge_messages can be set to True."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        # Create a concrete implementation for testing
        class TestAdapter(BasePipelineAdapter):
            async def get_messages(self, messages):
                return []

            async def stream(self, messages):
                return

        # Override the class attribute
        TestAdapter.merge_messages = True
        assert TestAdapter.merge_messages is True

    def test_merge_messages_type_is_bool(self):
        """Verify merge_messages is a boolean."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter

        assert isinstance(BasePipelineAdapter.merge_messages, bool)


class TestOpenAIAdapterConfig:
    """Test OpenAIAdapter specific configuration."""

    def test_openai_adapter_inherits_base(self):
        """Verify OpenAIAdapter extends BasePipelineAdapter."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter
        from django_ai_sdk.adapters.openai import OpenAIAdapter

        assert issubclass(OpenAIAdapter, BasePipelineAdapter)

    def test_openai_adapter_inherits_merge_messages(self):
        """Verify OpenAIAdapter inherits merge_messages from base."""
        from django_ai_sdk.adapters.openai import OpenAIAdapter

        # Should inherit from base (False by default)
        assert hasattr(OpenAIAdapter, "merge_messages")

    def test_openai_adapter_has_model_attribute(self):
        """Verify OpenAIAdapter has model attribute."""
        from django_ai_sdk.adapters.openai import OpenAIAdapter

        assert hasattr(OpenAIAdapter, "model")

    def test_openai_adapter_has_instructions_attribute(self):
        """Verify OpenAIAdapter has instructions attribute."""
        from django_ai_sdk.adapters.openai import OpenAIAdapter

        assert hasattr(OpenAIAdapter, "instructions")

    def test_openai_adapter_has_query_attribute(self):
        """Verify OpenAIAdapter has query attribute."""
        from django_ai_sdk.adapters.openai import OpenAIAdapter

        assert hasattr(OpenAIAdapter, "query")


class TestHaystackAdapterConfig:
    """Test HaystackAdapter specific configuration."""

    def test_haystack_adapter_inherits_base(self):
        """Verify HaystackAdapter extends BasePipelineAdapter."""
        from django_ai_sdk.adapters.base import BasePipelineAdapter
        from django_ai_sdk.adapters.haystack import HaystackAdapter

        assert issubclass(HaystackAdapter, BasePipelineAdapter)

    def test_haystack_adapter_inherits_merge_messages(self):
        """Verify HaystackAdapter inherits merge_messages from base."""
        from django_ai_sdk.adapters.haystack import HaystackAdapter

        assert hasattr(HaystackAdapter, "merge_messages")

    def test_haystack_adapter_has_required_methods(self):
        """Verify HaystackAdapter implements required methods."""
        from django_ai_sdk.adapters.haystack import HaystackAdapter

        required = ["get_messages", "stream"]
        for method in required:
            assert hasattr(HaystackAdapter, method), f"Missing: {method}"


class TestAdapterUtils:
    """Test adapter utilities."""

    def test_merge_messages_function_exists(self):
        """Verify merge_messages utility function exists."""
        from django_ai_sdk.adapters.utils import merge_messages

        assert callable(merge_messages)

    def test_merge_messages_accepts_messages_list(self):
        """Verify merge_messages accepts list of messages."""
        from django_ai_sdk.adapters.utils import merge_messages
        from django_ai_sdk.common import ChatMessage

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="user", content="World"),
        ]

        result = merge_messages(messages)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_merge_messages_returns_tuples(self):
        """Verify merge_messages returns list of tuples."""
        from django_ai_sdk.adapters.utils import merge_messages
        from django_ai_sdk.common import ChatMessage

        messages = [ChatMessage(role="user", content="Test")]
        result = merge_messages(messages)

        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        assert result[0][0] == "user"
        assert result[0][1] == "Test"

    def test_merge_messages_respects_max_history(self):
        """Verify merge_messages respects max_history parameter."""
        from django_ai_sdk.adapters.utils import merge_messages
        from django_ai_sdk.common import ChatMessage

        messages = [
            ChatMessage(role="user", content="1"),
            ChatMessage(role="user", content="2"),
            ChatMessage(role="user", content="3"),
        ]

        result = merge_messages(messages, max_history=2)
        assert len(result) <= 2


class TestAssistantAdapterIntegration:
    """Test Assistant integration with adapters."""

    def test_assistant_storage_attribute(self):
        """Verify Assistant has storage class attribute (named 'storage')."""
        from django_ai_sdk import Assistant

        # The class attribute is named 'storage' (not 'storage_adapter')
        assert hasattr(Assistant, "storage")

    def test_assistant_subclass_can_set_storage_adapter(self):
        """Verify Assistant subclass can configure storage_adapter."""
        from django_ai_sdk import Assistant
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        class TestAssistant(Assistant):
            name = "Test"
            storage_adapter = MemoryStorageAdapter

        # Subclass can set storage_adapter (used by __init__)
        assert hasattr(TestAssistant, "storage_adapter")
        assert TestAssistant.storage_adapter is MemoryStorageAdapter

    def test_assistant_protocol_attribute(self):
        """Verify Assistant has protocol class attribute."""
        from django_ai_sdk import Assistant

        assert hasattr(Assistant, "protocol")

    def test_assistant_can_override_adapter_config(self):
        """Verify Assistant can override adapter configuration."""
        from django_ai_sdk import Assistant
        from django_ai_sdk.storage.memory import MemoryStorageAdapter

        class TestAssistant(Assistant):
            name = "Test"
            storage_adapter = MemoryStorageAdapter

        assert TestAssistant.storage_adapter is MemoryStorageAdapter
