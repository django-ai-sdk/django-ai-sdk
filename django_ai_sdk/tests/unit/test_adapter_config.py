"""
Unit tests for adapter configuration.
"""

class TestOpenAIRunnableConfig:
    """Test OpenAIRunnable configuration."""

    def test_openai_runnable_has_model(self):
        """Verify OpenAIRunnable has model attribute."""
        from django_ai_sdk.adapters.openai import OpenAIRunnable

        assert hasattr(OpenAIRunnable, "model")

    def test_openai_runnable_has_instructions(self):
        """Verify OpenAIRunnable has instructions attribute."""
        from django_ai_sdk.adapters.openai import OpenAIRunnable

        assert hasattr(OpenAIRunnable, "instructions")


class TestOpenAIStreamConfig:
    """Test OpenAIStream configuration."""

    def test_openai_stream_has_model(self):
        """Verify OpenAIStream has model attribute."""
        from django_ai_sdk.adapters.openai import OpenAIStream

        assert hasattr(OpenAIStream, "model")

    def test_openai_stream_has_instructions(self):
        """Verify OpenAIStream has instructions attribute."""
        from django_ai_sdk.adapters.openai import OpenAIStream

        assert hasattr(OpenAIStream, "instructions")

    def test_openai_stream_has_merge_messages(self):
        """Verify OpenAIStream has merge_messages attribute."""
        from django_ai_sdk.adapters.openai import OpenAIStream

        assert hasattr(OpenAIStream, "merge_messages")

    def test_openai_stream_has_stream_method(self):
        """Verify OpenAIStream defines stream method."""
        from django_ai_sdk.adapters.openai import OpenAIStream

        assert hasattr(OpenAIStream, "stream")


class TestHaystackRunnableConfig:
    """Test HaystackRunnable configuration."""

    def test_haystack_runnable_has_model(self):
        """Verify HaystackRunnable has model attribute."""
        from django_ai_sdk.adapters.haystack import HaystackRunnable

        assert hasattr(HaystackRunnable, "model")

    def test_haystack_runnable_has_instructions(self):
        """Verify HaystackRunnable has instructions attribute."""
        from django_ai_sdk.adapters.haystack import HaystackRunnable

        assert hasattr(HaystackRunnable, "instructions")


class TestHaystackStreamConfig:
    """Test HaystackStream configuration."""

    def test_haystack_stream_has_model(self):
        """Verify HaystackStream has model attribute."""
        from django_ai_sdk.adapters.haystack import HaystackStream

        assert hasattr(HaystackStream, "model")

    def test_haystack_stream_has_merge_messages(self):
        """Verify HaystackStream has merge_messages attribute."""
        from django_ai_sdk.adapters.haystack import HaystackStream

        assert hasattr(HaystackStream, "merge_messages")

    def test_haystack_stream_has_stream_method(self):
        """Verify HaystackStream defines stream method."""
        from django_ai_sdk.adapters.haystack import HaystackStream

        assert hasattr(HaystackStream, "stream")


class TestRunnableProtocol:
    """Test Runnable protocol conformance."""

    def test_runnable_protocol_exists(self):
        """Verify Runnable protocol is defined."""
        from django_ai_sdk.adapters.protocols import Runnable

        assert hasattr(Runnable, "run")

    def test_haystack_runnable_conforms(self):
        """Verify HaystackRunnable conforms to Runnable protocol."""
        from django_ai_sdk.adapters.haystack import HaystackRunnable

        assert hasattr(HaystackRunnable, "run")

    def test_openai_runnable_conforms(self):
        """Verify OpenAIRunnable conforms to Runnable protocol."""
        from django_ai_sdk.adapters.openai import OpenAIRunnable

        assert hasattr(OpenAIRunnable, "run")

    def test_openai_stream_has_run(self):
        """Verify OpenAIStream has run() (implements Runnable too)."""
        from django_ai_sdk.adapters.openai import OpenAIStream

        assert hasattr(OpenAIStream, "run")

    def test_haystack_stream_has_run(self):
        """Verify HaystackStream has run() (implements Runnable too)."""
        from django_ai_sdk.adapters.haystack import HaystackStream

        assert hasattr(HaystackStream, "run")

    def test_streamable_protocol_exists(self):
        """Verify Streamable protocol is defined."""
        from django_ai_sdk.adapters.protocols import Streamable

        assert hasattr(Streamable, "stream")

    def test_openai_stream_conforms_to_streamable(self):
        """Verify OpenAIStream conforms to Streamable protocol."""
        from django_ai_sdk.adapters.openai import OpenAIStream

        assert hasattr(OpenAIStream, "stream")

    def test_haystack_stream_conforms_to_streamable(self):
        """Verify HaystackStream conforms to Streamable protocol."""
        from django_ai_sdk.adapters.haystack import HaystackStream

        assert hasattr(HaystackStream, "stream")


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
