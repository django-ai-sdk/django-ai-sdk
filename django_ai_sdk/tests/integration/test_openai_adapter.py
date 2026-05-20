"""
Integration tests for OpenAI Adapter.
"""

import pytest
import pytest_asyncio
import uuid
import json
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from django_ai_sdk.adapters.openai import OpenAIStream, OpenAIAgentStream
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.events import (
    MessageStartEvent,
    TextChunkEvent,
    MessageEndEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
    ReasoningChunkEvent,
)
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.tests.factories.message_factory import ChatMessageFactory


class TestOpenAIStream:
    """Test suite for OpenAIStream."""

    @pytest_asyncio.fixture
    async def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        return client

    @pytest_asyncio.fixture
    async def adapter(self, mock_openai_client):
        """Create OpenAIAdapter with mock client."""
        return OpenAIStream(
            client=mock_openai_client,
            model="gpt-4o-mini",
            instructions="You are a test assistant",
            store=True,
            storage_adapter=MemoryStorageAdapter(thread_id="test-thread"),
        )

    @pytest.mark.asyncio
    async def test_message_id_generation_consistency(self, adapter, mock_openai_client):
        """Test that message ID is generated once and used consistently."""
        # Mock streaming response with single chunk
        chunk = MagicMock()
        chunk.choices = [
            MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)
        ]
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.tool_calls = None

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=async_generator([chunk])
        )

        # Collect all events
        events = []
        async for event in adapter.stream([ChatMessageFactory.build(user=True)]):
            events.append(event)

        # Find message_start event
        start_events = [e for e in events if isinstance(e, MessageStartEvent)]
        assert len(start_events) == 1

        message_id = start_events[0].message_id

        # Verify it's a valid UUID
        assert isinstance(uuid.UUID(message_id), uuid.UUID)

        # Verify ID is consistent (no other message_start events)
        assert len([e for e in events if isinstance(e, MessageStartEvent)]) == 1

    @pytest.mark.asyncio
    async def test_text_streaming_events(self, adapter, mock_openai_client):
        """Test that text streaming produces correct event sequence."""
        # Mock multiple text chunks
        chunks = [
            MagicMock(
                choices=[
                    MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)
                ]
            ),
            MagicMock(
                choices=[
                    MagicMock(delta=MagicMock(content="world!"), finish_reason=None)
                ]
            ),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content=""), finish_reason="stop")]
            ),
        ]

        for chunk in chunks:
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=async_generator(chunks)
        )

        events = []
        async for event in adapter.stream([ChatMessageFactory.build(user=True)]):
            events.append(event)

        # Check event sequence
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], TextChunkEvent)
        assert events[1].content == "Hello "
        assert isinstance(events[2], TextChunkEvent)
        assert events[2].content == "world!"
        assert isinstance(events[-2], MessageEndEvent)

    @pytest.mark.asyncio
    async def test_reasoning_content_events(self, adapter, mock_openai_client):
        """Test that reasoning content produces reasoning events."""
        # Mock chunks with reasoning
        chunks = [
            MagicMock(
                choices=[
                    MagicMock(
                        delta=MagicMock(
                            content="", reasoning_content="Let me think..."
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            MagicMock(
                choices=[
                    MagicMock(delta=MagicMock(content="Answer!"), finish_reason=None)
                ]
            ),
        ]

        for chunk in chunks:
            if not hasattr(chunk.choices[0].delta, "reasoning_content"):
                chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=async_generator(chunks)
        )

        events = []
        async for event in adapter.stream([ChatMessageFactory.build(user=True)]):
            events.append(event)

        # Should have reasoning event
        reasoning_events = [e for e in events if isinstance(e, ReasoningChunkEvent)]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].content == "Let me think..."

    @pytest.mark.asyncio
    async def test_tool_call_event_sequence(self, adapter, mock_openai_client):
        """Test that tool calls produce start → input → output sequence."""
        # Mock tool call chunks - use proper attribute setting
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function = MagicMock()
        tool_call.function.name = (
            "search"  # Set actual attribute, not MagicMock name param
        )
        tool_call.function.arguments = '{"query": "test"}'

        chunks = [
            MagicMock(
                choices=[
                    MagicMock(
                        delta=MagicMock(content="", tool_calls=[tool_call]),
                        finish_reason=None,
                    )
                ]
            ),
        ]

        for chunk in chunks:
            chunk.choices[0].delta.reasoning_content = None

        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=async_generator(chunks)
        )

        events = []
        async for event in adapter.stream([ChatMessageFactory.build(user=True)]):
            events.append(event)

        # Check tool event sequence
        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_inputs = [e for e in events if isinstance(e, ToolInputCompleteEvent)]

        assert len(tool_starts) == 1
        assert tool_starts[0].tool_call_id == "call_123"
        assert len(tool_inputs) == 1
        assert tool_inputs[0].tool_input == {"query": "test"}

    @pytest.mark.asyncio
    async def test_error_event_on_exception(self, adapter, mock_openai_client):
        """Test that exceptions produce error events."""
        # Mock exception
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        events = []
        async for event in adapter.stream([ChatMessageFactory.build(user=True)]):
            events.append(event)

        # Should have error event
        from django_ai_sdk.events import ErrorEvent

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "API Error" in error_events[0].error_message


class TestOpenAIAgentStream:
    """Test suite for OpenAIAgentStream."""

    @pytest.mark.asyncio
    async def test_agent_adapter_generates_consistent_id(self):
        """Test that agent adapter generates message ID consistently."""
        from agents.agent import Agent

        agent = Agent(name="test", instructions="Test")
        adapter = OpenAIAgentStream(
            agent=agent,
            store=True,
            storage_adapter=MemoryStorageAdapter(thread_id="test-thread"),
        )

        # Mock the runner
        with patch.object(adapter, "runner") as mock_runner:
            mock_result = MagicMock()

            # stream_events should be an async generator, not AsyncMock
            async def mock_stream_events():
                return
                yield  # Makes this an async generator

            mock_result.stream_events = mock_stream_events
            mock_runner.run_streamed = MagicMock(return_value=mock_result)

            events = []
            async for event in adapter.stream([ChatMessageFactory.build(user=True)]):
                events.append(event)

            # Find message start
            start_events = [e for e in events if isinstance(e, MessageStartEvent)]
            assert len(start_events) == 1

            message_id = start_events[0].message_id
            assert isinstance(uuid.UUID(message_id), uuid.UUID)


# Helper function to create async generator
async def async_generator(items):
    """Convert list to async generator."""
    for item in items:
        yield item
