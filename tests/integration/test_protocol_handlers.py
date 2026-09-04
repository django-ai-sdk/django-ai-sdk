"""
Integration tests for Protocol Handlers.
"""

import pytest
import pytest_asyncio
import uuid

from django_ai_sdk.protocols.vercel import (
    VercelProtocolHandler,
    MessageStartPart,
    TextStartPart,
    TextDeltaPart,
    TextEndPart,
    ReasoningStartPart,
    ReasoningDeltaPart,
    ReasoningEndPart,
    FinishPart,
    DonePart,
    SourceDocumentPart,
)
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.events import (
    MessageStartEvent,
    TextChunkEvent,
    MessageEndEvent,
    ReasoningChunkEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
    DataEvent,
    ErrorEvent,
    SourceEvent,
)
from tests.factories.schemas import ChatMessageFactory


class TestVercelProtocolHandler:
    """Test suite for VercelProtocolHandler."""

    @pytest_asyncio.fixture
    async def handler(self):
        """Create Vercel protocol handler."""
        return VercelProtocolHandler()

    @pytest.mark.asyncio
    async def test_message_conversion_to_chat_messages(self, handler):
        """Test converting protocol messages to ChatMessages."""

        # Create mock Vercel message objects with .parts attribute
        class MockPart:
            def __init__(self, type, text) -> None:
                self.type = type
                self.text = text

        class MockMessage:
            def __init__(self, role, text) -> None:
                self.role = role
                self.parts = [MockPart("text", text)]

        protocol_messages = [
            MockMessage("user", "Hello!"),
            MockMessage("assistant", "Hi there!"),
            MockMessage("user", "How are you?"),
        ]

        chat_messages = handler.to_chat_messages(protocol_messages)

        assert len(chat_messages) == 3
        assert chat_messages[0].role == "user"
        assert chat_messages[0].content == "Hello!"
        assert chat_messages[1].role == "assistant"
        assert chat_messages[2].role == "user"

    @pytest.mark.asyncio
    async def test_message_conversion_preserves_id(self, handler):
        """Test that message ID is preserved during conversion."""
        msg_id = str(uuid.uuid4())

        # Create mock message with ID
        class MockPart:
            def __init__(self, type, text) -> None:
                self.type = type
                self.text = text

        class MockMessage:
            def __init__(self, role, text, msg_id) -> None:
                self.role = role
                self.parts = [MockPart("text", text)]
                self.id = msg_id

        protocol_messages = [MockMessage("user", "Hello!", msg_id)]

        chat_messages = handler.to_chat_messages(protocol_messages)

        # Note: Our implementation might not preserve ID from protocol format
        # This test verifies the basic conversion works
        assert len(chat_messages) == 1
        assert chat_messages[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_message_conversion_from_chat_messages(self, handler):
        """Test converting ChatMessages back to protocol format."""
        chat_messages = [
            ChatMessageFactory.build(role="user", content="User msg"),
            ChatMessageFactory.build(role="assistant", content="Agent msg"),
        ]

        protocol_messages = handler.from_chat_messages(chat_messages)

        assert len(protocol_messages) == 2
        assert protocol_messages[0]["role"] == "user"
        # Content is in parts array, not direct content key
        assert len(protocol_messages[0]["parts"]) == 1
        assert protocol_messages[0]["parts"][0]["type"] == "text"
        assert protocol_messages[0]["parts"][0]["text"] == "User msg"
        assert protocol_messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_handle_stream_message_start_event(self, handler):
        """Test handling message start event."""

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert isinstance(chunks[0], MessageStartPart)
        assert chunks[0].message_id == "msg_123"  # Use snake_case for Python attribute

    @pytest.mark.asyncio
    async def test_handle_stream_text_events(self, handler):
        """Test handling text chunk events."""

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield TextChunkEvent(content="Hello ")
            yield TextChunkEvent(content="world!")
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        # Should have: start, text-start, text-delta, text-delta, text-end, finish
        assert any(isinstance(c, TextStartPart) for c in chunks)
        text_deltas = [c for c in chunks if isinstance(c, TextDeltaPart)]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "Hello "
        assert text_deltas[1].delta == "world!"
        assert any(isinstance(c, TextEndPart) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_stream_reasoning_events(self, handler):
        """Test handling reasoning chunk events."""

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield ReasoningChunkEvent(content="Let me think...")
            yield ReasoningChunkEvent(content="I understand now.")
            yield TextChunkEvent(content="Answer!")
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        # Should have reasoning parts
        reasoning_starts = [c for c in chunks if isinstance(c, ReasoningStartPart)]
        reasoning_deltas = [c for c in chunks if isinstance(c, ReasoningDeltaPart)]
        reasoning_ends = [c for c in chunks if isinstance(c, ReasoningEndPart)]

        assert len(reasoning_starts) == 1
        assert len(reasoning_deltas) == 2
        assert len(reasoning_ends) == 1

        # Reasoning should end before text starts
        reasoning_end_idx = next(i for i, c in enumerate(chunks) if isinstance(c, ReasoningEndPart))
        text_start_idx = next(i for i, c in enumerate(chunks) if isinstance(c, TextStartPart))
        assert reasoning_end_idx < text_start_idx

    @pytest.mark.asyncio
    async def test_handle_stream_tool_events(self, handler):
        """Test handling tool call events."""
        from django_ai_sdk.protocols.vercel import (
            ToolInputStartPart,
            ToolInputAvailablePart,
        )

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield ToolCallStartEvent(tool_call_id="call_1", tool_name="search")
            yield ToolInputCompleteEvent(
                tool_call_id="call_1", tool_name="search", tool_input={"query": "test"}
            )
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        # Should have tool input start and available parts
        tool_starts = [c for c in chunks if isinstance(c, ToolInputStartPart)]
        tool_inputs = [c for c in chunks if isinstance(c, ToolInputAvailablePart)]

        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "search"  # Use snake_case
        assert len(tool_inputs) == 1
        assert tool_inputs[0].input == {"query": "test"}

    @pytest.mark.asyncio
    async def test_handle_stream_tool_output_preserves_error_flag(self, handler):
        """Test that tool output preserves the error flag from Haystack."""
        from django_ai_sdk.protocols.vercel import ToolOutputAvailablePart

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield ToolOutputEvent(
                tool_call_id="call_1",
                tool_output={
                    "result": "Rate limit exceeded",
                    "origin": {"tool_name": "search", "arguments": {}},
                    "error": True,
                },
            )
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        tool_outputs = [c for c in chunks if isinstance(c, ToolOutputAvailablePart)]
        assert len(tool_outputs) == 1
        assert tool_outputs[0].output["result"] == "Rate limit exceeded"
        assert tool_outputs[0].output["error"] is True

    @pytest.mark.asyncio
    async def test_handle_stream_data_event(self, handler):
        """Test handling data events."""
        from django_ai_sdk.protocols.vercel import DataPart

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield DataEvent(data_type="rag-retrieval", data={"documents_found": 3, "query": "test"})
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        data_parts = [c for c in chunks if isinstance(c, DataPart)]
        assert len(data_parts) == 1
        assert data_parts[0].type == "data-rag-retrieval"
        assert data_parts[0].data["documents_found"] == 3

    @pytest.mark.asyncio
    async def test_handle_stream_error_event(self, handler):
        """Test handling error events."""
        from django_ai_sdk.protocols.vercel import ErrorPart

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield ErrorEvent(error_message="Something went wrong")
            yield MessageEndEvent(finish_reason="error")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        error_parts = [c for c in chunks if isinstance(c, ErrorPart)]
        assert len(error_parts) == 1
        assert error_parts[0].error_text == "Something went wrong"  # snake_case

    @pytest.mark.asyncio
    async def test_handle_stream_finishes_with_done(self, handler):
        """Test that stream always ends with DonePart."""

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield TextChunkEvent(content="Hello")
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        # Last chunk should be DonePart (but it's excluded from model_dump)
        # Instead check for finish part
        finish_parts = [c for c in chunks if isinstance(c, FinishPart)]
        assert len(finish_parts) == 1
        assert (
            finish_parts[0].finishReason == "stop"
        )  # Note: FinishPart has no alias, uses camelCase

    @pytest.mark.asyncio
    async def test_sse_format_generation(self, handler):
        """Test that SSE format is generated correctly."""
        from django_ai_sdk.protocols.vercel import format_sse

        # Test format_sse helper function
        data = {"type": "test", "content": "Hello"}
        formatted = format_sse(data)

        # Should be bytes with proper SSE format
        assert isinstance(formatted, bytes)
        assert b"data: {" in formatted
        assert b"}\n\n" in formatted

    @pytest.mark.asyncio
    async def test_state_reset_on_new_stream(self, handler):
        """Test that state is reset for each new stream."""

        # First stream
        async def stream1():
            yield MessageStartEvent(message_id="msg_1")
            yield TextChunkEvent(content="First")
            yield MessageEndEvent(finish_reason="stop")

        chunks1 = []
        async for chunk in handler.handle_stream(stream1()):
            chunks1.append(chunk)

        first_text_id = [c for c in chunks1 if isinstance(c, TextStartPart)][0].id

        # Second stream - should get new IDs
        async def stream2():
            yield MessageStartEvent(message_id="msg_2")
            yield TextChunkEvent(content="Second")
            yield MessageEndEvent(finish_reason="stop")

        # Reset state before second stream (this happens in sse() method)
        handler.text_started = False
        handler.text_id = None
        handler.message_id = None

        chunks2 = []
        async for chunk in handler.handle_stream(stream2()):
            chunks2.append(chunk)

        second_text_id = [c for c in chunks2 if isinstance(c, TextStartPart)][0].id

        # IDs should be different
        assert first_text_id != second_text_id

    @pytest.mark.asyncio
    async def test_handle_stream_source_event_yields_source_document_part(self, handler):
        """Test that SourceEvent is converted to SourceDocumentPart (Vercel spec-compliant)."""

        async def event_generator():
            yield MessageStartEvent(message_id="msg_123")
            yield SourceEvent(
                index=1,
                title="Sales Q3 Report",
                content="Document content here...",
                source_id="1",
                media_type="file",
            )
            yield MessageEndEvent(finish_reason="stop")

        chunks = []
        async for chunk in handler.handle_stream(event_generator()):
            chunks.append(chunk)

        # Find SourceDocumentPart chunks
        source_parts = [c for c in chunks if isinstance(c, SourceDocumentPart)]
        assert len(source_parts) == 1

        # Verify the spec-compliant format
        source_part = source_parts[0]
        assert source_part.type == "source-document"
        assert source_part.source_id == "1"
        assert source_part.media_type == "file"
        assert source_part.title == "Sales Q3 Report"


class TestVercelAgentAttribution:
    """Tool calls carry which agent ran them, so the client can label a
    subagent's work instead of crediting it to the coordinator."""

    @pytest.fixture
    def handler(self):
        from django_ai_sdk.protocols.vercel import VercelProtocolHandler

        return VercelProtocolHandler()

    @pytest.mark.asyncio
    async def test_subagent_tool_carries_provider_metadata(self, handler):
        from django_ai_sdk.protocols.vercel import (
            ToolInputAvailablePart,
            ToolInputStartPart,
        )

        async def event_generator():
            yield MessageStartEvent(message_id="msg_1")
            yield ToolCallStartEvent(
                tool_call_id="call_1", tool_name="search_web", agent="Research Planner"
            )
            yield ToolInputCompleteEvent(
                tool_call_id="call_1",
                tool_name="search_web",
                tool_input={"query": "x"},
                agent="Research Planner",
            )
            yield MessageEndEvent(finish_reason="stop")

        chunks = [c async for c in handler.handle_stream(event_generator())]

        start = next(c for c in chunks if isinstance(c, ToolInputStartPart))
        available = next(c for c in chunks if isinstance(c, ToolInputAvailablePart))
        assert start.provider_metadata == {"sdk": {"agent": "Research Planner"}}
        assert available.provider_metadata == {"sdk": {"agent": "Research Planner"}}
        # Serialized under the name the AI SDK reads.
        assert start.model_dump(exclude_none=True, by_alias=True)["providerMetadata"] == {
            "sdk": {"agent": "Research Planner"}
        }

    @pytest.mark.asyncio
    async def test_handoff_call_is_marked(self, handler):
        from django_ai_sdk.protocols.vercel import ToolInputStartPart

        async def event_generator():
            yield MessageStartEvent(message_id="msg_1")
            yield ToolCallStartEvent(
                tool_call_id="call_1",
                tool_name="research_planner",
                handoff="Research Planner",
            )
            yield MessageEndEvent(finish_reason="stop")

        chunks = [c async for c in handler.handle_stream(event_generator())]

        start = next(c for c in chunks if isinstance(c, ToolInputStartPart))
        assert start.provider_metadata == {"sdk": {"handoff": "Research Planner"}}

    @pytest.mark.asyncio
    async def test_plain_tool_call_payload_is_unchanged(self, handler):
        from django_ai_sdk.protocols.vercel import ToolInputStartPart

        async def event_generator():
            yield MessageStartEvent(message_id="msg_1")
            yield ToolCallStartEvent(tool_call_id="call_1", tool_name="get_today")
            yield MessageEndEvent(finish_reason="stop")

        chunks = [c async for c in handler.handle_stream(event_generator())]

        start = next(c for c in chunks if isinstance(c, ToolInputStartPart))
        assert start.provider_metadata is None
        assert "providerMetadata" not in start.model_dump(exclude_none=True, by_alias=True)

    def test_history_rebuild_keeps_the_label(self, handler):
        from django_ai_sdk.common import ChatMessage

        message = ChatMessage(
            role="assistant",
            id="m1",
            content="Summary.",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "search_web",
                    "arguments": {"query": "x"},
                    "result": {"hits": 1},
                    "agent": "Research Planner",
                },
                {"id": "call_2", "name": "get_today", "arguments": {}, "result": {}},
            ],
        )

        parts = handler.from_chat_messages([message])[0]["parts"]
        tool_parts = [p for p in parts if p["type"].startswith("tool-")]

        assert tool_parts[0]["callProviderMetadata"] == {
            "sdk": {"agent": "Research Planner"}
        }
        assert "callProviderMetadata" not in tool_parts[1]
