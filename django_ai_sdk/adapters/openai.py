import hashlib
import json
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from agents.items import ItemHelpers
from agents.run import RunConfig, Runner
from openai.types.chat import ChatCompletionMessageParam

from django_ai_sdk.adapters.base import BasePipelineAdapter
from django_ai_sdk.adapters.utils import merge_messages, normalize_usage
from django_ai_sdk.common import (
    ChatMessage,
    MessageChunk,
    StreamWriter,
)
from django_ai_sdk.events import (
    DataEvent,
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    ReasoningChunkEvent,
    StreamEndEvent,
    StreamEvent,
    TextChunkEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
)
from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI


logger = get_logger(__name__)


class OpenAIAdapter(BasePipelineAdapter):
    """
    OpenAI adapter that emits normalized events.
    """

    def __init__(
        self,
        client: "AsyncOpenAI",
        model: str | None = None,
        instructions: str | None = None,
        store: bool = True,
        storage_adapter: Any = None,
        rag_pipeline: Any = None,
    ) -> None:
        self.client = client
        self.model = model
        self.instructions = instructions
        self.store = store
        self.storage_adapter = storage_adapter
        self.rag_pipeline = rag_pipeline  # RAG adapter with .retrieve() method
        self.message_result: ChatMessage | None = None  # Complete result
        self._rag_sources: list[dict] = []

        logger.debug(
            f"""
            OpenAI adapter initialized with
            model={model}, store={store},
            storage_adapter={type(storage_adapter).__name__ if storage_adapter else None},
            rag_pipeline={type(rag_pipeline).__name__ if rag_pipeline else None}
            """
        )

    def get_messages(self, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
        """Convert internal conversation messages to OpenAI format.

        Note: This method returns ONLY the conversation messages (user/assistant).
        System messages (instructions, RAG context) are added separately by the stream() method.

        Message merging is controlled by merge_messages flag (default: False).
        """
        logger.debug(
            f"Converting {len(messages)} internal messages to OpenAI format (merge={self.merge_messages})"
        )

        # Filter to user/assistant only
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        result: list[ChatCompletionMessageParam] = []

        if self.merge_messages:
            for role, content in merge_messages(conversation):
                result.append(
                    cast("ChatCompletionMessageParam", {"role": role, "content": content})
                )
        else:
            result.extend(
                [
                    cast("ChatCompletionMessageParam", {"role": msg.role, "content": msg.content})
                    for msg in conversation
                ]
            )

        logger.debug(f"Message conversion complete: {len(result)} conversation messages")
        return result

    def _convert_to_message_chunk(self, openai_chunk: Any) -> list[MessageChunk]:
        """Convert OpenAI chunk to MessageChunk(s)."""
        chunks = []

        if not openai_chunk.choices:
            return chunks

        choice = openai_chunk.choices[0]
        delta = choice.delta

        # Reasoning content (for o1, o3-mini, DeepSeek, etc.)
        if getattr(delta, "reasoning_content", None):
            chunks.append(
                MessageChunk(
                    type="reasoning",
                    content=delta.reasoning_content,
                    metadata={"id": getattr(openai_chunk, "id", None)},
                )
            )

        # Text content
        if delta.content:
            chunks.append(
                MessageChunk(
                    type="text",
                    content=delta.content,
                    metadata={"id": getattr(openai_chunk, "id", None)},
                )
            )

        # Tool calls
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.function and tool_call.id:
                    tool_call_id = tool_call.id  # Use OpenAI's ID - never generate

                    # Tool call start
                    if tool_call.function.name:
                        chunks.append(
                            MessageChunk(
                                type="tool_call_start",
                                content={
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_call.function.name,
                                },
                                metadata={"model": self.model},
                            )
                        )

                    # Tool input
                    if tool_call.function.arguments:
                        try:
                            tool_input = json.loads(tool_call.function.arguments)
                            chunks.append(
                                MessageChunk(
                                    type="tool_input",
                                    content={
                                        "tool_call_id": tool_call_id,
                                        "tool_name": tool_call.function.name,
                                        "tool_input": tool_input,
                                    },
                                    metadata={"model": self.model},
                                )
                            )
                        except json.JSONDecodeError as e:
                            # Emit error chunk for malformed JSON
                            chunks.append(
                                MessageChunk(
                                    type="error",
                                    content={
                                        "error_message": f"Failed to parse tool arguments: {e}",
                                        "tool_call_id": tool_call_id,
                                    },
                                    metadata={"model": self.model},
                                )
                            )

        return chunks

    def _chunk_to_event(self, message_chunk: MessageChunk) -> StreamEvent | None:
        """Convert MessageChunk to appropriate StreamEvent."""
        match message_chunk.type:
            case "reasoning":
                return ReasoningChunkEvent(content=message_chunk.content)

            case "text":
                return TextChunkEvent(content=message_chunk.content)

            case "tool_call_start":
                return ToolCallStartEvent(
                    tool_call_id=message_chunk.content["tool_call_id"],
                    tool_name=message_chunk.content["tool_name"],
                )

            case "tool_input":
                return ToolInputCompleteEvent(
                    tool_call_id=message_chunk.content["tool_call_id"],
                    tool_name=message_chunk.content["tool_name"],
                    tool_input=message_chunk.content["tool_input"],
                )

            case "error":
                return ErrorEvent(error_message=message_chunk.content["error_message"])

        return None

    async def stream(  # type: ignore
        self,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[StreamEvent, None]:
        logger.debug(f"Starting OpenAI stream with {len(messages)} input messages")

        # Build message list fresh to ensure proper structure
        openai_messages: list[ChatCompletionMessageParam] = []

        # 1. Add instructions as separate system message
        if self.instructions:
            openai_messages.append(
                cast("ChatCompletionMessageParam", {"role": "system", "content": self.instructions})
            )
            logger.debug("Added system instructions (separate message)")

        # 2. Inject RAG context if configured (as separate, clearly marked system message)
        self.query: str | None = None  # Store last user query for reuse in events
        if self.rag_pipeline and messages:
            # Get the last user message as query
            for msg in reversed(list(messages)):
                if msg.role == "user":
                    self.query = msg.content
                    break

            if self.query:
                # Retrieve documents using RAG adapter
                rag_result = await self.rag_pipeline.retrieve(self.query)

                if rag_result.sources:
                    # Format context and add as separate system message with clear delimiter
                    context = self.rag_pipeline.format_context(rag_result)
                    rag_system_message = cast(
                        "ChatCompletionMessageParam",
                        {
                            "role": "system",
                            "content": f"Retrieved context for this query:\n\n{context}\n\nUse the above context to answer the user's question. If the context doesn't contain relevant information, rely on your training data.",
                        },
                    )
                    openai_messages.append(rag_system_message)

                    # Store sources for StreamWriter
                    self._rag_sources = [s.model_dump() for s in rag_result.sources]
                    logger.debug(
                        f"Added RAG context as separate system message with {len(rag_result.sources)} sources"
                    )

        # 3. Add conversation history (get_messages now returns only conversation, no system messages)
        conversation_messages = self.get_messages(messages)
        openai_messages.extend(cast("list[ChatCompletionMessageParam]", conversation_messages))

        logger.debug(f"Built final message list: {len(openai_messages)} total messages")

        # Generate message ID at adapter level (single source of truth)
        message_id = str(uuid.uuid4())
        logger.debug(f"Generated message ID: {message_id}")

        # Create StreamWriter with pre-generated ID (if storage enabled)
        stream_writer = None
        if self.store and self.storage_adapter:
            stream_writer = StreamWriter(
                adapter_type="openai",
                message_id=message_id,
                model=self.model or "",
                role="assistant",
                storage_callback=self.storage_adapter.storage_callback,
            )
            # Add RAG sources if available
            if self._rag_sources:
                stream_writer.message.sources = self._rag_sources
            logger.debug(f"StreamWriter configured with ID: {message_id}")
        else:
            logger.debug("Message storage disabled or no storage adapter")

        _finalize_called = False

        try:
            # Start message - use same ID for SSE
            yield MessageStartEvent(message_id=message_id)

            # Emit RAG retrieval data event if sources were found
            if self._rag_sources and self.query:
                # Build sources array for Sources component
                sources_data = []
                for i, source in enumerate(self._rag_sources):
                    # Extract title from meta or generate from content/id
                    title = None
                    if isinstance(source.get("meta"), dict):
                        title = (
                            source["meta"].get("title")
                            or source["meta"].get("topic")
                            or source["meta"].get("id")
                        )
                    if not title:
                        title = source.get("id") or f"Document {i + 1}"

                    # Get preview text
                    preview = ""
                    if isinstance(source.get("content"), str):
                        preview = source["content"][:200]

                    sources_data.append(
                        {
                            "title": title,
                            "preview": preview,
                            "score": source.get("score"),
                        }
                    )

                yield DataEvent(
                    data_type="rag-retrieval",
                    data={
                        "query": self.query,
                        "documents_found": len(self._rag_sources),
                        "sources": sources_data,
                    },
                )
                logger.debug(
                    f"Emitted data-rag-retrieval event with {len(self._rag_sources)} sources"
                )

            # Make streaming request
            assert self.model is not None, "OpenAI model is required but not provided"

            logger.debug(f"Making OpenAI streaming request with model: {self.model}")
            response_stream = await self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=openai_messages,
                stream=True,
            )

            logger.debug("OpenAI streaming request initiated successfully")

            chunk_count = 0
            text_chunks = 0
            tool_chunks = 0

            # Process chunks
            async for openai_chunk in response_stream:
                chunk_count += 1
                logger.debug(f"Processing OpenAI chunk #{chunk_count}")

                # Convert OpenAI chunk to MessageChunk(s)
                message_chunks = self._convert_to_message_chunk(openai_chunk)
                logger.debug(f"Converted to {len(message_chunks)} message chunks")

                # Process each MessageChunk
                for chunk in message_chunks:
                    if chunk.type == "text":
                        text_chunks += 1
                    elif chunk.type in ("tool_call_start", "tool_input"):
                        tool_chunks += 1

                    # Add to stream writer
                    if stream_writer:
                        stream_writer.add_chunk(chunk)

                    # Convert to event and yield
                    event = self._chunk_to_event(chunk)
                    if event:
                        yield event

                # Handle finish reason
                if openai_chunk.choices and openai_chunk.choices[0].finish_reason:
                    finish_reason = openai_chunk.choices[0].finish_reason
                    logger.debug(f"Stream finished with reason: {finish_reason}")

                    # Extract usage from final chunk
                    usage = None
                    if hasattr(openai_chunk, "usage") and openai_chunk.usage:
                        usage = normalize_usage(
                            {
                                "prompt_tokens": openai_chunk.usage.prompt_tokens,
                                "completion_tokens": openai_chunk.usage.completion_tokens,
                                "total_tokens": openai_chunk.usage.total_tokens,
                            }
                        )
                        logger.debug(f"Usage: {usage}")

                    # Finalize message
                    if stream_writer:
                        logger.debug("Finalizing stored message")
                        self.message_result = await stream_writer.finalize(
                            finish_reason, usage=usage
                        )
                        _finalize_called = True
                        logger.debug(
                            f"Message stored with {len(self.message_result.content)} characters"
                        )

                    yield MessageEndEvent(finish_reason=finish_reason, usage=usage)
                    break

            logger.debug(
                f"OpenAI stream processing complete: {chunk_count} chunks, {text_chunks} text, {tool_chunks} tools"
            )

        except Exception as openai_error:
            logger.error(f"OpenAI streaming error: {type(openai_error).__name__}: {openai_error}")
            logger.debug(
                f"OpenAI error occurred after processing chunks, stream_writer={'exists' if stream_writer else 'none'}"
            )

            # Handle error in stream writer
            if stream_writer:
                logger.debug("Adding error chunk to stream writer")
                error_chunk = MessageChunk(
                    type="error",
                    content={"error_message": str(openai_error)},
                    metadata={"model": self.model, "source": "openai_adapter"},
                )
                stream_writer.add_chunk(error_chunk)
                self.message_result = await stream_writer.finalize("error")
                _finalize_called = True
                logger.debug("Error message finalized in storage")

            yield ErrorEvent(error_message=str(openai_error))
            return

        finally:
            if stream_writer and not _finalize_called:
                logger.debug("Finalizing with cancelled reason - connection may have dropped")
                self.message_result = await stream_writer.finalize("cancelled")

        logger.info("OpenAI stream completed successfully")

        # Stream termination
        yield StreamEndEvent()


class OpenAIAgentAdapter(BasePipelineAdapter):
    """OpenAI agents adapter that emits normalized events."""

    def __init__(
        self,
        agent: Any,
        runner_config: RunConfig | None = None,
        store: bool = True,
        storage_adapter: Any = None,
    ) -> None:
        self.agent = agent
        self.runner = Runner
        self.runner_config = runner_config or None
        self.store: bool = store
        self.storage_adapter = storage_adapter

        # Track tool calls: {tool_call_id: (tool_name, tool_input)}
        # TODO: move into schema, typed object
        self._tool_call_registry: dict[str, tuple[str, Any]] = {}
        self.message_result: ChatMessage | None = None

    def _make_tool_id(self, tool_name: str, tool_input: Any) -> str:
        """Create deterministic tool ID."""
        raw = tool_name + json.dumps(tool_input, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"tool_{digest}"

    def get_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert internal ChatMessage format to OpenAI Agents format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def get_input(self, messages: list["ChatMessage"]) -> str:
        """Extract the last user message as input for the agent."""
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            raise ValueError(
                "No user message found - agent requires at least one user message to process"
            )
        return user_messages[-1].content

    async def stream(  # type: ignore
        self,
        messages: list["ChatMessage"],
    ) -> AsyncGenerator["StreamEvent", None]:
        agent_input = self.get_input(messages)
        self._tool_call_registry.clear()  # Clear registry for new stream
        pending_tool_calls = []  # Track tool calls waiting for output

        # Generate message ID at adapter level (single source of truth)
        message_id = str(uuid.uuid4())
        logger.debug(f"Generated message ID: {message_id}")

        # Create StreamWriter with pre-generated ID
        stream_writer = None
        if self.store and self.storage_adapter:
            stream_writer = StreamWriter(
                adapter_type="openai_agent",
                message_id=message_id,
                model="openai-agent",
                role="assistant",
                storage_callback=self.storage_adapter.storage_callback,
            )
            logger.debug(f"StreamWriter configured with ID: {message_id}")
        else:
            logger.debug("Message storage disabled or no storage adapter")

        try:
            yield MessageStartEvent(message_id=message_id)

            result = self.runner.run_streamed(
                self.agent,
                input=agent_input,
                run_config=self.runner_config,  # type: ignore[arg-type] # OpenAI RunConfig type not available
            )

            async for event in result.stream_events():
                if event.type != "run_item_stream_event":
                    continue

                item = event.item

                # Tool call input
                if item.type == "tool_call_item":
                    tool_name = None
                    if hasattr(item, "function") and hasattr(item.function, "name"):
                        tool_name = item.function.name
                    elif hasattr(item, "tool") and hasattr(item.tool, "name"):
                        tool_name = item.tool.name
                    elif hasattr(item, "name"):
                        tool_name = item.name
                    else:
                        tool_name = "unknown"

                    tool_input = getattr(item, "input", {})
                    tool_call_id = self._make_tool_id(tool_name, tool_input)

                    # Register the tool call
                    self._tool_call_registry[tool_call_id] = (tool_name, tool_input)
                    pending_tool_calls.append(tool_call_id)

                    yield ToolCallStartEvent(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )

                    if tool_input:
                        yield ToolInputCompleteEvent(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            tool_input=tool_input
                            if isinstance(tool_input, dict)
                            else {"input": tool_input},
                        )

                    # Add to StreamWriter if available
                    if stream_writer:
                        stream_writer.add_chunk(
                            MessageChunk(
                                type="tool_call_start",
                                content={
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name,
                                },
                            )
                        )

                # Tool call output
                elif item.type == "tool_call_output_item":
                    tool_output = getattr(item, "output", {})

                    # Try to get tool_call_id from the item itself
                    tool_call_id = getattr(item, "tool_call_id", None)

                    if not tool_call_id and pending_tool_calls:
                        # Fallback: use the most recent pending tool call (LIFO)
                        tool_call_id = pending_tool_calls.pop()
                    elif not tool_call_id:
                        # Last resort: generate fallback
                        logger.error("No tool_call_id in output item and no pending calls")
                        tool_call_id = str(uuid.uuid4())
                        self._tool_call_registry[tool_call_id] = ("unknown", {})

                    yield ToolOutputEvent(
                        tool_call_id=tool_call_id,
                        tool_output=tool_output
                        if isinstance(tool_output, dict)
                        else {"output": tool_output},
                    )

                    # Add to StreamWriter if available
                    if stream_writer:
                        from django_ai_sdk.common import MessageChunk

                        stream_writer.add_chunk(
                            MessageChunk(
                                type="tool_output",
                                content={
                                    "tool_call_id": tool_call_id,
                                    "tool_output": tool_output
                                    if isinstance(tool_output, dict)
                                    else {"output": tool_output},
                                },
                            )
                        )

                # Text output
                elif item.type == "message_output_item":
                    content = ItemHelpers.text_message_output(item)
                    if content and content.strip():
                        yield TextChunkEvent(content=content)

                        # Add to StreamWriter if available
                        if stream_writer:
                            from django_ai_sdk.common import MessageChunk

                            stream_writer.add_chunk(
                                MessageChunk(
                                    type="text",
                                    content=content,
                                )
                            )

            # Finalize message and store if StreamWriter available
            if stream_writer:
                await stream_writer.finalize(finish_reason="stop")
                self.message_result = stream_writer.message
                logger.debug(f"Message finalized and stored with ID: {message_id}")

            yield MessageEndEvent(finish_reason="stop")

        except Exception as e:
            yield ErrorEvent(error_message=f"Agent execution error: {str(e)}")
            return

        yield StreamEndEvent()
