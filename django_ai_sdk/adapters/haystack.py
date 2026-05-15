import ast
import asyncio
import json
import traceback
import uuid
from asyncio import Queue
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Union

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import StreamingChunk

from django_ai_sdk.adapters.base import BasePipelineAdapter
from django_ai_sdk.adapters.utils import merge_messages, normalize_usage
from django_ai_sdk.citations import CitationRegistry
from django_ai_sdk.common import (
    ChatMessage,
    MessageChunk,
    StreamWriter,
)
from django_ai_sdk.events import (
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    SourceEvent,
    StreamEndEvent,
    StreamEvent,
    SuggestionEvent,
    TextChunkEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
)
from django_ai_sdk.logger import get_logger
from django_ai_sdk.suggestions import SuggestionGenerator

if TYPE_CHECKING:
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = get_logger(__name__)


def parse_tool_input(arguments: str | None) -> dict[str, Any] | str:
    """
    Parse tool result as JSON if valid, otherwise return as string.
    """
    try:
        return json.loads(str(arguments))
    except (json.JSONDecodeError, ValueError):
        return str(arguments)


def parse_tool_output(obj: Any) -> Any:
    """
    Parse tool output from haystack to ensure it's JSON serializable.
    """
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str):
            try:
                return ast.literal_eval(obj)
            except (ValueError, SyntaxError):
                return obj
        return obj

    if isinstance(obj, dict):
        return {k: parse_tool_output(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [parse_tool_output(item) for item in obj]

    return str(obj)


class HaystackAdapter(BasePipelineAdapter):
    """
    Adapter for Haystack pipelines that emits events.
    """

    def __init__(
        self,
        pipeline: Any,
        generator_component: Any,
        store: bool = True,
        storage_adapter: Union["BaseStorageAdapter", None] = None,
        rag_pipeline: Any = None,
        citation_registry: CitationRegistry | None = None,
        suggestion_generator: "SuggestionGenerator | None" = None,
    ) -> None:
        self.pipeline = pipeline
        self.generator = generator_component
        self.store = store
        self.storage_adapter = storage_adapter
        self.rag_pipeline = (
            rag_pipeline  # TODO: we probably don't need this, we can check pipeline directly
        )
        self.citation_registry = citation_registry
        self.suggestion_generator = suggestion_generator
        self._sources_emitted = 0
        self.message_result: ChatMessage | None = None
        self.first_component = list(pipeline.graph.nodes())[0] if pipeline.graph.nodes() else None

        # Check if first component is an Agent
        self.agent_component = None
        self.model_name = None

        if self.first_component:
            component = pipeline.get_component(self.first_component)

            # Check if first component is an Agent
            # TODO: this logic should be refactored, it should be moved to a separate method
            if isinstance(component, Agent):
                self.agent_component = component
                # Try to get model name from chat_generator
                if hasattr(component, "chat_generator"):
                    cg = component.chat_generator
                    self.model_name = getattr(cg, "model", None) or getattr(cg, "model_name", None)
                logger.debug(
                    f"Detected Agent component: {self.first_component}, model: {self.model_name}"
                )

        logger.debug(
            f"""
            Haystack adapter initialized:
                store={store},
                first_component={self.first_component},
                agent_component={self.agent_component is not None},
                storage_adapter={type(storage_adapter).__name__ if storage_adapter else None}
            """
        )

    @staticmethod
    def _run_agent(
        agent: Any,
        messages: list[Any],
        callback: Any,
    ) -> Any:
        """Helper method to run agent."""
        return agent.run(messages=messages, streaming_callback=callback)

    def get_messages(self, messages: list[ChatMessage]) -> list["HaystackChatMessage"]:
        """Convert internal messages to Haystack ChatMessage format.

        Message merging is controlled by merge_messages flag (default: False).
        """
        logger.debug(
            f"Converting {len(messages)} internal messages to Haystack format (merge={self.merge_messages})"
        )

        # Filter to user/assistant only
        conversation = [m for m in messages if m.role in ("user", "assistant")]

        converted_messages: list[HaystackChatMessage] = []

        # Determine which messages to convert
        if self.merge_messages:
            filtered_messages = merge_messages(conversation)
        else:
            filtered_messages = [(msg.role, msg.content) for msg in conversation]

        # Convert to Haystack format
        for role, content in filtered_messages:
            if role == "user":
                converted_messages.append(HaystackChatMessage.from_user(content))
            elif role == "assistant":
                converted_messages.append(HaystackChatMessage.from_assistant(content))

        logger.debug(f"Haystack message conversion complete: final_count={len(converted_messages)}")

        # Ensure we have at least one message
        # TODO: we now return a system message instead of a user message, this is a temporary fix
        # In real-world scenarios: the assistant could be working without user input, it might be
        # handed of from another assistant or system. We should support this case gracefully.
        if not converted_messages:
            logger.warning("No messages available for Haystack pipeline!")
            converted_messages.append(HaystackChatMessage.from_system("No messages available."))

        return converted_messages

    def _create_text_chunk(self, content: str) -> MessageChunk:
        """Create a text MessageChunk."""
        return MessageChunk(
            type="text",
            content=content,
            metadata={"source": "haystack_adapter"},
        )

    def _create_tool_chunks_from_message(self, message: Any) -> list[MessageChunk]:
        """Convert Haystack message tool calls to MessageChunks."""
        chunks = []

        # Tool call starts and inputs
        for tool_call in message.tool_calls:
            chunks.append(
                MessageChunk(
                    type="tool_call_start",
                    content={
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.tool_name,
                    },
                    metadata={"source": "haystack_adapter"},
                )
            )

            chunks.append(
                MessageChunk(
                    type="tool_input",
                    content={
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.tool_name,
                        "tool_input": tool_call.arguments,
                    },
                    metadata={"source": "haystack_adapter"},
                )
            )

        # Tool outputs
        for tool_result in message.tool_call_results:
            # Ensure the output is JSON serializable
            tool_output = parse_tool_output(tool_result.to_dict())
            chunks.append(
                MessageChunk(
                    type="tool_output",
                    content={
                        "tool_call_id": tool_result.origin.id,
                        "tool_output": tool_output,
                    },
                    metadata={"source": "haystack_adapter"},
                )
            )

        return chunks

    async def stream(
        self,
        input: list[ChatMessage],
    ) -> AsyncGenerator[StreamEvent, None]:
        logger.debug(f"Starting Haystack stream with {len(input)} input messages")

        # Convert messages
        messages = self.get_messages(input)

        # Note: RAG is handled at pipeline level via get_rag_pipeline() in Assistant
        # The retriever is added to the pipeline before creating the adapter
        logger.debug(f"Converted to {len(messages)} Haystack messages")

        # Generate message ID
        message_id = str(uuid.uuid4())
        logger.debug(f"Generated message ID: {message_id}")

        # Create StreamWriter with pre-generated ID
        stream_writer = None
        if self.store and self.storage_adapter:
            stream_writer = StreamWriter(
                adapter_type="haystack",
                message_id=message_id,
                model="haystack-pipeline",
                role="assistant",
                storage_callback=self.storage_adapter.storage_callback,
            )
            logger.debug(f"StreamWriter configured with ID: {message_id}")
        else:
            logger.debug("Message storage disabled or no storage adapter")

        _finalize_called = False

        # Queue for token chunks and tool events
        queue: Queue = Queue()
        pipeline_finished = asyncio.Event()
        loop = asyncio.get_running_loop()
        logger.debug("Token queue and pipeline event initialized")

        # Create callback as bound method to avoid closure capturing issues
        def make_callback(
            queue_ref: Queue, event_ref: asyncio.Event, loop_ref: asyncio.AbstractEventLoop
        ) -> Callable[[StreamingChunk], None]:
            """Factory function to create callback with explicit references."""

            def callback(chunk: StreamingChunk) -> None:
                if chunk.content:
                    queue_ref.put_nowait(chunk.content)
                    logger.debug(f"Token queued: {chunk.content[:20]}...")

                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        if tool_call.tool_name:
                            queue_ref.put_nowait(("tool_call", tool_call))
                            logger.debug(f"Tool call queued: {tool_call.tool_name}")

                if chunk.tool_call_result:
                    queue_ref.put_nowait(("tool_result", chunk.tool_call_result))
                    logger.debug("Tool result queued")

                if chunk.finish_reason == "stop":
                    logger.debug("Pipeline finished with reason: stop")
                    loop_ref.call_soon_threadsafe(event_ref.set)

                # Capture usage from chunk metadata (when stream_options={"include_usage": True})
                # This handles the final chunk with choices=[] and usage data
                if hasattr(chunk, "meta") and chunk.meta:
                    usage_data = (
                        chunk.meta.get("usage")
                        if isinstance(chunk.meta, dict)
                        else getattr(chunk.meta, "usage", None)
                    )
                    if usage_data:
                        queue_ref.put_nowait(("usage", usage_data))
                        logger.debug(f"Usage chunk queued from meta: {usage_data}")

            return callback

        # Create callback with explicit references for thread safety
        streaming_callback = make_callback(queue, pipeline_finished, loop)
        self.generator.streaming_callback = streaming_callback
        logger.debug("Streaming callback attached to generator component")

        # Warm up the generator for async compatibility
        if hasattr(self.generator, "warm_up"):
            logger.debug("Warming up generator component")
            self.generator.warm_up()

        try:
            # Start message, use same ID generated at adapter level
            logger.debug(f"Starting message stream with ID: {message_id}")
            yield MessageStartEvent(message_id=message_id)

            # Run the pipeline or Agent directly
            # When we have an Agent component, run it directly with streaming_callback
            # TODO: we might need async support here, its kinda new, but we need to investigate
            # how to properly use async pipeline execution.
            if self.agent_component:
                logger.debug("Running Agent component directly with streaming_callback")
                pipeline_task = loop.run_in_executor(
                    None,
                    lambda: self._run_agent(self.agent_component, messages, streaming_callback),
                )
            else:
                pipeline_input = {"messages": messages}
                logger.debug("Using default pipeline input format")
                logger.debug("Starting pipeline execution in thread executor")
                pipeline_task = loop.run_in_executor(
                    None, lambda: self.pipeline.run(pipeline_input)
                )

            # Yield tokens
            pipeline_result = None
            usage = None
            while not pipeline_finished.is_set() or not queue.empty():
                # Check if pipeline task has completed
                if pipeline_task.done():
                    try:
                        pipeline_result = pipeline_task.result()
                        logger.debug("Pipeline task completed successfully, exiting token loop")

                        # Extract usage from result metadata if available
                        if pipeline_result:
                            replies = pipeline_result.get("replies", [])
                            if replies and replies[0].meta:
                                usage = normalize_usage(replies[0].meta.get("usage"))
                    except Exception as pipeline_error:
                        logger.error(f"Pipeline task failed: {pipeline_error}")
                        # Emit error and exit loop
                        yield ErrorEvent(
                            error_message=f"Pipeline failed: {type(pipeline_error).__name__}: {str(pipeline_error)}"
                        )
                        yield StreamEndEvent()
                        return
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.08)

                    # Handle tool events in real-time
                    if isinstance(item, tuple) and len(item) == 2:
                        event_type, payload = item

                        if event_type == "tool_call":
                            # Tool call started - use ToolCallDelta attributes directly
                            tool_call_id = payload.id or str(uuid.uuid4())
                            tool_name = payload.tool_name

                            yield ToolCallStartEvent(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                            )

                            # Emit ToolInputCompleteEvent
                            if payload.arguments:
                                yield ToolInputCompleteEvent(
                                    tool_call_id=tool_call_id,
                                    tool_name=tool_name,
                                    tool_input=parse_tool_input(payload.arguments),
                                )

                        elif event_type == "tool_result":
                            # Tool result received
                            tool_call_id = payload.origin.id
                            # Ensure the output is JSON serializable
                            tool_output = parse_tool_output(payload.to_dict())

                            yield ToolOutputEvent(
                                tool_call_id=tool_call_id,
                                tool_output=tool_output,
                            )

                            if self.citation_registry is not None:
                                all_sources = self.citation_registry.all_sources
                                for source in all_sources[self._sources_emitted :]:
                                    yield SourceEvent(
                                        index=source.index,
                                        title=source.title,
                                        content=source.content,
                                        tool_call_id=tool_call_id,
                                        source_id=str(source.index),  # Ties to [N] citation in text
                                        media_type="file",  # Local document source
                                    )
                                self._sources_emitted = len(all_sources)

                        elif event_type == "usage":
                            # Usage data from streaming chunk (when stream_options={"include_usage": True})
                            usage = normalize_usage(payload)

                    else:
                        # Plain text token
                        if not isinstance(item, str):
                            logger.warning(f"Skipping non-string item in text branch: {type(item)}")
                            continue

                        if stream_writer:
                            text_chunk = self._create_text_chunk(item)
                            stream_writer.add_chunk(text_chunk)

                        yield TextChunkEvent(content=item)

                except TimeoutError:
                    # No token available yet, continue loop
                    continue
                except Exception as streaming_error:
                    logger.error(
                        f"Exception occurred in streaming loop: {type(streaming_error).__name__}: {streaming_error}"
                    )
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    # Check if pipeline completed
                    if pipeline_task.done() and not pipeline_finished.is_set():
                        logger.warning("Pipeline completed during exception, setting finished flag")
                        pipeline_finished.set()

            try:
                # Get pipeline results to handle tool events
                logger.debug("Retrieving pipeline results for tool processing")
                pipeline_result = await pipeline_task
                logger.debug("Pipeline execution completed, processing results")

                # Extract response messages from pipeline result
                if self.first_component and self.first_component in pipeline_result:
                    response_messages = pipeline_result[self.first_component].get("messages", [])
                    logger.debug(
                        f"Found {len(response_messages)} response messages from component {self.first_component}"
                    )
                else:
                    response_messages = pipeline_result.get("messages", [])
                    logger.debug(
                        f"Found {len(response_messages)} response messages from pipeline root"
                    )

                # Process tool events from response messages for storage
                for message_index, message in enumerate(response_messages):
                    if hasattr(message, "tool_calls") or hasattr(message, "tool_call_results"):
                        tool_call_count = len(getattr(message, "tool_calls", []))
                        tool_result_count = len(getattr(message, "tool_call_results", []))
                        logger.debug(
                            f"Message {message_index}: {tool_call_count} tool calls, {tool_result_count} tool results"
                        )

                        # Create tool chunks and add to stream writer for storage
                        if stream_writer:
                            tool_chunks = self._create_tool_chunks_from_message(message)
                            logger.debug(
                                f"Created {len(tool_chunks)} tool chunks for message {message_index}"
                            )
                            for chunk in tool_chunks:
                                stream_writer.add_chunk(chunk)

                logger.debug("Tool storage complete: stored tool chunks from pipeline result")

            except Exception as pipeline_processing_error:
                logger.error(
                    f"Exception processing pipeline results: {type(pipeline_processing_error).__name__}: {pipeline_processing_error}"
                )
                logger.debug(f"Stack trace: {traceback.format_exc()}")

                # Handle error in stream writer
                if stream_writer:
                    error_chunk = MessageChunk(
                        type="error",
                        content={
                            "error_message": str(pipeline_processing_error),
                            "error_type": type(pipeline_processing_error).__name__,
                        },
                        metadata={"source": "haystack_adapter"},
                    )
                    stream_writer.add_chunk(error_chunk)
                    self.message_result = await stream_writer.finalize("error")
                    _finalize_called = True

                yield ErrorEvent(
                    error_message=f"{type(pipeline_processing_error).__name__}: {str(pipeline_processing_error)}"
                )
                yield StreamEndEvent()
                return

            # Persist numbered sources from citation registry
            if stream_writer and self.citation_registry and self.citation_registry.all_sources:
                stream_writer.message.sources = [
                    {
                        "index": src.index,
                        "title": src.title,
                        "content": src.content,
                        "metadata": src.metadata,
                    }
                    for src in self.citation_registry.all_sources
                ]
                logger.debug(f"Persisted {len(stream_writer.message.sources)} numbered sources")

            # Finalize message after all chunks processed
            logger.info(
                f"Finalizing message: stream_writer={stream_writer is not None}, usage={usage}"
            )
            if stream_writer:
                logger.debug("Finalizing stored message")
                self.message_result = await stream_writer.finalize("stop", usage=usage)
                logger.debug(
                    f"Message finalized with {len(self.message_result.content)} characters"
                )
            _finalize_called = True  # Mark as finalized even if no stream_writer (success path)

            # End message
            logger.info(f"Emitting message end event with usage: {usage}")
            yield MessageEndEvent(usage=usage)

            # Generate suggestions if generator is configured
            if self.suggestion_generator and self.message_result:
                try:
                    # Use only recent context for faster, cheaper suggestions
                    recent_messages = messages[-6:] if len(messages) > 6 else messages
                    suggestions = self.suggestion_generator.generate(
                        messages=recent_messages,
                        response=self.message_result.content,
                    )
                    if suggestions:
                        yield SuggestionEvent(suggestions=suggestions)
                except Exception as e:
                    logger.error(f"Error generating suggestions: {e}", exc_info=True)

        except Exception as critical_error:
            logger.error(
                f"Critical error in stream method: {type(critical_error).__name__}: {critical_error}"
            )
            logger.error(f"Critical error traceback: {traceback.format_exc()}")

            # Handle error in stream writer
            # TODO: same as above we should consider hiding this behind a debug flag
            if stream_writer:
                error_chunk = MessageChunk(
                    type="error",
                    content={
                        "error_message": str(critical_error),
                        "error_type": type(critical_error).__name__,
                    },
                    metadata={"source": "haystack_adapter"},
                )
                stream_writer.add_chunk(error_chunk)
                self.message_result = await stream_writer.finalize("error")
                _finalize_called = True

            yield ErrorEvent(
                error_message=f"{type(critical_error).__name__}: {str(critical_error)}"
            )
            return

        finally:
            if stream_writer and not _finalize_called:
                logger.debug("Finalizing with cancelled reason - connection may have dropped")
                self.message_result = await stream_writer.finalize("cancelled")

        logger.info("Haystack pipeline completed successfully")

        yield StreamEndEvent()
