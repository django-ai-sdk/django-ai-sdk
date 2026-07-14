from __future__ import annotations

import ast
import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, cast, overload

from django.conf import settings
from haystack import AsyncPipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import StreamingChunk

from django_ai_sdk.adapters.utils import merge_messages
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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.protocols import T
    from django_ai_sdk.citations import CitationRegistry, NumberedSource
    from django_ai_sdk.storage.base import BaseStorageAdapter
    from django_ai_sdk.suggestions import SuggestionGenerator


logger = get_logger(__name__)

_SENTINEL: object = object()


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


def get_error_chunk(e: Exception) -> MessageChunk:
    return MessageChunk(
        type="error",
        content={"error_message": str(e), "error_type": type(e).__name__},
    )


class Run:
    """
    Runnable Haystack adapter.
    """

    model: str | None = None
    instructions: str | None = None

    def __init__(
        self,
        generator: Any,
        model: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self.generator = generator
        self.model = model
        self.instructions = instructions

    def get_messages(self, messages: list[ChatMessage]) -> list[HaystackChatMessage]:
        """Quick conversion."""
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        converted: list[HaystackChatMessage] = []
        for msg in conversation:
            if msg.role == "user":
                converted.append(HaystackChatMessage.from_user(msg.content))  # type: ignore[arg-type]
            elif msg.role == "assistant":
                converted.append(HaystackChatMessage.from_assistant(msg.content))  # type: ignore[arg-type]
        return converted

    @overload
    async def run(
        self, messages: list[ChatMessage], *, response_format: None = None
    ) -> str | None: ...
    @overload
    async def run(self, messages: list[ChatMessage], *, response_format: type[T]) -> T | None: ...

    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        response_format: type[T] | None = None,
    ) -> T | str | None:
        user_messages = self.get_messages(messages)
        if system_prompt:
            user_messages = [HaystackChatMessage.from_system(system_prompt), *user_messages]

        if response_format:
            schema = response_format.model_json_schema()
            response = self.generator.run(
                messages=user_messages,
                generation_kwargs={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": response_format.__name__, "schema": schema},
                    }
                },
            )
            return response_format.model_validate_json(response["replies"][0].text)

        response = self.generator.run(messages=user_messages)
        return response["replies"][0].text


class Stream:
    """
    Adapter for Haystack AsyncPipelines that emits streaming events.
    """

    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None

    # Message processing configuration
    merge_messages: bool = False

    def __init__(
        self,
        pipeline: Any,
        generator: Any,
        store: bool = True,
        storage_adapter: BaseStorageAdapter | None = None,
        citation_registry: CitationRegistry | None = None,
        suggestion_generator: SuggestionGenerator | None = None,
    ) -> None:
        self.pipeline = pipeline
        if not isinstance(pipeline, AsyncPipeline):
            raise TypeError(
                f"Stream requires an AsyncPipeline, got {type(pipeline).__name__}. "
                "Construct your pipeline with AsyncPipeline() from haystack."
            )
        self.generator = generator
        self.store = store
        self.storage_adapter = storage_adapter
        self.citation_registry = citation_registry
        self.suggestion_generator = suggestion_generator
        self._sources_emitted = 0
        self.message_result: ChatMessage | None = None
        self.first_component = list(pipeline.graph.nodes())[0] if pipeline.graph.nodes() else None

        self.agent_component = None
        self.model_name = None

        if self.first_component:
            component = pipeline.get_component(self.first_component)
            if isinstance(component, Agent):
                self.agent_component = component
                if hasattr(component, "chat_generator"):
                    cg = component.chat_generator
                    self.model_name = getattr(cg, "model", None) or getattr(cg, "model_name", None)

    def get_messages(self, messages: list[ChatMessage]) -> list[HaystackChatMessage]:
        """Convert internal messages to Haystack ChatMessage format."""
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        converted_messages: list[HaystackChatMessage] = []

        if self.merge_messages:
            filtered_messages = merge_messages(conversation)
        else:
            filtered_messages = [(msg.role, msg.content) for msg in conversation]

        for role, content in filtered_messages:
            if role == "user":
                converted_messages.append(HaystackChatMessage.from_user(content))
            elif role == "assistant":
                converted_messages.append(HaystackChatMessage.from_assistant(content))

        # Ensure we have at least one message
        # TODO: we now return a system message instead of a user message, this is a temporary fix
        # In real-world scenarios: the assistant could be working without user input, it might be
        # handed of from another assistant or system. We should support this case gracefully.
        if not converted_messages:
            logger.warning("No messages available for Haystack pipeline!")
            converted_messages.append(HaystackChatMessage.from_system("No messages available."))

        return converted_messages

    @staticmethod
    def get_source_id(source: NumberedSource) -> str | None:
        if not source.doc_id:
            return None
        if source.chunk_id:
            return f"{source.doc_id}:{source.chunk_id}"
        return source.doc_id

    def get_text_chunk(self, content: str) -> MessageChunk:
        """Create a text MessageChunk."""
        return MessageChunk(type="text", content=content)

    def get_tool_chunks(self, message: HaystackChatMessage) -> list[MessageChunk]:
        """Convert Haystack message tool calls to MessageChunks."""
        chunks = []

        # Tool call starts and inputs
        for tool_call in message.tool_calls:
            chunks.append(
                MessageChunk(
                    type="tool_call_start",
                    content={"tool_call_id": tool_call.id, "tool_name": tool_call.tool_name},
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
                )
            )

        for tool_result in message.tool_call_results:
            tool_output = parse_tool_output(tool_result.to_dict())
            chunks.append(
                MessageChunk(
                    type="tool_output",
                    content={"tool_call_id": tool_result.origin.id, "tool_output": tool_output},
                )
            )

        return chunks

    def get_task(
        self,
        haystack_messages: list[HaystackChatMessage],
        streaming_callback: Any,
    ) -> asyncio.Task[Any]:
        """Create and schedule the pipeline or agent coroutine as a Task."""
        if self.agent_component:
            coro = self.agent_component.run_async(
                messages=haystack_messages, streaming_callback=streaming_callback
            )
        else:
            coro = self.pipeline.run_async(
                {"messages": haystack_messages}, streaming_callback=streaming_callback
            )
        return asyncio.create_task(coro)

    async def get_events(
        self,
        queue: asyncio.Queue[StreamingChunk | object],
        stream_writer: StreamWriter | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Consume queue and yield stream events."""
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break

            chunk = cast("StreamingChunk", item)

            if chunk.content:
                if stream_writer:
                    stream_writer.add_chunk(self.get_text_chunk(chunk.content))
                yield TextChunkEvent(content=chunk.content)

            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    if not tc.tool_name:
                        continue
                    tc_id = tc.id or str(uuid.uuid4())
                    yield ToolCallStartEvent(tool_call_id=tc_id, tool_name=tc.tool_name)
                    if tc.arguments:
                        yield ToolInputCompleteEvent(
                            tool_call_id=tc_id,
                            tool_name=tc.tool_name,
                            tool_input=parse_tool_input(tc.arguments),
                        )

            if chunk.tool_call_result:
                result = chunk.tool_call_result
                tool_call_id = result.origin.id or str(uuid.uuid4())
                if result.error:
                    logger.error(
                        f"Tool call failed: tool={result.origin.tool_name}, result={result.result}"
                    )
                yield ToolOutputEvent(
                    tool_call_id=tool_call_id,
                    tool_output=parse_tool_output(result.to_dict()),
                )

                if self.citation_registry is not None:
                    all_sources = self.citation_registry.all_sources
                    for source in all_sources[self._sources_emitted :]:
                        source_id = self.get_source_id(source)
                        if source_id is not None:
                            yield SourceEvent(
                                index=source.index,
                                title=source.title,
                                content=source.content,
                                tool_call_id=tool_call_id,
                                source_id=source_id,
                                media_type="file",
                            )
                    self._sources_emitted = len(all_sources)

    async def get_pipeline_result(
        self,
        pipeline_task: asyncio.Task[Any],
        stream_writer: StreamWriter | None,
    ) -> None:
        """Await pipeline completion and store tool chunks from response messages."""
        pipeline_result = await pipeline_task

        if self.first_component and self.first_component in pipeline_result:
            response_messages = pipeline_result[self.first_component].get("messages", [])
        else:
            response_messages = pipeline_result.get("messages", [])

        for message in response_messages:
            if hasattr(message, "tool_calls") or hasattr(message, "tool_call_results"):
                if stream_writer:
                    for chunk in self.get_tool_chunks(message):
                        stream_writer.add_chunk(chunk)

    async def get_final_message(
        self,
        stream_writer: StreamWriter | None,
    ) -> ChatMessage | None:
        """Persist citation sources and finalize the stream writer."""
        if not stream_writer:
            return None

        if self.citation_registry and self.citation_registry.all_sources:
            sources_list = []
            for src in self.citation_registry.all_sources:
                source_id = self.get_source_id(src)
                if source_id is None:
                    continue
                # Reference fields only - deliberately no chunk `content`:
                # readers resolve it fresh by source_id, so inlining it just
                # duplicated multi-KB chunks into every message row.
                sources_list.append(
                    {
                        "index": src.index,
                        "title": src.title,
                        "source_id": source_id,
                        "memory_id": src.memory_id,
                        "page_number": src.page_number,
                    }
                )
            stream_writer.message.sources = sources_list

        result = await stream_writer.finalize("stop")
        self.message_result = result
        return result

    async def get_suggestions(
        self,
        messages: list[ChatMessage],
    ) -> SuggestionEvent | None:
        """Generate follow-up suggestions, returning a SuggestionEvent or None."""
        if not (self.suggestion_generator and self.message_result):
            return None
        try:
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            timeout = getattr(settings, "AI_SDK_SUGGESTION_TIMEOUT", 5.0)
            suggestions = await asyncio.wait_for(
                self.suggestion_generator.generate(
                    messages=recent_messages,
                    response=self.message_result.content,
                ),
                timeout=timeout,
            )
            if suggestions:
                return SuggestionEvent(suggestions=suggestions)
        except TimeoutError:
            logger.warning("Suggestion generation timed out, skipping")
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}", exc_info=True)
        return None

    async def stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[StreamEvent, None]:
        haystack_messages = self.get_messages(messages)
        message_id = str(uuid.uuid4())

        stream_writer = None
        if self.store and self.storage_adapter:
            stream_writer = StreamWriter(
                message_id=message_id,
                model="haystack-pipeline",
                role="assistant",
                storage_callback=self.storage_adapter.storage_callback,
            )

        _finalize_called = False
        queue: asyncio.Queue[StreamingChunk | object] = asyncio.Queue()

        async def streaming_callback(chunk: StreamingChunk) -> None:
            await queue.put(chunk)
            if chunk.tool_call_result and chunk.tool_call_result.error:
                logger.error(
                    f"Tool call failed: "
                    f"tool={chunk.tool_call_result.origin.tool_name}, "
                    f"result={chunk.tool_call_result.result}"
                )

        self.generator.streaming_callback = streaming_callback

        if hasattr(self.generator, "warm_up"):
            await asyncio.to_thread(self.generator.warm_up)

        pipeline_task: asyncio.Task[Any] | None = None

        try:
            yield MessageStartEvent(message_id=message_id)

            pipeline_task = self.get_task(haystack_messages, streaming_callback)
            pipeline_task.add_done_callback(lambda _: queue.put_nowait(_SENTINEL))

            async for event in self.get_events(queue, stream_writer):
                yield event

            try:
                await self.get_pipeline_result(pipeline_task, stream_writer)
            except Exception as pipeline_error:
                logger.error(f"Pipeline task failed: {pipeline_error}", exc_info=True)
                if stream_writer:
                    stream_writer.add_chunk(get_error_chunk(pipeline_error))
                    self.message_result = await stream_writer.finalize("error")
                    _finalize_called = True
                yield ErrorEvent(
                    error_message=f"Pipeline failed: {type(pipeline_error).__name__}: {str(pipeline_error)}"
                )
                yield StreamEndEvent()
                return

            await self.get_final_message(stream_writer)
            _finalize_called = True

            yield MessageEndEvent()

            if suggestion := await self.get_suggestions(messages):
                yield suggestion

        except Exception as critical_error:
            logger.error(
                f"Critical error in stream: {type(critical_error).__name__}: {critical_error}",
                exc_info=True,
            )
            if stream_writer and not _finalize_called:
                stream_writer.add_chunk(get_error_chunk(critical_error))
                self.message_result = await stream_writer.finalize("error")
                _finalize_called = True
            yield ErrorEvent(
                error_message=f"{type(critical_error).__name__}: {str(critical_error)}"
            )
            return

        finally:
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=1.0)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    pass
            if stream_writer and not _finalize_called:
                self.message_result = await stream_writer.finalize("cancelled")

        yield StreamEndEvent()
