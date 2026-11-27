from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, cast

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import (
    ReasoningContent,
    StreamingChunk,
    ToolCall,
    ToolCallResult,
)

from django_ai_sdk.adapters.citations.grounding import Grounder, rebase, uncited
from django_ai_sdk.adapters.citations.registry import current_registry
from django_ai_sdk.adapters.run import (
    ATTACHMENT_CONTEXT_META_KEY,
    get_user_message,
    get_user_text,
)
from django_ai_sdk.adapters.run import Run as Run
from django_ai_sdk.adapters.utils import merge_messages
from django_ai_sdk.agents.subagent import SUBAGENT_META_KEY, SubagentStreamFilter
from django_ai_sdk.agents.tool_agent import decode_tool_result, returned_error
from django_ai_sdk.artifacts.schemas import turn_artifacts
from django_ai_sdk.artifacts.tool_artifacts import build_artifact
from django_ai_sdk.common import (
    ChatMessage,
    MessageChunk,
    StreamWriter,
)
from django_ai_sdk.errors import ErrorInfo, describe_error
from django_ai_sdk.events import (
    ErrorEvent,
    MessageEndEvent,
    MessageStartEvent,
    ReasoningChunkEvent,
    StreamEndEvent,
    StreamEvent,
    SuggestionEvent,
    TextChunkEvent,
    TextEndEvent,
    ToolCallStartEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
)
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.stream import (
    CONTENT_FILTER,
    StreamTurn,
    withdrawn_citations,
)
from django_ai_sdk.receipts import receipt_keys
from django_ai_sdk.tracing import bind as bind_trace
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.artifacts.tool_artifacts import ToolArtifact
    from django_ai_sdk.oversight.plugin import TurnRequest
    from django_ai_sdk.storage.base import BaseStorageAdapter


logger = get_logger(__name__)

_SENTINEL: object = object()
_CITED: object = object()


def parse_tool_input(arguments: str | None) -> dict[str, Any] | str:
    try:
        return json.loads(str(arguments))
    except (json.JSONDecodeError, ValueError):
        return str(arguments)


def parse_tool_output(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return decode_tool_result(obj)

    if isinstance(obj, dict):
        return {k: parse_tool_output(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [parse_tool_output(item) for item in obj]

    return str(obj)


def stored_tool_output(result: ToolCallResult) -> dict[str, Any]:
    output = parse_tool_output(result.to_dict())
    output["error"] = bool(result.error) or returned_error(output.get("result"))
    return output


_TRUNCATION_NOTE = "\n\n... truncated, {omitted} chars omitted"


def get_tool_result_text(tool_call: dict[str, Any]) -> str:
    result = tool_call.get("result")
    if result is None:
        return ""
    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    return result if isinstance(result, str) else json.dumps(result, default=str)


def truncate_tool_result(text: str) -> str:
    limit = resolve_setting("AI_SDK_HISTORY_TOOL_OUTPUT_LIMIT", 4000)
    if not isinstance(limit, int) or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_NOTE.format(omitted=len(text) - limit)


def _get_message_pairs(
    messages: list[ChatMessage],
    has_handoff: Callable[[ChatMessage], list[dict[str, Any]]],
) -> list[tuple[ChatMessage | None, list[ChatMessage]]]:
    runs: list[tuple[ChatMessage | None, list[ChatMessage]]] = []
    current: list[ChatMessage] = []
    for message in messages:
        if has_handoff(message):
            if current:
                runs.append((None, current))
                current = []
            runs.append((message, []))
        else:
            current.append(message)
    if current:
        runs.append((None, current))
    return runs


def get_reasoning_text(reasoning: ReasoningContent) -> str:
    if reasoning.reasoning_text:
        return reasoning.reasoning_text
    parts = (reasoning.extra or {}).get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "reasoning_text")


def get_error_chunk(e: Exception, info: ErrorInfo) -> MessageChunk:
    return MessageChunk(
        type="error",
        content={
            "error_message": str(e) or type(e).__name__,
            "error_type": type(e).__name__,
            "error_code": info.spec.code,
            "error_ref": info.ref,
        },
    )


def get_error_event(info: ErrorInfo) -> ErrorEvent:
    return ErrorEvent(error_code=info.spec.code, ref=info.ref)


class Stream:
    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None
    hook_context: dict[str, Any] | None = None
    turn_request: TurnRequest | None = None
    tool_artifacts: dict[str, ToolArtifact] = {}
    thread_id: str | None = None
    user: Any = None
    citation_agent: Any = None
    _grounder: Grounder | None = None
    _citations_call_id: str | None = None
    _turn: StreamTurn | None = None

    merge_messages: bool = False

    def __init__(
        self,
        pipeline: Any,
        generator: Any,
        store: bool = True,
        storage_adapter: BaseStorageAdapter | None = None,
        citation_registry: SourceRegistry | None = None,
        suggestion_generator: SuggestionGenerator | None = None,
    ) -> None:
        self.pipeline = pipeline
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"Stream requires an Pipeline, got {type(pipeline).__name__}. "
                "Construct your pipeline with Pipeline() from haystack."
            )
        self.generator = generator
        self.store = store
        self.storage_adapter = storage_adapter
        self.citation_registry = citation_registry
        self.suggestion_generator = suggestion_generator
        self._persisted_tool_ids: set[str] = set()
        self._persisted_tool_output_ids: set[str] = set()
        self._artifact_ids: list[str] = []
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

        self._handoff_tools: dict[str, str] = {
            tool.name: getattr(tool._component, "name", "") or tool.name
            for tool in (getattr(self.agent_component, "tools", None) or [])
            if isinstance(getattr(tool, "_component", None), SubagentStreamFilter)
        }
        self._receipt_keys = receipt_keys(getattr(self.agent_component, "tools", None))

    @staticmethod
    def get_handoff_calls(message: ChatMessage) -> list[dict[str, Any]]:
        if message.role != "assistant":
            return []
        return [tc for tc in message.tool_calls if tc.get("handoff")]

    def replay_handoff(self, message: ChatMessage) -> list[HaystackChatMessage]:
        calls = [
            ToolCall(
                id=tc.get("id") or str(uuid.uuid4()),
                tool_name=tc.get("name") or "",
                arguments=tc.get("arguments") or {},
            )
            for tc in self.get_handoff_calls(message)
        ]
        ids = [call.id for call in calls if call.id]
        self._persisted_tool_ids.update(ids)
        self._persisted_tool_output_ids.update(ids)
        replayed = [HaystackChatMessage.from_assistant(message.content or None, tool_calls=calls)]
        for tool_call, call in zip(self.get_handoff_calls(message), calls, strict=True):
            replayed.append(
                HaystackChatMessage.from_tool(
                    truncate_tool_result(get_tool_result_text(tool_call)),
                    origin=call,
                )
            )
        return replayed

    def get_messages(self, messages: list[ChatMessage]) -> list[HaystackChatMessage]:
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        converted_messages: list[HaystackChatMessage] = []

        for message, group in _get_message_pairs(conversation, self.get_handoff_calls):
            if message is not None:
                converted_messages.extend(self.replay_handoff(message))
                continue
            if not self.merge_messages:
                for msg in group:
                    if msg.role == "user":
                        converted_messages.append(get_user_message(msg))
                    elif msg.role == "assistant":
                        converted_messages.append(HaystackChatMessage.from_assistant(msg.content))
                continue
            texts = [m.model_copy(update={"content": get_user_text(m)}) for m in group]
            hinted = any(a.context for m in group for a in m.attachments)
            meta = {ATTACHMENT_CONTEXT_META_KEY: 0} if hinted else None
            for role, content in merge_messages(texts):
                if role == "user":
                    converted_messages.append(HaystackChatMessage.from_user(content, meta=meta))
                elif role == "assistant":
                    converted_messages.append(HaystackChatMessage.from_assistant(content))

        if not converted_messages:
            logger.warning("No messages available for Haystack pipeline!")
            converted_messages.append(HaystackChatMessage.from_system("No messages available."))

        return converted_messages

    def get_attribution(self, tool_name: str, subagent: str | None = None) -> dict[str, str]:
        if subagent:
            return {"agent": subagent}
        if handoff := self._handoff_tools.get(tool_name):
            return {"handoff": handoff}
        return {}

    def get_text_chunk(self, content: str) -> MessageChunk:
        return MessageChunk(type="text", content=content)

    def get_tool_chunks(self, message: HaystackChatMessage) -> list[MessageChunk]:
        chunks = []

        for tool_call in message.tool_calls:
            chunks.append(
                MessageChunk(
                    type="tool_call_start",
                    content={
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.tool_name,
                    },
                    metadata=self.get_attribution(tool_call.tool_name),
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
            tool_output = stored_tool_output(tool_result)
            chunks.append(
                MessageChunk(
                    type="tool_output",
                    content={
                        "tool_call_id": tool_result.origin.id,
                        "tool_output": tool_output,
                    },
                )
            )

        return chunks

    def _persist_tool(self, stream_writer: StreamWriter | None, chunk: MessageChunk) -> None:
        if not stream_writer:
            return
        ids = (
            self._persisted_tool_output_ids
            if chunk.type == "tool_output"
            else self._persisted_tool_ids
        )
        ids.add(chunk.content["tool_call_id"])
        stream_writer.add_chunk(chunk)

    async def _tool_artifact(
        self, stream_writer: StreamWriter | None, result: ToolCallResult
    ) -> AsyncGenerator[StreamEvent, None]:
        spec = self.tool_artifacts.get(result.origin.tool_name)
        if spec is None or not self.thread_id:
            return
        tool_result = result.result
        if isinstance(tool_result, str):
            try:  # Haystack sends lists and dicts as JSON
                tool_result = json.loads(tool_result)
            except json.JSONDecodeError:
                pass
        built = await build_artifact(
            spec, result.origin.arguments, tool_result, self.thread_id, self.user
        )
        if built is None:
            return
        data, payload = built
        self._artifact_ids.append(str(payload["artifact_id"]))
        name = spec.artifact.tool_name()
        tool_call_id = f"{name}-{payload['artifact_id']}"
        self._persist_tool(
            stream_writer,
            MessageChunk(
                type="tool_call_start",
                content={"tool_call_id": tool_call_id, "tool_name": name},
            ),
        )
        yield ToolCallStartEvent(tool_call_id=tool_call_id, tool_name=name)
        yield self._tool_input(stream_writer, tool_call_id, name, {}, data)
        self._persist_tool(
            stream_writer,
            MessageChunk(
                type="tool_output",
                content={"tool_call_id": tool_call_id, "tool_output": payload},
            ),
        )
        yield ToolOutputEvent(tool_call_id=tool_call_id, tool_output=payload)

    async def get_citations_artifact(
        self, stream_writer: StreamWriter | None
    ) -> AsyncGenerator[StreamEvent, None]:
        """The answer's citations as a citations artifact, before the message is stored"""
        from django_ai_sdk.artifacts.schemas import CitationsArtifact  # noqa: PLC0415

        registry = self.citation_registry
        if not (registry and registry.all and self.thread_id and self.citation_agent):
            return
        answer = stream_writer.text if stream_writer else ""
        timeout = resolve_setting("AI_SDK_CITATION_TIMEOUT", 10.0)
        grounder = self._grounder or Grounder(registry, self.citation_agent)
        try:
            data = None
            if answer.strip():
                async for _ in grounder.wait(answer, timeout):
                    for event in self._partial_citations(grounder):
                        yield event
                data = rebase(grounder.citations(), answer)
        except Exception:
            logger.exception("Placing citations failed; the answer has none")
            data = None
        data = data or uncited(registry)
        if not data.sources:
            return
        try:
            payload = await CitationsArtifact.store(
                data.model_dump(mode="json", by_alias=True), self.thread_id, self.user
            )
        except Exception:
            logger.exception("Storing citations failed")
            return

        name = CitationsArtifact.tool_name()
        shown = self._citations_call_id
        tool_call_id = shown or f"{name}-{payload['artifact_id']}"
        self._persist_tool(
            stream_writer,
            MessageChunk(
                type="tool_call_start",
                content={"tool_call_id": tool_call_id, "tool_name": name},
            ),
        )
        tool_input = self._tool_input(stream_writer, tool_call_id, name, {}, {})
        if not shown:
            yield ToolCallStartEvent(tool_call_id=tool_call_id, tool_name=name)
            yield tool_input
        self._persist_tool(
            stream_writer,
            MessageChunk(
                type="tool_output",
                content={"tool_call_id": tool_call_id, "tool_output": payload},
            ),
        )
        yield ToolOutputEvent(tool_call_id=tool_call_id, tool_output=payload)

    def _partial_citations(self, grounder: Grounder | None) -> list[StreamEvent]:
        from django_ai_sdk.artifacts.schemas import CitationsArtifact  # noqa: PLC0415

        data = grounder.citations() if grounder else None
        if not data or not data.citations:
            return []
        events: list[StreamEvent] = []
        if not self._citations_call_id:
            name = CitationsArtifact.tool_name()
            self._citations_call_id = f"{name}-{uuid.uuid4()}"
            events += [
                ToolCallStartEvent(tool_call_id=self._citations_call_id, tool_name=name),
                ToolInputCompleteEvent(
                    tool_call_id=self._citations_call_id, tool_name=name, tool_input={}
                ),
            ]
        output = data.model_dump(mode="json", by_alias=True)
        events.append(ToolOutputEvent(tool_call_id=self._citations_call_id, tool_output=output))
        return events

    def _tool_input(
        self,
        stream_writer: StreamWriter | None,
        tool_call_id: str,
        tool_name: str,
        attribution: dict[str, str],
        tool_input: Any,
    ) -> ToolInputCompleteEvent:
        self._persist_tool(
            stream_writer,
            MessageChunk(
                type="tool_input",
                content={
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                },
            ),
        )
        return ToolInputCompleteEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            agent=attribution.get("agent"),
            handoff=attribution.get("handoff"),
        )

    def get_task(
        self,
        haystack_messages: list[HaystackChatMessage],
        streaming_callback: Any,
    ) -> asyncio.Task[Any]:
        if self.agent_component:
            coro = self._run_agent(self.agent_component, haystack_messages, streaming_callback)
        else:
            coro = self.pipeline.run_async({"messages": haystack_messages})
        return asyncio.create_task(coro)

    async def _run_agent(self, agent: Any, messages: list[Any], callback: Any) -> Any:
        from django_ai_sdk.oversight.plugin import run_with_oversight

        turn = self._turn.turn if self._turn is not None else self.turn_request
        return await run_with_oversight(
            agent,
            messages,
            turn=turn,
            hook_context=self.hook_context,
            streaming_callback=callback,
        )

    async def get_events(
        self,
        queue: asyncio.Queue[StreamingChunk | object],
        stream_writer: StreamWriter | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        pending: dict[str, tuple[str, dict[str, str], str]] = {}
        at_index: dict[int, str] = {}
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break

            if item is _CITED:
                for event in self._partial_citations(self._grounder):
                    yield event
                continue

            chunk = cast("StreamingChunk", item)

            text = chunk.content or ""
            if text:
                if stream_writer:
                    stream_writer.add_chunk(self.get_text_chunk(text))
                if self._grounder:
                    self._grounder.feed(text)
                yield TextChunkEvent(content=text)

            if chunk.reasoning and (reasoning := get_reasoning_text(chunk.reasoning)):
                if stream_writer:
                    stream_writer.add_chunk(MessageChunk(type="reasoning", content=reasoning))
                yield ReasoningChunkEvent(content=reasoning)

            if chunk.tool_calls:
                subagent = chunk.meta.get(SUBAGENT_META_KEY)
                for tc in chunk.tool_calls:
                    if tc.tool_name:
                        tc_id = tc.id or str(uuid.uuid4())
                        at_index[tc.index] = tc_id
                        attribution = self.get_attribution(tc.tool_name, subagent)
                        pending[tc_id] = (tc.tool_name, attribution, "")
                        self._persist_tool(
                            stream_writer,
                            MessageChunk(
                                type="tool_call_start",
                                content={
                                    "tool_call_id": tc_id,
                                    "tool_name": tc.tool_name,
                                },
                                metadata=attribution,
                            ),
                        )
                        yield ToolCallStartEvent(
                            tool_call_id=tc_id,
                            tool_name=tc.tool_name,
                            agent=attribution.get("agent"),
                            handoff=attribution.get("handoff"),
                        )
                    else:
                        # A later fragment has no name, and Chat Completions drops its id too.
                        tc_id = tc.id or at_index.get(tc.index, "")
                    if tc_id not in pending:
                        continue
                    name, attribution, streamed = pending[tc_id]
                    streamed += tc.arguments or ""
                    try:
                        tool_input = json.loads(streamed)
                    except json.JSONDecodeError:
                        pending[tc_id] = (name, attribution, streamed)
                        continue
                    del pending[tc_id]
                    yield self._tool_input(stream_writer, tc_id, name, attribution, tool_input)

            if chunk.tool_call_result:
                result = chunk.tool_call_result
                tool_call_id = result.origin.id or str(uuid.uuid4())
                if tool_call_id in pending:
                    name, attribution, _ = pending.pop(tool_call_id)
                    yield self._tool_input(
                        stream_writer,
                        tool_call_id,
                        name,
                        attribution,
                        result.origin.arguments,
                    )
                if result.error:
                    logger.warning(f"Tool call failed: tool={result.origin.tool_name}")
                    logger.debug(f"Tool call result: {result.result}")
                if self._turn is not None:
                    result = await self._turn.mask_tool_result(result)
                tool_output = stored_tool_output(result)
                self._persist_tool(
                    stream_writer,
                    MessageChunk(
                        type="tool_output",
                        content={
                            "tool_call_id": tool_call_id,
                            "tool_output": tool_output,
                        },
                    ),
                )
                yield ToolOutputEvent(tool_call_id=tool_call_id, tool_output=tool_output)
                if not tool_output["error"]:
                    async for event in self._tool_artifact(stream_writer, result):
                        yield event

    async def get_pipeline_result(
        self,
        pipeline_task: asyncio.Task[Any],
        stream_writer: StreamWriter | None,
    ) -> list[Any]:
        """Await pipeline completion, store tool chunks and return the response messages."""
        pipeline_result = await pipeline_task

        if self.first_component and self.first_component in pipeline_result:
            response_messages = pipeline_result[self.first_component].get("messages", [])
        else:
            response_messages = pipeline_result.get("messages", [])

        for message in response_messages:
            if hasattr(message, "tool_calls") or hasattr(message, "tool_call_results"):
                if stream_writer:
                    for chunk in self.get_tool_chunks(message):
                        cid = chunk.content["tool_call_id"]
                        if chunk.type == "tool_output":
                            if cid in self._persisted_tool_output_ids:
                                continue
                        elif cid in self._persisted_tool_ids:
                            continue
                        stream_writer.add_chunk(chunk)
        return response_messages

    async def get_final_message(
        self,
        stream_writer: StreamWriter | None,
        finish_reason: str = "stop",
    ) -> ChatMessage | None:
        """Finalize the stream writer: the message is stored."""
        if not stream_writer:
            return None

        result = await stream_writer.finalize(finish_reason)
        self.message_result = result
        return result

    async def get_suggestions(
        self,
        messages: list[ChatMessage],
    ) -> SuggestionEvent | None:
        if not (self.suggestion_generator and self.message_result):
            return None
        try:
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            timeout = resolve_setting("AI_SDK_SUGGESTION_TIMEOUT", 5.0)
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
            logger.opt(exception=e).error("Error generating suggestions: {}", e)
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
                model=self.model_name or "haystack-pipeline",
                role="assistant",
                storage_callback=self.storage_adapter.storage_callback,
            )

        _finalize_called = False
        queue: asyncio.Queue[StreamingChunk | object] = asyncio.Queue()

        async def streaming_callback(chunk: StreamingChunk) -> None:
            await queue.put(chunk)
            if chunk.tool_call_result and chunk.tool_call_result.error:
                logger.warning(f"Tool call failed: tool={chunk.tool_call_result.origin.tool_name}")
                logger.debug(f"Tool call result: {chunk.tool_call_result.result}")

        self.generator.streaming_callback = streaming_callback

        if hasattr(self.generator, "warm_up"):
            await asyncio.to_thread(self.generator.warm_up)

        pipeline_task: asyncio.Task[Any] | None = None
        turn = self._turn = StreamTurn(
            self.turn_request,
            message_id,
            stream_writer,
            self.citation_registry,
            self._artifact_ids,
            queue,
            receipts=self._receipt_keys,
        )
        turn_error = ""

        try:
            yield MessageStartEvent(message_id=message_id)
            for event in await turn.start():
                yield event
            # The task copies the context: tools see this turn's registry and artifacts.
            registry_token = current_registry.set(self.citation_registry)
            artifacts_token = turn_artifacts.set(self._artifact_ids)
            try:
                with (
                    bind_trace(
                        thread_id=getattr(self.storage_adapter, "thread_id", None),
                        message_id=message_id,
                    ),
                ):
                    pipeline_task = self.get_task(haystack_messages, streaming_callback)
            finally:
                current_registry.reset(registry_token)
                turn_artifacts.reset(artifacts_token)
            pipeline_task.add_done_callback(lambda _: queue.put_nowait(_SENTINEL))

            registry = self.citation_registry
            if registry is not None and self.thread_id and self.citation_agent and stream_writer:

                def cited() -> None:
                    if not turn.hold:
                        queue.put_nowait(_CITED)

                self._grounder = Grounder(registry, self.citation_agent, on_block=cited)

            async for event in self.get_events(queue, stream_writer):
                for sent in turn.pass_event(event):
                    yield sent

            try:
                await self.get_pipeline_result(pipeline_task, stream_writer)
            except Exception as pipeline_error:
                info = describe_error(pipeline_error)
                logger.opt(exception=pipeline_error).error(
                    "Pipeline task failed [{}] {}: {}: {}",
                    info.ref,
                    info.code,
                    type(pipeline_error).__name__,
                    pipeline_error,
                )
                turn_error = type(pipeline_error).__name__
                if stream_writer:
                    stream_writer.add_chunk(get_error_chunk(pipeline_error, info))
                    turn.end_early()
                    self.message_result = await stream_writer.finalize("error")
                    _finalize_called = True
                yield get_error_event(info)
                yield StreamEndEvent()
                return

            if turn.blocked_gate_result is not None:
                _finalize_called = True
                for event in turn.blocked_reply_events():
                    yield event
                return

            events, finished = await turn.finish()
            for event in events:
                yield event
            finish_reason = CONTENT_FILTER if finished.blocked else None

            yield TextEndEvent()
            if finished.blocked:
                for event in withdrawn_citations(self._grounder, self._citations_call_id):
                    yield event
            else:
                async for event in self.get_citations_artifact(stream_writer):
                    yield event
            await self.get_final_message(stream_writer, finish_reason or "stop")
            _finalize_called = True

            yield MessageEndEvent(finish_reason=finish_reason)

            if not finished.changed and (
                suggestion := await self.get_suggestions(turn.as_gated(messages))
            ):
                yield suggestion

        except Exception as critical_error:
            # Don't f-string `critical_error`: braces in its text would crash this log call.
            info = describe_error(critical_error)
            logger.opt(exception=critical_error).error(
                "Critical error in stream [{}] {}: {}: {}",
                info.ref,
                info.code,
                type(critical_error).__name__,
                critical_error,
            )
            turn_error = type(critical_error).__name__
            if stream_writer and not _finalize_called:
                stream_writer.add_chunk(get_error_chunk(critical_error, info))
                turn.end_early()
                self.message_result = await stream_writer.finalize("error")
                _finalize_called = True
            yield get_error_event(info)

        finally:
            if self._grounder:
                self._grounder.cancel()
                self._grounder = None
            self._citations_call_id = None
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=1.0)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    pass
            try:
                if stream_writer and not _finalize_called:
                    turn.end_early()
                    self.message_result = await stream_writer.finalize("cancelled")
            finally:
                await turn.aclose(turn_error)

        yield StreamEndEvent()
