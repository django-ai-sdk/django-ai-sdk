from __future__ import annotations

import contextvars
import dataclasses
import inspect
import re
from typing import TYPE_CHECKING, Any

from django.utils.text import slugify
from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole, StreamingCallbackT, StreamingChunk
from haystack.tracing import tracer as haystack_tracer

from django_ai_sdk.common import prompt
from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

from .tool_agent import (
    SKIPPED_META_KEY,
    ToolAgent,
    default_hooks,
)

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from haystack.components.agents import Agent as HaystackAgent

    from django_ai_sdk.agent import Agent

logger = get_logger(__name__)

# meta key naming the subagent
SUBAGENT_META_KEY = "django_ai_sdk.subagent"
SUBAGENT_NAME_TAG = "django_ai_sdk.subagent.name"
SUBAGENT_ID_TAG = "django_ai_sdk.subagent.id"

# class paths currently being built
_build_chain: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "_subagent_build_chain",
    default=(),
)


def subagent_tool_name(subagent_cls: type[Agent]) -> str:
    """Return the tool name for a subagent, derived from its display name."""
    base = getattr(subagent_cls, "name", "") or ""
    slug = slugify(base).replace("-", "_")
    if slug:
        return slug
    return re.sub(r"(?<!^)(?=[A-Z])", "_", subagent_cls.__name__).lower()


# final prompt for the subagent when its loop ended
FINAL_PROMPT = prompt("""\
    You are not able to continue due to agent limitations. 
    Write your final response now, from what you have already gathered above. 
    Do not ask for more tools.
""")

_PARTIAL_HEADER = "The run was cut short before a response was finished."


def _is_usable_final_response(message: Any) -> bool:
    """Whether a message is a usable final answer from the subagent."""
    return (
        message is not None
        and message.is_from(ChatRole.ASSISTANT)
        and bool(getattr(message, "text", None))
        and not message.tool_calls
    )


def _final_response(messages: list[ChatMessage]) -> str:
    """The subagent's written answer, or "" when it never produced one."""
    last = messages[-1] if messages else None
    return last.text or "" if last is not None and _is_usable_final_response(last) else ""


def _tool_line(tool_call: Any) -> str:
    """One compact `tool(args)` line: names and arguments"""
    args = ", ".join(f"{key}={value!r}" for key, value in (tool_call.arguments or {}).items())
    return f"{tool_call.tool_name}({args})"


def _sources(messages: list[ChatMessage]) -> str:
    """List what the subagent actually called, so the coordinator can cite it."""
    lines = [_tool_line(call) for message in messages for call in (message.tool_calls or [])]
    if not lines:
        return ""
    return "Sources consulted:\n" + "\n".join(f"- {line}" for line in lines)


def _digest(messages: list[ChatMessage]) -> str:
    """The gathered material itself, for when no response was ever written."""

    limit = resolve_setting("AI_SDK_SUBAGENT_DIGEST_LIMIT", 6000)
    budget = limit if isinstance(limit, int) and limit > 0 else None

    results = {
        result.origin.id: result
        for message in messages
        for result in (message.tool_call_results or [])
        if not message.meta.get(SKIPPED_META_KEY)
    }

    blocks: list[str] = []
    for message in messages:
        for call in message.tool_calls or []:
            result = results.get(call.id)
            if result is None:
                continue
            body = result.result if isinstance(result.result, str) else str(result.result)
            if budget is not None:
                if budget <= 0:
                    break
                if len(body) > budget:
                    body = body[:budget] + " …"
                budget -= len(body)
            blocks.append(f"{_tool_line(call)}\n→ {body}")

    if not blocks:
        return ""
    return "\n\n".join([_PARTIAL_HEADER, *blocks])


def subagent_response(messages: list[ChatMessage] | None) -> str:
    """Build the tool output the coordinator receives for a delegation."""
    messages = messages or []
    parts = [_final_response(messages) or _digest(messages), _sources(messages)]
    response = "\n\n".join(part for part in parts if part)
    return response or (
        "The sub-task could not be completed and produced no usable findings. "
        "Tell the user the work is unfinished."
    )


def _tag_subagent(chunk: StreamingChunk, name: str) -> StreamingChunk:
    """Return a copy of ``chunk`` stamped with the subagent it came from."""
    return dataclasses.replace(chunk, meta={**chunk.meta, SUBAGENT_META_KEY: name})


def _tool_chunks_only(
    callback: StreamingCallbackT | None,
    *,
    name: str = "",
    async_sink: bool = True,
    seen: set[str] | None = None,
) -> StreamingCallbackT | None:
    """Return a callback forwarding only tool chunks, tagged with ``name``.
    ``seen`` collects the ids of forwarded tool results.
    """
    if callback is None:
        return None

    def record(chunk: StreamingChunk) -> None:
        if seen is not None and chunk.tool_call_result:
            seen.add(chunk.tool_call_result.origin.id or "")

    if async_sink:

        async def filtered(chunk: StreamingChunk) -> None:
            if not (chunk.tool_calls or chunk.tool_call_result):
                return
            record(chunk)
            result = callback(_tag_subagent(chunk, name))
            if inspect.isawaitable(result):
                await result

    else:

        def filtered(chunk: StreamingChunk) -> None:
            if chunk.tool_calls or chunk.tool_call_result:
                record(chunk)
                callback(_tag_subagent(chunk, name))

    return filtered


def _unstreamed_results(result: dict[str, Any], seen: set[str]) -> list[StreamingChunk]:
    """Chunks for tool results the run produced without streaming them."""
    chunks = []
    for message in result.get("messages") or []:
        for tool_result in message.tool_call_results or []:
            origin_id = tool_result.origin.id or ""
            if origin_id in seen:
                continue
            seen.add(origin_id)
            chunks.append(StreamingChunk(content="", index=0, tool_call_result=tool_result))
    return chunks


@component
class SubagentStreamFilter:
    """Wrap a subagent Agent so only its tool calls and results reach the stream."""

    def __init__(self, agent: Any, name: str = "", agent_id: str = "") -> None:
        self._agent = agent
        self.name = name
        self.agent_id = agent_id

    def _span(self) -> Any:
        """Open a named span so the subagent's own loop is a readable subtree."""
        return haystack_tracer.trace(
            "django_ai_sdk.subagent.run",
            tags={SUBAGENT_NAME_TAG: self.name, SUBAGENT_ID_TAG: self.agent_id},
        )

    def _needs_response(self, result: dict[str, Any]) -> bool:
        """Whether the loop ended without the subagent writing its answer."""
        messages = result.get("messages") or []
        last = messages[-1] if messages else result.get("last_message")
        return not _is_usable_final_response(last)

    def _get_messages(self, result: dict[str, Any]) -> list[ChatMessage]:
        return [*(result.get("messages") or []), ChatMessage.from_user(FINAL_PROMPT)]

    def _accept_response(self, result: dict[str, Any], replies: list[ChatMessage]) -> None:
        """Fold a reply in so downstream sees a normal finished run."""
        if not replies or not _is_usable_final_response(replies[0]):
            return
        result["messages"] = [*(result.get("messages") or []), replies[0]]
        result["last_message"] = replies[0]

    @component.output_types(messages=list[ChatMessage], last_message=ChatMessage)
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        seen: set[str] = set()
        with self._span():
            result = self._agent.run(
                messages=messages,
                streaming_callback=_tool_chunks_only(
                    streaming_callback, name=self.name, async_sink=False, seen=seen
                ),
                task=task,
                **kwargs,
            )
            if self._needs_response(result):
                # One tool-less call, so the outcome does not depend on there
                # being a spare step left after the tool budget fired.
                try:
                    replies = self._agent.chat_generator.run(messages=self._get_messages(result))
                    self._accept_response(result, replies.get("replies") or [])
                except Exception as exc:  # noqa: BLE001 — digest still carries the work
                    logger.warning(
                        "Subagent {!r} could not synthesize a final response: {}",
                        self.name,
                        exc,
                    )
        self._flush(result, seen, streaming_callback)
        return result

    def _flush(
        self,
        result: dict[str, Any],
        seen: set[str],
        callback: StreamingCallbackT | None,
    ) -> None:
        """Send tool results the run never streamed, so no UI part hangs open."""
        if callback is None:
            return
        for chunk in _unstreamed_results(result, seen):
            callback(_tag_subagent(chunk, self.name))

    @component.output_types(messages=list[ChatMessage], last_message=ChatMessage)
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        seen: set[str] = set()
        with self._span():
            result = await self._agent.run_async(
                messages=messages,
                streaming_callback=_tool_chunks_only(streaming_callback, name=self.name, seen=seen),
                task=task,
                **kwargs,
            )
            if self._needs_response(result):
                try:
                    replies = await self._agent.chat_generator.run_async(
                        messages=self._get_messages(result)
                    )
                    self._accept_response(result, replies.get("replies") or [])
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Subagent {!r} could not reach a final response: {}",
                        self.name,
                        exc,
                    )
        await self._aflush(result, seen, streaming_callback)
        return result

    async def _aflush(
        self,
        result: dict[str, Any],
        seen: set[str],
        callback: StreamingCallbackT | None,
    ) -> None:
        """Send tool results the run never streamed, so no UI part hangs open."""
        if callback is None:
            return
        for chunk in _unstreamed_results(result, seen):
            sent = callback(_tag_subagent(chunk, self.name))
            if inspect.isawaitable(sent):
                await sent


async def build_subagent(
    subagent_cls: type[Agent],
    thread_id: str = "",
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> tuple[HaystackAgent, str] | None:
    """Build a bare Haystack Agent from an SDK Agent subclass."""
    path = f"{subagent_cls.__module__}.{subagent_cls.__qualname__}"
    chain = _build_chain.get()
    if path in chain:
        logger.error(
            f"Cyclic subagent delegation detected: {' -> '.join(chain)} -> {path}. "
            f"Skipping {subagent_cls.__name__!r}."
        )
        return None

    token = _build_chain.set((*chain, path))
    try:
        subagent = subagent_cls()
        tools = await subagent.get_tools(thread_id=thread_id, user=user)
        agent = ToolAgent.build_agent(
            subagent.get_llm(),
            tools,
            subagent.get_system_prompt(),
            user_prompt=prompt("""\
                This is a task delegated to you by your coordinator.
                Complete it to the best of your ability using your tools,
                then reply with a concise summary.
                
                Task: {{task}}
            """),
            required_variables=["task"],
            max_agent_steps=subagent.max_agent_steps,
            hooks=default_hooks(subagent),
            stream_subagent_tools=True,
        )
        return agent, subagent.agent_id
    finally:
        _build_chain.reset(token)
