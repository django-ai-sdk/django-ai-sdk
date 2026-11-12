"""Haystack hooks that check tool calls with the turn's guard.

They are installed on every tool agent, including subagents, and do nothing
without an active guard. Every call and result is recorded in the turn.

If any call in a step is refused, the whole step is answered with stubs, because
Haystack runs a step's calls all together or not at all. The model is told which
call was refused and that it may retry the others. Results of calls that did run
are masked like the question before the model reads them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy
from haystack.dataclasses import ChatMessage, TextContent

from django_ai_sdk.agents.tool_agent import (
    MASKED_META_KEY,
    REFUSED_META_KEY,
    answer_calls_with,
    pending_tool_calls,
    replace_messages,
    tool_results,
)
from django_ai_sdk.oversight.guard import (
    TOOL_UNAVAILABLE,
    UNCHECKED_TEXT,
    current_guard,
    outlived_turn,
)
from django_ai_sdk.oversight.types import ToolCall

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State
    from haystack.dataclasses import ToolCallResult

    from django_ai_sdk.oversight.guard import TurnGuard

# Told to the model for a call skipped because another call in the step was refused.
SKIPPED = gettext_lazy(
    "Not run: another call in this step was refused. Call it again on its own if you still need it."
)

# Sync hooks cannot await the guard, so every tool call in a sync run is refused.
UNGUARDABLE = gettext_lazy("This tool cannot be checked in a sync run, so it is not run.")


class OversightToolHook:
    """Refuse tool calls the turn's policy says no to, before they run."""

    # The SDK's seam for tool calls. agents/tool_agent.py installs this hook as a
    # Haystack before_tool hook on every tool agent.
    allowed_hook_points = ["before_tool"]

    def run(self, state: State) -> None:
        """Refuse every call when inside a turn.

        The guard only works from async code, so a sync call cannot be checked.
        """
        if _refused_after_turn(state):
            return
        step = _tool_step(state)
        if step is None:
            return
        guard, messages = step
        calls = messages[-1].tool_calls
        for call in calls:
            guard.add_tool_call(_as_tool_call(call))
        guard.note_error("tool called in a sync run")
        _answer_with_stubs(state, messages, {i: str(UNGUARDABLE) for i in range(len(calls))})

    async def run_async(self, state: State) -> None:
        if _refused_after_turn(state):
            return
        step = _tool_step(state)
        if step is None:
            return
        guard, messages = step
        refusals: dict[int, str] = {}
        for index, call in enumerate(messages[-1].tool_calls):
            seen = _as_tool_call(call)
            guard.add_tool_call(seen)
            reason = await guard.tool_call(seen)
            if reason:
                refusals[index] = reason
        if not refusals:
            return
        stubs = _answer_with_stubs(state, messages, refusals)
        if guard.on_tool_results is not None:
            await guard.on_tool_results(stubs)


def _tool_step(state: State) -> tuple[TurnGuard, list[ChatMessage]] | None:
    """The guard and messages, when inside a turn and the last message calls tools."""
    guard = current_guard()
    messages = pending_tool_calls(state) if guard is not None else None
    if guard is None or messages is None:
        return None
    return guard, messages


def _refused_after_turn(state: State) -> bool:
    """Refuse every pending call of a run that outlived its turn. True when it did."""
    if not outlived_turn():
        return False
    if (messages := pending_tool_calls(state)) is not None:
        calls = messages[-1].tool_calls
        _answer_with_stubs(state, messages, {i: str(TOOL_UNAVAILABLE) for i in range(len(calls))})
    return True


def _as_tool_call(call: Any) -> ToolCall:
    return ToolCall(id=call.id or "", name=call.tool_name, arguments=call.arguments)


def _answer_with_stubs(
    state: State, messages: list[ChatMessage], refusals: dict[int, str]
) -> list[ChatMessage]:
    """Answer the step's calls with stubs. Refused calls say why, the rest say skipped."""
    return answer_calls_with(
        state,
        messages,
        [
            (
                {"refused": refusals[index]} if index in refusals else {"skipped": str(SKIPPED)},
                {REFUSED_META_KEY: index in refusals},
            )
            for index in range(len(messages[-1].tool_calls))
        ],
    )


class OversightToolResultHook:
    """Mask tool results before the model reads them, like the gate masks the question.

    A tool can return data the gate would hide, such as a record number or a
    client's name. Each result is checked once, when it arrives.
    """

    # The SDK's seam for tool results. agents/tool_agent.py installs this hook as a
    # Haystack after_tool hook on every tool agent.
    allowed_hook_points = ["after_tool"]

    def run(self, state: State) -> None:
        """Withhold every result that is not a stub when inside a turn.

        The sync before-tool hook refuses every call, so any other result ran unchecked.
        """
        if current_guard() is not None or outlived_turn():
            _withhold_results(state)

    async def run_async(self, state: State) -> None:
        guard = current_guard()
        if guard is None:
            # A run that outlived its turn: nothing can check its results.
            if outlived_turn():
                _withhold_results(state)
            return
        messages = state.data.get("messages") or []
        changed: dict[int, ChatMessage] = {}
        for index, message, result in tool_results(messages, MASKED_META_KEY):
            masked = await masked_result(guard, result.origin.id or "", result.result)
            guard.add_tool_call(
                ToolCall(
                    id=result.origin.id or "",
                    name="" if result.origin.id else result.origin.tool_name,
                    result=masked if isinstance(masked, str) else _text_of(masked),
                    error=bool(result.error),
                )
            )
            changed[index] = _with_result(message, result, masked)
        replace_messages(state, messages, changed)


def _withhold_results(state: State) -> None:
    """Replace every unchecked result that is not a stub with UNCHECKED_TEXT."""
    messages = state.data.get("messages") or []
    withheld = {
        index: _with_result(message, result, str(UNCHECKED_TEXT))
        for index, message, result in tool_results(messages, MASKED_META_KEY)
        if REFUSED_META_KEY not in message.meta
    }
    replace_messages(state, messages, withheld)


def _with_result(message: ChatMessage, result: ToolCallResult, content: Any) -> ChatMessage:
    """`message` with new content, marked as checked so it is not checked again."""
    return ChatMessage.from_tool(
        content,
        origin=result.origin,
        error=result.error,
        meta={**message.meta, MASKED_META_KEY: True},
    )


async def masked_result(guard: TurnGuard, call_id: str, result: Any) -> Any:
    """Mask the text in a tool result. Images and files pass through unchanged.

    A result is a string or a list of content parts. Any other value is judged as
    JSON, and is replaced by the masked text only when masking changed it.
    """
    if isinstance(result, str):
        return await guard.mask_result(call_id, result)
    if result is None:
        return result
    if not isinstance(result, list):
        text = json.dumps(result, default=str)
        masked = await guard.mask_result(call_id, text)
        return result if masked == text else masked
    return [
        TextContent(text=await guard.mask_result(f"{call_id}#{index}", part.text))
        if isinstance(part, TextContent)
        else part
        for index, part in enumerate(result)
    ]


def _text_of(parts: list[Any]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextContent))


__all__ = [
    "OversightToolHook",
    "OversightToolResultHook",
    "masked_result",
]
