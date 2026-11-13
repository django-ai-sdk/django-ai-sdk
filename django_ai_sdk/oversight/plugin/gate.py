"""The gate, as before_run: judge the history and the question before the model reads them."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole, TextContent

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import GateResult
from django_ai_sdk.oversight.plugin.turn import OversightHook, _context, turn_of

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State

# The meta key on the assistant message that stands in for a refused or replaced answer.
REPLACED_META_KEY = "django_ai_sdk.oversight.replaced"


class GateHook(OversightHook):
    """Judge the run's history and question, then mask them in place or refuse the run.

    The history is every earlier user and assistant message, judged unless the
    hook_context says the stored thread vouches for it (`history_vouched`). A
    refusal ends the run before the first model call, with the policy's message
    as the answer. Otherwise the gate's notice waits on the guard for the stream,
    and the hook_context's `on_gated(turn)` runs, so a caller can store the
    question only once it passed.
    """

    allowed_hook_points = ["before_run"]  # noqa: RUF012

    def run(self, state: State) -> None:
        """A sync run cannot await the gate, so a turn of its own is refused."""
        turn = turn_of(_context(state))
        if turn is None or not turn.owns:
            return
        turn.guard.refuse("gate in a sync run")
        turn.gated = GateResult(blocked=True, message=str(Block.DEFAULT_MESSAGE))
        answer_with(state, turn.gated.message, stop="blocked")

    async def run_async(self, state: State) -> None:
        context = _context(state)
        turn = turn_of(context)
        if turn is not None and turn.owns:
            gate = turn.gated = await _gate(
                state, turn.guard, vouched=bool(context.get("history_vouched"))
            )
            if gate.blocked:
                answer_with(state, gate.message, stop="blocked")
                return
            if gate.notice:
                turn.guard.notices.append(gate.notice)
            await _mask_seeded_sources(turn.guard)
        if turn is not None:
            turn.first = len(state.data.get("messages") or [])
        if (on_gated := context.get("on_gated")) is not None:
            await on_gated(turn)


async def _gate(state: State, guard: Any, *, vouched: bool) -> GateResult:
    """Judge the history, then the question, masking each in the run's messages."""
    messages = list(state.data.get("messages") or [])
    said = [
        i
        for i, m in enumerate(messages)
        if m.role in (ChatRole.USER, ChatRole.ASSISTANT) and m.text
    ]
    judged = [
        ChatMessage(role=messages[i].role.value, content=messages[i].text or "") for i in said
    ]
    if judged:
        checked = await guard.gate_history(judged, history=not vouched)
        if checked.blocked:
            return checked
    gate = await guard.gate()
    for index, copy in zip(said, judged, strict=True):
        if copy.content != messages[index].text:
            messages[index] = with_text(messages[index], copy.content)
    if gate.masked and said and messages[said[-1]].is_from(ChatRole.USER):
        messages[said[-1]] = with_text(messages[said[-1]], gate.input)
    state.set("messages", messages, handler_override=replace_values)
    return gate


async def _mask_seeded_sources(guard: Any) -> None:
    """Mask the sources the thread already cited, before any citations card shows them."""
    from django_ai_sdk.adapters.citations.registry import current_registry
    from django_ai_sdk.oversight.plugin.finish import mask_sources

    if (registry := current_registry.get()) is not None:
        await mask_sources(guard, registry)


def with_text(message: HaystackChatMessage, text: str) -> HaystackChatMessage:
    """`message` with its text replaced, its images and other parts kept."""
    parts: list[Any] = []
    for part in message._content:
        if isinstance(part, TextContent):
            if not any(isinstance(p, TextContent) for p in parts):
                parts.append(TextContent(text=text))
        else:
            parts.append(part)
    return dataclasses.replace(message, _content=parts or [TextContent(text=text)])


def stand_in(text: str) -> HaystackChatMessage:
    return HaystackChatMessage.from_assistant(text, meta={REPLACED_META_KEY: True})


def answer_with(state: State, text: str, *, stop: str) -> None:
    """End the run with `text` as its answer."""
    messages = list(state.data.get("messages") or [])
    state.set("messages", [*messages, stand_in(text)], handler_override=replace_values)
    state.set("stop_run", stop)


__all__ = ["REPLACED_META_KEY", "GateHook", "answer_with", "stand_in", "with_text"]
