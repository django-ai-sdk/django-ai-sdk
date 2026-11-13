"""The gate, as before_run: judge the history and the question before the model reads them."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole, TextContent

from django_ai_sdk.common import ChatMessage, last_user_message
from django_ai_sdk.oversight.plugin.turn import OversightHook, request_in

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State

    from django_ai_sdk.oversight.guard import GateResult, TurnGuard
    from django_ai_sdk.oversight.plugin.turn import TurnRequest

# The meta key on the assistant message that stands in for a refused or replaced answer.
REPLACED_META_KEY = "django_ai_sdk.oversight.replaced"


class GateHook(OversightHook):
    """Judge the run's history and question, then mask them in place or refuse the run.

    The history is every earlier user and assistant message, judged unless the
    TurnRequest says the stored thread vouches for it (`history_vouched`). A
    refusal ends the run before the first model call, with the policy's message
    as the answer. Otherwise the gate's notice waits on the guard for the stream,
    and the request's `on_gated(request)` runs, so a caller can store the question
    only once it passed.
    """

    allowed_hook_points = ["before_run"]  # noqa: RUF012

    async def run_async(self, state: State) -> None:
        request = request_in(state)
        if (guard := request.guard) is not None:
            gate = request.gated = await _gate(state, guard, vouched=request.history_vouched)
            if gate.blocked:
                answer_with(state, gate.message, stop="blocked")
                return
            if gate.notice:
                guard.notices.append(gate.notice)
            await _mask_seeded_sources(guard)
        request.first = len(state.data.get("messages") or [])
        if request.on_gated is not None:
            await request.on_gated(request)


async def _gate(state: State, guard: TurnGuard, *, vouched: bool) -> GateResult:
    """Judge the history, then the question, masking each in the run's messages.

    A history the stored thread `vouched` for is not judged again, but attachment
    context always is: it is rebuilt from the document each turn and the model reads it.
    """
    messages = list(state.data.get("messages") or [])
    # The question as the run rendered it, such as a subagent's task from its
    # user_prompt: the turn opened on the messages before Haystack rendered it.
    question = next((m.text or "" for m in reversed(messages) if m.is_from(ChatRole.USER)), "")
    guard.original_input = guard.context.input = question
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
    if gate.blocked:
        return gate
    if gate.masked and (question := last_user_message(judged)) is not None:
        question.content = gate.input
    for index, copy in zip(said, judged, strict=True):
        if copy.content != messages[index].text:
            messages[index] = with_text(messages[index], copy.content)
    state.set("messages", messages, handler_override=replace_values)
    return gate


def keep_masked(messages: list[ChatMessage], request: TurnRequest) -> None:
    """Mask the caller's last user message as the gate masked it, so the caller stores that copy.

    Only its content is stored. When the masked text no longer splits back into
    the content and the attachments' context, the content is withheld rather
    than stored with what may be unmasked.
    """
    from django_ai_sdk.adapters.run import user_content
    from django_ai_sdk.oversight.guard import UNCHECKED_TEXT

    gated = request.gated
    if gated is not None and gated.masked and (question := last_user_message(messages)):
        content = user_content(question, gated.input)
        question.content = content if content is not None else str(UNCHECKED_TEXT)


async def _mask_seeded_sources(guard: TurnGuard) -> None:
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


__all__ = ["REPLACED_META_KEY", "GateHook", "answer_with", "keep_masked", "stand_in", "with_text"]
