"""The oversight hooks a Haystack Agent runs. Each does nothing outside a turn.

If any tool call in a step is blocked, the whole step is answered with stubs,
because Haystack runs a step's calls all together or not at all.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy
from haystack.components.agents.state.state import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.core.serialization_security import allow_deserialization_module
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole, TextContent
from pydantic import ValidationError

from django_ai_sdk.adapters.citations.registry import current_registry
from django_ai_sdk.adapters.run import ATTACHMENT_CONTEXT_META_KEY
from django_ai_sdk.agents.tool_agent import (
    BLOCKED_META_KEY,
    MASKED_META_KEY,
    STREAMED_META_KEY,
    answer_calls_with,
    pending_tool_calls,
    replace_messages,
    tool_failed,
    tool_results,
)
from django_ai_sdk.common import Attachment, ChatMessage
from django_ai_sdk.oversight.guard import current_guard, outlived_turn
from django_ai_sdk.oversight.guard.notices import TOOL_UNAVAILABLE_MESSAGE, UNCHECKED_TEXT
from django_ai_sdk.oversight.guard.stages import StageResult
from django_ai_sdk.oversight.plugin.request import REQUEST_KEY, TurnRequest
from django_ai_sdk.oversight.rules import Block
from django_ai_sdk.oversight.types import Stage, ToolCall
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State
    from haystack.dataclasses import StreamingChunk, ToolCallResult
    from pydantic import BaseModel

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.oversight.guard import TurnGuard

# An Agent saved with these hooks loads again.
allow_deserialization_module(__name__)


def turn_request(state: State) -> TurnRequest:
    return (state.data.get("hook_context") or {}).get(REQUEST_KEY) or TurnRequest()


class OversightHook:
    """Serializable, so an Agent with them still saves to YAML.

    A sync run never opens a turn of its own, so `run` does nothing unless overridden.
    """

    allowed_hook_points: list[str] = []  # noqa: RUF012

    def run(self, state: State) -> None:
        """A sync run has no turn of its own."""

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OversightHook:
        return default_from_dict(cls, data)


class StopClosedTurnHook(OversightHook):
    allowed_hook_points = ["before_llm"]  # noqa: RUF012

    def run(self, state: State) -> None:
        if outlived_turn():
            state.set("stop_run", "turn closed")

    async def run_async(self, state: State) -> None:
        self.run(state)


# On the system message that tells the model which tools oversight took away.
TOOLS_REMOVED_META_KEY = "django_ai_sdk.oversight.tools_removed"
TOOLS_REMOVED_MESSAGE = gettext_lazy(
    "These tools are not available for this message: {names}. Answer without them, and "
    "don't say you used them."
)


class LimitToolChoiceHook(OversightHook):
    allowed_hook_points = ["before_llm"]  # noqa: RUF012

    def run(self, state: State) -> None:
        # No guard once the turn has closed: StopClosedTurnHook stops that run.
        if (guard := current_guard()) is None:
            return
        # Haystack hands this same list to the model and the ToolInvoker for the
        # step, so filtering it in place takes the tools away.
        tools = state.data["tools"]
        kept = [tool for tool in tools if guard.allows_tool(tool.name)]
        removed = sorted({tool.name for tool in tools} - {tool.name for tool in kept})
        tools[:] = kept
        messages = state.data.get("messages") or []
        told = [m.meta.get(TOOLS_REMOVED_META_KEY) for m in messages]
        # Each step gets the full list again, so the model is told once per set removed.
        if removed and removed not in told and "messages" in state.schema:
            # The system prompt may still describe them; told nothing, a model says it used them.
            note = HaystackChatMessage.from_system(
                str(TOOLS_REMOVED_MESSAGE).format(names=", ".join(removed)),
                meta={TOOLS_REMOVED_META_KEY: removed},
            )
            state.set("messages", [note])

    async def run_async(self, state: State) -> None:
        self.run(state)


# The meta key on the assistant message that stands in for a blocked or replaced answer.
REPLACED_META_KEY = "django_ai_sdk.oversight.replaced"


# The call id a joined run's question is masked under: the same question is judged once.
JOINED = "joined"


class GateQuestionHook(OversightHook):
    """Judge history and question, then mask them in place or block the run.

    A block ends the run before the first model call. `on_gated` runs only once
    the question passed, so a caller stores only gated questions. A run that
    joined an outer turn has its question judged there, like a tool result.
    """

    allowed_hook_points = ["before_run"]  # noqa: RUF012

    async def run_async(self, state: State) -> None:
        turn = turn_request(state)
        if (guard := turn.guard) is not None:
            gate = turn.gate_result = await judge_run_input(
                state, guard, vouched=turn.history_vouched
            )
            if gate.blocked:
                end_run_with_answer(state, gate.message, stop="blocked")
                return
            if gate.notice:
                guard.notices.pending.append(gate.notice)
            # Before any citations card shows them.
            await mask_sources(guard, current_registry.get())
        elif (outer := current_guard()) is not None:
            turn.gate_result = await mask_joined_question(state, outer)
        turn.first_produced_index = len(state.data.get("messages") or [])
        if turn.on_gated is not None:
            await turn.on_gated(turn)


async def judge_run_input(state: State, guard: TurnGuard, *, vouched: bool) -> StageResult:
    """Judge history, then question, masking each in the run's messages."""
    messages = list(state.data.get("messages") or [])
    asked = question_index(messages)
    # The question as rendered (e.g. a subagent's user_prompt); the turn opened before that.
    question = ChatMessage(
        role="user", content=said_text(messages[asked]) if asked is not None else ""
    )
    guard.original_input = guard.context.input = question.content
    earlier = [
        i
        for i, m in enumerate(messages)
        if i != asked and m.role in (ChatRole.USER, ChatRole.ASSISTANT) and said_text(m)
    ]
    judged = [as_judged(messages[i]) for i in earlier]
    checked = await guard.gate_history(judged, question, history=not vouched)
    if checked.blocked:
        return checked
    gate = await guard.gate()
    if gate.blocked:
        return gate
    if gate.changed:
        question.content = gate.text
    for index, copy in zip([*earlier, asked], [*judged, question], strict=True):
        text = copy.content + "".join(a.context for a in copy.attachments)
        if index is not None and text != said_text(messages[index]):
            messages[index] = masked_message(messages[index], text)
    state.set("messages", messages, handler_override=replace_values)
    return gate


async def mask_joined_question(state: State, guard: TurnGuard) -> StageResult:
    """Judge the question of a run that joined `guard`'s turn like a tool result.

    Such a run has no gate of its own. The messages before its question are the
    outer turn's, judged there. Returns what the caller's copy should read.
    """
    messages = list(state.data.get("messages") or [])
    asked = question_index(messages)
    text = said_text(messages[asked]) if asked is not None else ""
    if asked is None or not text:
        return StageResult(Stage.GATE, text=text)
    masked = await guard.mask_result(JOINED, text)
    if masked != text:
        messages[asked] = masked_message(messages[asked], masked)
        state.set("messages", messages, handler_override=replace_values)
    return StageResult(Stage.GATE, text=masked, changed=masked != text)


def question_index(messages: list[HaystackChatMessage]) -> int | None:
    """The last message a person sent. Every other text, a reply after it too, is history."""
    return next(
        (i for i in reversed(range(len(messages))) if messages[i].is_from(ChatRole.USER)), None
    )


def said_text(message: HaystackChatMessage) -> str:
    """Every text part: the model reads them all."""
    return "\n\n".join(message.texts)


def as_judged(message: HaystackChatMessage) -> ChatMessage:
    """A copy to judge, with any attachment context apart from what the person wrote."""
    text = said_text(message)
    role = "user" if message.is_from(ChatRole.USER) else "assistant"
    at = message.meta.get(ATTACHMENT_CONTEXT_META_KEY)
    if at is None:
        return ChatMessage(role=role, content=text)
    context = Attachment(document_id="", context=text[at:])
    return ChatMessage(role=role, content=text[:at], attachments=[context])


def masked_message(message: HaystackChatMessage, text: str) -> HaystackChatMessage:
    parts: list[Any] = []
    for part in message._content:
        if isinstance(part, TextContent):
            if not any(isinstance(p, TextContent) for p in parts):
                parts.append(TextContent(text=text))
        else:
            parts.append(part)
    return dataclasses.replace(message, _content=parts or [TextContent(text=text)])


def replacement_answer(text: str) -> HaystackChatMessage:
    return HaystackChatMessage.from_assistant(text, meta={REPLACED_META_KEY: True})


def end_run_with_answer(state: State, text: str, *, stop: str) -> None:
    messages = list(state.data.get("messages") or [])
    state.set("messages", [*messages, replacement_answer(text)], handler_override=replace_values)
    state.set("stop_run", stop)


SKIPPED_CALL_MESSAGE = gettext_lazy(
    "Not run: another call in this step was blocked. Call it again on its own if you still need it."
)

# Sync hooks cannot await the guard, so every tool call in a sync run is blocked.
UNGUARDABLE_CALL_MESSAGE = gettext_lazy(
    "This tool cannot be checked in a sync run, so it is not run."
)


class JudgeToolCallHook(OversightHook):
    allowed_hook_points = ["before_tool"]  # noqa: RUF012

    def run(self, state: State) -> None:
        """Inside a turn, block every call: the guard cannot check sync calls."""
        if _blocked_after_turn(state):
            return
        step = _tool_step(state)
        if step is None:
            return
        guard, messages = step
        calls = messages[-1].tool_calls
        for call in calls:
            guard.add_tool_call(_as_tool_call(call, state))
        guard.note_error("tool called in a sync run")
        _block_tool_step(
            state, messages, {i: str(UNGUARDABLE_CALL_MESSAGE) for i in range(len(calls))}
        )

    async def run_async(self, state: State) -> None:
        if _blocked_after_turn(state):
            return
        step = _tool_step(state)
        if step is None:
            return
        guard, messages = step
        # A model may use a call id again; only this step's own stubs go unjudged.
        guard.masks.answered_here.difference_update(c.id for c in messages[-1].tool_calls)
        block_messages: dict[int, str] = {}
        for index, call in enumerate(messages[-1].tool_calls):
            seen = _as_tool_call(call, state)
            guard.add_tool_call(seen)
            reason = await guard.tool_call(seen)
            if reason:
                block_messages[index] = reason
        if not block_messages:
            return
        # The whole step is answered with stubs, which are oversight's own words.
        guard.masks.answered_here.update(call.id for call in messages[-1].tool_calls if call.id)
        streamed = guard.on_tool_results is not None
        stubs = _block_tool_step(state, messages, block_messages, streamed=streamed)
        if streamed:
            await guard.on_tool_results(stubs)


def _tool_step(state: State) -> tuple[TurnGuard, list[HaystackChatMessage]] | None:
    guard = current_guard()
    messages = pending_tool_calls(state) if guard is not None else None
    if guard is None or messages is None:
        return None
    return guard, messages


def _blocked_after_turn(state: State) -> bool:
    if not outlived_turn():
        return False
    if (messages := pending_tool_calls(state)) is not None:
        calls = messages[-1].tool_calls
        _block_tool_step(
            state, messages, {i: str(TOOL_UNAVAILABLE_MESSAGE) for i in range(len(calls))}
        )
    return True


def _as_tool_call(call: Any, state: State) -> ToolCall:
    return ToolCall(
        id=call.id or "", name=call.tool_name, arguments=call.arguments, agent=_agent_name(state)
    )


def _agent_name(state: State) -> str:
    """The agent of a run that joined an outer turn, recorded on its tool calls."""
    turn = turn_request(state)
    if turn.guard is not None or turn.agent is None:
        return ""
    return str(getattr(turn.agent, "name", "") or type(turn.agent).__name__)


def _block_tool_step(
    state: State,
    messages: list[HaystackChatMessage],
    block_messages: dict[int, str],
    *,
    streamed: bool = False,
) -> list[HaystackChatMessage]:
    return answer_calls_with(
        state,
        messages,
        [
            (
                {"refused": block_messages[index]}
                if index in block_messages
                else {"skipped": str(SKIPPED_CALL_MESSAGE)},
                {BLOCKED_META_KEY: index in block_messages, STREAMED_META_KEY: streamed},
            )
            for index in range(len(messages[-1].tool_calls))
        ],
    )


class MaskToolResultHook(OversightHook):
    """Mask tool results before the model reads them. Each result is checked once."""

    allowed_hook_points = ["after_tool"]  # noqa: RUF012

    def run(self, state: State) -> None:
        """Inside a turn, withhold every non-stub result: it ran unchecked."""
        if current_guard() is not None or outlived_turn():
            _withhold_results(state)

    async def run_async(self, state: State) -> None:
        guard = current_guard()
        if guard is None:
            if outlived_turn():
                _withhold_results(state)
            return
        messages = state.data.get("messages") or []
        changed: dict[int, HaystackChatMessage] = {}
        for index, message, result in tool_results(messages, MASKED_META_KEY):
            masked = await mask_tool_result(guard, result.origin.id or "", result.result)
            guard.add_tool_call(
                ToolCall(
                    id=result.origin.id or "",
                    name="" if result.origin.id else result.origin.tool_name,
                    result=masked
                    if isinstance(masked, str)
                    else "\n".join(p.text for p in masked if isinstance(p, TextContent)),
                    error=tool_failed(result),
                    agent=_agent_name(state),
                )
            )
            changed[index] = _with_result(message, result, masked)
        replace_messages(state, messages, changed)


def _withhold_results(state: State) -> None:
    messages = state.data.get("messages") or []
    withheld = {
        index: _with_result(message, result, str(UNCHECKED_TEXT))
        for index, message, result in tool_results(messages, MASKED_META_KEY)
        if BLOCKED_META_KEY not in message.meta
    }
    replace_messages(state, messages, withheld)


def _with_result(
    message: HaystackChatMessage, result: ToolCallResult, content: Any
) -> HaystackChatMessage:
    return HaystackChatMessage.from_tool(
        content,
        origin=result.origin,
        error=result.error,
        meta={**message.meta, MASKED_META_KEY: True},
    )


async def mask_tool_result(guard: TurnGuard, call_id: str, result: Any) -> Any:
    """Mask the text in a tool result; images and files pass through.

    Non-str, non-list values are judged as JSON and replaced only when masking changed them.
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


async def mask_streamed_result(chunk: StreamingChunk) -> StreamingChunk:
    """`chunk` with its tool result masked by the current turn, as its model reads it.

    Haystack streams a tool result before the after_tool hooks mask it, so a stream
    that forwards it elsewhere (a subagent's to its coordinator) masks it here first.
    """
    guard = current_guard()
    result = chunk.tool_call_result
    if guard is None or result is None:
        return chunk
    masked = await mask_tool_result(guard, result.origin.id or "", result.result)
    if masked is result.result:
        return chunk
    return dataclasses.replace(chunk, tool_call_result=dataclasses.replace(result, result=masked))


class RecordUsageHook(OversightHook):
    allowed_hook_points = ["after_run"]  # noqa: RUF012

    def run(self, state: State) -> None:
        guard = current_guard()
        if guard is None:
            return
        prompt, completion = token_counts(state.data.get("token_usage") or None)
        model = next(
            (
                message.meta["model"]
                for message in state.data.get("messages") or []
                if message.is_from(ChatRole.ASSISTANT) and message.meta.get("model")
            ),
            None,
        )
        guard.add_usage(model, prompt, completion)

    async def run_async(self, state: State) -> None:
        self.run(state)


class JudgeAnswerHook(OversightHook):
    """Judge the answer of a run with its own turn: pass, mask or withhold it.

    Sources are masked and given to the turn first, so finish judges against them.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    async def run_async(self, state: State) -> None:
        turn = turn_request(state)
        guard = turn.guard
        if guard is None:
            return
        if (registry := current_registry.get()) is not None:
            await mask_sources(guard, registry)
            guard.set_sources(registry.all, seeded=registry.seeded)
        if turn.question_blocked:
            return
        messages = list(state.data.get("messages") or [])
        produced = messages[turn.first_produced_index :]
        text = turn.read_answer(produced) if turn.read_answer else assistant_text(produced)
        if not text:
            guard.note_error("no result")
            return
        finished = await guard.finish(text)
        if turn.answer_schema is not None:
            withhold_unfit(guard, finished, turn.answer_schema)
        if finished.changed:
            if turn.read_answer:
                messages = [
                    *messages[: turn.first_produced_index],
                    replacement_answer(finished.text),
                ]
            else:
                messages = replace_answer(messages, turn.first_produced_index, finished.text)
            state.set("messages", messages, handler_override=replace_values)


def withhold_unfit(guard: TurnGuard, finished: StageResult, schema: type[BaseModel]) -> None:
    """Withhold a masked answer that no longer fits `schema`; the original would leak."""
    if not finished.changed or finished.blocked:
        return
    try:
        schema.model_validate_json(finished.text)
    except ValidationError:
        guard.block("masked result does not fit its schema")
        finished.blocked = True
        finished.message = finished.text = str(Block.DEFAULT_MESSAGE)


async def mask_sources(guard: TurnGuard, registry: SourceRegistry | None) -> None:
    """Mask source text in the citation registry; card, artifact and lenses all read it."""
    for source in registry.all if registry is not None else []:
        text = source.content or ""
        if not text or guard.masks.sources.get(source.key) == text:
            continue
        # mask_result keys on the text too, so a changed source is masked again.
        masked = await guard.mask_result(f"source:{source.key}", text)
        source.content = guard.masks.sources[source.key] = masked


def assistant_text(produced: list[HaystackChatMessage]) -> str:
    return "".join(m.text or "" for m in produced if m.is_from(ChatRole.ASSISTANT))


def replace_answer(
    messages: list[HaystackChatMessage], first: int, output: str
) -> list[HaystackChatMessage]:
    for index in range(len(messages) - 1, first - 1, -1):
        message = messages[index]
        if message.is_from(ChatRole.ASSISTANT) and message.text and not message.tool_calls:
            meta = {**message.meta, REPLACED_META_KEY: True}
            replaced = HaystackChatMessage.from_assistant(output, meta=meta)
            return [*messages[:index], replaced, *messages[index + 1 :]]
    return [*messages, replacement_answer(output)]
