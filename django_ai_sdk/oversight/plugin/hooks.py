"""The oversight hooks a Haystack Agent runs, and the TurnRequest they read and fill in.

    before_run   GateHook           judge the history and question; mask them, or refuse the run
    before_llm   OversightStopHook  stop a run whose turn has closed; offer only the tools it allows
    before_tool  OversightToolHook  refuse a tool call the policy says no to
    after_tool   OversightToolResultHook  mask a tool result before the model reads it
    after_run    RecordUsageHook    add the run's model and token usage to its turn
    after_run    FinishHook         give the sources to the turn; judge the answer, pass, mask or withhold it

The run's `hook_context` carries the `TurnRequest`, so the hooks read and fill in
the same object the caller holds. Each hook does nothing when the turn's policy
has nothing for its stage, or when there is no turn.

If any tool call in a step is refused, the whole step is answered with stubs,
because Haystack runs a step's calls all together or not at all. The model is
told which call was refused and that it may retry the others.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from django.utils.translation import gettext_lazy
from haystack.components.agents.state.state import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.core.serialization_security import allow_deserialization_module
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole, TextContent
from pydantic import ValidationError

from django_ai_sdk.agents.tool_agent import (
    MASKED_META_KEY,
    REFUSED_META_KEY,
    answer_calls_with,
    pending_tool_calls,
    replace_messages,
    tool_results,
)
from django_ai_sdk.common import ChatMessage, last_user_message
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.guard import current_guard, outlived_turn
from django_ai_sdk.oversight.guard.notices import TOOL_UNAVAILABLE, UNCHECKED_TEXT
from django_ai_sdk.oversight.rules import Block
from django_ai_sdk.oversight.types import Stage, ToolCall
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from haystack.components.agents.state.state import State
    from haystack.dataclasses import ToolCallResult
    from pydantic import BaseModel

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.guard.stages import StageResult
    from django_ai_sdk.oversight.policy import Policy

# An Agent saved with these hooks loads again.
allow_deserialization_module(__name__)

# Where the run's hook_context holds its TurnRequest.
REQUEST_KEY = "django_ai_sdk.oversight"


@dataclass
class TurnRequest:
    """A run's turn: what the caller wants of it, then what oversight decided.

    Pass it to `run_agent(request=...)`. Once the run started, `guard` is the
    turn it opened, or None when it joined an outer turn or has no policy.
    """

    # The SDK agent whose policy the run is under.
    agent: Any = None
    channel: str = "run"
    user: Any = None
    thread_id: str | None = None
    # A policy for this run only, as in `Agent.run(oversight=...)`.
    oversight: type[Policy] | str | Literal[False] | None = None
    # The stored thread vouches for the history, so the gate does not judge it again.
    history_vouched: bool = False
    # Called once the gate passed the question, such as to store it.
    on_gated: Callable[[TurnRequest], Awaitable[None]] | None = None
    # Reads the answer finish judges off the messages the run produced, such as
    # what a coordinator reads of a subagent. A replaced answer then takes the
    # place of every produced message. By default finish judges all the text the
    # model wrote, as a stream shows it, and a replacement takes the last reply's place.
    read: Callable[[list[HaystackChatMessage]], str] | None = None
    # A Pydantic model the answer must still fit once masked.
    schema: type[BaseModel] | None = None
    # The id of the message that will store the answer.
    message_id: str = ""
    # The caller closes the turn itself, such as a stream that still stores the answer.
    keep_open: bool = False
    # Called with the stubs of refused tool calls, which never reach the ToolInvoker.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    guard: TurnGuard | None = None
    gated: StageResult | None = None
    # Index of the first message the run produced, after the gate.
    first: int = 0

    @property
    def refused(self) -> bool:
        return self.gated is not None and self.gated.blocked

    @property
    def finished(self) -> StageResult | None:
        return self.guard.finished if self.guard is not None else None

    def raise_if_blocked(self) -> None:
        """Raise `Blocked` when the gate refused the question or finish withheld the answer."""
        if self.guard is None:
            return
        if self.refused:
            assert self.gated is not None
            raise Blocked(self.gated.message, stage=str(Stage.GATE), turn_id=self.guard.id)
        if (finished := self.finished) is not None and finished.blocked:
            raise Blocked(finished.message, stage=str(Stage.FINISH), turn_id=self.guard.id)

    def context(self, **extra: Any) -> dict[str, Any]:
        """A hook_context holding this request, beside `extra` for other hooks."""
        return {**extra, REQUEST_KEY: self}


def turn_request(state: State) -> TurnRequest:
    """The run's TurnRequest. A run started without one gets the defaults."""
    return (state.data.get("hook_context") or {}).get(REQUEST_KEY) or TurnRequest()


class OversightHook:
    """Base for the oversight hooks: serializable, so an Agent with them still saves to YAML.

    Haystack calls `run_async` in an async run, and `run` in a sync one. A sync
    run never opens a turn of its own (it cannot await the guard), so `run`
    does nothing unless a hook overrides it.
    """

    allowed_hook_points: list[str] = []  # noqa: RUF012

    def run(self, state: State) -> None:
        """A sync run: nothing to do."""

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OversightHook:
        return default_from_dict(cls, data)


class OversightStopHook(OversightHook):
    """Before each model call: stop a run whose turn has closed, and offer only the tools it allows."""

    allowed_hook_points = ["before_llm"]  # noqa: RUF012

    def run(self, state: State) -> None:
        if outlived_turn():
            state.set("stop_run", "turn closed")
        elif (guard := current_guard()) is not None:
            # Haystack hands this same list to the model and the ToolInvoker for
            # the step, so filtering it in place takes the tools away. The tool
            # hook refuses a call to a removed tool anyway.
            tools = state.data["tools"]
            tools[:] = [tool for tool in tools if guard.allows_tool(tool.name)]

    async def run_async(self, state: State) -> None:
        self.run(state)


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
        from django_ai_sdk.adapters.citations.registry import current_registry

        request = turn_request(state)
        if (guard := request.guard) is not None:
            gate = request.gated = await judge_run_input(
                state, guard, vouched=request.history_vouched
            )
            if gate.blocked:
                end_run_with_answer(state, gate.message, stop="blocked")
                return
            if gate.notice:
                guard.notices.pending.append(gate.notice)
            # Mask the sources the thread already cited, before any citations card shows them.
            if (registry := current_registry.get()) is not None:
                await mask_sources(guard, registry)
        request.first = len(state.data.get("messages") or [])
        if request.on_gated is not None:
            await request.on_gated(request)


async def judge_run_input(state: State, guard: TurnGuard, *, vouched: bool) -> StageResult:
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
    if gate.changed and (question := last_user_message(judged)) is not None:
        question.content = gate.text
    for index, copy in zip(said, judged, strict=True):
        if copy.content != messages[index].text:
            messages[index] = masked_message(messages[index], copy.content)
    state.set("messages", messages, handler_override=replace_values)
    return gate


def masked_message(message: HaystackChatMessage, text: str) -> HaystackChatMessage:
    """`message` with its text replaced, its images and other parts kept."""
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
    """End the run with `text` as its answer."""
    messages = list(state.data.get("messages") or [])
    state.set("messages", [*messages, replacement_answer(text)], handler_override=replace_values)
    state.set("stop_run", stop)


# Told to the model for a call skipped because another call in the step was refused.
SKIPPED = gettext_lazy(
    "Not run: another call in this step was refused. Call it again on its own if you still need it."
)

# Sync hooks cannot await the guard, so every tool call in a sync run is refused.
UNGUARDABLE = gettext_lazy("This tool cannot be checked in a sync run, so it is not run.")


class OversightToolHook(OversightHook):
    """Refuse tool calls the turn's policy says no to, before they run."""

    allowed_hook_points = ["before_tool"]  # noqa: RUF012

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
            guard.add_tool_call(_as_tool_call(call, state))
        guard.note_error("tool called in a sync run")
        _refuse_tool_step(state, messages, {i: str(UNGUARDABLE) for i in range(len(calls))})

    async def run_async(self, state: State) -> None:
        if _refused_after_turn(state):
            return
        step = _tool_step(state)
        if step is None:
            return
        guard, messages = step
        refusals: dict[int, str] = {}
        for index, call in enumerate(messages[-1].tool_calls):
            seen = _as_tool_call(call, state)
            guard.add_tool_call(seen)
            reason = await guard.tool_call(seen)
            if reason:
                refusals[index] = reason
        if not refusals:
            return
        stubs = _refuse_tool_step(state, messages, refusals)
        if guard.on_tool_results is not None:
            await guard.on_tool_results(stubs)


def _tool_step(state: State) -> tuple[TurnGuard, list[HaystackChatMessage]] | None:
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
        _refuse_tool_step(state, messages, {i: str(TOOL_UNAVAILABLE) for i in range(len(calls))})
    return True


def _as_tool_call(call: Any, state: State) -> ToolCall:
    return ToolCall(
        id=call.id or "", name=call.tool_name, arguments=call.arguments, agent=_agent_name(state)
    )


def _agent_name(state: State) -> str:
    """The agent of a run that joined an outer turn, recorded with its tool calls."""
    request = turn_request(state)
    if request.guard is not None or request.agent is None:
        return ""
    return str(getattr(request.agent, "name", "") or type(request.agent).__name__)


def _refuse_tool_step(
    state: State, messages: list[HaystackChatMessage], refusals: dict[int, str]
) -> list[HaystackChatMessage]:
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


class OversightToolResultHook(OversightHook):
    """Mask tool results before the model reads them, like the gate masks the question.

    A tool can return data the gate would hide, such as a record number or a
    client's name. Each result is checked once, when it arrives.
    """

    allowed_hook_points = ["after_tool"]  # noqa: RUF012

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
        changed: dict[int, HaystackChatMessage] = {}
        for index, message, result in tool_results(messages, MASKED_META_KEY):
            masked = await masked_result(guard, result.origin.id or "", result.result)
            guard.add_tool_call(
                ToolCall(
                    id=result.origin.id or "",
                    name="" if result.origin.id else result.origin.tool_name,
                    result=masked if isinstance(masked, str) else _text_of(masked),
                    error=bool(result.error),
                    agent=_agent_name(state),
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


def _with_result(
    message: HaystackChatMessage, result: ToolCallResult, content: Any
) -> HaystackChatMessage:
    """`message` with new content, marked as checked so it is not checked again."""
    return HaystackChatMessage.from_tool(
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


class RecordUsageHook(OversightHook):
    """Add the run's model and token usage to the turn it runs in, sync or async."""

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


class FinishHook(OversightHook):
    """Judge the answer of a run with a turn of its own: pass, mask or withhold it.

    The citation sources are masked and given to the turn first, so finish can
    judge the answer against them. The answer is then passed, masked or withheld
    in the run's messages. The TurnRequest's `read` says which text that is. With
    its `schema` (a Pydantic model), a masked answer that no longer fits is withheld.
    A sync run has no turn of its own, so it has nothing to finish.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    async def run_async(self, state: State) -> None:
        from django_ai_sdk.adapters.citations.registry import current_registry

        request = turn_request(state)
        guard = request.guard
        if guard is None:
            return
        if (registry := current_registry.get()) is not None:
            await mask_sources(guard, registry)
            guard.set_sources(registry.all, seeded=registry.seeded)
        if request.refused:
            return
        messages = list(state.data.get("messages") or [])
        produced = messages[request.first :]
        text = request.read(produced) if request.read else assistant_text(produced)
        if not text:
            guard.note_error("no result")
            return
        finished = await guard.finish(text)
        if request.schema is not None:
            withhold_unfit(guard, finished, request.schema)
        if finished.changed:
            if request.read:
                messages = [*messages[: request.first], replacement_answer(finished.text)]
            else:
                messages = replace_answer(messages, request.first, finished.text)
            state.set("messages", messages, handler_override=replace_values)


def withhold_unfit(guard: TurnGuard, finished: StageResult, schema: type[BaseModel]) -> None:
    """Withhold a masked structured answer whose JSON no longer fits `schema`.

    Returning the unmasked original instead would leak what the mask hid.
    """
    if not finished.changed or finished.blocked:
        return
    try:
        schema.model_validate_json(finished.text)
    except ValidationError:
        guard.refuse("masked result does not fit its schema")
        finished.blocked = True
        finished.message = finished.text = str(Block.DEFAULT_MESSAGE)


async def mask_sources(guard: TurnGuard, registry: SourceRegistry) -> None:
    """Mask source text in the citation registry, like tool results.

    The citations card, the stored citations artifact and the lenses all read
    the registry, so none of them should see unmasked text. The gate, the stream
    and the finish hook call this; each text is masked once.
    """
    for source in registry.all:
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
