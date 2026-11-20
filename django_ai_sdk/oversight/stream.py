"""Oversight in a streamed chat turn; every intervention reaches the client as `data-oversight`."""

from __future__ import annotations

import dataclasses
import uuid
from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy
from haystack.dataclasses import StreamingChunk, ToolCallResult

from django_ai_sdk.agents.tool_agent import BLOCKED_META_KEY
from django_ai_sdk.artifacts.schemas import CitationsData
from django_ai_sdk.events import (
    DataEvent,
    MessageEndEvent,
    MessageStartEvent,
    ReasoningChunkEvent,
    StreamEndEvent,
    TextChunkEvent,
    TextEndEvent,
    ToolCallStartEvent,
    ToolInputChunkEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
)
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.guard.notices import notice_base
from django_ai_sdk.oversight.guard.stages import StageResult
from django_ai_sdk.oversight.plugin.hooks import mask_sources, mask_tool_result
from django_ai_sdk.oversight.plugin.request import TurnRequest
from django_ai_sdk.oversight.rules import Block
from django_ai_sdk.oversight.types import Outcome, Stage
from django_ai_sdk.receipts import receipt

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, Mapping

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.adapters.citations.grounding import Grounder
    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.common import ChatMessage, StreamWriter
    from django_ai_sdk.events import StreamEvent

# The guard a stream reads before the run opened its turn: allows everything.
_INERT = TurnGuard()

CONTENT_FILTER = "content-filter"

# While an answer is held, tool cards show only that a tool ran.
HELD_OUTPUT = gettext_lazy("Checking the answer first. The result shows here once it passes.")
WITHHELD_OUTPUT = gettext_lazy("Not shown: oversight withheld this answer.")


def notice_event(notice: dict[str, Any]) -> DataEvent:
    return DataEvent(data_type="oversight", data=notice)


def blocked_reply(message: str) -> list[StreamEvent]:
    return [
        TextChunkEvent(content=message),
        TextEndEvent(),
        MessageEndEvent(finish_reason=CONTENT_FILTER),
        StreamEndEvent(),
    ]


def stored_notices(notices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ "replace" becomes "replaced": True; a stored copy would go stale when reflect masks."""
    return [
        {**{k: v for k, v in n.items() if k != "replace"}, "replaced": True}
        if "replace" in n
        else n
        for n in notices
    ]


class BlockedStream:
    """The full reply when the gate blocks a question. Stores nothing."""

    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None

    def __init__(self, message: str, notice: dict[str, Any] | None = None) -> None:
        self.message = message
        self.notice = notice or {}

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[StreamEvent, None]:
        yield MessageStartEvent(message_id=str(uuid.uuid4()))
        yield notice_event({**self.notice, "message": self.message})
        for event in blocked_reply(self.message):
            yield event


async def close_unguardable(guard: TurnGuard) -> BlockedStream | None:
    """Close a turn whose adapter is not a Stream (unguardable past the gate).

    Returns the blocked reply to stream instead, unless the policy fails open.
    """
    error = "adapter cannot be guarded"
    if not guard.resolved.fails_closed:
        guard.note_error(error)
        await guard.close()
        return None
    guard.block(error)
    await guard.close()
    notice = notice_base(guard.id, Stage.FINISH, Outcome.BLOCKED)
    return BlockedStream(str(Block.DEFAULT_MESSAGE), notice)


async def run_side_task(runner: Any, messages: list[ChatMessage], **kwargs: Any) -> Any:
    """Run a side task (suggestions, citations, a title) unguarded: the turn itself is judged."""
    from django_ai_sdk.agents.base import Agent

    if isinstance(runner, Agent):
        kwargs["oversight"] = False
    else:
        kwargs["request"] = TurnRequest(oversight=False)
    return await runner.run(messages, **kwargs)


def withdrawn_citations(grounder: Grounder | None, call_id: str | None) -> list[StreamEvent]:
    if grounder is not None:
        grounder.cancel()
    if not call_id:
        return []
    empty = CitationsData().model_dump(mode="json", by_alias=True)
    return [ToolOutputEvent(tool_call_id=call_id, tool_output=empty)]


class StreamTurn:
    """The oversight side of one streamed chat turn: the run decides, this shows it.

    Order: `start()`, `pass_event` per event, then `finish` (or
    `blocked_reply_events`, or `end_early`), and always `aclose` once stored.
    """

    def __init__(
        self,
        turn: TurnRequest | None,
        message_id: str,
        writer: StreamWriter | None,
        registry: SourceRegistry | None,
        artifact_ids: list[str],
        queue: asyncio.Queue[Any],
        *,
        receipts: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.writer = writer
        # Receipt keys by tool name: what of a result a withheld answer keeps.
        self.receipts = receipts
        self.registry = registry
        # Artifacts stored during this answer, deleted if the answer is withheld.
        self.artifact_ids = artifact_ids
        self.finished = False
        # Stored on the message so a reload shows them, even after an error or disconnect.
        self.sent: list[dict[str, Any]] = []
        self.held: list[StreamEvent] = []
        # A call's arguments by its id, kept back until the tool stage has judged the call.
        self.unjudged: dict[str, ToolInputCompleteEvent] = {}
        # Their arguments are neither stored nor released.
        self.blocked_call_ids: set[str] = set()

        async def stream_blocked_results(stubs: list[Any]) -> None:
            # Blocked calls never reach the ToolInvoker, so stream their stubs here.
            for index, stub in enumerate(stubs):
                result = stub.tool_call_result
                if stub.meta.get(BLOCKED_META_KEY):
                    self.blocked_call_ids.add(result.origin.id or "")
                    origin = dataclasses.replace(result.origin, arguments={})
                    result = ToolCallResult(result=result.result, origin=origin, error=result.error)
                chunk = StreamingChunk(content="", index=index, tool_call_result=result)
                await queue.put(chunk)

        # A copy, so the adapter's TurnRequest is never shared between streams.
        self.turn = dataclasses.replace(turn or TurnRequest())
        self.turn.message_id = message_id
        self.turn.keep_open = True
        self.turn.on_tool_results = stream_blocked_results

    @property
    def guard(self) -> TurnGuard:
        return self.turn.guard or _INERT

    @property
    def hold(self) -> bool:
        """The answer streams only progress until it is checked."""
        policy = self.guard.policy
        return policy is not None and policy.holds_answer()

    def _emit_notice(self, notice: dict[str, Any]) -> DataEvent:
        self.sent.append(notice)
        return notice_event(notice)

    async def start(self) -> list[StreamEvent]:
        if not self.guard.active:
            return []
        await mask_sources(self.guard, self.registry)
        return self._emit_pending_notices()

    def _emit_pending_notices(self) -> list[StreamEvent]:
        events: list[StreamEvent] = [
            self._emit_notice(notice) for notice in self.guard.notices.pending
        ]
        self.guard.notices.pending.clear()
        return events

    @property
    def blocked_gate_result(self) -> StageResult | None:
        return self.turn.gate_result if self.turn.question_blocked else None

    def blocked_reply_events(self) -> list[StreamEvent]:
        gate_result = self.blocked_gate_result
        assert gate_result is not None
        self.finished = True
        notice = {**(gate_result.notice or {}), "message": gate_result.message}
        return [
            *self._emit_pending_notices(),
            self._emit_notice(notice),
            *blocked_reply(gate_result.message),
        ]

    def pass_event(self, event: StreamEvent) -> list[StreamEvent]:
        if not self.guard.active:
            return [event]
        pending = self._emit_pending_notices()
        if pending:
            return [*pending, *self.pass_event(event)]
        if isinstance(event, TextChunkEvent):
            self.guard.add_text(event.content)
        elif not isinstance(event, ReasoningChunkEvent):
            if self.hold:
                self.held.append(event)
                progress = _held_progress(event)
                return [progress] if progress is not None else []
            return self._once_judged(event)
        # Held text is sent after the check. Held reasoning is never shown.
        return [] if self.hold else [event]

    def _once_judged(self, event: StreamEvent) -> list[StreamEvent]:
        """Keep a call's arguments back until its result shows the tool stage let it run.

        The model streams them before that stage judges the call, and a blocked
        call's may be what the block is about. Only under a policy that may block one.
        """
        policy = self.guard.policy
        if policy is None or not policy.blocks_tool_calls():
            return [event]
        if isinstance(event, ToolInputCompleteEvent):
            self.unjudged[event.tool_call_id] = event
            return [event.model_copy(update={"tool_input": {}})]
        if isinstance(event, ToolOutputEvent):
            asked = self.unjudged.pop(event.tool_call_id, None)
            if asked is not None and asked.tool_call_id not in self.blocked_call_ids:
                return [asked, event]
        return [event]

    async def mask_tool_result(self, result: ToolCallResult) -> ToolCallResult:
        """Once, so the card, storage and model all see the same text."""
        if not self.guard.active:
            return result
        masked = await mask_tool_result(self.guard, result.origin.id or "", result.result)
        if masked is not result.result:
            result = ToolCallResult(result=masked, origin=result.origin, error=result.error)
        await mask_sources(self.guard, self.registry)
        return result

    async def finish(self) -> tuple[list[StreamEvent], StageResult]:
        """Apply the run's finish decision to the answer. Returns the events to send."""
        guard, writer = self.guard, self.writer
        if not guard.active:
            return [], StageResult(Stage.FINISH)
        finished = await guard.finish(writer.text if writer else guard.context.output)
        self.finished = True
        if writer and finished.changed:
            writer.replace_text(finished.text)
        if writer and (finished.changed or self.hold):
            # Reasoning may reveal what was masked or withheld, and held reasoning
            # was never judged, so do not store it.
            writer.drop_reasoning()
        if finished.blocked:
            # Tool details first: if the artifacts fail to delete, they are still gone.
            if writer:
                writer.drop_tool_details(self.receipts)
            await self.drop_artifacts()
        self._drop_blocked_inputs()
        events: list[StreamEvent] = self._emit_pending_notices()
        if finished.notice:
            events.append(self._emit_notice(finished.notice))
        if self.sent and writer:
            writer.message.metadata["oversight"] = stored_notices(self.sent)
        events += _released(self.held, withheld=finished.blocked, receipts=self.receipts)
        if self.hold:
            events.append(TextChunkEvent(content=finished.text))
        return events, finished

    def _drop_blocked_inputs(self) -> None:
        """The block may be about what the arguments hold."""
        if not self.blocked_call_ids:
            return
        for call in self.writer.message.tool_calls if self.writer else []:
            if call.get("id") in self.blocked_call_ids:
                call["arguments"] = {}
        self.held = [
            e.model_copy(update={"tool_input": {}})
            if isinstance(e, ToolInputCompleteEvent) and e.tool_call_id in self.blocked_call_ids
            else e
            for e in self.held
        ]

    async def drop_artifacts(self) -> None:
        """They may hold what the tools found."""
        if not self.artifact_ids:
            return
        from django_ai_sdk.artifacts.models import Artifact

        await Artifact.objects.filter(id__in=self.artifact_ids).adelete()
        self.artifact_ids.clear()

    def end_early(self) -> None:
        """Keep sent notices; clear an answer oversight did not finish checking."""
        if self.writer is None or not self.guard.active:
            return
        if self.sent:
            self.writer.message.metadata["oversight"] = stored_notices(self.sent)
        if not self.finished:
            self.writer.replace_text("")
            self.writer.drop_reasoning()
            self.writer.drop_tool_details(self.receipts)

    async def aclose(self, error: str) -> None:
        if self.turn.guard is not None:
            await self.turn.guard.close(error=error or ("" if self.finished else "disconnected"))


def _held_progress(event: StreamEvent) -> StreamEvent | None:
    """A held tool event stripped to its name and progress. None hides it."""
    if isinstance(event, ToolCallStartEvent):
        return event
    if isinstance(event, ToolInputCompleteEvent):
        return event.model_copy(update={"tool_input": {}})
    if isinstance(event, ToolOutputEvent):
        return event.model_copy(update={"tool_output": {"result": str(HELD_OUTPUT)}})
    return None


def _released(
    held: list[StreamEvent], *, withheld: bool, receipts: Mapping[str, tuple[str, ...]]
) -> list[StreamEvent]:
    if withheld:
        # A tool's receipt, else that it was withheld, in place of each result.
        names = {e.tool_call_id: e.tool_name for e in held if isinstance(e, ToolCallStartEvent)}
        released: list[StreamEvent] = []
        for event in held:
            if not isinstance(event, ToolOutputEvent):
                continue
            kept = receipt(event.tool_output, receipts.get(names.get(event.tool_call_id, "")))
            output = {"result": kept or str(WITHHELD_OUTPUT)}
            released.append(event.model_copy(update={"tool_output": output}))
        return released
    # Starts were already sent. The complete input replaces the input chunks.
    return [e for e in held if not isinstance(e, ToolCallStartEvent | ToolInputChunkEvent)]
