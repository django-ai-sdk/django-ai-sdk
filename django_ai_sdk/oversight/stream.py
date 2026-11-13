"""Oversight in a streamed chat turn, and the stream events for its interventions.

Every intervention reaches the client as a `data-oversight` part. A UI can then
show that a question was masked, a tool refused, or an answer withheld or replaced.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy
from haystack.dataclasses import StreamingChunk, ToolCallResult

from django_ai_sdk.agents.tool_agent import REFUSED_META_KEY
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
from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import FinishResult, TurnGuard
from django_ai_sdk.oversight.guard.notices import notice_base
from django_ai_sdk.oversight.plugin import (
    REQUEST_KEY,
    TurnRequest,
    close_turn,
    mask_sources,
    masked_result,
    request_of,
    turn_of,
)
from django_ai_sdk.oversight.types import Outcome, Stage

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.adapters.citations.grounding import Grounder
    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.common import ChatMessage, StreamWriter
    from django_ai_sdk.events import StreamEvent
    from django_ai_sdk.oversight.guard.gate import GateResult

# The guard a stream reads before the run opened its turn: allows everything.
_INERT = TurnGuard()

# The finish reason sent when oversight withholds an answer.
CONTENT_FILTER = "content-filter"

# While an answer is held, tool cards show that a tool ran but not its input or
# output: just this line. After the check they get the real data, or WITHHELD_OUTPUT.
HELD_OUTPUT = gettext_lazy("Checking the answer first. The result shows here once it passes.")
WITHHELD_OUTPUT = gettext_lazy("Not shown: oversight withheld this answer.")


def notice_event(notice: dict[str, Any]) -> DataEvent:
    return DataEvent(data_type="oversight", data=notice)


def stored_notices(notices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn notices into the form stored on the message. "replace" becomes "replaced": True.

    The stored content already holds the replacement. A second copy in the
    metadata would go stale when reflect masks the message later.
    """
    return [
        {**{k: v for k, v in n.items() if k != "replace"}, "replaced": True}
        if "replace" in n
        else n
        for n in notices
    ]


class RefusalStream:
    """The full reply when the gate refuses a question. Streams the policy's message and stores nothing."""

    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None

    def __init__(self, message: str, notice: dict[str, Any] | None = None) -> None:
        self.message = message
        self.notice = notice or {}

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[StreamEvent, None]:
        yield MessageStartEvent(message_id=str(uuid.uuid4()))
        yield notice_event({**self.notice, "message": self.message})
        yield TextChunkEvent(content=self.message)
        yield TextEndEvent()
        yield MessageEndEvent(finish_reason=CONTENT_FILTER)
        yield StreamEndEvent()


async def close_unguardable(guard: TurnGuard) -> RefusalStream | None:
    """Close a chat turn whose adapter is not a Stream, so it cannot be guarded past the gate.

    Returns the refusal to stream instead, unless the policy fails open.
    """
    error = "adapter cannot be guarded"
    if not guard.resolved.fails_closed:
        guard.note_error(error)
        await guard.close()
        return None
    guard.refuse(error)
    await guard.close()
    notice = notice_base(guard.id, Stage.FINISH, Outcome.BLOCKED)
    return RefusalStream(str(Block.DEFAULT_MESSAGE), notice)


async def run_side_task(runner: Any, messages: list[ChatMessage], **kwargs: Any) -> Any:
    """Run a side task of a turn, such as suggestions, citations or a title, on an Agent or a Run."""
    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.oversight.plugin import TurnRequest

    # Oversight: off on purpose. The turn itself is judged, not its side tasks.
    if isinstance(runner, Agent):
        kwargs["oversight"] = False
    else:
        kwargs["request"] = TurnRequest(oversight=False)
    return await runner.run(messages, **kwargs)


def withdrawn_citations(grounder: Grounder | None, call_id: str | None) -> list[StreamEvent]:
    """Stop grounding a withheld answer and clear the citations card shown while it streamed."""
    from django_ai_sdk.artifacts.schemas import CitationsData

    if grounder is not None:
        grounder.cancel()
    if not call_id:
        return []
    empty = CitationsData().model_dump(mode="json", by_alias=True)
    return [ToolOutputEvent(tool_call_id=call_id, tool_output=empty)]


class StreamTurn:
    """The oversight side of one streamed chat turn.

    The run's hooks open and decide the turn (see `oversight.plugin`); this shows
    what they decided. The Stream runs the agent with `context`, which holds the
    run's TurnRequest, as its hook_context, sends the `start()` events, passes every event through
    `pass_event`, then calls `afinish`, or `refusal_events()` when the gate
    refused, or `end_early` when the turn ends before that, and always `aclose`,
    which closes the turn once the answer is stored. Until the run opened a turn,
    and without a policy, every call passes the stream through unchanged.
    """

    def __init__(
        self,
        context: dict[str, Any],
        message_id: str,
        writer: StreamWriter | None,
        registry: SourceRegistry | None,
        artifact_ids: list[str],
        queue: asyncio.Queue[Any],
    ) -> None:
        # The run's hook_context. The stream closes the turn itself, after storing.
        self.context = context
        self.writer = writer
        self.registry = registry
        # Artifacts stored during this answer, deleted if the answer is withheld.
        self.artifact_ids = artifact_ids
        self.finished = False
        # Notices sent this turn. Stored on the message so a reload shows them,
        # even after an error or a disconnect.
        self.sent: list[dict[str, Any]] = []
        # Tool events of a held answer, sent after the check.
        self.held: list[StreamEvent] = []
        # Ids of refused tool calls, whose arguments are neither stored nor released.
        self.refused: set[str] = set()

        async def refused_results(stubs: list[Any]) -> None:
            # Refused calls never reach the ToolInvoker, so stream their stubs here.
            for index, stub in enumerate(stubs):
                result = stub.tool_call_result
                if stub.meta.get(REFUSED_META_KEY):
                    self.refused.add(result.origin.id or "")
                    origin = dataclasses.replace(result.origin, arguments={})
                    result = ToolCallResult(result=result.result, origin=origin, error=result.error)
                chunk = StreamingChunk(content="", index=index, tool_call_result=result)
                await queue.put(chunk)

        # A copy, so the adapter's request is never shared between streams.
        request = dataclasses.replace(request_of(context) or TurnRequest())
        request.message_id = message_id
        request.keep_open = True
        request.on_tool_results = refused_results
        context[REQUEST_KEY] = request

    @property
    def guard(self) -> TurnGuard:
        """The run's own turn, or an inert guard until it opened one."""
        turn = turn_of(self.context)
        return turn.guard if turn is not None and turn.owns else _INERT

    @property
    def hold(self) -> bool:
        """Whether the answer streams only progress until it is checked."""
        policy = self.guard.policy
        return policy is not None and policy.holds_answer()

    def _send(self, notice: dict[str, Any]) -> DataEvent:
        self.sent.append(notice)
        return notice_event(notice)

    async def start(self) -> list[StreamEvent]:
        """Return the notices raised before the answer started, and mask the seeded sources."""
        if not self.guard.active:
            return []
        await self.mask_sources()
        return self._pending()

    def _pending(self) -> list[StreamEvent]:
        """Send the notices raised since the last event, such as the gate's."""
        events: list[StreamEvent] = [self._send(notice) for notice in self.guard.notices]
        self.guard.notices.clear()
        return events

    @property
    def refusal(self) -> GateResult | None:
        """The gate's refusal, when the run's gate hook refused the question."""
        turn = turn_of(self.context)
        return turn.gated if turn is not None and turn.refused else None

    def refusal_events(self) -> list[StreamEvent]:
        """The rest of the reply when the gate refused: the policy's message, stored nowhere."""
        refusal = self.refusal
        assert refusal is not None
        self.finished = True
        notice = {**(refusal.notice or {}), "message": refusal.message}
        return [
            *self._pending(),
            self._send(notice),
            TextChunkEvent(content=refusal.message),
            TextEndEvent(),
            MessageEndEvent(finish_reason=CONTENT_FILTER),
            StreamEndEvent(),
        ]

    def pass_event(self, event: StreamEvent) -> list[StreamEvent]:
        """Report `event` to the guard and return what to send now."""
        if not self.guard.active:
            return [event]
        pending = self._pending()
        if pending:
            return [*pending, *self.pass_event(event)]
        if isinstance(event, TextChunkEvent):
            self.guard.add_text(event.content)
        elif not isinstance(event, ReasoningChunkEvent):
            if self.hold:
                self.held.append(event)
                progress = _held_progress(event)
                return [progress] if progress is not None else []
        # Held text is sent after the check. Held reasoning is never shown.
        return [] if self.hold else [event]

    async def mask_tool_result(self, result: ToolCallResult) -> ToolCallResult:
        """Mask a tool result once, so the card, storage and model all see the same text."""
        if not self.guard.active:
            return result
        masked = await masked_result(self.guard, result.origin.id or "", result.result)
        if masked is not result.result:
            result = ToolCallResult(result=masked, origin=result.origin, error=result.error)
        # Mask new citation sources before any card shows them.
        await self.mask_sources()
        return result

    async def mask_sources(self) -> None:
        """Mask the citation sources before any card shows them."""
        if self.registry is not None:
            await mask_sources(self.guard, self.registry)

    async def afinish(self) -> tuple[list[StreamEvent], FinishResult]:
        """Pass, mask or withhold the answer as the run's finish decided. Returns the events to send."""
        guard, writer = self.guard, self.writer
        if not guard.active:
            return [], FinishResult()
        # The after_run hook finished the turn; this reads that decision.
        finished = await guard.finish(writer.text if writer else guard.context.output)
        self.finished = True
        if writer and finished.replaced:
            writer.replace_text(finished.output)
        if writer and (finished.replaced or self.hold):
            # Reasoning may reveal what was masked or withheld, and held reasoning
            # was never judged, so do not store it.
            writer.drop_reasoning()
        if finished.blocked:
            # Tool details first: if the artifacts fail to delete, they are still gone.
            if writer:
                writer.drop_tool_details()
            await self._drop_artifacts()
        self._drop_refused_inputs()
        events: list[StreamEvent] = self._pending()
        if finished.notice:
            events.append(self._send(finished.notice))
        if self.sent and writer:
            writer.message.metadata["oversight"] = stored_notices(self.sent)
        events += _released(self.held, withheld=finished.blocked)
        if self.hold:
            events.append(TextChunkEvent(content=finished.output))
        return events, finished

    def _drop_refused_inputs(self) -> None:
        """Clear the arguments of refused calls: a refusal may be about what they hold."""
        if not self.refused:
            return
        for call in self.writer.message.tool_calls if self.writer else []:
            if call.get("id") in self.refused:
                call["arguments"] = {}
        self.held = [
            e.model_copy(update={"tool_input": {}})
            if isinstance(e, ToolInputCompleteEvent) and e.tool_call_id in self.refused
            else e
            for e in self.held
        ]

    async def _drop_artifacts(self) -> None:
        """Delete this answer's artifacts. They may hold what the tools found."""
        if not self.artifact_ids:
            return
        from django_ai_sdk.artifacts.models import Artifact

        await Artifact.objects.filter(id__in=self.artifact_ids).adelete()
        self.artifact_ids.clear()

    def end_early(self) -> None:
        """Prepare a turn that ended early for storage.

        Keeps the notices already sent, so a reload shows them. Clears an answer
        that oversight did not finish checking.
        """
        if self.writer is None or not self.guard.active:
            return
        if self.sent:
            self.writer.message.metadata["oversight"] = stored_notices(self.sent)
        if not self.finished:
            self.writer.replace_text("")
            self.writer.drop_reasoning()
            self.writer.drop_tool_details()

    async def aclose(self, error: str) -> None:
        """Close the turn, saying why it ended early, if it did."""
        await close_turn(self.context, error=error or ("" if self.finished else "disconnected"))


def _held_progress(event: StreamEvent) -> StreamEvent | None:
    """Strip a held tool event down to its name and progress. None hides it."""
    if isinstance(event, ToolCallStartEvent):
        return event
    if isinstance(event, ToolInputCompleteEvent):
        return event.model_copy(update={"tool_input": {}})
    if isinstance(event, ToolOutputEvent):
        return event.model_copy(update={"tool_output": {"result": str(HELD_OUTPUT)}})
    return None


def _released(held: list[StreamEvent], *, withheld: bool) -> list[StreamEvent]:
    """Return the held tool events to send after the check, or WITHHELD outputs."""
    if withheld:
        output = {"result": str(WITHHELD_OUTPUT)}
        return [
            event.model_copy(update={"tool_output": output})
            for event in held
            if isinstance(event, ToolOutputEvent)
        ]
    # Starts were already sent. The complete input replaces the input chunks.
    return [e for e in held if not isinstance(e, ToolCallStartEvent | ToolInputChunkEvent)]


__all__ = [
    "CONTENT_FILTER",
    "RefusalStream",
    "StreamTurn",
    "close_unguardable",
    "run_side_task",
    "withdrawn_citations",
]
