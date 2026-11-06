"""Oversight in a streamed chat turn, and the stream events for its interventions.

Every intervention reaches the client as a `data-oversight` part. A UI can then
show that a question was masked, a tool refused, or an answer withheld or replaced.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy
from haystack.dataclasses import StreamingChunk, ToolCallResult

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
from django_ai_sdk.oversight.hooks import masked_result
from django_ai_sdk.oversight.types import ToolCall
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.common import ChatMessage, StreamWriter
    from django_ai_sdk.events import StreamEvent
    from django_ai_sdk.oversight.guard import FinishResult, TurnGuard

# The finish reason sent when oversight withholds an answer.
CONTENT_FILTER = "content-filter"

# While an answer is held, tool cards show that a tool ran but not its input or
# output. After the check they get the real data, or WITHHELD_OUTPUT.
HELD_OUTPUT = gettext_lazy("Shown once the answer is checked.")
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


def _observe_event(guard: TurnGuard, event: StreamEvent) -> None:
    """Report a streamed tool call or tool result to the guard."""
    if isinstance(event, ToolInputCompleteEvent):
        guard.add_tool_call(
            ToolCall(
                id=event.tool_call_id,
                name=event.tool_name,
                arguments=event.tool_input,
                agent=event.agent or "",
            )
        )
    elif isinstance(event, ToolOutputEvent):
        output = event.tool_output
        failed = isinstance(output, dict) and bool(output.get("error"))
        guard.add_tool_call(ToolCall(id=event.tool_call_id, name="", result=output, error=failed))


def report_usage(guard: TurnGuard, replies: list[Any]) -> None:
    """Add each reply's token usage to the turn."""
    for message in replies:
        meta = getattr(message, "meta", None) or {}
        prompt, completion = token_counts(meta.get("usage"))
        if prompt is not None or completion is not None:
            guard.add_usage(meta.get("model"), prompt, completion)


def observe_replies(guard: TurnGuard, replies: list[Any]) -> None:
    """Report a non-streaming run's usage and sources to the guard.

    The tool hooks already reported its tool calls and results.
    """
    from django_ai_sdk.adapters.citations.registry import current_registry

    report_usage(guard, replies)
    registry = current_registry.get()
    if registry is not None:
        guard.set_sources(registry.all, seeded=registry.seeded)


def restrict_agent_tools(agent_component: Any, guard: TurnGuard) -> None:
    """Remove the tools `guard` took away from a built Haystack agent."""
    if agent_component is None or guard.restrict is None:
        return
    tools = list(getattr(agent_component, "tools", None) or [])
    agent_component.tools = [t for t in tools if guard.allows_tool(getattr(t, "name", ""))]


class StreamTurn:
    """The oversight side of one streamed chat turn.

    The Stream sends the `start()` events, passes every event through
    `pass_event`, then calls `afinish` with the pipeline's replies, or
    `end_early` when the turn ends before that, and always `aclose`.
    """

    def __init__(
        self,
        guard: TurnGuard,
        message_id: str,
        writer: StreamWriter | None,
        registry: SourceRegistry | None,
        artifact_ids: list[str],
        queue: asyncio.Queue[Any],
    ) -> None:
        self.guard = guard
        self.writer = writer
        self.registry = registry
        # Artifacts stored during this answer, deleted if the answer is withheld.
        self.artifact_ids = artifact_ids
        # A held answer streams only progress until it is checked.
        self.hold = guard.policy is not None and guard.policy.holds_answer()
        self.finished = False
        # Notices sent this turn. Stored on the message so a reload shows them,
        # even after an error or a disconnect.
        self.sent: list[dict[str, Any]] = []
        # Tool events of a held answer, sent after the check.
        self.held: list[StreamEvent] = []
        # Masked source text by key, so each text is masked only once.
        self._masked_sources: dict[str, str] = {}
        guard.message_id = message_id

        async def refused_results(stubs: list[Any]) -> None:
            # Refused calls never reach the ToolInvoker, so stream their stubs here.
            for index, stub in enumerate(stubs):
                chunk = StreamingChunk(
                    content="", index=index, tool_call_result=stub.tool_call_result
                )
                await queue.put(chunk)

        guard.on_tool_results = refused_results

    def _send(self, notice: dict[str, Any]) -> DataEvent:
        self.sent.append(notice)
        return notice_event(notice)

    async def start(self) -> list[StreamEvent]:
        """Return the notices raised before the answer started, and mask the seeded sources."""
        events: list[StreamEvent] = [self._send(notice) for notice in self.guard.notices]
        self.guard.notices.clear()
        await self.mask_sources()
        return events

    def pass_event(self, event: StreamEvent) -> list[StreamEvent]:
        """Report `event` to the guard and return what to send now."""
        if isinstance(event, TextChunkEvent):
            self.guard.add_text(event.content)
        elif not isinstance(event, ReasoningChunkEvent):
            _observe_event(self.guard, event)
            if self.hold:
                self.held.append(event)
                progress = _held_progress(event)
                return [progress] if progress is not None else []
        # Held text is sent after the check. Held reasoning is never shown.
        return [] if self.hold else [event]

    async def mask_tool_result(self, result: ToolCallResult) -> ToolCallResult:
        """Mask a tool result once, so the card, storage and model all see the same text."""
        masked = await masked_result(self.guard, result.origin.id or "", result.result)
        if masked is not result.result:
            result = ToolCallResult(result=masked, origin=result.origin, error=result.error)
        # Mask new citation sources before any card shows them.
        await self.mask_sources()
        return result

    async def mask_sources(self) -> None:
        """Mask source text in the citation registry, like tool results.

        The citations card, the stored citations artifact and the lenses all read
        the registry, so none of them should see unmasked text.
        """
        for source in self.registry.all if self.registry is not None else []:
            text = source.content or ""
            if not text or self._masked_sources.get(source.key) == text:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()[:16]
            masked = await self.guard.mask_result(f"source:{source.key}:{digest}", text)
            source.content = masked
            self._masked_sources[source.key] = masked

    async def afinish(
        self, replies: list[Any], model: str | None
    ) -> tuple[list[StreamEvent], FinishResult]:
        """Check the whole answer, then pass, mask or withhold it. Returns the events to send."""
        guard, writer = self.guard, self.writer
        report_usage(guard, replies)
        # The model the stream ran, when the replies do not name one.
        guard.add_usage(model, None, None)
        if self.registry is not None:
            await self.mask_sources()
            guard.set_sources(self.registry.all, seeded=self.registry.seeded)
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
        events: list[StreamEvent] = [self._send(finished.notice)] if finished.notice else []
        if self.sent and writer:
            writer.message.metadata["oversight"] = stored_notices(self.sent)
        events += _released(self.held, withheld=finished.blocked)
        if self.hold:
            events.append(TextChunkEvent(content=finished.output))
        return events, finished

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
        if self.writer is None:
            return
        if self.sent:
            self.writer.message.metadata["oversight"] = stored_notices(self.sent)
        if not self.finished:
            self.writer.replace_text("")
            self.writer.drop_reasoning()
            self.writer.drop_tool_details()

    async def aclose(self, error: str) -> None:
        """Close the turn, saying why it ended early, if it did."""
        await self.guard.close(error=error or ("" if self.finished else "disconnected"))


def _held_progress(event: StreamEvent) -> StreamEvent | None:
    """Strip a held tool event down to its name and progress. None hides it."""
    if isinstance(event, ToolCallStartEvent):
        return event
    if isinstance(event, ToolInputCompleteEvent):
        return event.model_copy(update={"tool_input": {}})
    if isinstance(event, ToolOutputEvent):
        return event.model_copy(update={"tool_output": {"result": {"held": str(HELD_OUTPUT)}}})
    return None


def _released(held: list[StreamEvent], *, withheld: bool) -> list[StreamEvent]:
    """Return the held tool events to send after the check, or WITHHELD outputs."""
    if withheld:
        output = {"result": {"withheld": str(WITHHELD_OUTPUT)}}
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
    "observe_replies",
    "report_usage",
    "restrict_agent_tools",
]
