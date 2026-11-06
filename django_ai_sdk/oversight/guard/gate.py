"""Gate stage. Judge the question and its history before the model sees them."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.conf import deadline_ms, setting
from django_ai_sdk.oversight.guard.masks import Masking
from django_ai_sdk.oversight.guard.notices import (
    TOO_LONG_MESSAGE,
    UNCHECKED_TEXT,
    Notices,
    acted_on,
    notice_base,
)
from django_ai_sdk.oversight.guard.state import merge_restrict
from django_ai_sdk.oversight.patterns import mask_all
from django_ai_sdk.oversight.types import Outcome, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Callable

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.guard.asking import Matched
    from django_ai_sdk.oversight.guard.masks import Judged

# How many of the latest history items gate_history judges. Older ones are withheld
# unread, or passed as is when the policy fails open. Every gate lens judges each
# item, so this caps the cost of one turn.
MAX_HISTORY_JUDGED = 20
# How many history items are judged at once.
HISTORY_CONCURRENCY = 4


@dataclass
class GateResult:
    """What the gate stage decided about the question."""

    blocked: bool = False
    message: str = ""
    # The question as the model should see it, masked if a rule says so.
    input: str = ""
    masked: bool = False
    # Tool names taken away. None keeps every tool, [] removes all.
    restrict: list[str] | None = None
    # The `data-oversight` payload for the client, if there is anything to say.
    notice: dict[str, Any] | None = None


class GateStage(Masking, Notices):
    """Adds `gate()` and `gate_history()`, which judge and mask the question and its history."""

    async def gate(self) -> GateResult:
        """Judge the question before the model sees it.

        Always writes the turn ahead (`write_ahead`) afterwards, refused or not.
        """
        result = await self._gate_checked()
        await self.write_ahead()
        return result

    async def _gate_checked(self) -> GateResult:
        result = GateResult(input=self.context.input)
        if not self.active:
            return result
        if self._unloaded:
            return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
        limit = setting("MAX_INPUT_CHARS")
        if limit and len(self.context.input) > limit:
            # Refuse now, because later stages like reflect would only read part of it.
            self.too_long = True
            return self._refuse_gate(result, str(TOO_LONG_MESSAGE))
        try:
            return await self._gate(result)
        except Exception:
            # A broken lens or rule follows the fail mode, as in the other stages.
            if self._stage_failed("gate failed", refuse_turn=True):
                return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
            return result

    def _refuse_gate(self, result: GateResult, message: str) -> GateResult:
        result.blocked = True
        result.message = message
        result.notice = {**notice_base(self.id, Stage.GATE, Outcome.BLOCKED), "message": message}
        return result

    async def _gate(self, result: GateResult) -> GateResult:
        matched = await self._stage(
            Stage.GATE, subjects={Subject.INPUT}, deadline=deadline_ms(Stage.GATE)
        )
        self.matched.extend(matched)
        masks = self._masks(matched, Subject.INPUT)
        # Tools taken away by gate_history stay taken away.
        block, result.restrict = acted_on(matched, self.restrict)
        if block is not None:
            result.blocked = True
            result.message = block
        if masks:
            masked = mask_all(self.context.input, masks)
            result.masked = masked != self.context.input
            self.context.input = masked
        result.input = self.context.input
        self.restrict = result.restrict
        result.notice = self._notice(Stage.GATE, matched, result.blocked, result.message)
        return result

    async def gate_history(
        self, messages: list[ChatMessage], *, history: bool = True
    ) -> GateResult:
        """Judge earlier messages and attachment context that come with the question.

        Call it before `gate()` for history the server did not store itself, such
        as a stateless run, a chat without a thread, or a workflow's `history`
        input. Each earlier user or assistant message and each attachment context
        (file names, summaries, captions, including the question's own) is judged
        separately by the gate's input lenses and masked in place. Text a mask
        cannot point into is withheld whole. With `history=False` only the last
        message's attachments are judged, because a stored thread was already
        judged turn by turn.

        Only the latest MAX_HISTORY_JUDGED items are judged, HISTORY_CONCURRENCY
        at a time. Older ones are withheld unless the policy fails open. The
        question and the judged items count together against MAX_INPUT_CHARS.

        The result is blocked when an enforced rule that blocks, or removes every
        tool, matches any item. `gate()` runs such a question without tools
        instead, but history is resent every turn, so the match would quietly
        leave every later turn without tools. A failed lens counts as a match unless the policy
        fails open. Removed tools stay removed for the turn. Findings are recorded
        aside, once per lens and rule, and never steer masks or rules on the
        question itself. Rules that only alert, queue or notify are skipped,
        because clients resend history on every turn.
        """
        result = GateResult(input=self.context.input)
        if not self.active:
            return result
        if self._unloaded:
            return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
        lenses = self.resolved.active_lenses(Stage.GATE, Subject.INPUT)
        items = _history_items(messages, history=history) if lenses else []
        if not items:
            self.set_history(messages)
            return result
        unread, items = items[:-MAX_HISTORY_JUDGED], items[-MAX_HISTORY_JUDGED:]
        limit = setting("MAX_INPUT_CHARS")
        if limit and len(self.context.input) + sum(len(get()) for get, _ in items) > limit:
            self.too_long = True
            return self._refuse_gate(result, str(TOO_LONG_MESSAGE))
        if unread and self._fails_closed:
            for _get, put in unread:
                put(str(UNCHECKED_TEXT))
        elif unread:
            self.note_error("history passed unchecked")
        slots = asyncio.Semaphore(HISTORY_CONCURRENCY)

        async def judge(text: str) -> Judged:
            async with slots:
                return await self._judged(lenses, text, Stage.GATE, blocks=True)

        try:
            judged = await asyncio.gather(*(judge(get()) for get, _ in items))
        except Exception:
            if self._stage_failed("history check failed", refuse_turn=True):
                return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
            return result
        found: list[Matched] = []
        for (_read, put), one in zip(items, judged, strict=True):
            put(one.text)
            found.extend(one.matched)
            if one.block is not None and not result.blocked:
                result.blocked = True
                result.message = one.block
            if one.restrict is not None:
                self.restrict = merge_restrict(self.restrict, one.restrict)
        recorded = self._record_aside(
            Stage.GATE, [o for one in judged for o in one.answered], found
        )
        self.set_history(messages)
        if result.blocked:
            result.restrict = self.restrict
            result.notice = self._notice(Stage.GATE, recorded, True, result.message)
            # gate() will not run after a refusal, so write the turn ahead here.
            await self.write_ahead()
        return result


def _history_items(
    messages: list[ChatMessage], *, history: bool
) -> list[tuple[Callable[[], str], Callable[[str], None]]]:
    """Return a (read, write back) pair for each text gate_history judges.

    Earlier user and assistant messages (with `history`), plus the context of
    each attachment, including the question's own.
    """
    items: list[tuple[Callable[[], str], Callable[[str], None]]] = []
    for index, message in enumerate(messages):
        earlier = index < len(messages) - 1
        if history and earlier and message.role in {"user", "assistant"} and message.content:
            items.append(
                (
                    lambda m=message: m.content,
                    lambda text, m=message: setattr(m, "content", text),
                )
            )
        if not history and earlier:
            continue
        for attachment in message.attachments:
            if attachment.context:
                items.append(
                    (
                        lambda a=attachment: a.context,
                        lambda text, a=attachment: setattr(a, "context", text),
                    )
                )
    return items
