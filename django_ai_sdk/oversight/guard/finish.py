"""Finish stage. Judge the whole answer before it is stored and shown."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.conf import deadline_ms
from django_ai_sdk.oversight.guard.masks import Masking
from django_ai_sdk.oversight.guard.notices import Notices, acted_on
from django_ai_sdk.oversight.patterns import mask_all
from django_ai_sdk.oversight.types import Stage, Subject


@dataclass
class FinishResult:
    """What the finish stage decided about the answer."""

    blocked: bool = False
    message: str = ""
    # The answer that may be stored and shown.
    output: str = ""
    replaced: bool = False
    notice: dict[str, Any] | None = None


class FinishStage(Masking, Notices):
    """Adds `finish()`, which blocks, masks or passes the complete answer."""

    async def finish(self, output: str | None = None) -> FinishResult:
        """Judge the finished answer before it is stored. Later calls return the same result.

        The run's after_run hook usually finishes the turn. The code around the run
        then reads that decision instead of judging again.
        """
        if self.finished is not None:
            return self.finished
        result = await self._finish(output)
        if self.active:
            self.finished = result
        return result

    async def _finish(self, output: str | None) -> FinishResult:
        if output is not None:
            self.context.output = output
        self.original_output = self.original_output or self.context.output
        result = FinishResult(output=self.context.output)
        if not self.active:
            return result
        if self._unloaded:
            return self._refuse_output(result)
        try:
            matched = await self._stage(
                Stage.FINISH,
                subjects={Subject.OUTPUT, Subject.EXCHANGE, Subject.SOURCES},
                deadline=deadline_ms(Stage.FINISH),
            )
            self.matched.extend(matched)
            masks = self._masks(matched, Subject.OUTPUT)
            block, _ = acted_on(matched)
            if block is not None:
                result.blocked = result.replaced = True
                result.message = result.output = block
            elif masks:
                masked = mask_all(result.output, masks)
                result.replaced = masked != result.output
                result.output = masked
            if result.replaced:
                self.context.output = result.output
            result.notice = self._notice(
                Stage.FINISH, matched, result.blocked, result.message, result
            )
            return result
        except Exception:
            # Keep the answer, unless the policy fails closed.
            if self._stage_failed("finish failed", refuse_turn=True):
                return self._refuse_output(result)
            return result

    def _refuse_output(self, result: FinishResult) -> FinishResult:
        result.blocked = result.replaced = True
        result.message = result.output = self.context.output = str(Block.DEFAULT_MESSAGE)
        return result
