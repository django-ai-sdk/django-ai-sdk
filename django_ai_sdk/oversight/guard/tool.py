"""Tool stage. Judge each tool call before it runs, and enforce removed tools."""

from __future__ import annotations

from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.conf import deadline_ms
from django_ai_sdk.oversight.guard.asking import errored_lenses, evaluate_rules, run_lenses
from django_ai_sdk.oversight.guard.notices import TOOL_UNAVAILABLE, acted_on, restrict_refusal
from django_ai_sdk.oversight.guard.state import GuardState
from django_ai_sdk.oversight.types import Stage, Subject, ToolCall


class ToolStage(GuardState):
    """Adds `tool_call()`, which judges one tool call before it runs."""

    async def tool_call(self, call: ToolCall) -> str | None:
        """Judge one tool call. Returns the refusal message, or None to allow it."""
        if not self.active:
            return None
        if self._unloaded:
            return str(Block.DEFAULT_MESSAGE)
        try:
            return await self._tool_call(call)
        except Exception:
            return (
                str(Block.DEFAULT_MESSAGE)
                if self._stage_failed("tool check failed", refuse_turn=False)
                else None
            )

    async def _tool_call(self, call: ToolCall) -> str | None:
        if not self.allows_tool(call.name):
            return str(TOOL_UNAVAILABLE)
        lenses = self.resolved.active_lenses(Stage.TOOL, Subject.TOOL_CALL)
        if not lenses:
            return None
        view = self.context.model_copy(update={"tool_call": call})
        answered = await run_lenses(lenses, view, Stage.TOOL, deadline_ms=deadline_ms(Stage.TOOL))
        self._keep_answers(answered)
        # Keep this call's answers out of context.answers so the next call is judged on its own.
        answers = {o.lens.name: o.answer for o in answered}
        tool_lenses = {lens.name for lens in lenses}
        # Failures of other lenses still count. A tool lens failure counts only for this call.
        errored = {name for name in self._errored_lenses() if name not in tool_lenses}
        # Judge every call, not just the first, so a retried call is refused again.
        matched = [
            m
            for m in evaluate_rules(
                self.resolved,
                {**self.context.answers, **answers},
                Stage.TOOL,
                done=set(),
                errored=errored | errored_lenses(answered),
                fail_closed=self._fails_closed,
            )
            if m.rule.lenses() & tool_lenses
        ]
        # One recorded match per rule, however many calls it refused.
        recorded = {m.rule.name for m in self.matched if not m.aside}
        self.matched.extend(m for m in matched if m.rule.name not in recorded)
        # Removed tools stay removed. Later calls are refused and get_tools() leaves them out.
        block, self.restrict = acted_on(matched, self.restrict)
        if block is not None:
            return block
        if not self.allows_tool(call.name):
            restricting = next(
                (m for m in matched if any(a.kind == "restrict_tools" for a in m.acted)), None
            )
            return restrict_refusal(restricting) if restricting else str(TOOL_UNAVAILABLE)
        return None
