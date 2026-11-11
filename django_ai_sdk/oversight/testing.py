"""Test helpers. Run a policy over one exchange as a turn would, without recording it.

    from django_ai_sdk.oversight.testing import dry_run

    result = await dry_run("support", input="Where is my refund?", output="It is on its way.")
    assert result.outcome == "allowed"
    assert result.ended_at == ""
    result.stage("gate").answers["prompt_injection"].value  # what the gate's lens said

A dry run opens a real TurnGuard and runs the gate, each tool call and finish.
It then runs reflect right away as the worker would, ignoring sample rates. The
guard is never closed, so no turn is stored.

Judges are real, so a model judge makes real calls. In tests, use a CheckLens, or
give a judge lens an `LLMJudge(llm=FakeLLM(...))` from `django_ai_sdk.testing`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.guard.asking import Answered, Matched, should_retry
from django_ai_sdk.oversight.types import Answer, Outcome, Stage, ToolCall

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.policy import ResolvedPolicy


@dataclass
class StageResult:
    """The answers and rule matches from one stage of a dry run."""

    stage: Stage
    answered: list[Answered] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    blocked: bool = False
    message: str = ""
    # The text after this stage: the question at the gate, the answer at finish.
    text: str = ""
    # The tool call's name, for a tool stage.
    tool: str = ""

    @property
    def answers(self) -> dict[str, Answer]:
        """The latest answer per lens at this stage."""
        return {o.lens.name: o.answer for o in self.answered}

    @property
    def rules(self) -> list[str]:
        """Rules that matched at this stage, whether they acted or not."""
        return [m.rule.name for m in self.matched]


@dataclass
class TurnResult:
    """The result of a dry run."""

    policy: str
    stages: list[StageResult] = field(default_factory=list)
    # "gate" or "finish" when that stage refused, else "". A refused tool call does
    # not end the turn. Only its own stage result is blocked.
    ended_at: str = ""
    outcome: Outcome = Outcome.ALLOWED
    # What the turn would have become with every report-mode rule enforced.
    report_outcome: Outcome = Outcome.ALLOWED

    def stage(self, name: Stage | str) -> StageResult:
        """The last result for stage `name`. KeyError if the run never reached it."""
        for result in reversed(self.stages):
            if result.stage == name:
                return result
        raise KeyError(name)

    @property
    def matched(self) -> list[Matched]:
        return [m for s in self.stages for m in s.matched]


async def dry_run(
    policy: str,
    *,
    input: str,
    output: str = "",
    tool_calls: list[ToolCall | dict[str, Any]] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> TurnResult:
    """Run `policy` over one exchange as a turn would, without recording it.

    `tool_calls` ({"name", "arguments"} or ToolCall) are checked as the model's
    calls, in order, before the answer. LookupError when the policy does not load.
    """
    from django_ai_sdk.oversight.guard import TurnGuard
    from django_ai_sdk.oversight.resolve import aget_policy, broken_reason

    resolved = await aget_policy(policy)
    if resolved is None:
        raise LookupError(broken_reason(policy) or f"No oversight policy {policy!r}")
    # Opened directly, so a dry run checks even when oversight is switched off.
    guard = TurnGuard._opened(
        resolved,
        agent=SimpleNamespace(name="Dry run", oversight=resolved.name),
        channel="dry_run",
        user=user,
        thread_id=None,
        messages=None,
        input=input,
    )
    guard.recording = False
    return await _run(guard, resolved, output=output, tool_calls=tool_calls or [])


async def _run(
    guard: TurnGuard,
    resolved: ResolvedPolicy,
    *,
    output: str,
    tool_calls: list[ToolCall | dict[str, Any]],
) -> TurnResult:
    result = TurnResult(policy=resolved.name)

    def stage_out(stage: Stage, before: int, matched_before: int, **extra: Any) -> None:
        result.stages.append(
            StageResult(
                stage=stage,
                answered=guard.answered[before:],
                matched=guard.matched[matched_before:],
                **extra,
            )
        )

    gate = await guard.gate()
    stage_out(Stage.GATE, 0, 0, blocked=gate.blocked, message=gate.message, text=gate.input)
    if gate.blocked:
        result.ended_at = "gate"

    if not result.ended_at:
        for raw in tool_calls:
            call = raw if isinstance(raw, ToolCall) else ToolCall.model_validate(raw)
            answered, matched = len(guard.answered), len(guard.matched)
            refused = await guard.tool_call(call)
            guard.context.tool_calls.append(call)
            stage_out(
                Stage.TOOL,
                answered,
                matched,
                blocked=bool(refused),
                message=refused or "",
                tool=call.name,
            )

    if not result.ended_at and output:
        answered, matched = len(guard.answered), len(guard.matched)
        finished = await guard.finish(output)
        stage_out(
            Stage.FINISH,
            answered,
            matched,
            blocked=finished.blocked,
            message=finished.message,
            text=finished.output,
        )
        if finished.blocked:
            result.ended_at = "finish"

    # Reflect runs on every closed turn, a blocked one too.
    answered, matched = len(guard.answered), len(guard.matched)
    await _reflect(guard, resolved)
    stage_out(Stage.REFLECT, answered, matched)

    result.outcome = guard.outcome
    result.report_outcome = guard.report_outcome
    return result


async def _reflect(guard: TurnGuard, resolved: ResolvedPolicy) -> None:
    """Run what reflect would on the open guard: its lenses, and inline lenses to ask again."""
    from django_ai_sdk.oversight.reflect import ajudge_turn

    retry = {
        item.lens.name
        for item in guard.answered
        if should_retry(item.answer.error, [step.error for step in item.answer.trail])
    }
    answered, matched = await ajudge_turn(
        resolved,
        guard.context,
        retry,
        {m.rule.name for m in guard.matched},
        outcome=guard.outcome,
        sample=False,
    )
    guard.answered.extend(answered)
    guard.matched.extend(matched)


__all__ = ["StageResult", "TurnResult", "dry_run"]
