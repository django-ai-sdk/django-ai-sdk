"""Dry runs: run a policy over one exchange as a turn would, without recording it.

    from django_ai_sdk.oversight import dry_run

    result = await dry_run("support", input="Where is my refund?", output="It is on its way.")
    assert result.outcome == "allowed"
    assert result.blocked_stage == ""
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

from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.guard.stages import StageResult
from django_ai_sdk.oversight.policy import ResolvedPolicy
from django_ai_sdk.oversight.reflection import judge_reflect_stage
from django_ai_sdk.oversight.registry import broken_reason, get_resolved_policy
from django_ai_sdk.oversight.types import Outcome, Stage, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.oversight.guard.judging import Matched


@dataclass
class DryRunResult:
    """The result of a dry run."""

    policy: str
    stages: list[StageResult] = field(default_factory=list)
    # "gate" or "finish" when that stage blocked, else "". A refused tool call does
    # not end the turn. Only its own stage result is blocked.
    blocked_stage: str = ""
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
    policy: str | ResolvedPolicy,
    *,
    input: str,
    output: str = "",
    tool_calls: Sequence[ToolCall | dict[str, Any]] | None = None,
    user: AbstractBaseUser | AnonymousUser | None = None,
) -> DryRunResult:
    """Run `policy` over one exchange as a turn would, without recording it.

    `policy` is a registered policy name or a `ResolvedPolicy` from `policy.build`,
    so a draft can run before it is registered. `tool_calls` ({"name", "arguments"}
    or ToolCall) are checked in order before the answer. Raises LookupError when a
    named policy does not load.
    """
    if isinstance(policy, ResolvedPolicy):
        resolved = policy
    else:
        resolved = await get_resolved_policy(policy)
        if resolved is None:
            raise LookupError(broken_reason(policy) or f"No oversight policy {policy!r}")
    # Opened directly, so a dry run checks even when oversight is switched off.
    guard = TurnGuard.under_policy(
        resolved,
        agent=SimpleNamespace(name="Dry run", oversight=resolved.name),
        channel="dry_run",
        user=user,
        thread_id=None,
        messages=None,
        input=input,
    )
    guard.recording = False
    return await _run_stages(guard, resolved, output=output, tool_calls=tool_calls or [])


async def _run_stages(
    guard: TurnGuard,
    resolved: ResolvedPolicy,
    *,
    output: str,
    tool_calls: Sequence[ToolCall | dict[str, Any]],
) -> DryRunResult:
    result = DryRunResult(policy=resolved.name)

    def add_stage(stage: Stage, before: int, matched_before: int, **extra: Any) -> None:
        result.stages.append(
            StageResult(
                stage=stage,
                answered=guard.answered[before:],
                matched=guard.matched[matched_before:],
                **extra,
            )
        )

    gate = await guard.gate()
    add_stage(Stage.GATE, 0, 0, blocked=gate.blocked, message=gate.message, text=gate.text)
    if gate.blocked:
        result.blocked_stage = "gate"

    if not result.blocked_stage:
        for raw in tool_calls:
            call = raw if isinstance(raw, ToolCall) else ToolCall.model_validate(raw)
            answered, matched = len(guard.answered), len(guard.matched)
            refused = await guard.tool_call(call)
            guard.context.tool_calls.append(call)
            add_stage(
                Stage.TOOL,
                answered,
                matched,
                blocked=bool(refused),
                message=refused or "",
                tool=call.name,
            )

    if not result.blocked_stage and output:
        answered, matched = len(guard.answered), len(guard.matched)
        finished = await guard.finish(output)
        add_stage(
            Stage.FINISH,
            answered,
            matched,
            blocked=finished.blocked,
            message=finished.message,
            text=finished.text,
        )
        if finished.blocked:
            result.blocked_stage = "finish"

    # Reflect runs on every closed turn, a blocked one too.
    answered, matched = len(guard.answered), len(guard.matched)
    await _run_reflect_stage(guard, resolved)
    add_stage(Stage.REFLECT, answered, matched)

    result.outcome = guard.outcome
    result.report_outcome = guard.report_outcome
    return result


async def _run_reflect_stage(guard: TurnGuard, resolved: ResolvedPolicy) -> None:
    """Run what reflect would on the open guard: its lenses, and inline lenses to ask again."""
    retry = {item.lens.name for item in guard.answered if item.needs_retry}
    answered, matched, _masks = await judge_reflect_stage(
        resolved,
        guard.context,
        retry,
        {m.rule.name for m in guard.matched},
        outcome=guard.outcome,
        sample=False,
    )
    guard.answered.extend(answered)
    guard.matched.extend(matched)
