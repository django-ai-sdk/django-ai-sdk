"""Dry runs: run a policy over one exchange as a turn would, without recording it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    policy: str
    stages: list[StageResult] = field(default_factory=list)
    # "gate" or "finish" when that stage blocked. A blocked tool call does not end the turn.
    blocked_stage: str = ""
    outcome: Outcome = Outcome.ALLOWED
    # As if every report-mode rule were enforced.
    report_outcome: Outcome = Outcome.ALLOWED

    def stage(self, name: Stage | str) -> StageResult:
        """The last result for stage `name`; KeyError if never reached."""
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
    """Run `policy` over one exchange as a turn would, then reflect, without recording it.

        result = await dry_run("support", input="Where is my refund?", output="On its way.")
        result.stage("gate").answers["injection_question"].value

    `policy` is a name or a `ResolvedPolicy` (e.g. a draft from `build_policy`).
    `tool_calls` are checked in order before the answer. Judges are real: in tests
    use a CheckLens or `LLMJudge(llm=FakeLLM(...))`. Raises LookupError when a named
    policy does not load.
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

    def add_stage(stage_result: StageResult, before: int, matched_before: int) -> None:
        result.stages.append(
            replace(
                stage_result,
                answered=guard.answered[before:],
                matched=guard.matched[matched_before:],
            )
        )

    gate = await guard.gate()
    add_stage(gate, 0, 0)
    if gate.blocked:
        result.blocked_stage = "gate"

    if not result.blocked_stage:
        for raw in tool_calls:
            call = raw if isinstance(raw, ToolCall) else ToolCall.model_validate(raw)
            answered, matched = len(guard.answered), len(guard.matched)
            message = await guard.tool_call(call)
            guard.context.tool_calls.append(call)
            tool_result = StageResult(stage=Stage.TOOL, tool=call.name)
            if message:
                tool_result.block(message)
            add_stage(tool_result, answered, matched)

    if not result.blocked_stage and output:
        answered, matched = len(guard.answered), len(guard.matched)
        finish = await guard.finish(output)
        add_stage(finish, answered, matched)
        if finish.blocked:
            result.blocked_stage = "finish"

    # Reflect runs on every closed turn, a blocked one too.
    answered, matched = len(guard.answered), len(guard.matched)
    await _run_reflect_stage(guard, resolved)
    add_stage(StageResult(stage=Stage.REFLECT), answered, matched)

    result.outcome = guard.outcome
    result.report_outcome = guard.report_outcome
    return result


async def _run_reflect_stage(guard: TurnGuard, resolved: ResolvedPolicy) -> None:
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
