"""Running a Haystack Agent with oversight: the turn around the run, the hooks inside it."""

from __future__ import annotations

from typing import Any

from django_ai_sdk.oversight.guard import TurnGuard, closing_turn, guarding
from django_ai_sdk.oversight.plugin.finish import FinishHook
from django_ai_sdk.oversight.plugin.gate import GateHook
from django_ai_sdk.oversight.plugin.tools import OversightToolHook, OversightToolResultHook
from django_ai_sdk.oversight.plugin.turn import (
    OversightHook,
    OversightStopHook,
    TurnRequest,
    agent_of,
    open_turn,
)


def oversight_hooks() -> dict[str, list[Any]]:
    """The oversight hooks, at the points they run. The run's TurnRequest names the agent."""
    return {
        "before_run": [GateHook()],
        "before_llm": [OversightStopHook()],
        "before_tool": [OversightToolHook()],
        "after_tool": [OversightToolResultHook()],
        "after_run": [FinishHook()],
    }


def with_oversight(hooks: dict[str, list[Any]] | None) -> dict[str, list[Any]]:
    """Merge the oversight hooks into `hooks`, once, where each must run.

    They come first, so a refused call is neither budgeted nor logged and a
    result is masked before any other hook reads or cuts it. Only finish comes
    last, after any hook that completes the answer. Oversight hooks already in
    `hooks` are moved to these places.
    """
    merged = {
        point: [h for h in items if not isinstance(h, OversightHook)]
        for point, items in (hooks or {}).items()
    }
    for point, ours in oversight_hooks().items():
        others = merged.get(point, [])
        merged[point] = [*others, *ours] if point == "after_run" else [*ours, *others]
    return merged


async def run_agent(
    agent: Any,
    messages: list[Any],
    *,
    request: TurnRequest | None = None,
    hook_context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a Haystack Agent with oversight. Read what it decided on `request` afterwards.

    The turn opens before the run, under the policy of `request.agent`, or the
    run joins the turn it runs inside. It closes after the run, also when the
    run raises, since Haystack then skips after_run.
    """
    request = request or TurnRequest()
    kwargs["hook_context"] = request.context(**(hook_context or {}))
    agent = _overseen(agent)
    if request.oversight is False:
        # Unguarded on purpose: it does not join the outer turn either.
        with guarding(None):
            return await agent.run_async(messages=messages, **kwargs)
    guard = request.guard = await _open(request, messages)
    if guard is None:
        return await agent.run_async(messages=messages, **kwargs)
    async with closing_turn(guard, keep_open=request.keep_open):
        with guarding(guard):
            return await agent.run_async(messages=messages, **kwargs)


def run_agent_sync(
    agent: Any, messages: list[Any], *, request: TurnRequest, **kwargs: Any
) -> dict[str, Any]:
    """Run a Haystack Agent with oversight from sync code, which cannot open a turn.

    Inside a turn the run joins it, and its tool calls are refused, since the
    guard only works from async code. Ask `needs_own_turn(request)` first: a run
    that needs a turn of its own must not run here.
    """
    return _overseen(agent).run(messages=messages, hook_context=request.context(), **kwargs)


def _overseen(agent: Any) -> Any:
    """A copy of `agent` with the oversight hooks in their places."""
    return agent.clone(hooks=with_oversight(agent.hooks))


async def _open(request: TurnRequest, messages: list[Any]) -> TurnGuard | None:
    """Open the run's own turn. None when it joins the turn it runs inside, or has no policy."""
    agent = agent_of(request)
    if agent is None:
        return None
    # GateHook takes the question again once Haystack rendered any user_prompt.
    question = next((m.text or "" for m in reversed(messages) if m.is_from("user")), "")
    guard, opened = await open_turn(
        agent,
        channel=request.channel,
        user=request.user,
        thread_id=request.thread_id,
        input=question,
        oversight=request.oversight,
    )
    if not opened:
        return None
    guard.message_id = request.message_id or guard.message_id
    guard.on_tool_results = request.on_tool_results
    return guard


__all__ = ["oversight_hooks", "run_agent", "run_agent_sync", "with_oversight"]
