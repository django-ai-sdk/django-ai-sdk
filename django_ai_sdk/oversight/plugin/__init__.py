"""Oversight as a Haystack plugin: hooks a Haystack Agent runs, one per concern.

    from django_ai_sdk.oversight.plugin import install, oversight_hooks

    agent = Agent(chat_generator=..., tools=..., hooks=oversight_hooks(my_sdk_agent))
    result = await agent.run_async(messages, hook_context={"user": user, "channel": "api"})

The hooks run at these points, in this order:

    before_run   TurnOpenHook   open the turn under the agent's policy, or join the outer one
                 GateHook       judge the history and question; mask them, or refuse the run
    before_llm   OversightStopHook  stop a run whose turn has closed
    before_tool  OversightToolHook  refuse a tool call the policy says no to
    after_tool   OversightToolResultHook  mask a tool result before the model reads it
    after_run    UsageHook      record usage and sources
                 FinishHook     judge the answer; pass, mask or withhold it
                 TurnCloseHook  record the turn

`install(agent)` adds them to an Agent built without them, and wraps its tools in
a `GuardedToolset`, which offers only the tools the turn still allows. Each hook
does nothing when the turn's policy has nothing for its stage, or when there is
no turn. `run_agent` runs an Agent this way and closes its turn even when the run
fails; the turn is then in `hook_context[TURN_KEY]` (see `turn_of`).
"""

from __future__ import annotations

from typing import Any

from haystack.core.serialization_security import allow_deserialization_module

from django_ai_sdk.oversight.plugin.finish import FinishHook, UsageHook, mask_sources
from django_ai_sdk.oversight.plugin.gate import GateHook, keep_masked
from django_ai_sdk.oversight.plugin.tools import (
    OversightToolHook,
    OversightToolResultHook,
    masked_result,
)
from django_ai_sdk.oversight.plugin.toolset import GuardedToolset
from django_ai_sdk.oversight.plugin.turn import (
    TURN_KEY,
    OversightHook,
    OversightStopHook,
    Turn,
    TurnCloseHook,
    TurnOpenHook,
    close_turn,
    run_agent,
    turn_of,
)

# An Agent saved with these hooks loads again. Only the hook classes are trusted: the
# agent a TurnOpenHook names must be on the allowlist on its own.
allow_deserialization_module(__name__)


def oversight_hooks(agent: Any = None) -> dict[str, list[Any]]:
    """The oversight hooks for `agent`, an SDK Agent whose policy they enforce.

    Without `agent`, the run's hook_context names it (`"agent"`).
    """
    return {
        "before_run": [TurnOpenHook(agent), GateHook()],
        "before_llm": [OversightStopHook()],
        "before_tool": [OversightToolHook()],
        "after_tool": [OversightToolResultHook()],
        "after_run": [UsageHook(), FinishHook(), TurnCloseHook()],
    }


def with_oversight(hooks: dict[str, list[Any]] | None, agent: Any = None) -> dict[str, list[Any]]:
    """Merge the oversight hooks into `hooks`, once, where each must run.

    Opening, the gate and the stop check come first, and so does the tool check,
    so a refused call is neither budgeted nor logged. Results are masked before
    ToolOutputLimitHook cuts them. Usage, finish and closing come last, after
    any hook that completes the answer.
    """
    from django_ai_sdk.agents.tool_agent import ToolOutputLimitHook

    if agent is None:
        # Keep the agent an installed TurnOpenHook already enforces.
        opening = (hooks or {}).get("before_run", [])
        agent = next((h.agent for h in opening if isinstance(h, TurnOpenHook)), None)
    ours = oversight_hooks(agent)
    merged = {
        point: [h for h in items if not isinstance(h, OversightHook)]
        for point, items in (hooks or {}).items()
    }
    after_tool = merged.get("after_tool", [])
    cut = next(
        (i for i, h in enumerate(after_tool) if isinstance(h, ToolOutputLimitHook)),
        len(after_tool),
    )
    for point in ("before_run", "before_llm", "before_tool"):
        merged[point] = [*ours[point], *merged.get(point, [])]
    merged["after_tool"] = [*after_tool[:cut], *ours["after_tool"], *after_tool[cut:]]
    merged["after_run"] = [*merged.get("after_run", []), *ours["after_run"]]
    return merged


def has_oversight(hooks: dict[str, list[Any]] | None) -> bool:
    return any(isinstance(h, TurnOpenHook) for h in (hooks or {}).get("before_run", []))


def install(haystack_agent: Any, agent: Any = None) -> Any:
    """`haystack_agent` with the oversight hooks and a GuardedToolset. A copy, unless it has both."""
    if has_oversight(haystack_agent.hooks) and (
        isinstance(haystack_agent.tools, GuardedToolset) or not haystack_agent.tools
    ):
        return haystack_agent
    return haystack_agent.clone(
        hooks=with_oversight(haystack_agent.hooks, agent),
        tools=GuardedToolset.wrap(haystack_agent.tools),
    )


__all__ = [
    "TURN_KEY",
    "FinishHook",
    "GateHook",
    "GuardedToolset",
    "OversightStopHook",
    "OversightToolHook",
    "OversightToolResultHook",
    "Turn",
    "TurnCloseHook",
    "TurnOpenHook",
    "UsageHook",
    "close_turn",
    "has_oversight",
    "install",
    "keep_masked",
    "mask_sources",
    "masked_result",
    "oversight_hooks",
    "run_agent",
    "turn_of",
    "with_oversight",
]
