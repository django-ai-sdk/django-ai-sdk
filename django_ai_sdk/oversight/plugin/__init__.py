"""Oversight as a Haystack plugin: hooks a Haystack Agent runs, one per concern.

    from django_ai_sdk.oversight.plugin import TurnRequest, run_agent

    request = TurnRequest(agent=my_sdk_agent, user=user, channel="api")
    result = await run_agent(Agent(chat_generator=..., tools=...), messages, request=request)
    request.raise_if_blocked()

`run_agent` opens the turn under the agent's policy, or joins the outer one,
runs a copy of the Agent with the oversight hooks in their places, and closes
the turn after the run, also when it raises. The hooks run at these points:

    before_run   GateHook           judge the history and question; mask them, or refuse the run
    before_llm   OversightStopHook  stop a run whose turn has closed; offer only the tools it allows
    before_tool  OversightToolHook  refuse a tool call the policy says no to
    after_tool   OversightToolResultHook  mask a tool result before the model reads it
    after_run    FinishHook         record usage and sources; judge the answer, pass, mask or withhold it

Each hook does nothing when the turn's policy has nothing for its stage, or when
there is no turn. A run that names no SDK agent joins the turn it runs in, or
runs under DEFAULT_POLICY outside one; a side call opts out with
`TurnRequest(oversight=False)`. A sync run (`run_agent_sync`) cannot open a
turn, so one that needs its own must not run. The SDK runs every Haystack Agent
through `run_agent`.
"""

from __future__ import annotations

from haystack.core.serialization_security import allow_deserialization_module

from django_ai_sdk.oversight.plugin.agent import (
    oversight_hooks,
    run_agent,
    run_agent_sync,
    with_oversight,
)
from django_ai_sdk.oversight.plugin.finish import FinishHook, WholeRun, mask_sources
from django_ai_sdk.oversight.plugin.gate import GateHook, keep_masked
from django_ai_sdk.oversight.plugin.tools import (
    OversightToolHook,
    OversightToolResultHook,
    masked_result,
)
from django_ai_sdk.oversight.plugin.turn import (
    OversightStopHook,
    TurnRequest,
    needs_own_turn,
    opens_own_turn,
)

# An Agent saved with these hooks loads again.
allow_deserialization_module(__name__)


__all__ = [
    "FinishHook",
    "GateHook",
    "OversightStopHook",
    "OversightToolHook",
    "OversightToolResultHook",
    "TurnRequest",
    "WholeRun",
    "keep_masked",
    "mask_sources",
    "masked_result",
    "needs_own_turn",
    "opens_own_turn",
    "oversight_hooks",
    "run_agent",
    "run_agent_sync",
    "with_oversight",
]
