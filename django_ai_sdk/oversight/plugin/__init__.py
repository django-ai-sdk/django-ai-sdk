"""Oversight as a Haystack plugin: hooks a Haystack Agent runs, one per concern.

    from django_ai_sdk.oversight.plugin import TurnRequest, install, turn_of

    agent = install(Agent(chat_generator=..., tools=...), my_sdk_agent)
    context = TurnRequest(user=user, channel="api").context()
    result = await agent.run_async(messages, hook_context=context)
    turn_of(context).raise_if_blocked()

`install(agent)` returns a copy of a Haystack Agent with the oversight hooks in
their places and its tools in a `GuardedToolset`, which offers only the tools the
turn still allows. Each async run of the copy gets a task of its own, and a run
that raises has its turn closed with the error. The hooks run at these points,
in this order:

    before_run   TurnOpenHook   open the turn under the agent's policy, or join the outer one
                 GateHook       judge the history and question; mask them, or refuse the run
    before_llm   OversightStopHook  stop a run whose turn has closed
    before_tool  OversightToolHook  refuse a tool call the policy says no to
    after_tool   OversightToolResultHook  mask a tool result before the model reads it
    after_run    UsageHook      record usage and sources
                 FinishHook     judge the answer; pass, mask or withhold it
                 TurnCloseHook  record the turn

Each hook does nothing when the turn's policy has nothing for its stage, or when
there is no turn. A run that names no SDK agent, in `install()` or a TurnRequest,
joins the turn it runs in, or runs under DEFAULT_POLICY outside one; a side call
opts out with `TurnRequest(oversight=False)`. A sync run cannot open a turn, so
one that needs its own refuses to run. The SDK runs every Haystack Agent through
`run_agent`, which installs it first.
"""

from __future__ import annotations

from haystack.core.serialization_security import allow_deserialization_module

from django_ai_sdk.oversight.plugin.agent import (
    InstalledAgent,
    install,
    oversight_hooks,
    run_agent,
    with_oversight,
)
from django_ai_sdk.oversight.plugin.finish import (
    Answer,
    FinishHook,
    UsageHook,
    WholeRun,
    mask_sources,
)
from django_ai_sdk.oversight.plugin.gate import GateHook, keep_masked
from django_ai_sdk.oversight.plugin.tools import (
    OversightToolHook,
    OversightToolResultHook,
    masked_result,
)
from django_ai_sdk.oversight.plugin.toolset import GuardedToolset
from django_ai_sdk.oversight.plugin.turn import (
    REQUEST_KEY,
    OversightStopHook,
    Turn,
    TurnCloseHook,
    TurnOpenHook,
    TurnRequest,
    close_turn,
    request_of,
    turn_of,
)

# An Agent saved with these hooks loads again. Only the hook classes are trusted: the
# agent a TurnOpenHook names must be on the allowlist on its own.
allow_deserialization_module(__name__)


__all__ = [
    "REQUEST_KEY",
    "Answer",
    "FinishHook",
    "GateHook",
    "GuardedToolset",
    "InstalledAgent",
    "OversightStopHook",
    "OversightToolHook",
    "OversightToolResultHook",
    "Turn",
    "TurnCloseHook",
    "TurnOpenHook",
    "TurnRequest",
    "UsageHook",
    "WholeRun",
    "close_turn",
    "install",
    "keep_masked",
    "mask_sources",
    "masked_result",
    "oversight_hooks",
    "request_of",
    "run_agent",
    "turn_of",
    "with_oversight",
]
