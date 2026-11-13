"""Joining oversight to a Haystack Agent: the hooks in their places, guarded tools, a task per run."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any

from haystack.components.agents import Agent as HaystackAgent
from haystack.core.serialization import generate_qualified_class_name

from django_ai_sdk.oversight.plugin.finish import FinishHook, UsageHook
from django_ai_sdk.oversight.plugin.gate import GateHook
from django_ai_sdk.oversight.plugin.tools import OversightToolHook, OversightToolResultHook
from django_ai_sdk.oversight.plugin.toolset import GuardedToolset
from django_ai_sdk.oversight.plugin.turn import (
    REQUEST_KEY,
    OversightHook,
    OversightStopHook,
    TurnCloseHook,
    TurnOpenHook,
    TurnRequest,
    close_turn,
)


def oversight_hooks(agent: Any = None) -> dict[str, list[Any]]:
    """The oversight hooks for `agent`, an SDK Agent whose policy they enforce.

    Without `agent`, the run's TurnRequest names it.
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
    any hook that completes the answer. Oversight hooks already in `hooks` are
    moved to these places.
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


class InstalledAgent(HaystackAgent):
    """A Haystack Agent with oversight installed: see `install`.

    Each async run gets a task of its own, so the turn its hooks make current
    ends with the run, and the turn is closed with the error when the run
    raises (Haystack then skips after_run).
    """

    async def run_async(
        self,
        messages: Any,
        streaming_callback: Any = None,
        *,
        generation_kwargs: Any = None,
        tools: Any = None,
        hook_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Never empty: Haystack swaps an empty hook_context for its own, and the
        # turn would be lost to the caller.
        context = {} if hook_context is None else hook_context
        context.setdefault(REQUEST_KEY, TurnRequest())
        run = super().run_async(
            messages,
            streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            hook_context=context,
            **kwargs,
        )
        try:
            return await asyncio.create_task(run)
        except BaseException as exc:
            await close_turn(context, error=_run_error(exc))
            raise

    def to_dict(self) -> dict[str, Any]:
        """Saved as the Agent it was, with the hooks: install() it again once loaded."""
        data = super().to_dict()
        data["type"] = generate_qualified_class_name(_uninstalled(type(self)))
        return data


# Haystack requires run_async to declare the same parameters, typed the same, as run.
InstalledAgent.run_async.__signature__ = inspect.signature(HaystackAgent.run_async)  # ty: ignore[unresolved-attribute]


def install(haystack_agent: Any, agent: Any = None) -> InstalledAgent:
    """A copy of `haystack_agent` with the oversight hooks in their places and guarded tools.

    The one place oversight joins a Haystack Agent: `run_agent` calls it on every
    run. `agent` is the SDK Agent whose policy applies, unless the run's
    TurnRequest or an installed TurnOpenHook already names it. The copy keeps
    the class of `haystack_agent`, with each async run in a task of its own.
    """
    # Like Agent.clone(), into the installed class.
    params = inspect.signature(type(haystack_agent).__init__).parameters
    init: dict[str, Any] = {
        name: getattr(haystack_agent, name) for name in params if name != "self"
    }
    init["hooks"] = with_oversight(haystack_agent.hooks, agent)
    init["tools"] = GuardedToolset.wrap(haystack_agent.tools)
    return _installed(type(haystack_agent))(**init)


async def run_agent(
    agent: Any, messages: list[Any], *, hook_context: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Run a Haystack Agent with oversight. Read the turn with `turn_of(hook_context)`."""
    return await install(agent).run_async(messages=messages, hook_context=hook_context, **kwargs)


@functools.cache
def _installed(cls: type[HaystackAgent]) -> type[InstalledAgent]:
    """`cls` with oversight installed: InstalledAgent itself, or a subclass of both."""
    if issubclass(cls, InstalledAgent):
        return cls
    if cls is HaystackAgent:
        return InstalledAgent
    return type(f"Installed{cls.__name__}", (InstalledAgent, cls), {"__module__": cls.__module__})


def _uninstalled(cls: type) -> type:
    return next(c for c in cls.__mro__ if not issubclass(c, InstalledAgent))


def _run_error(exc: BaseException) -> str:
    if isinstance(exc, GeneratorExit | asyncio.CancelledError):
        return "disconnected"
    return type(exc).__name__


__all__ = ["InstalledAgent", "install", "oversight_hooks", "run_agent", "with_oversight"]
