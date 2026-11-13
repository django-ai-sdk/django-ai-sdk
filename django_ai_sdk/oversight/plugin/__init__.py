"""Oversight as a Haystack plugin: the turn around a run, and the hooks inside it.

    from django_ai_sdk.oversight.plugin import TurnRequest, run_agent

    request = TurnRequest(agent=my_sdk_agent, user=user, channel="api")
    result = await run_agent(Agent(chat_generator=..., tools=...), messages, request=request)
    request.raise_if_blocked()

`run_agent` opens the turn under the agent's policy, or joins the outer one,
runs a copy of the Agent with the oversight hooks (`hooks.py`) in their places,
and closes the turn after the run, also when it raises.

A turn inside another one joins it when its policy is the same (or there is
none). Another policy opens a child turn, linked to the outer one. A run that
names no SDK agent joins the turn it runs in, or runs under DEFAULT_POLICY
outside one; a side call opts out with `TurnRequest(oversight=False)`. A sync
run (`run_agent_sync`) cannot open a turn, so one that needs its own must not
run. The SDK runs every Haystack Agent through `run_agent`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from django_ai_sdk.adapters.run import user_content
from django_ai_sdk.common import last_user_message
from django_ai_sdk.oversight.agents import acode_policy, code_policy, policy_needs_await
from django_ai_sdk.oversight.conf import is_enabled
from django_ai_sdk.oversight.guard import TurnGuard, closing_turn, current_guard, guarding
from django_ai_sdk.oversight.guard.notices import UNCHECKED_TEXT, merge_restrict
from django_ai_sdk.oversight.plugin.hooks import (
    FinishHook,
    GateHook,
    OversightHook,
    OversightStopHook,
    OversightToolHook,
    OversightToolResultHook,
    RecordUsageHook,
    TurnRequest,
)
from django_ai_sdk.oversight.registry import policy_name

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import Policy

# The agent of a Haystack run that names none: it has no policy of its own. So it
# runs under DEFAULT_POLICY, like any agent without one, and is recorded under
# this name. Without DEFAULT_POLICY it opens no turn.
UNNAMED_AGENT = SimpleNamespace(name="Haystack agent", oversight=None)


def policy_agent(request: TurnRequest) -> Any:
    """The agent whose policy the run is under. Named by nothing, outside a turn:
    the unnamed agent, under DEFAULT_POLICY. Inside one, the run joins it."""
    if request.agent is None and current_guard() is None:
        return UNNAMED_AGENT
    return request.agent


async def open_turn(
    agent: Any,
    *,
    channel: str,
    user: Any = None,
    thread_id: str | None = None,
    messages: list[ChatMessage] | None = None,
    input: str | None = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> tuple[TurnGuard, bool]:
    """Start a turn of `agent`. Returns its guard, and whether it opened a turn of its own.

    It opens none without a policy, or inside a turn of the same policy, which
    it joins. Inside a turn of another policy it opens a child turn, which is
    still the outer turn's person and keeps the tools the outer turn took away.
    """
    outer = current_guard()
    guard = await TurnGuard.start(
        agent,
        channel=channel,
        user=user,
        thread_id=thread_id,
        messages=messages,
        input=input,
        oversight=oversight,
    )
    if not guard.active or joins_turn(outer, guard.resolved.name):
        return guard, False
    if outer is not None:
        guard.parent_turn_id = outer.id
        guard.actor_id = guard.actor_id or outer.actor_id
        if outer.restrict is not None:
            guard.restrict = merge_restrict(guard.restrict, outer.restrict)
    return guard, True


def joins_turn(outer: TurnGuard | None, policy_name: str) -> bool:
    """Whether a run under `policy_name` joins the open turn `outer` instead of its own."""
    return outer is not None and outer.policy is not None and outer.policy.name == policy_name


def needs_own_turn(request: TurnRequest) -> bool:
    """Whether the run needs a turn of its own here, which only an async run opens.

    Sync code cannot await `get_oversight()`, so an agent that overrides it is
    assumed to need one, and so is a policy name that does not resolve.
    """
    agent, oversight = policy_agent(request), request.oversight
    if agent is None or oversight in (False, "") or not is_enabled():
        return False
    try:
        if oversight:
            name = policy_name(oversight)
        elif policy_needs_await(agent):
            return True
        else:
            name = code_policy(agent)[0]
    except Exception:
        return True
    return bool(name) and not joins_turn(current_guard(), name)


async def opens_own_turn(request: TurnRequest) -> bool:
    """Whether the run of `request.agent` opens a turn of its own here. True when its policy cannot be told."""
    agent, oversight = request.agent, request.oversight
    if agent is None or oversight in (False, "") or not is_enabled():
        return False
    try:
        name = policy_name(oversight) if oversight else ""
        name = name or (await acode_policy(agent, channel=request.channel, user=request.user))[0]
    except Exception:
        return True
    return bool(name) and not joins_turn(current_guard(), name)


def build_oversight_hooks() -> dict[str, list[Any]]:
    """The oversight hooks, at the points they run. The run's TurnRequest names the agent."""
    return {
        "before_run": [GateHook()],
        "before_llm": [OversightStopHook()],
        "before_tool": [OversightToolHook()],
        "after_tool": [OversightToolResultHook()],
        "after_run": [RecordUsageHook(), FinishHook()],
    }


def merge_oversight_hooks(hooks: dict[str, list[Any]] | None) -> dict[str, list[Any]]:
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
    for point, ours in build_oversight_hooks().items():
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
    agent = apply_oversight(agent)
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
    return apply_oversight(agent).run(messages=messages, hook_context=request.context(), **kwargs)


def apply_oversight(agent: Any) -> Any:
    """A copy of `agent` with the oversight hooks in their places."""
    return agent.clone(hooks=merge_oversight_hooks(agent.hooks))


async def _open(request: TurnRequest, messages: list[Any]) -> TurnGuard | None:
    """Open the run's own turn. None when it joins the turn it runs inside, or has no policy."""
    agent = policy_agent(request)
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


def keep_masked(messages: list[ChatMessage], request: TurnRequest) -> None:
    """Mask the caller's last user message as the gate masked it, so the caller stores that copy.

    Only its content is stored. When the masked text no longer splits back into
    the content and the attachments' context, the content is withheld rather
    than stored with what may be unmasked.
    """
    gated = request.gated
    if gated is not None and gated.changed and (question := last_user_message(messages)):
        content = user_content(question, gated.text)
        question.content = content if content is not None else str(UNCHECKED_TEXT)
