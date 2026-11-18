"""Oversight as a Haystack plugin: the turn around a run, and the hooks inside it.

    from django_ai_sdk.oversight import TurnRequest, run_with_oversight

    turn = TurnRequest(agent=my_sdk_agent, user=user, channel="api")
    result = await run_with_oversight(Agent(chat_generator=..., tools=...), messages, turn=turn)
    turn.raise_if_blocked()

`run_with_oversight` opens the turn under the agent's policy, runs a copy of the
Agent with the oversight hooks in their places, and closes the turn after the
run, also when it raises.

A run inside a turn of the same policy (or none) joins it; another policy opens
a child turn linked to the outer one. A run that names no SDK agent joins the
turn it runs in, or runs under DEFAULT_POLICY outside one; a side call opts out
with `TurnRequest(oversight=False)`. A sync run cannot open a turn, so one that
needs its own raises. The SDK runs every Haystack Agent through here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from django_ai_sdk.adapters.run import user_content
from django_ai_sdk.common import last_user_message
from django_ai_sdk.oversight.agents import (
    attribute_policy,
    declared_policy,
    overrides_get_oversight,
)
from django_ai_sdk.oversight.calls import settled
from django_ai_sdk.oversight.conf import is_enabled
from django_ai_sdk.oversight.guard import (
    TurnGuard,
    closing_turn,
    current_guard,
    guarding,
    note_opened,
)
from django_ai_sdk.oversight.guard.notices import UNCHECKED_TEXT, merge_restrict
from django_ai_sdk.oversight.plugin.hooks import (
    GateQuestionHook,
    JudgeAnswerHook,
    JudgeToolCallHook,
    LimitToolChoiceHook,
    MaskToolResultHook,
    OversightHook,
    RecordUsageHook,
    StopClosedTurnHook,
    question_index,
    said_text,
)
from django_ai_sdk.oversight.plugin.hooks import (
    mask_streamed_result as mask_streamed_result,
)
from django_ai_sdk.oversight.plugin.request import TurnRequest
from django_ai_sdk.oversight.registry import policy_name

if TYPE_CHECKING:
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import Policy

# A Haystack run that names no agent runs under DEFAULT_POLICY, recorded under this name.
UNNAMED_AGENT = SimpleNamespace(name="Haystack agent", oversight=None)


def policy_agent(turn: TurnRequest) -> Any:
    if turn.agent is None and current_guard() is None:
        return UNNAMED_AGENT
    return turn.agent


async def open_turn(
    agent: Any,
    *,
    channel: str | None,
    user: Any = None,
    thread_id: str | None = None,
    messages: list[ChatMessage] | None = None,
    input: str | None = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> tuple[TurnGuard, bool]:
    """Start a turn of `agent`. Returns its guard and whether it opened a turn of its own.

    A child turn (another policy inside a turn) keeps the outer turn's person and
    the tools the outer turn took away.
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
    note_opened(guard)
    return guard, True


def joins_turn(outer: TurnGuard | None, policy_name: str) -> bool:
    return outer is not None and outer.policy is not None and outer.policy.name == policy_name


def needs_own_turn_sync(turn: TurnRequest) -> bool:
    """Whether the run needs a turn of its own, which only an async run opens.

    Sync code cannot await `get_oversight()`, so an override of it, or a policy
    name that does not resolve, is assumed to need one.
    """
    agent, oversight = policy_agent(turn), turn.oversight
    if agent is None or oversight in (False, "") or not is_enabled():
        return False
    try:
        if oversight:
            name = policy_name(oversight)
        elif overrides_get_oversight(agent):
            return True
        else:
            name = attribute_policy(agent)[0]
    except Exception:
        return True
    return bool(name) and not joins_turn(current_guard(), name)


async def needs_own_turn(turn: TurnRequest) -> bool:
    """True also when the policy cannot be determined."""
    agent, oversight = policy_agent(turn), turn.oversight
    if agent is None or oversight in (False, "") or not is_enabled():
        return False
    try:
        name = policy_name(oversight) if oversight else ""
        name = name or (await declared_policy(agent, channel=turn.channel, user=turn.user))[0]
    except Exception:
        return True
    return bool(name) and not joins_turn(current_guard(), name)


def build_oversight_hooks() -> dict[str, list[Any]]:
    return {
        "before_run": [GateQuestionHook()],
        "before_llm": [StopClosedTurnHook(), LimitToolChoiceHook()],
        "before_tool": [JudgeToolCallHook()],
        "after_tool": [MaskToolResultHook()],
        "after_run": [RecordUsageHook(), JudgeAnswerHook()],
    }


def merge_oversight_hooks(hooks: dict[str, list[Any]] | None) -> dict[str, list[Any]]:
    """Merge the oversight hooks into `hooks`, once, where each must run.

    They come first, so a blocked call is neither budgeted nor logged and a
    result is masked before any other hook reads or cuts it. Only after_run comes
    last, after any hook that completes the answer.
    """
    merged = {
        point: [h for h in items if not isinstance(h, OversightHook)]
        for point, items in (hooks or {}).items()
    }
    for point, ours in build_oversight_hooks().items():
        others = merged.get(point, [])
        merged[point] = [*others, *ours] if point == "after_run" else [*ours, *others]
    return merged


async def run_with_oversight(
    agent: Any,
    messages: list[Any],
    *,
    turn: TurnRequest | None = None,
    hook_context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a Haystack Agent with oversight. Read what it decided on `turn` afterwards.

    The turn closes after the run, also when it raises, since Haystack then skips
    after_run.
    """
    turn = turn or TurnRequest()
    kwargs["hook_context"] = turn.hook_context(**(hook_context or {}))
    agent = apply_oversight(agent)
    if turn.oversight is False:
        # Unguarded on purpose: it does not join the outer turn either.
        with guarding(None), settled():
            return await agent.run_async(messages=messages, **kwargs)
    guard = turn.guard = await _open(turn, messages)
    if guard is None:
        # No policy, or joined the outer turn: its model calls are accounted for.
        with settled():
            return await agent.run_async(messages=messages, **kwargs)
    async with closing_turn(guard, keep_open=turn.keep_open):
        with guarding(guard):
            return await agent.run_async(messages=messages, **kwargs)


def run_with_oversight_sync(
    agent: Any, messages: list[Any], *, turn: TurnRequest, **kwargs: Any
) -> dict[str, Any]:
    """Run with oversight from sync code, which cannot open a turn.

    Inside a turn its tool calls are blocked (the guard is async only). A run that
    needs its own turn raises RuntimeError; ask `needs_own_turn_sync(turn)` first
    to answer instead.
    """
    if needs_own_turn_sync(turn):
        raise RuntimeError("This run needs a turn of its own: use run_with_oversight.")
    return apply_oversight(agent).run(messages=messages, hook_context=turn.hook_context(), **kwargs)


def apply_oversight(agent: Any) -> Any:
    return agent.clone(hooks=merge_oversight_hooks(agent.hooks))


async def _open(turn: TurnRequest, messages: list[Any]) -> TurnGuard | None:
    """None when the run joins the turn it runs inside, or has no policy."""
    agent = policy_agent(turn)
    if agent is None:
        return None
    # GateQuestionHook takes the question again once Haystack rendered any user_prompt.
    asked = question_index(messages)
    question = said_text(messages[asked]) if asked is not None else ""
    guard, opened = await open_turn(
        agent,
        channel=turn.channel,
        user=turn.user,
        thread_id=turn.thread_id,
        input=question,
        oversight=turn.oversight,
    )
    if not opened:
        return None
    guard.message_id = turn.message_id or guard.message_id
    guard.on_tool_results = turn.on_tool_results
    return guard


def mask_stored_question(messages: list[ChatMessage], turn: TurnRequest) -> None:
    """Mask the caller's last user message as the gate did, so the caller stores that copy.

    When the masked text no longer splits back into content and attachment
    context, the content is withheld rather than stored possibly unmasked.
    """
    gate = turn.gate_result
    if gate is not None and gate.changed and (question := last_user_message(messages)):
        content = user_content(question, gate.text)
        question.content = content if content is not None else str(UNCHECKED_TEXT)
