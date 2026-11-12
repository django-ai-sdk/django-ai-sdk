"""Oversight as Haystack Agent hooks. Every model run of an SDK agent goes through `run_attached`.

The run's hooks do the oversight work inside the Haystack loop:

- before_run gates the question and its history (`OversightGateHook`).
- before_llm stops a run whose turn already closed (`OversightStopHook`).
- before_tool and after_tool check tool calls and mask results (see hooks.py).
- after_run records usage and sources, then finishes the answer (`OversightRunHook`).

A `GuardedToolset` around the agent's tools offers the model only the tools the
turn still allows, checked again on every step.

Opening and closing the turn stays outside the run: Haystack skips after_run when
a run raises or is cancelled, and a turn must close either way. The code that
opens the turn passes a `TurnRun` to say what the hooks should gate and finish.
A run without one (a joined turn, a side task) is only checked and counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole
from haystack.tools import Tool, Toolset

from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import current_guard, outlived_turn
from django_ai_sdk.oversight.guard.gate import GateResult
from django_ai_sdk.oversight.hooks import OversightToolHook, OversightToolResultHook
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from haystack.components.agents import Agent as HaystackAgent
    from haystack.components.agents.state.state import State

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.oversight.guard import TurnGuard

# The hook_context key the oversight hooks read.
OVERSIGHT_KEY = "django_ai_sdk.oversight"
# On the assistant message that stands in for a refused or replaced answer.
REPLACED_META_KEY = "django_ai_sdk.oversight.replaced"


@dataclass
class TurnRun:
    """What the hooks gate and finish for the turn this run belongs to.

    Only the code that opened the turn makes one. `gate` judges the question and
    masks the caller's messages in place; `to_haystack` then rebuilds the run's
    input from them. Without `to_haystack` the question is masked in the last user
    message. `answer` picks the text to finish from the messages the run produced;
    when given, a replaced answer replaces all of them.
    """

    guard: TurnGuard
    gate: Callable[[], Awaitable[GateResult]] | None = None
    to_haystack: Callable[[], list[HaystackChatMessage]] | None = None
    answer: Callable[[list[HaystackChatMessage]], str] | None = None
    # Set by the hooks.
    gated: GateResult | None = None
    sent: int = 0
    first: int = 0

    @property
    def refused(self) -> bool:
        return self.gated is not None and self.gated.blocked


def attach(agent: HaystackAgent) -> HaystackAgent:
    """`agent` as oversight runs it: with the oversight hooks and a GuardedToolset.

    Returns a copy; the built agent and its Toolsets stay as they are. It works the
    same for agents the SDK built and ones built by hand.
    """
    return agent.clone(hooks=with_oversight(agent.hooks), tools=GuardedToolset.wrap(agent.tools))


async def run_attached(
    agent: HaystackAgent,
    messages: list[HaystackChatMessage],
    *,
    turn: TurnRun | None = None,
    hook_context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run `agent` with oversight attached, under the current guard."""
    if turn is not None:
        turn.sent = len(messages)
    context = {
        **(hook_context or {}),
        OVERSIGHT_KEY: {"turn": turn, "model": _model_name(agent.chat_generator)},
    }
    return await attach(agent).run_async(messages=messages, hook_context=context, **kwargs)


def with_oversight(hooks: dict[str, list[Any]] | None) -> dict[str, list[Any]]:
    """Add the oversight hooks to `hooks`, once each, where they must run.

    The gate and the stop check run first. The tool check runs before any other
    before_tool hook, so a refused call is neither budgeted nor logged. Results
    are masked before ToolOutputLimitHook cuts them, so both see one masking.
    Finishing runs last, after any hook that completes the answer.
    """
    from django_ai_sdk.agents.tool_agent import ToolOutputLimitHook

    ours = (
        OversightGateHook,
        OversightStopHook,
        OversightToolHook,
        OversightToolResultHook,
        OversightRunHook,
    )
    added = {
        point: [h for h in items if not isinstance(h, ours)]
        for point, items in (hooks or {}).items()
    }
    after_tool = added.get("after_tool", [])
    cut = next(
        (i for i, h in enumerate(after_tool) if isinstance(h, ToolOutputLimitHook)), len(after_tool)
    )
    added["before_run"] = [OversightGateHook(), *added.get("before_run", [])]
    added["before_llm"] = [OversightStopHook(), *added.get("before_llm", [])]
    added["before_tool"] = [OversightToolHook(), *added.get("before_tool", [])]
    added["after_tool"] = [*after_tool[:cut], OversightToolResultHook(), *after_tool[cut:]]
    added["after_run"] = [*added.get("after_run", []), OversightRunHook()]
    return added


class GuardedToolset(Toolset):
    """The agent's tools, less the ones the current turn took away.

    Haystack flattens the tools again on every step, so a tool a rule takes away
    mid-run is gone from the model's next call. Warm-up, close and per-run copies
    are passed on to the wrapped Tools and Toolsets.
    """

    def __init__(self, items: list[Tool | Toolset]) -> None:
        # Not the dataclass init: the wrapped items are Toolsets too, not only Tools.
        self.tools = []
        self.items = items

    @classmethod
    def wrap(cls, tools: Any) -> Any:
        """Wrap `tools` once. No tools stay no tools."""
        if isinstance(tools, GuardedToolset) or not _has_items(tools):
            return tools
        return cls([tools] if isinstance(tools, Toolset) else list(tools))

    def _all(self) -> list[Tool]:
        return [
            tool for item in self.items for tool in (item if isinstance(item, Toolset) else [item])
        ]

    def __iter__(self) -> Iterator[Tool]:
        guard = current_guard()
        return iter([t for t in self._all() if guard is None or guard.allows_tool(t.name)])

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __contains__(self, item: str | Tool) -> bool:
        name = item if isinstance(item, str) else item.name
        return any(tool.name == name for tool in self)

    def get_selectable_tools(self) -> list[Tool]:
        return [
            tool
            for item in self.items
            for tool in (item.get_selectable_tools() if isinstance(item, Toolset) else [item])
        ]

    def spawn(self, selected_tool_names: set[str] | None = None) -> GuardedToolset:
        return GuardedToolset(
            [
                item.spawn(selected_tool_names) if isinstance(item, Toolset) else item
                for item in self.items
            ]
        )

    def warm_up(self) -> None:
        from haystack.tools.utils import warm_up_tools

        warm_up_tools(self.items)

    async def warm_up_async(self) -> None:
        from haystack.tools.utils import warm_up_tools_async

        await warm_up_tools_async(self.items)

    def close(self) -> None:
        from haystack.tools.utils import close_tools

        close_tools(self.items)

    async def close_async(self) -> None:
        from haystack.tools.utils import close_tools_async

        await close_tools_async(self.items)


class OversightGateHook:
    """Judge the question and its history before the model reads them.

    A masked question replaces the run's input. A refusal ends the run before the
    first model call, with the policy's message as the answer.
    """

    allowed_hook_points = ["before_run"]

    def run(self, state: State) -> None:
        """A sync run cannot await the gate, so a turn that needs one is refused."""
        turn = _turn(state)
        if turn is None:
            return
        turn.guard.refuse("gate in a sync run")
        turn.gated = GateResult(blocked=True, message=str(Block.DEFAULT_MESSAGE))
        _answer_with(state, turn.gated.message, stop="blocked")

    async def run_async(self, state: State) -> None:
        turn = _turn(state)
        if turn is None:
            return
        guard = turn.guard
        gate = turn.gated = (
            await turn.gate() if turn.gate else GateResult(input=guard.context.input)
        )
        messages = list(state.data.get("messages") or [])
        if gate.blocked:
            _answer_with(state, gate.message, stop="blocked")
            return
        if turn.to_haystack is not None:
            # The caller's messages were masked in place: rebuild what the model reads.
            messages = [*messages[: len(messages) - turn.sent], *turn.to_haystack()]
            state.set("messages", messages, handler_override=replace_values)
        elif gate.masked:
            messages = _masked_question(messages, guard.original_input, gate.input)
            state.set("messages", messages, handler_override=replace_values)
        turn.first = len(messages)


class OversightStopHook:
    """Stop a run whose turn has closed, before it spends another model call."""

    allowed_hook_points = ["before_llm"]

    def run(self, state: State) -> None:
        if outlived_turn():
            state.set("stop_run", "turn closed")

    async def run_async(self, state: State) -> None:
        self.run(state)


class OversightRunHook:
    """Record the run's usage, then the sources and the finished answer of its turn."""

    allowed_hook_points = ["after_run"]

    def run(self, state: State) -> None:
        """A sync run reports its usage; it never owns a turn to finish."""
        _record_usage(state)

    async def run_async(self, state: State) -> None:
        _record_usage(state)
        turn = _turn(state)
        if turn is None or turn.gated is None or turn.refused:
            return
        guard = turn.guard
        from django_ai_sdk.adapters.citations.registry import current_registry

        if (registry := current_registry.get()) is not None:
            await mask_sources(guard, registry)
            guard.set_sources(registry.all, seeded=registry.seeded)
        messages = list(state.data.get("messages") or [])
        produced = messages[turn.first :]
        text = turn.answer(produced) if turn.answer else _assistant_text(produced)
        finished = await guard.finish(text)
        if not finished.replaced:
            return
        if turn.answer is not None:
            messages = [*messages[: turn.first], _stand_in(finished.output)]
        else:
            messages = _replaced_answer(messages, turn.first, finished.output)
        state.set("messages", messages, handler_override=replace_values)


async def mask_sources(guard: TurnGuard, registry: SourceRegistry) -> None:
    """Mask source text in the citation registry, like tool results.

    The citations card, the stored citations artifact and the lenses all read
    the registry, so none of them should see unmasked text. The stream and the
    run's hook both call this; each text is masked once.
    """
    for source in registry.all:
        text = source.content or ""
        if not text or guard.masked_sources.get(source.key) == text:
            continue
        # mask_result keys on the text too, so a changed source is masked again.
        masked = await guard.mask_result(f"source:{source.key}", text)
        source.content = guard.masked_sources[source.key] = masked


def _turn(state: State) -> TurnRun | None:
    context = (state.data.get("hook_context") or {}).get(OVERSIGHT_KEY) or {}
    turn = context.get("turn")
    return turn if turn is not None and turn.guard.active else None


def _record_usage(state: State) -> None:
    """Add the run's token usage to the current turn, joined or its own."""
    guard = current_guard()
    if guard is None:
        return
    context = (state.data.get("hook_context") or {}).get(OVERSIGHT_KEY) or {}
    prompt, completion = token_counts(state.data.get("token_usage") or None)
    guard.add_usage(_reply_model(state) or context.get("model"), prompt, completion)


def _reply_model(state: State) -> str | None:
    for message in state.data.get("messages") or []:
        if message.is_from(ChatRole.ASSISTANT) and (model := message.meta.get("model")):
            return model
    return None


def _model_name(generator: Any) -> str | None:
    return getattr(generator, "model", None) or getattr(generator, "model_name", None)


def _assistant_text(messages: list[HaystackChatMessage]) -> str:
    """All text the model wrote in this run, as a stream shows it."""
    return "".join(m.text or "" for m in messages if m.is_from(ChatRole.ASSISTANT))


def _stand_in(text: str) -> HaystackChatMessage:
    return HaystackChatMessage.from_assistant(text, meta={REPLACED_META_KEY: True})


def _answer_with(state: State, text: str, *, stop: str) -> None:
    """End the run with `text` as its answer."""
    messages = list(state.data.get("messages") or [])
    state.set("messages", [*messages, _stand_in(text)], handler_override=replace_values)
    state.set("stop_run", stop)


def _replaced_answer(
    messages: list[HaystackChatMessage], first: int, output: str
) -> list[HaystackChatMessage]:
    """Put the finished `output` in place of the run's last written answer."""
    for index in range(len(messages) - 1, first - 1, -1):
        message = messages[index]
        if message.is_from(ChatRole.ASSISTANT) and message.text and not message.tool_calls:
            replaced = HaystackChatMessage.from_assistant(
                output, meta={**message.meta, REPLACED_META_KEY: True}
            )
            return [*messages[:index], replaced, *messages[index + 1 :]]
    return [*messages, _stand_in(output)]


def _masked_question(
    messages: list[HaystackChatMessage], original: str, masked: str
) -> list[HaystackChatMessage]:
    """Mask the question in the last user message, wherever a template put it.

    If the question cannot be found there, the whole message becomes the masked question.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not message.is_from(ChatRole.USER):
            continue
        text = message.text or ""
        rewritten = text.replace(original, masked) if original and original in text else masked
        return [*messages[:index], HaystackChatMessage.from_user(rewritten), *messages[index + 1 :]]
    return messages


def _has_items(tools: Any) -> bool:
    # Not truthiness: it would iterate a lazy Toolset and warm it up early.
    return tools is not None and (isinstance(tools, Toolset) or bool(list(tools)))


__all__ = [
    "OVERSIGHT_KEY",
    "GuardedToolset",
    "OversightGateHook",
    "OversightRunHook",
    "OversightStopHook",
    "TurnRun",
    "attach",
    "mask_sources",
    "run_attached",
    "with_oversight",
]
