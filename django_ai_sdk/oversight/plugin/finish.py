"""After the run: what it cost, which sources it used, and the finished answer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole

from django_ai_sdk.oversight.agent_run import withhold_unfit
from django_ai_sdk.oversight.guard import current_guard
from django_ai_sdk.oversight.plugin.gate import REPLACED_META_KEY, stand_in
from django_ai_sdk.oversight.plugin.turn import OversightHook, request_in
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    from collections.abc import Callable

    from haystack.components.agents.state.state import State

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.oversight.guard import TurnGuard


class Answer:
    """What finish judges in a run, and where a masked or withheld answer goes.

    By default: all the text the model wrote in the run, as a stream shows it.
    A replaced answer takes the place of the last reply.
    """

    def text(self, produced: list[HaystackChatMessage]) -> str:
        return "".join(m.text or "" for m in produced if m.is_from(ChatRole.ASSISTANT))

    def replace(
        self, messages: list[HaystackChatMessage], first: int, output: str
    ) -> list[HaystackChatMessage]:
        for index in range(len(messages) - 1, first - 1, -1):
            message = messages[index]
            if message.is_from(ChatRole.ASSISTANT) and message.text and not message.tool_calls:
                meta = {**message.meta, REPLACED_META_KEY: True}
                replaced = HaystackChatMessage.from_assistant(output, meta=meta)
                return [*messages[:index], replaced, *messages[index + 1 :]]
        return [*messages, stand_in(output)]


class WholeRun(Answer):
    """The text `read` makes of the run's messages, such as what a coordinator reads of a subagent.

    A replaced answer takes the place of every message the run produced, so
    `read` finds only the finished text.
    """

    def __init__(self, read: Callable[[list[HaystackChatMessage]], str]) -> None:
        self.read = read

    def text(self, produced: list[HaystackChatMessage]) -> str:
        return self.read(produced)

    def replace(
        self, messages: list[HaystackChatMessage], first: int, output: str
    ) -> list[HaystackChatMessage]:
        return [*messages[:first], stand_in(output)]


class UsageHook(OversightHook):
    """Add the run's token usage to its turn, its own or the one it joined.

    Then, for a turn of its own, mask the citation sources and give them to the
    turn, so finish can judge the answer against them.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    def run(self, state: State) -> None:
        _record_usage(state)

    async def run_async(self, state: State) -> None:
        _record_usage(state)
        turn = request_in(state).turn
        if turn is None or not turn.owns:
            return
        from django_ai_sdk.adapters.citations.registry import current_registry

        if (registry := current_registry.get()) is not None:
            await mask_sources(turn.guard, registry)
            turn.guard.set_sources(registry.all, seeded=registry.seeded)


class FinishHook(OversightHook):
    """Judge the answer the run wrote, then pass, mask or withhold it in the run's messages.

    The TurnRequest's `answer` says which text that is (see `Answer`). With its
    `schema` (a Pydantic model), a masked answer that no longer fits is withheld.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    async def run_async(self, state: State) -> None:
        request = request_in(state)
        turn = request.turn
        if turn is None or not turn.owns or turn.refused:
            return
        guard = turn.guard
        answer = request.answer or Answer()
        messages = list(state.data.get("messages") or [])
        text = answer.text(messages[turn.first :])
        if not text:
            guard.note_error("no result")
            return
        finished = await guard.finish(text)
        if request.schema is not None:
            withhold_unfit(guard, finished, request.schema)
        if finished.replaced:
            messages = answer.replace(messages, turn.first, finished.output)
            state.set("messages", messages, handler_override=replace_values)


async def mask_sources(guard: TurnGuard, registry: SourceRegistry) -> None:
    """Mask source text in the citation registry, like tool results.

    The citations card, the stored citations artifact and the lenses all read
    the registry, so none of them should see unmasked text. The stream and the
    usage hook both call this; each text is masked once.
    """
    for source in registry.all:
        text = source.content or ""
        if not text or guard.masked_sources.get(source.key) == text:
            continue
        # mask_result keys on the text too, so a changed source is masked again.
        masked = await guard.mask_result(f"source:{source.key}", text)
        source.content = guard.masked_sources[source.key] = masked


def _record_usage(state: State) -> None:
    guard = current_guard()
    if guard is None:
        return
    prompt, completion = token_counts(state.data.get("token_usage") or None)
    guard.add_usage(_model(state), prompt, completion)


def _model(state: State) -> str | None:
    for message in state.data.get("messages") or []:
        if message.is_from(ChatRole.ASSISTANT) and (model := message.meta.get("model")):
            return model
    return None


__all__ = ["Answer", "FinishHook", "UsageHook", "WholeRun", "mask_sources"]
