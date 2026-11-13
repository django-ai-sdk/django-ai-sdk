"""After the run: what it cost, which sources it used, and the finished answer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole
from pydantic import ValidationError

from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import current_guard
from django_ai_sdk.oversight.plugin.gate import REPLACED_META_KEY, stand_in
from django_ai_sdk.oversight.plugin.turn import OversightHook, _context, turn_of
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.oversight.guard import FinishResult, TurnGuard


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
        turn = turn_of(_context(state))
        if turn is None or not turn.owns:
            return
        from django_ai_sdk.adapters.citations.registry import current_registry

        if (registry := current_registry.get()) is not None:
            await mask_sources(turn.guard, registry)
            turn.guard.set_sources(registry.all, seeded=registry.seeded)


class FinishHook(OversightHook):
    """Judge the answer the run wrote, then pass, mask or withhold it in the run's messages.

    The answer is all text the model wrote in this run, as a stream shows it. The
    hook_context may change that: `answer(messages)` picks the text, and a
    replaced answer then replaces every message the run produced. With `schema`
    (a Pydantic model), a masked answer that no longer fits it is withheld.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    def run(self, state: State) -> None:
        """A sync run never owns a turn, so there is nothing to finish."""

    async def run_async(self, state: State) -> None:
        context = _context(state)
        turn = turn_of(context)
        if turn is None or not turn.owns or turn.refused:
            return
        guard = turn.guard
        messages = list(state.data.get("messages") or [])
        produced = messages[turn.first :]
        answer = context.get("answer")
        text = answer(produced) if answer else _assistant_text(produced)
        if not text:
            guard.note_error("no result")
            return
        finished = await guard.finish(text)
        if (schema := context.get("schema")) is not None:
            _withhold_unfit(guard, finished, schema)
        if not finished.replaced:
            return
        if answer is not None:
            messages = [*messages[: turn.first], stand_in(finished.output)]
        else:
            messages = _replaced(messages, turn.first, finished.output)
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


def _withhold_unfit(guard: TurnGuard, finished: FinishResult, schema: Any) -> None:
    """Withhold a masked structured answer whose JSON no longer fits `schema`."""
    if not finished.replaced or finished.blocked:
        return
    try:
        schema.model_validate_json(finished.output)
    except ValidationError:
        guard.refuse("masked result does not fit its schema")
        finished.blocked = True
        finished.message = finished.output = str(Block.DEFAULT_MESSAGE)


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


def _assistant_text(messages: list[HaystackChatMessage]) -> str:
    return "".join(m.text or "" for m in messages if m.is_from(ChatRole.ASSISTANT))


def _replaced(
    messages: list[HaystackChatMessage], first: int, output: str
) -> list[HaystackChatMessage]:
    """Put the finished `output` in place of the run's last written answer."""
    for index in range(len(messages) - 1, first - 1, -1):
        message = messages[index]
        if message.is_from(ChatRole.ASSISTANT) and message.text and not message.tool_calls:
            meta = {**message.meta, REPLACED_META_KEY: True}
            replaced = HaystackChatMessage.from_assistant(output, meta=meta)
            return [*messages[:index], replaced, *messages[index + 1 :]]
    return [*messages, stand_in(output)]


__all__ = ["FinishHook", "UsageHook", "mask_sources"]
