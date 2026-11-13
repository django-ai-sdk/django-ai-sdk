"""After the run: what it cost, which sources it used, and the finished answer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole
from pydantic import ValidationError

from django_ai_sdk.oversight.actions import Block
from django_ai_sdk.oversight.guard import current_guard
from django_ai_sdk.oversight.plugin.gate import REPLACED_META_KEY, stand_in
from django_ai_sdk.oversight.plugin.turn import OversightHook, request_in
from django_ai_sdk.pricing import token_counts

if TYPE_CHECKING:
    from collections.abc import Callable

    from haystack.components.agents.state.state import State
    from pydantic import BaseModel

    from django_ai_sdk.adapters.citations import SourceRegistry
    from django_ai_sdk.oversight.guard import FinishResult, TurnGuard


class WholeRun:
    """The text `read` makes of the run's messages, such as what a coordinator reads of a subagent.

    A replaced answer takes the place of every message the run produced, so
    `read` finds only the finished text. Without it, finish judges all the text
    the model wrote, as a stream shows it, and a replaced answer takes the place
    of the last reply.
    """

    def __init__(self, read: Callable[[list[HaystackChatMessage]], str]) -> None:
        self.read = read

    def text(self, produced: list[HaystackChatMessage]) -> str:
        return self.read(produced)

    def replace(
        self, messages: list[HaystackChatMessage], first: int, output: str
    ) -> list[HaystackChatMessage]:
        return [*messages[:first], stand_in(output)]


class FinishHook(OversightHook):
    """Add the run's token usage to its turn, then judge the answer of a turn of its own.

    The citation sources are masked and given to the turn first, so finish can
    judge the answer against them. The answer is then passed, masked or withheld
    in the run's messages. The TurnRequest's `answer` says which text that is
    (see `WholeRun`). With its `schema` (a Pydantic model), a masked answer that
    no longer fits is withheld.
    """

    allowed_hook_points = ["after_run"]  # noqa: RUF012

    def run(self, state: State) -> None:
        _record_usage(state)

    async def run_async(self, state: State) -> None:
        _record_usage(state)
        request = request_in(state)
        guard = request.guard
        if guard is None:
            return
        from django_ai_sdk.adapters.citations.registry import current_registry

        if (registry := current_registry.get()) is not None:
            await mask_sources(guard, registry)
            guard.set_sources(registry.all, seeded=registry.seeded)
        if request.refused:
            return
        messages = list(state.data.get("messages") or [])
        produced = messages[request.first :]
        text = request.answer.text(produced) if request.answer else _written(produced)
        if not text:
            guard.note_error("no result")
            return
        finished = await guard.finish(text)
        if request.schema is not None:
            withhold_unfit(guard, finished, request.schema)
        if finished.replaced:
            replace = request.answer.replace if request.answer else _replace_last_reply
            messages = replace(messages, request.first, finished.output)
            state.set("messages", messages, handler_override=replace_values)


def withhold_unfit(guard: TurnGuard, finished: FinishResult, schema: type[BaseModel]) -> None:
    """Withhold a masked structured answer whose JSON no longer fits `schema`.

    Returning the unmasked original instead would leak what the mask hid.
    """
    if not finished.replaced or finished.blocked:
        return
    try:
        schema.model_validate_json(finished.output)
    except ValidationError:
        guard.refuse("masked result does not fit its schema")
        finished.blocked = True
        finished.message = finished.output = str(Block.DEFAULT_MESSAGE)


async def mask_sources(guard: TurnGuard, registry: SourceRegistry) -> None:
    """Mask source text in the citation registry, like tool results.

    The citations card, the stored citations artifact and the lenses all read
    the registry, so none of them should see unmasked text. The gate, the stream
    and the finish hook call this; each text is masked once.
    """
    for source in registry.all:
        text = source.content or ""
        if not text or guard.masked_sources.get(source.key) == text:
            continue
        # mask_result keys on the text too, so a changed source is masked again.
        masked = await guard.mask_result(f"source:{source.key}", text)
        source.content = guard.masked_sources[source.key] = masked


def _written(produced: list[HaystackChatMessage]) -> str:
    return "".join(m.text or "" for m in produced if m.is_from(ChatRole.ASSISTANT))


def _replace_last_reply(
    messages: list[HaystackChatMessage], first: int, output: str
) -> list[HaystackChatMessage]:
    for index in range(len(messages) - 1, first - 1, -1):
        message = messages[index]
        if message.is_from(ChatRole.ASSISTANT) and message.text and not message.tool_calls:
            meta = {**message.meta, REPLACED_META_KEY: True}
            replaced = HaystackChatMessage.from_assistant(output, meta=meta)
            return [*messages[:index], replaced, *messages[index + 1 :]]
    return [*messages, stand_in(output)]


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


__all__ = ["FinishHook", "WholeRun", "mask_sources", "withhold_unfit"]
