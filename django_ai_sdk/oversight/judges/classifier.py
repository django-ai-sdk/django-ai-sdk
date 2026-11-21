"""Classifiers: fast models that answer typed questions, batched per stage and state.

A subclass implements `question(lens)` and `classify(state, questions, timeout=)`
and sets `external`; see `judges.typesafe`.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.oversight.judges.base import (
    HISTORY_ITEM_CHARS,
    HISTORY_MESSAGES,
    NO_VERDICT,
    TITLE_CHARS,
    TOO_LONG,
    Judge,
    clip,
    masked_clip,
    too_long,
)
from django_ai_sdk.oversight.lenses import JudgeLens
from django_ai_sdk.oversight.patterns import mask_json, mask_secrets
from django_ai_sdk.oversight.types import Answer, Source, Subject

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterable

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnContext

# Seconds the first question of a stage waits for others before sending.
BATCH_DELAY = 0.005
RATIONALE_ANSWERS = 3


def certainty(probability: float) -> float:
    """Distance from 0.5, scaled to 0..1."""
    return min(1.0, abs(float(probability) - 0.5) * 2)


def yes_answer(probability: float, model: str, rationale: str) -> Answer:
    return Answer(
        value=probability, confidence=certainty(probability), model=model, rationale=rationale
    )


def ranked_rationale(pairs: Iterable[tuple[str, float]], n: int = RATIONALE_ANSWERS) -> str:
    """The `n` likeliest pairs, as "label 0.80, other 0.15"."""
    top = sorted(pairs, key=lambda pair: -pair[1])[:n]
    return ", ".join(f"{label} {probability:.2f}" for label, probability in top)


TAKEN_FROM_YES = "The reply states at least one fact that is in the source and relies on it."
TAKEN_FROM_NO = (
    "The reply's facts are not in the source: it only shares the topic, or it says something else."
)

# Asked when there are no sources. A reply that states facts then has no source.
STATES_FACTS = JudgeLens(
    name="facts",
    question="Does `assistant_reply` state facts as its answer to `user_message`?",
    yes="The reply gives a fact, rule, figure, date, amount or step as what is so.",
    no=(
        "The reply only greets, thanks, asks something back, offers help, or says it doesn't know."
    ),
)


class Batch:
    """Groups a stage's classifier questions so those about the same state share one call.

    The flush runs in its own task, so a lens cancelled at the deadline does not
    cancel the call other lenses wait on.
    """

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.groups: dict[tuple[str, str], tuple[Any, Any, dict[str, Any]]] = {}
        self._flushes: set[asyncio.Task] = set()

    def left(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.05, self.deadline - asyncio.get_running_loop().time())

    def ask(
        self,
        key: str,
        classify: Callable[..., Awaitable[dict[str, Answer]]],
        state: Any,
        name: str,
        question: Any,
    ) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        if not self.groups:
            loop.call_later(BATCH_DELAY, self._start_flush)
        group = (key, json.dumps(state, sort_keys=True, default=str))
        future = loop.create_future()
        self.groups.setdefault(group, (classify, state, {}))[2][name] = (question, future)
        return future

    def _start_flush(self) -> None:
        task = asyncio.ensure_future(self.flush())
        self._flushes.add(task)
        task.add_done_callback(self._flushes.discard)

    async def flush(self) -> None:
        groups, self.groups = self.groups, {}
        await asyncio.gather(
            *(self._send(classify, state, waiting) for classify, state, waiting in groups.values())
        )

    async def _send(self, classify: Any, state: Any, waiting: dict[str, Any]) -> None:
        """Send one call; cancel it once nobody waits, so it never outlives its stage."""
        questions = {name: question for name, (question, _) in waiting.items()}
        futures = [future for _, future in waiting.values()]
        call = asyncio.ensure_future(classify(state, questions, timeout=self.left()))

        def abandon(_: asyncio.Future) -> None:
            if all(f.done() for f in futures) and not call.done():
                call.cancel()

        for future in futures:
            future.add_done_callback(abandon)
        abandon(call)
        try:
            result: Any = await call
        except asyncio.CancelledError:
            if not call.cancelled():
                raise
            return
        except Exception as exc:
            result = exc
        for future in futures:
            if future.done():
                continue
            if isinstance(result, Exception):
                future.set_exception(result)
            else:
                future.set_result(result)


_batch: contextvars.ContextVar[Batch | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_classifier_batch", default=None
)


@contextlib.contextmanager
def batched(deadline_ms: float | None = None) -> Generator[Batch]:
    """Batch the classifier questions of lens tasks created inside this block."""
    deadline = asyncio.get_running_loop().time() + deadline_ms / 1000 if deadline_ms else None
    batch = Batch(deadline)
    token = _batch.set(batch)
    try:
        yield batch
    finally:
        _batch.reset(token)


class Classifier(Judge):
    """Base for judges that classify yes/no, label, score and per-source lenses."""

    external: ClassVar[str] = "classifier"
    timeout = 20.0

    TEXT_CHARS: ClassVar[int] = 30_000
    SOURCE_CHARS: ClassVar[int] = 8_000
    SOURCES_CHARS: ClassVar[int] = 32_000
    TOOL_CALL_CHARS: ClassVar[int] = 4_000
    HISTORY: ClassVar[int] = HISTORY_MESSAGES

    @abstractmethod
    def question(self, lens: JudgeLens) -> Any | None:
        """Turn `lens` into this classifier's question, or None if it can't answer it."""

    @abstractmethod
    async def classify(
        self, state: dict[str, Any], questions: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Answer]:
        """Answer `questions` about `state` in one call. `timeout`: seconds left, or None."""

    def batch_key(self) -> str:
        """Only judges with the same configuration share a call."""
        described = {k: v for k, v in self.describe().items() if k != "name"}
        return f"{type(self).__qualname__}:{json.dumps(described, sort_keys=True, default=str)}"

    def answers(self, lens: Lens) -> bool:
        assert isinstance(lens, JudgeLens)  # Only judge lenses have judges.
        per_source = lens.answer.kind == "entities" and lens.per_source
        return per_source or self.question(lens) is not None

    async def answer(self, lens: Lens, context: TurnContext) -> Answer | None:
        assert isinstance(lens, JudgeLens)  # Only judge lenses have judges.
        if lens.answer.kind == "entities" and lens.per_source:
            return await self.per_source(lens, context)
        question = self.question(lens)
        if question is None:
            return None
        if self.too_long(lens, context):
            return Answer(error=TOO_LONG)
        answers = await self._classify_batched(self.state(lens, context), {lens.name: question})
        # Asked and not answered is a failure, as with the model judge, so fail-closed holds.
        return answers.get(lens.name) or Answer(error=NO_VERDICT)

    async def _classify_batched(
        self, state: dict[str, Any], questions: dict[str, Any], *, alone: bool = False
    ) -> dict[str, Answer]:
        batch = _batch.get()
        if batch is None:
            return await self.classify(state, questions)
        if alone:
            return await self.classify(state, questions, timeout=batch.left())
        futures = [
            batch.ask(self.batch_key(), self.classify, state, name, question)
            for name, question in questions.items()
        ]
        results = await asyncio.gather(*futures)
        merged: dict[str, Answer] = {}
        for result in results:
            merged.update(result)
        return {name: merged[name] for name in questions if name in merged}

    def text(self, text: str, limit: int) -> str:
        return masked_clip(text, limit)

    def labelled_sources(
        self, context: TurnContext, *, earlier: bool = False
    ) -> list[tuple[str, Source]]:
        """The turn's sources labelled S1, S2, ..., masked, clipped, up to SOURCES_CHARS."""
        labelled: list[tuple[str, Source]] = []
        total = 0
        for source in context.grounding(earlier=earlier):
            title = self.text(source.title, TITLE_CHARS)
            content = self.text(source.content, self.SOURCE_CHARS - len(title))
            if labelled and total + len(title) + len(content) > self.SOURCES_CHARS:
                break
            total += len(title) + len(content)
            kept = source.model_copy(update={"title": title, "content": content})
            labelled.append((f"S{len(labelled) + 1}", kept))
        return labelled

    def state(self, lens: JudgeLens, context: TurnContext) -> dict[str, Any]:
        """Only the parts the lens's subject covers."""
        subject = lens.subject
        state: dict[str, Any] = {}
        if context.history and subject in {Subject.QUESTION, Subject.EXCHANGE}:
            state["history"] = [
                {"role": h.role, "content": self.text(h.content, HISTORY_ITEM_CHARS)}
                for h in context.history[-self.HISTORY :]
            ]
        if subject is not Subject.TOOL_CALL or lens.reads_question:
            state["user_message"] = self.text(context.input, self.TEXT_CHARS)
        if subject in {Subject.ANSWER, Subject.EXCHANGE}:
            state["assistant_reply"] = self.text(context.output, self.TEXT_CHARS)
        # Unlike the model judge, an exchange gets no sources: a classifier scores
        # what it is sent, so sources would turn it into a groundedness score.
        if subject in {Subject.SOURCES, Subject.ANSWER} and (
            labelled := self.labelled_sources(context, earlier=subject is Subject.ANSWER)
        ):
            state["sources"] = [
                {"id": label, "title": source.title, "content": source.content}
                for label, source in labelled
            ]
        if subject is Subject.TOOL_CALL and context.tool_call:
            arguments = mask_json(context.tool_call.arguments, mask_secrets)
            compact = json.dumps(arguments, ensure_ascii=False, default=str)
            state["tool_call"] = {
                "name": self.text(context.tool_call.name, TITLE_CHARS),
                "arguments": arguments
                if len(compact) <= self.TOOL_CALL_CHARS
                else clip(compact, self.TOOL_CALL_CHARS),
            }
        return state

    def too_long(self, lens: Lens, context: TurnContext) -> bool:
        return too_long(
            lens,
            context,
            input_chars=self.TEXT_CHARS,
            output_chars=self.TEXT_CHARS,
            tool_chars=self.TOOL_CALL_CHARS,
        )

    def source_question(self, source: Source) -> Any | None:
        return self.question(
            JudgeLens(
                name="taken_from",
                question="Is `assistant_reply` taken from this source?",
                guidance=f"Source: {source.title}\n{source.content}",
                yes=TAKEN_FROM_YES,
                no=TAKEN_FROM_NO,
            )
        )

    async def per_source(self, lens: JudgeLens, context: TurnContext) -> Answer | None:
        """Which sources the answer is taken from, one yes/no per source.

        Confidence is the lowest per-source one. Without sources, a reply that
        states facts is taken from none.
        """
        if not context.output.strip():
            return Answer(value=lens.answer.parse(lens, []), confidence=1.0, rationale="No answer.")
        if self.too_long(lens, context):
            return Answer(error=TOO_LONG)
        labelled = self.labelled_sources(context, earlier=True)
        state = {
            "user_message": self.text(context.input, self.TEXT_CHARS),
            "assistant_reply": self.text(context.output, self.TEXT_CHARS),
        }
        questions = {label: self.source_question(source) for label, source in labelled}
        questions = {label: q for label, q in questions.items() if q is not None}
        if labelled and not questions:
            return None
        if not labelled:
            facts = self.question(STATES_FACTS)
            if facts is None:
                return None
            answers = await self._classify_batched(state, {"facts": facts}, alone=True)
            return _unsourced(lens, answers.get("facts"))
        answers = await self._classify_batched(state, questions, alone=True)

        found: list[str] = []
        sure: list[float] = []
        notes: list[str] = []
        items: dict[str, float] = {}
        model = ""
        for label, source in labelled:
            judged = answers.get(label)
            if judged is None or judged.error or not isinstance(judged.value, int | float):
                continue
            yes = float(judged.value)
            model = model or judged.model
            sure.append(judged.confidence)
            title = source.title or label
            if source.seeded:
                title = f"{title} (found earlier)"
            if yes >= 0.5:
                found.append(title)
            notes.append(f"{label} {title}: {yes:.2f}")
            items[source.key or label] = round(yes, 3)
        if not sure:
            return Answer(error=NO_VERDICT)
        return Answer(
            value=lens.answer.parse(lens, found),
            confidence=min(sure),
            model=model,
            rationale="; ".join(notes),
            items=items,
        )


def _unsourced(lens: Lens, judged: Answer | None) -> Answer | None:
    """A missing or failed answer is a failure, so fail-closed holds."""
    if judged is None or judged.error or not isinstance(judged.value, int | float):
        return Answer(error=(judged.error if judged else "") or NO_VERDICT)
    if float(judged.value) < 0.5:
        return None
    return Answer(
        value=lens.answer.parse(lens, []),
        confidence=judged.confidence,
        model=judged.model,
        rationale=f"No source found; states facts: {float(judged.value):.2f}",
    )
