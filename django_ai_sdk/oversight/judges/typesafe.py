"""TypeSafe as a judge: a hosted commercial classification service, optional.

A `Classifier` that maps a lens to a TypeSafe question and reads back its
judgments; the base handles what is sent, batching and per-source questions. It
needs django-ai-sdk[typesafe] and TYPESAFE_API_KEY, and abstains without them
(ai_sdk.oversight.W005). This module never logs what was sent or what came back.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.oversight.judges.base import (
    HISTORY_ITEM_CHARS,
    TITLE_CHARS,
    modules_installed,
    secret_setting,
    unit,
)
from django_ai_sdk.oversight.judges.classifier import (
    TAKEN_FROM_NO,
    TAKEN_FROM_YES,
    Classifier,
    ranked_rationale,
    yes_answer,
)
from django_ai_sdk.oversight.types import Answer

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import JudgeLens
    from django_ai_sdk.oversight.types import Source

# Seconds per request, retries included. Inline stages pass their remaining time instead.
TYPESAFE_TIMEOUT = 20.0
# Most characters of a turn per request. TypeSafe reads at most 32k tokens per
# question (state plus question) and 64k per request. At 3 to 4 characters per
# token, this keeps every question inside those limits.
REQUEST_CHARS = 100_000
# Confidence used when a judgment gives none.
UNSTATED_CONFIDENCE = 0.8


def typesafe_model() -> str | None:
    """TYPESAFE_MODEL, or None to let typesafe-sdk choose."""
    return secret_setting("TYPESAFE_MODEL") or None


# One client per event loop. Its connection pool is bound to the loop that opened it,
# and background tasks under async_to_sync run on their own loop.
_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[str, Any]] = (
    weakref.WeakKeyDictionary()
)


def _client(api_key: str) -> Any:
    from typesafe_sdk import AsyncTypeSafeClient

    loop = asyncio.get_running_loop()
    cached = _clients.get(loop)
    if cached is None or cached[0] != api_key:
        cached = (api_key, AsyncTypeSafeClient(api_key=api_key, model=typesafe_model()))
        _clients[loop] = cached
    return cached[1]


class TypeSafe(Classifier):
    """Answer with TypeSafe, a hosted commercial classification service.

    Needs the `typesafe` extra (pip install 'django-ai-sdk[typesafe]') and a
    TYPESAFE_API_KEY, and abstains without them. `model` pins a model. It defaults
    to TYPESAFE_MODEL, then to the SDK's default.

    Flag, Choice and Score lenses become TypeSafe questions. A yes/no is a "noul",
    a probability of yes. An Entities lens with `per_source = True` rates each
    source, as `Classifier.per_source` does.
    """

    external: ClassVar[str] = "TypeSafe"
    timeout = TYPESAFE_TIMEOUT

    def __init__(self, model: str | None = None, *, name: str = "") -> None:
        self.model = model
        self.name = name or "typesafe"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model or ""}

    def missing(self) -> str:
        if not secret_setting("TYPESAFE_API_KEY"):
            return "TYPESAFE_API_KEY is not set"
        if not modules_installed("typesafe_sdk"):
            return "typesafe-sdk is not installed (pip install django-ai-sdk[typesafe])"
        return ""

    def question(self, lens: JudgeLens) -> dict[str, Any] | None:
        """Map `lens` to a TypeSafe question, or None when TypeSafe has no matching type."""
        kind = lens.answer.kind
        if kind == "flag":
            kind, criteria = "noul", {"true": lens.yes, "false": lens.no}
        elif kind == "choice":
            criteria = {label: lens.criteria.get(label) or label for label in lens.labels}
        elif kind == "score":
            criteria = [lens.low, lens.middle, lens.high]
        else:
            return None
        instructions = f"{lens.question}\n{lens.guidance}" if lens.guidance else lens.question
        return {"type": kind, "instructions": instructions, "criteria": criteria}

    def source_question(self, source: Source) -> dict[str, Any]:
        """TypeSafe's yes/no question about whether the reply is taken from `source`."""
        return {
            "type": "noul",
            "instructions": {
                "source": {"title": source.title, "content": source.content},
                "question": "Is `assistant_reply` taken from `source`?",
            },
            "criteria": {"true": TAKEN_FROM_YES, "false": TAKEN_FROM_NO},
        }

    async def classify(
        self, state: dict[str, Any], questions: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Answer]:
        options: dict[str, Any] = {}
        if timeout is not None:
            from typesafe_sdk import RetryPolicy

            # Inline, a retry would land after the deadline, so make one attempt.
            options = {"timeout": timeout, "retry": RetryPolicy(max_retries=0)}
        response = await _client(secret_setting("TYPESAFE_API_KEY")).system_one(
            state=state, questions=questions, model=self.model or typesafe_model(), **options
        )
        answers = {}
        for name, question in questions.items():
            judged = judgment_answer(response, name, question, self.model)
            if judged is not None:
                answers[name] = judged
        return answers


# Every part of a turn TypeSafe can be sent must fit in one request.
assert (
    2 * TypeSafe.TEXT_CHARS
    + TypeSafe.SOURCES_CHARS
    + TITLE_CHARS
    + TypeSafe.TOOL_CALL_CHARS
    + TypeSafe.HISTORY * HISTORY_ITEM_CHARS
    <= REQUEST_CHARS
)


def judgment_answer(
    response: Any, name: str, question: dict[str, Any], model: str | None = None
) -> Answer | None:
    """TypeSafe's judgment on question `name` as an Answer, or None when missing."""
    # The pinned model wins over the setting, as it does when the request is sent.
    model = getattr(response, "model", None) or model or typesafe_model() or ""
    if (noul := getattr(response, "nouls", {}).get(name)) is not None:
        # A noul is a probability of yes. Its distance from 0.5 is its certainty.
        yes = float(noul.noul)
        return yes_answer(yes, model, f"yes {yes:.2f}")
    if (choice := getattr(response, "choices", {}).get(name)) is not None:
        return _rated(choice.choice, choice, question, model)
    if (score := getattr(response, "scores", {}).get(name)) is not None:
        # A score is a level index from 0 to levels - 1. Scale it to 0..1.
        value = float(score.score)
        levels = len(score.legend or {}) or len(question.get("criteria") or [])
        if levels > 1:
            value = unit(value / (levels - 1), 0.0)
        return _rated(value, score, question, model)
    return None


def _rated(value: Any, judged: Any, question: dict[str, Any], model: str) -> Answer:
    """A choice or score judgment as an Answer, with its likeliest answers as rationale."""
    probabilities = dict(judged.probabilities or {})
    confidence = getattr(judged, "confidence", None)
    if not confidence and probabilities:
        # Use the likeliest level's probability as the confidence.
        confidence = max(float(p or 0) for p in probabilities.values())
    return Answer(
        value=value,
        confidence=float(confidence) if confidence is not None else UNSTATED_CONFIDENCE,
        model=model,
        rationale=_rationale(probabilities, question),
    )


def _rationale(probabilities: dict[str, Any], question: dict[str, Any]) -> str:
    """List the likeliest answers with their probabilities. Score levels use their names."""
    criteria = question.get("criteria")
    names = list(criteria) if isinstance(criteria, list) else []

    def label_name(label: str) -> str:
        if names and str(label).isdigit() and int(label) < len(names):
            return names[int(label)]
        return str(label)

    return ranked_rationale(
        (label_name(label), float(probability or 0)) for label, probability in probabilities.items()
    )


__all__ = [
    "REQUEST_CHARS",
    "TypeSafe",
    "judgment_answer",
]
