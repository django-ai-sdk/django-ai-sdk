from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.oversight.judges.base import (
    UNSTATED_CONFIDENCE,
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

TYPESAFE_TIMEOUT = 20.0
# Keeps every question inside TypeSafe's 32k tokens per question, 64k per request.
REQUEST_CHARS = 100_000


def typesafe_model() -> str | None:
    return secret_setting("TYPESAFE_MODEL") or None


# One client per event loop: its pool is bound to the loop that opened it, and
# background tasks under async_to_sync run on their own loop.
_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[str, Any]] = (
    weakref.WeakKeyDictionary()
)


def api_client(api_key: str) -> Any:
    from typesafe_sdk import AsyncTypeSafeClient

    loop = asyncio.get_running_loop()
    cached = _clients.get(loop)
    if cached is None or cached[0] != api_key:
        cached = (api_key, AsyncTypeSafeClient(api_key=api_key))
        _clients[loop] = cached
    return cached[1]


class TypeSafe(Classifier):
    """Abstains without the `typesafe` extra and TYPESAFE_API_KEY."""

    external: ClassVar[str] = "TypeSafe"
    timeout = TYPESAFE_TIMEOUT
    name = "typesafe"

    def __init__(
        self, model: str | None = None, *, name: str = "", timeout: float | None = None
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        self.model = model

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model or ""}

    def missing(self) -> str:
        if not secret_setting("TYPESAFE_API_KEY"):
            return "TYPESAFE_API_KEY is not set"
        if not modules_installed("typesafe_sdk"):
            return "typesafe-sdk is not installed (pip install django-ai-sdk[typesafe])"
        return ""

    def question(self, lens: JudgeLens) -> dict[str, Any] | None:
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

            options = {"timeout": timeout, "retry": RetryPolicy(max_retries=0)}
        response = await api_client(secret_setting("TYPESAFE_API_KEY")).system_one(
            state=state, questions=questions, model=self.model or typesafe_model(), **options
        )
        answers = {}
        for name, question in questions.items():
            judged = judgment_answer(response, name, question, self.model)
            if judged is not None:
                answers[name] = judged
        return answers


def judgment_answer(
    response: Any, name: str, question: dict[str, Any], model: str | None = None
) -> Answer | None:
    model = getattr(response, "model", None) or model or typesafe_model() or ""
    if (noul := getattr(response, "nouls", {}).get(name)) is not None:
        yes = float(noul.noul)
        return yes_answer(yes, model, f"yes {yes:.2f}")
    if (choice := getattr(response, "choices", {}).get(name)) is not None:
        return _rated(choice.choice, choice, question, model)
    if (score := getattr(response, "scores", {}).get(name)) is not None:
        value = float(score.score)
        levels = len(score.legend or {}) or len(question.get("criteria") or [])
        if levels > 1:
            value = unit(value / (levels - 1), 0.0)
        return _rated(value, score, question, model)
    return None


def _rated(value: Any, judged: Any, question: dict[str, Any], model: str) -> Answer:
    probabilities = dict(judged.probabilities or {})
    confidence = getattr(judged, "confidence", None)
    if confidence is None and probabilities:
        confidence = max(float(p or 0) for p in probabilities.values())
    return Answer(
        value=value,
        confidence=float(confidence) if confidence is not None else UNSTATED_CONFIDENCE,
        model=model,
        rationale=_rationale(probabilities, question),
    )


def _rationale(probabilities: dict[str, Any], question: dict[str, Any]) -> str:
    criteria = question.get("criteria")
    names = list(criteria) if isinstance(criteria, list) else []

    def label_name(label: str) -> str:
        if names and str(label).isdigit() and int(label) < len(names):
            return names[int(label)]
        return str(label)

    return ranked_rationale(
        (label_name(label), float(probability or 0)) for label, probability in probabilities.items()
    )
