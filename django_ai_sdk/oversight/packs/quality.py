"""The quality pack: was the answer good, and did retrieval find the right things?

The RAG triad, "taken from" and helpfulness run in the background (reflect), so
they add no latency. Refusals, wrong-language replies, empty retrieval and failed
tools are checked at finish. Every judge is an optional alias: with none defined,
the judge lenses are skipped and the checks still run.
"""

from __future__ import annotations

from django_ai_sdk.oversight.actions import Alert, Queue
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Entities, JudgeLens, Score
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.presets import EmptyRetrievalLens, RefusalLens, ToolErrorsLens
from django_ai_sdk.oversight.registry import register_policy
from django_ai_sdk.oversight.rules import Rule
from django_ai_sdk.oversight.types import Stage, Subject


class ContextRelevance(JudgeLens):
    name = "context_relevance"
    answer = Score
    label = "Context relevance"
    question = "How relevant are the retrieved sources to the user's question?"
    guidance = (
        "1 when the sources contain what is needed to answer; 0.5 when they are on topic "
        "but miss the specific point; 0 when they are about something else."
    )
    low, high = "unrelated", "exactly what is needed"
    subject = Subject.SOURCES
    stages = (Stage.REFLECT,)
    judges = ["classifier?"]  # noqa: RUF012


class Groundedness(JudgeLens):
    name = "groundedness"
    answer = Score
    label = "Groundedness"
    question = "How much of the answer is supported by the retrieved sources?"
    guidance = (
        "Judge claims, not wording. 1 when every factual claim is in the sources; 0 when "
        "the answer's claims are absent from or contradict them. General knowledge and "
        "pleasantries neither help nor hurt."
    )
    low, high = "unsupported", "fully supported"
    subject = Subject.OUTPUT
    stages = (Stage.REFLECT,)
    needs_grounding = True
    judges = ["classifier?"]  # noqa: RUF012


class TakenFrom(JudgeLens):
    name = "taken_from"
    answer = Entities
    label = "Taken from"
    question = "Which of the retrieved sources is the answer taken from?"
    guidance = "Name each source a fact in the answer comes from; none when it rests on none."
    subject = Subject.OUTPUT
    stages = (Stage.REFLECT,)
    # The labels are the turn's sources. A classifier rates each one.
    per_source = True
    # Asked for every answer, since one without sources can be taken from none.
    judges = ["classifier?"]  # noqa: RUF012


class AnswerRelevance(JudgeLens):
    name = "answer_relevance"
    answer = Score
    label = "Answer relevance"
    question = "How directly does the answer address what the user asked?"
    low, high = "beside the point", "fully addresses it"
    subject = Subject.EXCHANGE
    stages = (Stage.REFLECT,)
    judges = ["classifier?"]  # noqa: RUF012


class Helpfulness(JudgeLens):
    name = "helpfulness"
    answer = Score
    label = "Helpfulness"
    question = "How helpful would this answer be to the person who asked?"
    guidance = "Consider correctness, completeness and clarity together."
    low, high = "useless", "excellent"
    subject = Subject.EXCHANGE
    stages = (Stage.REFLECT,)
    judges = ["classifier?", "judge?"]  # noqa: RUF012
    sample = 0.25


class LanguageMismatch(JudgeLens):
    name = "language_mismatch"
    label = "Language mismatch"
    question = "Is the answer in a different language than the question?"
    yes = "Answers in a different language than the question."
    no = "Same language."
    subject = Subject.EXCHANGE
    # LanguageLens is a check lens, not a judge, so it cannot lead this chain.
    judges = ["classifier?"]  # noqa: RUF012


@register_policy
class QualityPack(Policy):
    name = "quality"
    label = "Quality"
    description = "The RAG triad, refusals, language and tool health."
    origin = "pack:quality"
    lenses = [
        ContextRelevance,
        Groundedness,
        TakenFrom,
        AnswerRelevance,
        Helpfulness,
        RefusalLens,
        LanguageMismatch,
        EmptyRetrievalLens,
        ToolErrorsLens,
    ]
    rules = [
        Rule(
            "review_ungrounded",
            when=(L("groundedness") < 0.5) & L("groundedness").confident(0.7),
            then=Queue("quality"),
            description="Answers the sources do not support go to a person.",
        ),
        Rule(
            "review_unsourced",
            # No confidence gate. An unsure verdict is worth a look too.
            when=L("taken_from").present & ~L("taken_from").is_true,
            then=Queue("quality"),
            description="Answers taken from none of the sources, or no answer, go to a person.",
        ),
        Rule(
            "review_irrelevant_context",
            when=(L("context_relevance") < 0.3) & L("context_relevance").confident(0.7),
            then=Queue("retrieval"),
            description="Retrieval that found the wrong things, to tune the knowledge base.",
        ),
        Rule(
            "note_empty_retrieval",
            when=L("empty_retrieval").is_true,
            then=Alert("info"),
        ),
        Rule(
            "note_tool_failures",
            when=L("tool_errors") >= 0.5,
            then=Alert("warning"),
        ),
    ]


__all__ = [
    "AnswerRelevance",
    "ContextRelevance",
    "Groundedness",
    "Helpfulness",
    "LanguageMismatch",
    "QualityPack",
    "TakenFrom",
]
