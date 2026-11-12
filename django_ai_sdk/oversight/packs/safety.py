"""The safety pack: harmful questions, harmful answers, and people at risk.

The classifier answers all three lenses. When it is unsure about the answer, the
`judge` model is asked, since finish has time for it. A message suggesting someone
is at risk is never blocked. It goes to the front of a person's queue.
"""

from __future__ import annotations

from django_ai_sdk.oversight.actions import Alert, Notify, Queue
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import JudgeLens, Score
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register_policy
from django_ai_sdk.oversight.rules import Rule
from django_ai_sdk.oversight.types import Subject


class SelfHarm(JudgeLens):
    name = "self_harm"
    label = "Person at risk"
    question = "Does the message suggest the person may harm themselves?"
    yes = "Expresses intent, plan or wish to harm or kill themselves, directly or indirectly."
    no = "Mentions death or harm figuratively, about others, or as a topic."
    judges = ["classifier?"]  # noqa: RUF012


class HarmInput(JudgeLens):
    name = "harm_input"
    answer = Score
    label = "Harmful request"
    question = "How harmful is what the message asks for (violence, hate, sexual, illegal)?"
    low, high = "harmless", "clearly harmful"
    judges = ["classifier?"]  # noqa: RUF012


class HarmOutput(JudgeLens):
    name = "harm_output"
    answer = Score
    label = "Harmful answer"
    question = "How harmful is the answer's content (violence, hate, sexual, illegal)?"
    low, high = "harmless", "clearly harmful"
    subject = Subject.OUTPUT
    judges = ["classifier?", "judge?"]  # noqa: RUF012


@register_policy
class SafetyPack(Policy):
    name = "safety"
    label = "Safety"
    description = "Harmful content in and out, and people who may be at risk."
    origin = "pack:safety"
    lenses = [SelfHarm, HarmInput, HarmOutput]
    rules = [
        Rule(
            "person_at_risk",
            when=L("self_harm").is_true,
            then=[Queue("safety"), Notify(severity="critical")],
            description="Never blocked: someone looks, now.",
        ),
        Rule(
            "harmful_request",
            when=L("harm_input") >= 0.8,
            then=[Queue("safety"), Alert("high")],
        ),
        Rule(
            "harmful_answer",
            when=L("harm_output") >= 0.7,
            then=[Queue("safety"), Alert("high")],
        ),
    ]


__all__ = ["HarmInput", "HarmOutput", "SafetyPack", "SelfHarm"]
