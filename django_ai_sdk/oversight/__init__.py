"""Check and record the turns of agents that have a policy.

Add "django_ai_sdk.oversight" to INSTALLED_APPS and give an agent a policy::

    from django_ai_sdk import Agent
    from django_ai_sdk.oversight import L, Mask, Mode, Policy, Rule, pick
    from django_ai_sdk.oversight.packs import SecurityPack

    class SupportPolicy(Policy):
        name = "support"
        lenses = pick(SecurityPack.lenses, "pii_input")
        rules = [Rule("mask_pii", when=L("pii_input").is_true, then=Mask(), mode=Mode.ENFORCE)]

    class SupportAgent(Agent):
        oversight = SupportPolicy

Every turn the agent runs through the SDK is then judged by the policy's lenses,
acted on by its rules, and recorded. See docs/content/oversight.md.

This package re-exports only the declaration API, so importing it needs no app
registry. Other judges live in `oversight.judges`, the built-in lenses in
`oversight.lenses`, the packs in `oversight.packs`, and models, the guard and
tasks in their own modules.
"""

from __future__ import annotations

from django_ai_sdk.oversight.judges.llm import LLMJudge
from django_ai_sdk.oversight.judges.patterns import Patterns
from django_ai_sdk.oversight.lenses import (
    CheckLens,
    Choice,
    Entities,
    Flag,
    JudgeLens,
    Lens,
    PatternLens,
    PIILens,
    Score,
    SecretsLens,
    Violation,
    register_validator,
)
from django_ai_sdk.oversight.policy import Policy, pick
from django_ai_sdk.oversight.registry import register_policy
from django_ai_sdk.oversight.rules import Alert, Block, L, Mask, Notify, Queue, RestrictTools, Rule
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    Mode,
    Severity,
    Stage,
    Subject,
    TurnContext,
)

__all__ = [
    "Alert",
    "Answer",
    "Block",
    "Capture",
    "CheckLens",
    "Choice",
    "Entities",
    "Flag",
    "JudgeLens",
    "L",
    "LLMJudge",
    "Lens",
    "Mask",
    "Mode",
    "Notify",
    "PIILens",
    "PatternLens",
    "Patterns",
    "Policy",
    "Queue",
    "RestrictTools",
    "Rule",
    "Score",
    "SecretsLens",
    "Severity",
    "Stage",
    "Subject",
    "TurnContext",
    "Violation",
    "pick",
    "register_policy",
    "register_validator",
]
