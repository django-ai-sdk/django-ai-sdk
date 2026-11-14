"""Check and record the turns of agents that have a policy.

Add "django_ai_sdk.oversight" to INSTALLED_APPS and give an agent a policy::

    from django_ai_sdk import Agent
    from django_ai_sdk.oversight import L, Mask, Mode, Policy, Rule, SecurityPack, pick

    class SupportPolicy(Policy):
        name = "support"
        lenses = pick(SecurityPack.lenses, "pii_input")
        rules = [Rule("mask_pii", when=L("pii_input").is_true, then=Mask(), mode=Mode.ENFORCE)]

    class SupportAgent(Agent):
        oversight = SupportPolicy

Every turn the agent runs through the SDK is then judged by the policy's lenses,
acted on by its rules, and recorded. See docs/content/oversight.md.

Everything an app uses imports from here; the submodules are internal. The
declaration API loads with the package, so importing it needs no app registry.
The rest (judges, packs, the run plugin, the guard, the stream, the dry run,
notifiers and retention) loads on first use. The review API (`services`,
`schemas`, `permissions`) and the authoring app are opt-in layers with their own
modules.
"""

from __future__ import annotations

import importlib
import typing

from django_ai_sdk.oversight.judges.llm import LLMJudge
from django_ai_sdk.oversight.judges.patterns import Patterns
from django_ai_sdk.oversight.lenses import (
    CheckLens,
    Choice,
    EmptyRetrievalLens,
    Entities,
    Flag,
    JudgeLens,
    LanguageLens,
    LengthLens,
    Lens,
    PatternLens,
    PIILens,
    RefusalLens,
    Score,
    SecretsLens,
    ToolErrorsLens,
    Violation,
    register_validator,
)
from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy, pick
from django_ai_sdk.oversight.registry import (
    add_policy_loader,
    build_policy,
    invalidate_policy,
    register_policy,
    unregister_policy,
)
from django_ai_sdk.oversight.rules import Alert, Block, L, Mask, Notify, Queue, RestrictTools, Rule
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    Mode,
    Outcome,
    Severity,
    Stage,
    Subject,
    TextSpan,
    TurnContext,
)

if typing.TYPE_CHECKING:
    from django_ai_sdk.oversight.content import forget_actor, forget_thread, forget_turns
    from django_ai_sdk.oversight.guard import (
        TurnGuard,
        closing_turn,
        current_guard,
        guarding,
        outlived_turn,
        unguarded,
    )
    from django_ai_sdk.oversight.guard.stages import StageResult
    from django_ai_sdk.oversight.history import history_for
    from django_ai_sdk.oversight.judges.base import Judge, ask_judge, build_judge, judge_label
    from django_ai_sdk.oversight.judges.classifier import Classifier, certainty
    from django_ai_sdk.oversight.judges.local import LocalClassifier, load_local_classifier
    from django_ai_sdk.oversight.judges.typesafe import TypeSafe
    from django_ai_sdk.oversight.notifiers import (
        LogNotifier,
        Notifier,
        WebhookNotifier,
        send_to_notifiers,
    )
    from django_ai_sdk.oversight.packs import DefaultPolicy
    from django_ai_sdk.oversight.packs.quality import (
        AnswerRelevance,
        ContextRelevance,
        Groundedness,
        Helpfulness,
        QualityPack,
        TakenFrom,
    )
    from django_ai_sdk.oversight.packs.safety import HarmInput, HarmOutput, SafetyPack, SelfHarm
    from django_ai_sdk.oversight.packs.security import (
        EXFILTRATION_PATTERNS,
        INJECTION_PATTERNS,
        Exfiltration,
        PromptInjection,
        SecurityPack,
    )
    from django_ai_sdk.oversight.patterns import (
        DEFAULT_PII_KINDS,
        PII_PATTERNS,
        SECRET_PATTERNS,
        mask,
        mask_all,
        mask_secrets,
    )
    from django_ai_sdk.oversight.plugin import (
        TurnRequest,
        apply_oversight,
        build_oversight_hooks,
        merge_oversight_hooks,
        needs_own_turn,
        needs_own_turn_sync,
        run_with_oversight,
        run_with_oversight_sync,
    )
    from django_ai_sdk.oversight.plugin.hooks import (
        EnforceTurnLimitsHook,
        GateQuestionHook,
        JudgeAnswerHook,
        JudgeToolCallsHook,
        MaskToolResultsHook,
        OversightHook,
        RecordUsageHook,
    )
    from django_ai_sdk.oversight.retention import prune
    from django_ai_sdk.oversight.stream import CONTENT_FILTER, StreamTurn
    from django_ai_sdk.oversight.trial import DryRunResult, dry_run

# Each name loaded on first use, by the module that defines it. Loading them with the
# package would import model clients, Haystack and the app's models with a policy file.
_LAZY = {
    **dict.fromkeys(["forget_actor", "forget_thread", "forget_turns"], "content"),
    **dict.fromkeys(
        ["TurnGuard", "closing_turn", "current_guard", "guarding", "outlived_turn", "unguarded"],
        "guard",
    ),
    "StageResult": "guard.stages",
    "history_for": "history",
    **dict.fromkeys(["Judge", "ask_judge", "build_judge", "judge_label"], "judges.base"),
    **dict.fromkeys(["Classifier", "certainty"], "judges.classifier"),
    **dict.fromkeys(["LocalClassifier", "load_local_classifier"], "judges.local"),
    "TypeSafe": "judges.typesafe",
    **dict.fromkeys(
        ["LogNotifier", "Notifier", "WebhookNotifier", "send_to_notifiers"], "notifiers"
    ),
    "DefaultPolicy": "packs",
    **dict.fromkeys(
        [
            "AnswerRelevance",
            "ContextRelevance",
            "Groundedness",
            "Helpfulness",
            "QualityPack",
            "TakenFrom",
        ],
        "packs.quality",
    ),
    **dict.fromkeys(["HarmInput", "HarmOutput", "SafetyPack", "SelfHarm"], "packs.safety"),
    **dict.fromkeys(
        [
            "EXFILTRATION_PATTERNS",
            "INJECTION_PATTERNS",
            "Exfiltration",
            "PromptInjection",
            "SecurityPack",
        ],
        "packs.security",
    ),
    **dict.fromkeys(
        [
            "DEFAULT_PII_KINDS",
            "PII_PATTERNS",
            "SECRET_PATTERNS",
            "mask",
            "mask_all",
            "mask_secrets",
        ],
        "patterns",
    ),
    **dict.fromkeys(
        [
            "TurnRequest",
            "apply_oversight",
            "build_oversight_hooks",
            "merge_oversight_hooks",
            "needs_own_turn",
            "needs_own_turn_sync",
            "run_with_oversight",
            "run_with_oversight_sync",
        ],
        "plugin",
    ),
    **dict.fromkeys(
        [
            "EnforceTurnLimitsHook",
            "GateQuestionHook",
            "JudgeAnswerHook",
            "JudgeToolCallsHook",
            "MaskToolResultsHook",
            "OversightHook",
            "RecordUsageHook",
        ],
        "plugin.hooks",
    ),
    "prune": "retention",
    **dict.fromkeys(["CONTENT_FILTER", "StreamTurn"], "stream"),
    **dict.fromkeys(["DryRunResult", "dry_run"], "trial"),
}


def __getattr__(name: str) -> typing.Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "CONTENT_FILTER",
    "EmptyRetrievalLens",
    "LanguageLens",
    "LengthLens",
    "PII_PATTERNS",
    "RefusalLens",
    "SECRET_PATTERNS",
    "ToolErrorsLens",
    "DEFAULT_PII_KINDS",
    "EXFILTRATION_PATTERNS",
    "INJECTION_PATTERNS",
    "LLMJudge",
    "PIILens",
    "Alert",
    "Answer",
    "AnswerRelevance",
    "Block",
    "Capture",
    "CheckLens",
    "Choice",
    "Classifier",
    "ContextRelevance",
    "DefaultPolicy",
    "DryRunResult",
    "EnforceTurnLimitsHook",
    "Entities",
    "Exfiltration",
    "Flag",
    "GateQuestionHook",
    "Groundedness",
    "HarmInput",
    "HarmOutput",
    "Helpfulness",
    "Judge",
    "JudgeAnswerHook",
    "JudgeLens",
    "JudgeToolCallsHook",
    "L",
    "Lens",
    "LocalClassifier",
    "LogNotifier",
    "Mask",
    "MaskToolResultsHook",
    "Mode",
    "Notifier",
    "Notify",
    "Outcome",
    "OversightHook",
    "PatternLens",
    "Patterns",
    "Policy",
    "PromptInjection",
    "QualityPack",
    "Queue",
    "RecordUsageHook",
    "ResolvedPolicy",
    "RestrictTools",
    "Rule",
    "SafetyPack",
    "Score",
    "SecretsLens",
    "SecurityPack",
    "SelfHarm",
    "Severity",
    "Stage",
    "StageResult",
    "StreamTurn",
    "Subject",
    "TakenFrom",
    "TextSpan",
    "TurnContext",
    "TurnGuard",
    "TurnRequest",
    "TypeSafe",
    "Violation",
    "WebhookNotifier",
    "add_policy_loader",
    "apply_oversight",
    "ask_judge",
    "build_judge",
    "build_oversight_hooks",
    "build_policy",
    "certainty",
    "closing_turn",
    "current_guard",
    "dry_run",
    "forget_actor",
    "forget_thread",
    "forget_turns",
    "guarding",
    "history_for",
    "invalidate_policy",
    "judge_label",
    "load_local_classifier",
    "mask",
    "mask_all",
    "mask_secrets",
    "merge_oversight_hooks",
    "needs_own_turn",
    "needs_own_turn_sync",
    "outlived_turn",
    "pick",
    "prune",
    "register_policy",
    "register_validator",
    "run_with_oversight",
    "run_with_oversight_sync",
    "send_to_notifiers",
    "unguarded",
    "unregister_policy",
]
