from __future__ import annotations

from django_ai_sdk.oversight import (
    Alert,
    Choice,
    JudgeLens,
    L,
    LensPack,
    Maybe,
    Mode,
    Notify,
    PatternJudge,
    Policy,
    QualityPack,
    Rule,
    RulePack,
    SecurityPack,
    Stage,
    Subject,
    register_policy,
)

LEGAL_WORDS = {
    "legal": [
        r"\b(?:klacht(?:en)?|aansprakelijk(?:heid)?|schadevergoeding|letselschade)\b",
        r"\b(?:advocaat|rechtszaak|tuchtklacht|datalek)\b",
        r"\b(?:complaint|lawyer|lawsuit|liability|data breach)\b",
    ]
}


class LegalRisk(JudgeLens):
    name = "legal_risk"
    label = "Legal risk"
    question = (
        "Does the message mention a complaint, a liability or damages claim, "
        "a lawyer, a lawsuit or a data breach?"
    )
    subject = Subject.QUESTION
    stages = (Stage.GATE,)
    judges = [PatternJudge(LEGAL_WORDS), Maybe("judge")]


class UserTone(JudgeLens):
    name = "user_tone"
    label = "Tone"
    answer = Choice
    question = "What is the tone of the user's message?"
    labels = ["neutral", "friendly", "confused", "worried", "frustrated", "angry"]
    default = "neutral"
    subject = Subject.QUESTION
    stages = (Stage.REFLECT,)
    judges = [Maybe("judge")]


@register_policy
class StudioPolicy(Policy):
    name = "studio"
    label = "Studio"
    description = (
        "Checks every question, tool call and answer for prompt injection, "
        "secrets, personal data and legal risk."
    )
    lenses = [
        LensPack(
            SecurityPack.lenses,
            "injection_question",
            "secrets_question",
            "pii_question",
            "secrets_tool_call",
            "pii_answer",
        ),
        LegalRisk,
        LensPack(QualityPack.lenses, "helpfulness"),
        UserTone,
    ]
    rules = [
        RulePack(SecurityPack.rules, "injection_in_question", mode=Mode.ENFORCE),
        RulePack(
            SecurityPack.rules,
            "secrets_in_question",
            mode=Mode.ENFORCE,
            notice="A key or password in your message was hidden from the assistant.",
        ),
        RulePack(
            SecurityPack.rules,
            "pii_in_question",
            mode=Mode.ENFORCE,
            notice="Personal data in your message was hidden from the assistant.",
        ),
        RulePack(SecurityPack.rules, "secrets_in_tool_call", mode=Mode.ENFORCE),
        RulePack(SecurityPack.rules, "pii_in_answer", mode=Mode.REPORT),
        Rule(
            "alert_legal",
            when=L("legal_risk").is_true,
            then=[Notify(to=["legal"], severity="high"), Alert("high")],
            mode=Mode.ENFORCE,
            description="Legal hears about complaints, claims and data breaches.",
            notice="The legal team has been informed about this conversation.",
        ),
    ]
