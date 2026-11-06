"""Built-in lenses: subclass one, name it, and it is ready to use.

Pattern presets find spans, so a Mask rule can mask what they find::

    class InputPII(PIILens):
        name = "pii_input"
        kinds = [*DEFAULT_PII_KINDS, "bsn"]

Check presets answer structural facts about the turn in plain Python: its length,
its language, empty retrieval and failed tools. None of them calls a model, so
they add no cost and send no data out. `RefusalLens` is a judge lens: the shipped
patterns first, then `classifier?` when one is defined.

`PRESETS` names each one for the JSON form (`{"kind": "pattern", "preset": "pii"}`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.judges.patterns import Patterns
from django_ai_sdk.oversight.lenses import CheckLens, JudgeLens, Lens, PatternLens, Score
from django_ai_sdk.oversight.patterns import (
    DEFAULT_PII_KINDS,
    PII_PATTERNS,
    PII_VERIFY,
    REFUSAL_PATTERNS,
    SECRET_PATTERNS,
    WORD,
)
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from django_ai_sdk.oversight.patterns import PatternTable
    from django_ai_sdk.oversight.types import TurnContext

# Don't guess the language of texts shorter than this many words.
MIN_LANGUAGE_WORDS = 3


class PIILens(PatternLens):
    """Personal data by kind (`patterns.PII_PATTERNS`), checked by checksum where it can be.

    `kinds` defaults to the kinds that mean the same in every country
    (DEFAULT_PII_KINDS): email, iban (mod-97 checked), credit_card (Luhn checked),
    phone (international format) and ip_address. Name country kinds to add them
    (COUNTRY_PII_KINDS): bsn (11-check), phone_nl and postcode_nl. Your own
    `patterns` come after the kinds, without a checksum.
    """

    name = "pii"
    kinds: list[str] = list(DEFAULT_PII_KINDS)  # noqa: RUF012
    verify = PII_VERIFY
    flags = 0
    # A card number can also look like a phone number: the first kind wins.
    exclusive = True

    def pattern_table(self) -> PatternTable:
        # No kinds means the default kinds.
        kinds = self.kinds or DEFAULT_PII_KINDS
        unknown = sorted(set(kinds) - set(PII_PATTERNS))
        if unknown:
            raise ImproperlyConfigured(
                f"Unknown personal data kinds {unknown}; known: {sorted(PII_PATTERNS)}"
            )
        own = self.patterns if isinstance(self.patterns, Mapping) else {"": list(self.patterns)}
        return {**{kind: PII_PATTERNS[kind] for kind in kinds}, **own}


class SecretsLens(PatternLens):
    """API keys, tokens and private keys (`patterns.SECRET_PATTERNS`), as in a pasted message."""

    name = "secrets"
    patterns = SECRET_PATTERNS
    flags = 0


class RefusalLens(JudgeLens):
    """Whether the answer declines. Many refusals signal poor quality, not safety."""

    name = "refusal"
    label = "Refusal"
    question = "Does the assistant decline to answer?"
    yes = "Declines, deflects or says it cannot help."
    no = "Answers, even partly."
    subject = Subject.OUTPUT
    # The shipped patterns answer a plain refusal, so this works without a classifier.
    judges = [Patterns(REFUSAL_PATTERNS), "classifier?"]  # noqa: RUF012


class LengthLens(CheckLens):
    """Whether the subject is longer than `max_chars`."""

    name = "length"
    max_chars: int = 0

    async def check(self, text: str, view: TurnContext) -> Answer:
        return Answer(value=len(text) > self.max_chars, rationale=f"{len(text)} characters")

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.max_chars, int) or self.max_chars < 1:
            raise ImproperlyConfigured(f"Length lens {self.name!r} needs `max_chars` of 1 or more.")


# The most frequent words per language, enough to tell them apart.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and is are was to of in that it for you with on this have be not".split()),
    "nl": frozenset("de het een en is van dat niet je ik op te zijn met voor maar wat".split()),
    "de": frozenset("der die das und ist nicht ich sie es zu mit den von ein eine auf".split()),
    "fr": frozenset("le la les et est un une des que pas je vous il de pour dans sur".split()),
    "es": frozenset("el la los las y es un una que no de en por para con se lo".split()),
}


def guess_language(text: str) -> tuple[str, float]:
    """Guess (language code, confidence) from stopwords. ("", 0) when unsure."""
    words = [word.lower() for word in WORD.findall(text)]
    if len(words) < MIN_LANGUAGE_WORDS:
        return "", 0.0
    counts = {lang: sum(word in stop for word in words) for lang, stop in _STOPWORDS.items()}
    lang, best = max(counts.items(), key=lambda item: item[1])
    if not best:
        return "", 0.0
    return lang, round(best / sum(counts.values()), 3)


class LanguageLens(CheckLens):
    """Whether the answer is in another language than the question. Abstains when unsure.

    For the subject's language as a Choice, write a CheckLens over `guess_language`.
    """

    name = "language"
    subject = Subject.EXCHANGE

    async def check(self, text: str, view: TurnContext) -> Answer | None:
        asked, _ = guess_language(view.input)
        replied, confidence = guess_language(view.output)
        if not asked or not replied:
            return None
        return Answer(
            value=asked != replied,
            confidence=confidence,
            rationale=f"asked in {asked}, answered in {replied}",
        )


class EmptyRetrievalLens(CheckLens):
    """Whether the turn called a tool but found no sources."""

    name = "empty_retrieval"
    label = "Nothing retrieved"
    subject = Subject.SOURCES

    def applies(self, view: TurnContext) -> bool:
        # Runs on any tool call: an empty search leaves no grounding to judge.
        return bool(view.tool_calls)

    async def check(self, text: str, view: TurnContext) -> Answer:
        found = len(view.fresh_sources())
        return Answer(value=bool(view.tool_calls) and not found, rationale=f"{found} sources")


class ToolErrorsLens(CheckLens):
    """The share of the turn's tool calls that failed. `L("tool_errors") > 0`: any did."""

    name = "tool_errors"
    label = "Tool failures"
    answer = Score
    subject = Subject.EXCHANGE

    async def check(self, text: str, view: TurnContext) -> Answer | None:
        if not view.tool_calls:
            return None
        failed = sum(call.error for call in view.tool_calls)
        return Answer(value=failed / len(view.tool_calls), rationale=f"{failed} failed")


PRESETS: dict[str, type[Lens]] = {
    cls.name: cls
    for cls in (
        PIILens,
        SecretsLens,
        RefusalLens,
        LengthLens,
        LanguageLens,
        EmptyRetrievalLens,
        ToolErrorsLens,
    )
}


__all__ = [
    "PRESETS",
    "EmptyRetrievalLens",
    "LanguageLens",
    "LengthLens",
    "PIILens",
    "RefusalLens",
    "SecretsLens",
    "ToolErrorsLens",
    "guess_language",
]
