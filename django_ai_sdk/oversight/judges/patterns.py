"""Regexes and keywords as a judge. Deterministic, free, and fast enough for the gate.

A pattern lens is answered by one. In a judge lens it is the cheap first judge:

    judges = [Patterns(INJECTION_PATTERNS), "classifier?"]

A hit is sure (confidence 1). A miss has `miss_confidence`, by default below a
lens's `min_confidence`, so the next judge is asked.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.judges.base import Judge
from django_ai_sdk.oversight.patterns import _compile_patterns, _regex_spans, _scan, _verified
from django_ai_sdk.oversight.types import Answer, TextSpan

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.patterns import PatternTable, Verifier
    from django_ai_sdk.oversight.types import TurnContext

# Generous, since the text each regex reads is capped (`patterns.MAX_SCAN_CHARS`).
PATTERNS_TIMEOUT = 2.0
# Unsure, so a judge after it is asked.
MISS_CONFIDENCE = 0.5


class Patterns(Judge):
    """Finds labelled spans with regexes and whole-word keywords.

    `patterns` maps a label to its regexes; a plain list labels every hit with the
    lens's name. `keywords` maps a label to words, and the label with the most hits
    comes first. `verify` maps a label to a check its matches must pass.
    `exclusive` gives every stretch of text one label: the first pattern that
    matches it keeps it. Personal data needs that, since a card number can also
    look like a phone number.
    """

    timeout = PATTERNS_TIMEOUT

    def __init__(
        self,
        patterns: PatternTable = (),
        *,
        keywords: PatternTable = (),
        verify: Mapping[str, Verifier] | None = None,
        flags: int = re.IGNORECASE,
        exclusive: bool = False,
        miss_confidence: float = MISS_CONFIDENCE,
        name: str = "",
    ) -> None:
        groups = keywords.items() if isinstance(keywords, Mapping) else [("", keywords)]
        self.patterns = _compile_patterns(patterns, flags)
        # One whole-word regex per label, so a label's hits can be counted.
        self.keywords = [
            (label, re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", flags))
            for label, group in groups
            if (words := [group] if isinstance(group, str) else list(group))
        ]
        self.verify = verify or {}
        self.exclusive = exclusive
        self.miss_confidence = miss_confidence
        self.name = name or "patterns"

    def labels(self) -> list[str]:
        """The labels this judge looks for, so a miss can name them."""
        return list(dict.fromkeys(label for label, _ in (*self.patterns, *self.keywords) if label))

    def find(self, text: str, default_label: str = "") -> list[TextSpan]:
        """Every span in `text`, scanned in windows. Unlabelled ones get `default_label`."""
        return _scan(text, lambda chunk: self._find(chunk, default_label))

    def _find(self, text: str, default_label: str) -> list[TextSpan]:
        if self.verify or self.exclusive:
            spans = self._checked(text, default_label)
        else:
            # By position, as `_checked` gives them, so a Choice picks the same label either way.
            spans = sorted(
                _regex_spans(self.patterns, text, default_label), key=lambda span: span.start
            )
        # The keyword label with the most hits comes first, so a Choice picks it.
        hits = [_regex_spans([pair], text, default_label) for pair in self.keywords]
        return spans + [span for found in sorted(hits, key=len, reverse=True) for span in found]

    def _checked(self, text: str, default_label: str) -> list[TextSpan]:
        """Spans that pass their verifier, and with `exclusive`, don't overlap earlier ones."""
        spans: list[TextSpan] = []
        taken: list[tuple[int, int]] = []
        for label, pattern in self.patterns:
            for start, end in _verified(pattern, self.verify.get(label), text):
                if self.exclusive and any(start < until and since < end for since, until in taken):
                    continue
                taken.append((start, end))
                spans.append(TextSpan(start=start, end=end, label=label or default_label))
        return sorted(spans, key=lambda span: span.start)

    async def answer(self, lens: Lens, view: TurnContext) -> Answer | None:
        text = view.text(lens.subject)
        if not text:
            return None
        # In a thread, so a long text can't block the event loop.
        spans = await asyncio.to_thread(self.find, text, lens.name)
        labels = list(dict.fromkeys(span.label for span in spans))
        notes = [f"found {_listed(labels)}"] if labels else []
        if not labels and (kinds := self.labels()):
            notes.append(f"found none of: {_listed(kinds)}")
        return Answer(
            value=lens.answer.from_detections(lens, labels),
            confidence=1.0 if spans else self.miss_confidence,
            spans=spans,
            rationale="; ".join(notes),
        )


def _listed(kinds: list[str]) -> str:
    return ", ".join(kind.replace("_", " ") for kind in kinds)


__all__ = ["MISS_CONFIDENCE", "Patterns"]
