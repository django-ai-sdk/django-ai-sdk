"""Regexes and keywords as a judge. Deterministic, free, and fast enough for the gate.

A pattern lens is answered by one. In a judge lens it is the cheap first judge:

    judges = [Patterns(INJECTION_PATTERNS), "classifier?"]

A hit is sure (confidence 1). A miss has `miss_confidence`, by default below a
lens's `min_confidence`, so the next judge is asked.
"""

from __future__ import annotations

import asyncio
import bisect
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.judges.base import Judge
from django_ai_sdk.oversight.patterns import compile_patterns, verified
from django_ai_sdk.oversight.types import Answer, TextSpan

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.patterns import PatternTable, Verifier
    from django_ai_sdk.oversight.types import TurnContext

PATTERNS_TIMEOUT = 2.0
# Unsure, so a judge after it is asked.
MISS_CONFIDENCE = 0.5


class Patterns(Judge):
    """Finds labelled spans with regexes and whole-word keywords.

    `patterns` maps a label to its regexes; a plain list labels every hit with the
    lens's name. `keywords` maps a label to whole words. `verify` maps a label to a
    check its matches must pass. Every stretch of text gets one label: the first
    pattern that matches it keeps it, since a card number can also look like a
    phone number.
    """

    timeout = PATTERNS_TIMEOUT
    name = "patterns"

    def __init__(
        self,
        patterns: PatternTable = (),
        *,
        keywords: PatternTable = (),
        verify: Mapping[str, Verifier] | None = None,
        flags: int = re.IGNORECASE,
        miss_confidence: float = MISS_CONFIDENCE,
        name: str = "",
        timeout: float | None = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        groups = keywords.items() if isinstance(keywords, Mapping) else [("", keywords)]
        # Keywords after patterns, as one whole-word regex per label.
        self.patterns = compile_patterns(patterns, flags) + [
            (label, re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", flags))
            for label, group in groups
            if (words := [group] if isinstance(group, str) else list(group))
        ]
        self.verify = verify or {}
        self.miss_confidence = miss_confidence

    def labels(self) -> list[str]:
        """The labels this judge looks for, so a miss can name them."""
        return list(dict.fromkeys(label for label, _ in self.patterns if label))

    def find(self, text: str, default_label: str = "") -> list[TextSpan]:
        """Every verified span in `text`, in order. Unlabelled ones get `default_label`."""
        spans: list[TextSpan] = []
        # Spans kept so far, sorted and never overlapping, so only the neighbours can overlap.
        taken: list[tuple[int, int]] = []
        for label, pattern in self.patterns:
            for start, end in verified(pattern, self.verify.get(label), text):
                at = bisect.bisect(taken, (start, end))
                if (at and taken[at - 1][1] > start) or (at < len(taken) and taken[at][0] < end):
                    continue
                taken.insert(at, (start, end))
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
