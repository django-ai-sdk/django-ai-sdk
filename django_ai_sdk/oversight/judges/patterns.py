"""Regexes and keywords as a judge: deterministic, free, and fast enough for the gate."""

from __future__ import annotations

import asyncio
import bisect
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.judges.base import Judge
from django_ai_sdk.oversight.patterns import compile_patterns, valid_spans
from django_ai_sdk.oversight.types import Answer, TextSpan

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.patterns import PatternTable, Verifier
    from django_ai_sdk.oversight.types import TurnContext

PATTERNS_TIMEOUT = 2.0
# Unsure, so a judge after it is asked.
MISS_CONFIDENCE = 0.5


class PatternJudge(Judge):
    """Finds labelled spans with regexes and whole-word keywords.

    A hit is sure; a miss has `miss_confidence`, below `min_confidence` by default,
    so the next judge is asked. The first pattern to match a stretch of text keeps
    it, since a card number can also look like a phone number.
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
        self.patterns = compile_patterns(patterns, flags) + [
            (label, re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", flags))
            for label, group in groups
            if (words := [group] if isinstance(group, str) else list(group))
        ]
        self.verify = verify or {}
        self.miss_confidence = miss_confidence

    def labels(self) -> list[str]:
        return list(dict.fromkeys(label for label, _ in self.patterns if label))

    def find(self, text: str, default_label: str = "") -> list[TextSpan]:
        spans: list[TextSpan] = []
        # Sorted and never overlapping, so only the neighbours can overlap.
        taken: list[tuple[int, int]] = []
        for label, pattern in self.patterns:
            for start, end in valid_spans(pattern, self.verify.get(label), text):
                at = bisect.bisect(taken, (start, end))
                if (at and taken[at - 1][1] > start) or (at < len(taken) and taken[at][0] < end):
                    continue
                taken.insert(at, (start, end))
                spans.append(TextSpan(start=start, end=end, label=label or default_label))
        return sorted(spans, key=lambda span: span.start)

    async def answer(self, lens: Lens, context: TurnContext) -> Answer | None:
        text = context.text(lens.subject)
        if not text:
            return None
        spans = await asyncio.to_thread(self.find, text, lens.name)
        labels = list(dict.fromkeys(span.label for span in spans))
        notes = [f"found {', '.join(k.replace('_', ' ') for k in labels)}"] if labels else []
        if not labels and (sought := self.labels()):
            notes.append(f"found none of: {', '.join(k.replace('_', ' ') for k in sought)}")
        return Answer(
            value=lens.answer.from_detections(lens, labels),
            confidence=1.0 if spans else self.miss_confidence,
            spans=spans,
            rationale="; ".join(notes),
        )
