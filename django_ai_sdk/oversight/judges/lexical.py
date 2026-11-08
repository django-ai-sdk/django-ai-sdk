"""A local judge for RAG lenses, by word overlap. A weak fallback for model judges."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.judges.base import Judge
from django_ai_sdk.oversight.patterns import WORD
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnContext

# Words too common to say anything about support.
_FILLER = frozenset(
    "a an the and or but is are was were be been to of in on at for with by from as it its this "
    "that these those you your we our they their he she his her i me my not no yes do does did "
    "can could will would should may might de het een en of maar is zijn was van in op voor met "
    "door als dat dit die je jij u wij we zij hij ik niet geen wel".split()
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _terms(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if len(w) > 2 and w.lower() not in _FILLER}


class LexicalSupport(Judge):
    """Local RAG checks by word overlap. Nothing leaves the process.

    For OUTPUT or EXCHANGE it measures groundedness: the share of answer sentences
    whose content words mostly appear in one source. For SOURCES it measures
    relevance: the share of the question's content words found in the sources.
    A Score gets the share. A Flag is true when the share is below `threshold`.
    """

    def __init__(
        self,
        *,
        overlap: float = 0.5,
        threshold: float = 0.5,
        confidence: float = 0.6,
        name: str = "",
    ) -> None:
        self.overlap = overlap
        self.threshold = threshold
        # Overlap is a weak signal. Leave room for a judge to overrule it.
        self.confidence = confidence
        self.name = name or "lexical_support"

    async def answer(self, lens: Lens, view: TurnContext) -> Answer | None:
        sources = [_terms(s.content) for s in view.fresh_sources() if s.content]
        if not sources:
            return None
        if lens.subject is Subject.SOURCES:
            asked = _terms(view.input)
            if not asked:
                return None
            pooled = set().union(*sources)
            share = len(asked & pooled) / len(asked)
            detail = f"{len(asked & pooled)} of {len(asked)} question terms in sources"
        else:
            sentences = [s for s in _SENTENCE.split(view.output) if len(_terms(s)) >= 3]
            if not sentences:
                return None
            supported = sum(
                any(len(_terms(s) & src) / len(_terms(s)) >= self.overlap for src in sources)
                for s in sentences
            )
            share = supported / len(sentences)
            detail = f"{supported} of {len(sentences)} sentences supported"
        value: Any = round(share, 3) if lens.answer.kind == "score" else share < self.threshold
        return Answer(value=value, confidence=self.confidence, rationale=detail)


__all__ = ["LexicalSupport"]
