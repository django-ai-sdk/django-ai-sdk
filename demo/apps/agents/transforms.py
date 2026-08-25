from __future__ import annotations

import json
from typing import Any

from django_ai_sdk.common import ChatMessage
from django_ai_sdk.files.transforms import BaseTransform
from django_ai_sdk.memories.schemas import DocumentExtraction, Predicate

MAX_CONTENT_LENGTH = 10_000

_PREDICATES = "\n".join(f"- {p}" for p in Predicate.__members__.values())

EXTRACTION_PROMPT = f"""You are a structured information extractor.

Given the document text below, extract:

1) A concise summary.
2) A list of keywords.
3) All named entities (NER) with their types (e.g., PERSON, ORG, LOCATION, DATE, etc.).
4) A list of facts.

Allowed predicate values:

{_PREDICATES}

Guidelines:
- Map real relationships to these predicate IDs when appropriate.
- For each fact, provide a natural sentence under "text" that conveys the fact clearly.
- Use the specific document text that supports the fact as "evidence".
- Only use predicates from the allowed list when possible; if necessary for a fact not covered, choose the closest controlled predicate.

Return the output strictly in the DocumentExtraction format.
"""


class DocumentExtractionTransform(BaseTransform):
    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def run(self, data: Any, **kwargs: Any) -> DocumentExtraction | Any:
        content = data if isinstance(data, str) else json.dumps(data, default=str)
        result = await self._agent.run(
            [ChatMessage(role="user", content=content[:MAX_CONTENT_LENGTH])],
            system_prompt=EXTRACTION_PROMPT,
            response_format=DocumentExtraction,
            tools=False,
        )
        return result if result is not None else data
