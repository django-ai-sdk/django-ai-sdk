"""
Note: this is also a temporary tasks file until we have the package ready.
Same as for schemas: we need make sure we have the correct imports and
we must unify Caren version against the public one.
"""

from openai import AsyncOpenAI

from django_ai_sdk.logger import get_logger
from django_ai_sdk.memories.schemas import DocumentExtraction, Predicate
from django_ai_sdk.tracking.utils import track_llm

logger = get_logger(__name__)


def get_llm() -> AsyncOpenAI:
    """Create OpenAI client using settings."""
    from django.conf import settings

    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_URL,
    )


def get_predicates() -> list[str]:
    """Get a list of formatted predicate values."""
    return [f"- {predicate}" for predicate in Predicate.__members__.values()]


MAX_CONTENT_LENGTH = 10_000
PROMPT = f"""You are a structured information extractor.

Given the document text below, extract:

1) A concise summary.
2) A list of keywords.
3) All named entities (NER) with their types (e.g., PERSON, ORG, LOCATION, DATE, etc.).
4) A list of facts.

Allowed predicate values:

{"\n".join(get_predicates())}

Guidelines:
- Map real relationships to these predicate IDs when appropriate.
- For each fact, provide a natural sentence under "text" that conveys the fact clearly.
- Use the specific document text that supports the fact as "evidence".
- Only use predicates from the allowed list when possible; if necessary for a fact not covered, choose the closest controlled predicate.

Return the output strictly in the DocumentExtraction format.
"""


@track_llm
async def extract_document(content: str) -> DocumentExtraction | None:
    """Extract summary, keywords, and facts from content"""

    logger.info("Extracting content metadata")
    client = get_llm()

    try:
        logger.info(f"Prompt: \n\n{PROMPT}\n\n")
        response = await client.beta.chat.completions.parse(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": PROMPT,
                },
                {"role": "user", "content": content[:MAX_CONTENT_LENGTH]},
            ],
            response_format=DocumentExtraction,
        )

        return response.choices[0].message.parsed
    except Exception as e:
        logger.warning(f"Content extraction failed: {e}")
        return None


def get_prompt_metadata(
    content: str,
    extraction: DocumentExtraction,
) -> str:
    """
    Combine original content with extracted summary, facts, and keywords

    Args:
        content: The original document content.
        extraction: DocumentExtraction

    Returns:
        Combined text string formatted for RAG document ingestion.
    """

    parts = [content]

    if extraction.summary:
        parts.append(f"Summary: {extraction.summary}")

    if extraction.facts:
        facts = ". ".join(fact.text for fact in extraction.facts)
        parts.append(f"Key facts: {facts}")

    if extraction.keywords:
        keywords = ", ".join(extraction.keywords)
        parts.append(f"Keywords: {keywords}")

    if extraction.entities:
        entities = ", ".join(f"{ent.text} ({ent.type})" for ent in extraction.entities)
        parts.append(f"Named entities: {entities}")

    return "\n\n".join(parts)
