from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.conf import settings
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from django_ai_sdk.adapters.base import Run
from django_ai_sdk.logger import get_logger
from django_ai_sdk.prompts import prompt

if TYPE_CHECKING:
    from django_ai_sdk.artifacts import ArtifactSchema
    from django_ai_sdk.common import ChatMessage

logger = get_logger(__name__)


EXTRACTION_MODEL_SETTING = "AI_SDK_EXTRACTION_MODEL"


def llm_generator() -> Any:
    """
    Build a dedicated generator
    """
    model = getattr(settings, EXTRACTION_MODEL_SETTING, None) or getattr(
        settings, "AI_SDK_DEFAULT_MODEL", "gpt-4o-mini"
    )
    return OpenAIChatGenerator(
        model=model,
        api_key=Secret.from_token(settings.OPENAI_API_KEY),
        api_base_url=getattr(settings, "OPENAI_API_URL", None),
    )


async def extract_artifact(
    messages: list[ChatMessage],
    output_model: type[ArtifactSchema],
) -> ArtifactSchema | None:
    """
    Run a one-shot structured extraction
    """
    extractor = Run(generator=llm_generator())
    try:
        return await extractor.run(
            messages,
            system_prompt=prompt("""\
                You are a structured data extractor.
                Extract the requested information from the conversation.
                Return only the structured data — no commentary, no preamble.
            """),
            response_format=output_model,
        )
    except Exception as e:
        logger.warning("Artifact extraction failed for {}: {}", output_model.__name__, e)
        return None
