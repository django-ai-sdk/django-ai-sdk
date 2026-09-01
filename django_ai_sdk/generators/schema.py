from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.generators.anthropic import SCHEMA_KWARGS as ANTHROPIC_SCHEMA_KWARGS
from django_ai_sdk.generators.azure import SCHEMA_KWARGS as AZURE_SCHEMA_KWARGS
from django_ai_sdk.generators.huggingface import SCHEMA_KWARGS as HUGGINGFACE_SCHEMA_KWARGS
from django_ai_sdk.generators.mistral import SCHEMA_KWARGS as MISTRAL_SCHEMA_KWARGS
from django_ai_sdk.generators.ollama import SCHEMA_KWARGS as OLLAMA_SCHEMA_KWARGS
from django_ai_sdk.generators.openai import SCHEMA_KWARGS as OPENAI_SCHEMA_KWARGS
from django_ai_sdk.generators.openrouter import SCHEMA_KWARGS as OPENROUTER_SCHEMA_KWARGS

if TYPE_CHECKING:
    from pydantic import BaseModel

SCHEMA_KWARGS: dict[str, str | None] = {
    **OPENAI_SCHEMA_KWARGS,
    **AZURE_SCHEMA_KWARGS,
    **ANTHROPIC_SCHEMA_KWARGS,
    **HUGGINGFACE_SCHEMA_KWARGS,
    **MISTRAL_SCHEMA_KWARGS,
    **OLLAMA_SCHEMA_KWARGS,
    **OPENROUTER_SCHEMA_KWARGS,
}

# An unlisted generator default.
DEFAULT_SCHEMA_KWARG = "response_format"


def schema_kwargs(generator: Any, schema: type[BaseModel]) -> dict[str, Any]:
    """Return the generation kwargs that make `generator` answer in `schema`."""
    for cls in type(generator).__mro__:
        if cls.__name__ in SCHEMA_KWARGS:
            kwarg = SCHEMA_KWARGS[cls.__name__]
            if kwarg is None:
                raise ValueError(f"{cls.__name__} takes no structured-output kwarg.")
            return {kwarg: schema}
    return {DEFAULT_SCHEMA_KWARG: schema}
