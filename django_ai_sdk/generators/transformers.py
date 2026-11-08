from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.transformers import TransformersChatGenerator

from django_ai_sdk.generators.base import build_kwargs, resolve_secret
from django_ai_sdk.oversight.calls import watched_factory


@watched_factory
def transformers_chat(**kwargs: Any) -> TransformersChatGenerator:
    """Local Hugging Face transformers generator wired to Django settings."""
    return TransformersChatGenerator(
        **build_kwargs(
            {"token": resolve_secret("HUGGINGFACE_API_KEY")},
            kwargs,
        )
    )
