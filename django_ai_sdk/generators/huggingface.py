from __future__ import annotations

from typing import Any

from haystack_integrations.components.generators.huggingface_api import (
    HuggingFaceAPIChatGenerator,
)

from django_ai_sdk.generators.base import build_kwargs, resolve_secret, resolve_setting


def huggingface_api_chat(**kwargs: Any) -> HuggingFaceAPIChatGenerator:
    """Hugging Face Inference API generator wired to Django settings.

    A `model` is turned into serverless inference params; `HUGGINGFACE_API_URL`
    selects a dedicated inference endpoint instead.
    """
    url = resolve_setting("HUGGINGFACE_API_URL")
    model = kwargs.pop("model", None)
    defaults = {
        "api_type": "text_generation_inference" if url else "serverless_inference_api",
        "api_params": {"url": url} if url else {"model": model},
        "token": resolve_secret("HUGGINGFACE_API_KEY"),
    }
    return HuggingFaceAPIChatGenerator(**build_kwargs(defaults, kwargs))
