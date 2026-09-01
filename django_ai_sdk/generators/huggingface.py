from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.generators.base import (
    build_kwargs,
    requires_generator,
    resolve_secret,
    resolve_setting,
)

if TYPE_CHECKING:
    from haystack_integrations.components.generators.huggingface_api import (
        HuggingFaceAPIChatGenerator,
    )
    from haystack_integrations.components.generators.transformers import TransformersChatGenerator


# Neither Hugging Face generator takes a schema parameter.
SCHEMA_KWARGS: dict[str, str | None] = {
    "HuggingFaceAPIChatGenerator": None,
    "TransformersChatGenerator": None,
}


def huggingface_api_chat(**kwargs: Any) -> HuggingFaceAPIChatGenerator:
    """Hugging Face Inference API generator wired to Django settings.

    A `model` is turned into serverless inference params; `HUGGINGFACE_API_URL`
    selects a dedicated inference endpoint instead.
    """
    with requires_generator("huggingface"):
        from haystack_integrations.components.generators.huggingface_api import (
            HuggingFaceAPIChatGenerator,
        )

    url = resolve_setting("HUGGINGFACE_API_URL")
    model = kwargs.pop("model", None)
    defaults = {
        "api_type": "text_generation_inference" if url else "serverless_inference_api",
        "api_params": {"url": url} if url else {"model": model},
        "token": resolve_secret("HUGGINGFACE_API_KEY"),
    }
    return HuggingFaceAPIChatGenerator(**build_kwargs(defaults, kwargs))


def transformers_chat(**kwargs: Any) -> TransformersChatGenerator:
    """Local Hugging Face transformers generator wired to Django settings."""
    with requires_generator("huggingface"):
        from haystack_integrations.components.generators.transformers import (
            TransformersChatGenerator,
        )

    return TransformersChatGenerator(
        **build_kwargs(
            {"token": resolve_secret("HUGGINGFACE_API_KEY")},
            kwargs,
        )
    )
