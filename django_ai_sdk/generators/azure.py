from __future__ import annotations

from typing import Any

from haystack.components.generators.chat import (
    AzureOpenAIChatGenerator,
    AzureOpenAIResponsesChatGenerator,
)

from django_ai_sdk.generators.base import build_kwargs, resolve_secret
from django_ai_sdk.utils import resolve_setting


def _deployment(kwargs: dict[str, Any]) -> str | None:
    """Azure names the model a deployment, so accept either spelling."""
    return (
        kwargs.pop("azure_deployment", None)
        or kwargs.pop("model", None)
        or resolve_setting("AZURE_OPENAI_DEPLOYMENT")
    )


def azure_openai_chat(**kwargs: Any) -> AzureOpenAIChatGenerator:
    """Azure OpenAI Chat Completions generator wired to Django settings."""
    return AzureOpenAIChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": resolve_setting("AZURE_OPENAI_ENDPOINT"),
                "api_version": resolve_setting("AZURE_OPENAI_API_VERSION"),
                "azure_deployment": _deployment(kwargs),
            },
            kwargs,
        )
    )


def azure_openai_responses_chat(**kwargs: Any) -> AzureOpenAIResponsesChatGenerator:
    """Azure OpenAI Responses API generator wired to Django settings."""
    return AzureOpenAIResponsesChatGenerator(
        **build_kwargs(
            {
                "api_key": resolve_secret("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": resolve_setting("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": _deployment(kwargs),
            },
            kwargs,
        )
    )
