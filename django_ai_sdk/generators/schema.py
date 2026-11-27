from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

SCHEMA_KWARGS: dict[str, str | None] = {
    "OpenAIResponsesChatGenerator": "text_format",
    "OpenAIChatGenerator": "response_format",
    "OllamaChatGenerator": None,
    "AnthropicChatGenerator": None,
    "HuggingFaceAPIChatGenerator": None,
    "TransformersChatGenerator": None,
}

DEFAULT_SCHEMA_KWARG = "response_format"


def schema_kwargs(generator: Any, schema: type[BaseModel]) -> dict[str, Any]:
    for cls in type(generator).__mro__:
        if cls.__name__ in SCHEMA_KWARGS:
            kwarg = SCHEMA_KWARGS[cls.__name__]
            if kwarg is None:
                raise ValueError(f"{cls.__name__} takes no structured-output kwarg at run time.")
            return {kwarg: schema}
    return {DEFAULT_SCHEMA_KWARG: schema}
