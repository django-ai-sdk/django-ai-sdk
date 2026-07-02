from __future__ import annotations

from typing import TYPE_CHECKING

from agents.models.interface import Model, ModelProvider
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class NebulModelProvider(ModelProvider):
    def __init__(self, model: str, client: AsyncOpenAI) -> None:
        super().__init__()
        self.model = model
        self.client = client

    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or self.model,
            openai_client=self.client,
        )
