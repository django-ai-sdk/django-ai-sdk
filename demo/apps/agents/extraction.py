from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


class PirateExtractionAgent(Agent):
    """Extraction agent for the demo."""

    name = "Extraction Agent"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True

    def _build_generator(self) -> OpenAIChatGenerator:
        return OpenAIChatGenerator(
            model=self.model,
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self._build_generator(), model=self.model)

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self._build_generator(), model=self.model)
