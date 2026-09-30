from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run
from django_ai_sdk.generators import openai_responses_chat

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


class PirateExtractionAgent(Agent):
    """Extraction agent for the demo."""

    name = "Extraction Agent"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True
    llm = openai_responses_chat
    llm_kwargs = {"reasoning": {"effort": "low", "summary": "auto"}}

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self.get_llm(), model=self.model)

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self.get_llm(), model=self.model)
