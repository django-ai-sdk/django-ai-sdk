from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run
from django_ai_sdk.common import prompt
from django_ai_sdk.generators import openai_chat

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


class VisionAgent(Agent):
    """Answers every image question for the SDK (AI_SDK_VISION_AGENT).

    It captions uploaded images (ImageCaptionProcessor) and answers the ask_image
    tool. The image arrives attached to a user message; the text is the caption
    prompt or the question.
    """

    name = "Vision Agent"
    model = settings.AI_SDK_VISION_MODEL
    hidden = True
    llm = openai_chat
    instructions = prompt("""\
        You look at images and report what they show.
        Describe only what is visible, never guess at details you cannot see.
        Transcribe text in an image exactly as written.
        Answer in the language of the question.
    """)

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self.get_llm(), model=self.model)
