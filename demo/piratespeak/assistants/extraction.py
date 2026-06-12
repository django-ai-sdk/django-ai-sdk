from django.conf import settings
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.openai import OpenAIRunnable
from openai import AsyncOpenAI


class PirateExtractionAssistant(Assistant):
    """Extraction assistant for the demo."""

    name = "Extraction Assistant"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True

    async def get_pipeline_adapter(
        self, thread_id: str | None = None, **kwargs: object
    ) -> OpenAIRunnable:
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL,
        )
        return OpenAIRunnable(client=client, model=self.model)
