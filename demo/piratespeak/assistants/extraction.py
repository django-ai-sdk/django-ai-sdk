from django.conf import settings
from django_ai_sdk import Assistant
from django_ai_sdk.adapters.base import Run
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret


class PirateExtractionAssistant(Assistant):
    """Extraction assistant for the demo."""

    name = "Extraction Assistant"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True

    async def get_pipeline_adapter(
        self, thread_id: str | None = None, **kwargs: object
    ) -> Run:
        generator = OpenAIChatGenerator(
            model=self.model,
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )
        return Run(generator=generator, model=self.model)
