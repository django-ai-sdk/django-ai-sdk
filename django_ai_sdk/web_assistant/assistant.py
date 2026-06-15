from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.conf import settings as django_settings
from django.utils.module_loading import import_string
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.assistant import Assistant
from django_ai_sdk.common import prompt
from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.db import DbStorageAdapter
from django_ai_sdk.suggestions import DefaultSuggestionGenerator

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from .models import WebAssistantSettings


class WebAssistant(Assistant):
    """Assistant whose configuration is loaded from a WebAssistantSettings DB record.

    Constructed on demand by AssistantService — not registered in the class registry.
    Each instance reflects the live DB config at construction time.
    """

    _skip_auto_register = True

    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter

    def __init__(self, config: WebAssistantSettings) -> None:
        self._config = config
        self.name = config.name
        self.model = config.model
        self.instructions = prompt(config.system_prompt or "You are a helpful assistant.")
        self.mcp_servers = list(config.mcp_servers or [])
        self.title_generation = config.title_generation
        self.max_history = config.max_history
        self.file_upload = config.file_upload
        if config.suggestion_enabled:
            self.suggestion_generator = DefaultSuggestionGenerator
        super().__init__()

    @property
    def assistant_id(self) -> str:
        return str(self._config.id)

    def _build_generator(self) -> OpenAIChatGenerator:
        return OpenAIChatGenerator(
            model=self.get_model(),
            api_key=Secret.from_token(django_settings.OPENAI_API_KEY),
            api_base_url=getattr(django_settings, "OPENAI_API_URL", None),
        )

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self._build_generator())

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Stream:
        generator = self._build_generator()
        storage_adapter = await self.get_storage_adapter(thread_id)
        tools = await self.get_tools(thread_id=thread_id or "", user=user)
        mcp_tools = await self.get_mcp_tools(user)
        tools.extend(mcp_tools)

        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=tools,
            ),
            generator=generator,
        )

        return Stream(
            pipeline=tool_agent.pipeline(),
            generator=generator,
            storage_adapter=storage_adapter,
        )

    async def get_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        result = await super().get_tools(thread_id=thread_id, user=user)
        from django_ai_sdk.web_assistant.config import get_tool_registry

        tool_registry = get_tool_registry()
        for key in self._config.tools or []:
            path = tool_registry.get(key)
            if not path:
                continue
            tool_fn = import_string(path)
            items = tool_fn(thread_id=thread_id, user=user)
            if isinstance(items, list):
                result.extend(items)
            else:
                result.append(items)
        return result

    def __repr__(self) -> str:
        return f"<WebAssistant id={self.assistant_id!r} name={self.name!r}>"
