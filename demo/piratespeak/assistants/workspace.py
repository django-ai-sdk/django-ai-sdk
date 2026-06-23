from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django_ai_sdk import (
    ApprovalCardArtifact,
    Assistant,
    ChainOfThoughtArtifact,
    CodeBlockArtifact,
    ConfirmationArtifact,
    DataTableArtifact,
    ImageArtifact,
    OptionListArtifact,
    PlanArtifact,
    ProgressTrackerArtifact,
    QuestionFlowArtifact,
    SchemaDisplayArtifact,
    SnippetArtifact,
    StackTraceArtifact,
    TaskArtifact,
    TerminalArtifact,
    TestResultsArtifact,
)
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.assistants import auto_register
from django_ai_sdk.common import prompt
from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.db import DbStorageAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from piratespeak.assistants.tools import get_today

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


@auto_register
class WorkspaceAssistant(Assistant):
    name = "Workspace Assistant"
    model = settings.AI_SDK_DEFAULT_MODEL
    instructions = prompt("""\
        You are a professional AI assistant for workplace productivity.
        Help with tasks like drafting messages, summarising content, planning,
        brainstorming, and answering questions clearly and concisely.
        - Be direct and professional.
        - Format output with markdown when it aids readability.
        - Do not fabricate facts; say so when you don't know something.
        When you have gathered enough information to summarise the workspace context,
        call artifact_data_table_artifact() with columns for "Topic", "Summary", and
        "Action" and one row per key item.
        IMPORTANT: after the tool call succeeds (returns artifact_id), reply with ONE
        sentence that briefly describes what the data shows — do NOT repeat the data as text or markdown.
    """)

    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter
    max_history = 20

    tools: list = [get_today]
    artifacts: list = [
        ApprovalCardArtifact,
        ChainOfThoughtArtifact,
        CodeBlockArtifact,
        ConfirmationArtifact,
        DataTableArtifact,
        ImageArtifact,
        OptionListArtifact,
        PlanArtifact,
        ProgressTrackerArtifact,
        QuestionFlowArtifact,
        SchemaDisplayArtifact,
        SnippetArtifact,
        StackTraceArtifact,
        TaskArtifact,
        TerminalArtifact,
        TestResultsArtifact,
    ]

    def _build_generator(self) -> OpenAIChatGenerator:
        return OpenAIChatGenerator(
            model=self.get_model(),
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
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
        artifact_tools = await self.get_artifact_tools(thread_id=thread_id or "", user=user)

        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=[*tools, *artifact_tools],
            ),
            generator=generator,
        )

        return Stream(
            pipeline=tool_agent.pipeline(),
            generator=generator,
            storage_adapter=storage_adapter,
        )
