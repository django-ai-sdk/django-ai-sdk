from __future__ import annotations

import random
from typing import TYPE_CHECKING, Annotated

from django.conf import settings
from django.utils import timezone
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agents import auto_register
from django_ai_sdk.citations import DefaultCitationFormatter
from django_ai_sdk.common import prompt
from django_ai_sdk.permissions import IsAdminUser
from django_ai_sdk.suggestions import DefaultSuggestionGenerator
from haystack import Pipeline
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from haystack.utils import Secret

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


def pirate_boat_expert(topic: Annotated[str, "Topic about pirate boats"]) -> str:
    """Provide expert lore about pirate boats."""
    return f"Aye! On pirate boats: {topic}! They be swift and full o' tales."


def find_treasure(location: Annotated[str, "Location to search for treasure"]) -> dict:
    """Simulate treasure discovery."""
    treasure_id = random.randint(1000, 9999)
    value = random.randint(5000, 50000)
    return {
        "location": location,
        "treasure_id": treasure_id,
        "value": f"{value} gold coins",
    }


def get_datetime() -> dict:
    """Get current time and date in Europe/Amsterdam timezone."""
    tz = timezone.get_current_timezone()
    nowtz = timezone.now().astimezone(tz)

    return {
        "today": nowtz.date().isoformat(),
        "current_time": nowtz.timetz().isoformat(),
    }


@auto_register
class AgentSwarmAgent(Agent):
    """
    Agent swarm agent with specialized pirate expertise.
    """

    name = "Pirate Agent Swarm"
    description = "An agent swarm agent with specialized pirate expertise."
    model = settings.AI_SDK_DEFAULT_MODEL
    permissions = [IsAdminUser]
    instructions = prompt("""\
        You are a Triage Agent for a crew of pirate specialists.

        Decide whether the user wants:
        - pirate boat expertise (call pirate_boat_expert)
        - treasure finding (call find_treasure)
        - date and time information (call get_datetime)
        - or general pirate help

        Always respond as a pirate and use your tools when appropriate.
        Respond with text or tool calls as needed.
    """)

    citation_formatter_class = DefaultCitationFormatter
    suggestion_generator = DefaultSuggestionGenerator

    async def get_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list:
        """Return Haystack-compatible tools for agent swarm."""
        return [
            self._create_boat_expert_tool(),
            self._create_treasure_tool(),
            self._create_date_tool(),
        ]

    # TODO: convert in utility function
    def _create_boat_expert_tool(self, **kwargs: object) -> Tool:
        """Create Haystack tool for boat expertise."""
        return Tool(
            name="pirate_boat_expert",
            description="Expert knowledge about pirate boats",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Subject about pirate boats",
                    }
                },
                "required": ["topic"],
            },
            function=pirate_boat_expert,
        )

    def _create_treasure_tool(self, **kwargs: object) -> Tool:
        """Create Haystack tool for treasure finding."""
        return Tool(
            name="find_treasure",
            description="Find treasure at a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to search for treasure",
                    }
                },
                "required": ["location"],
            },
            function=find_treasure,
        )

    def _create_date_tool(self, **kwargs: object) -> Tool:
        """Get current time and date in Europe/Amsterdam timezone."""
        return Tool(
            name="Today, current date",
            description="Get current date and time",
            parameters={},
            function=get_datetime,
        )

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        generator = OpenAIChatGenerator(
            model=self.get_model(),
            api_key=Secret.from_token(settings.OPENAI_API_KEY),
            api_base_url=getattr(settings, "OPENAI_API_URL", None),
        )
        return Run(generator=generator)

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Stream:
        """Create Haystack agent swarm adapter."""
        storage_adapter = await self.get_storage_adapter(thread_id)

        pipeline = Pipeline()

        # Create triage agent
        triage_agent = HaystackAgent(
            chat_generator=OpenAIChatGenerator(
                model=self.get_model(),
                api_key=Secret.from_token(settings.OPENAI_API_KEY),
                api_base_url=getattr(settings, "OPENAI_API_URL", None),
            ),
            tools=await self.get_tools(thread_id=thread_id or "", user=user),
            system_prompt=self.get_system_prompt(),
            exit_conditions=["text"],
        )

        pipeline.add_component("triage_agent", triage_agent)

        return Stream(
            pipeline=pipeline,
            generator=triage_agent.chat_generator,
            storage_adapter=storage_adapter,
            citation_registry=self.get_citation_registry(),
            suggestion_generator=self.get_suggestion_generator(),
        )
