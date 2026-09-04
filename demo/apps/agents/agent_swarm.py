from __future__ import annotations

import random
from typing import TYPE_CHECKING, Annotated

from django.conf import settings
from django.utils import timezone
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agents import LogToolCallsHook, ToolAgent, ToolAgentConfig, auto_register
from django_ai_sdk.citations import DefaultCitationFormatter
from django_ai_sdk.common import prompt
from django_ai_sdk.generators import openai_responses_chat
from django_ai_sdk.permissions import IsAdminUser
from django_ai_sdk.suggestions import DefaultSuggestionGenerator
from haystack.tools import Tool

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


def _pirate_boat_expert(topic: Annotated[str, "Topic about pirate boats"]) -> str:
    """Provide expert lore about pirate boats."""
    return f"Aye! On pirate boats: {topic}! They be swift and full o' tales."


def pirate_boat_expert_tool(**kwargs: object) -> Tool:
    """Expert knowledge about pirate boats."""
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
        function=_pirate_boat_expert,
    )


def _find_treasure(location: Annotated[str, "Location to search for treasure"]) -> dict:
    """Simulate treasure discovery."""
    treasure_id = random.randint(1000, 9999)
    value = random.randint(5000, 50000)
    return {
        "location": location,
        "treasure_id": treasure_id,
        "value": f"{value} gold coins",
    }


def find_treasure_tool(**kwargs: object) -> Tool:
    """Find treasure at a given location."""
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
        function=_find_treasure,
    )


def get_datetime() -> dict:
    """Get current time and date in Europe/Amsterdam timezone."""
    tz = timezone.get_current_timezone()
    nowtz = timezone.now().astimezone(tz)

    return {
        "today": nowtz.date().isoformat(),
        "current_time": nowtz.timetz().isoformat(),
    }


def get_datetime_tool(**kwargs: object) -> Tool:
    """Current date and time tool."""
    return Tool(
        name="get_datetime",
        parameters={},
        description="Get current date and time",
        function=get_datetime,
    )


@auto_register
class PirateBoatExpertAgent(Agent):
    """Expert on pirate boats, galleons, and seafaring lore."""

    name = "Pirate Boat Expert"
    description = "Deep knowledge about pirate boats and seafaring lore."
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True
    tools: list = [pirate_boat_expert_tool]
    instructions = prompt("""\
        You are the crew's pirate boat expert.

        Use the pirate_boat_expert tool to answer any question about pirate
        boats, galleons, rigging, or seafaring lore.
        Reply with a concise summary in pirate style.
    """)


@auto_register
class TreasureHunterAgent(Agent):
    """Hunts for treasure at a given location."""

    name = "Treasure Hunter"
    description = "Simulates discovering treasure at a location."
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True
    tools: list = [find_treasure_tool]
    instructions = prompt("""\
        You are the crew's treasure hunter.

        Use the find_treasure tool to search for treasure at a location.
        Reply with a concise summary of what you found in pirate style.
    """)


@auto_register
class PirateSwarmAgent(Agent):
    """
    Agent swarm agent with a crew of specialized pirate agents.
    """

    name = "Pirate Agent Swarm"
    description = "An agent swarm agent that delegates to a crew of pirate subagents."
    model = settings.AI_SDK_DEFAULT_MODEL
    llm = openai_responses_chat
    llm_kwargs = {"reasoning": {"effort": "low", "summary": "auto"}}
    permissions = [IsAdminUser]
    instructions = prompt("""\
        You are a Triage Agent for a crew of pirate subagents.

        Decide whether the user wants:
        - pirate boat expertise (delegate to the Pirate Boat Expert)
        - treasure finding (delegate to the Treasure Hunter)
        - date and time information (call get_datetime)
        - or general pirate help

        Always respond as a pirate and delegate to the subagents via the
        available subagent tools when appropriate. Summarize their findings
        for the user. Respond with text or tool calls as needed.
    """)

    agents = [PirateBoatExpertAgent, TreasureHunterAgent]

    tools: list = [get_datetime_tool]

    citation_formatter_class = DefaultCitationFormatter
    suggestion_generator = DefaultSuggestionGenerator

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Run:
        return Run(generator=self.get_llm())

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Stream:
        """Create Haystack agent swarm adapter."""

        generator = self.get_llm()
        storage_adapter = await self.get_storage_adapter(thread_id)

        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=await self.get_tools(thread_id=thread_id or "", user=user),
                exit_conditions=["text"],
                max_agent_steps=10,
                hooks={"before_tool": [LogToolCallsHook()]},
            ),
            generator=generator,
        )

        pipeline = tool_agent.pipeline()

        return Stream(
            pipeline=pipeline,
            generator=generator,
            storage_adapter=storage_adapter,
            citation_registry=self.get_citation_registry(),
            suggestion_generator=self.get_suggestion_generator(),
        )
