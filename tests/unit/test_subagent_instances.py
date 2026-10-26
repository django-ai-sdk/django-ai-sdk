"""Subagents are agent instances: classes, or agents by id (registered or runtime)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.components.generators.chat import MockChatGenerator
from haystack.tools import Tool

from django_ai_sdk import Agent
from django_ai_sdk.agents.models import AgentSettings
from django_ai_sdk.agents.subagent import resolve_subagents
from django_ai_sdk.memories.models import Memory
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter

MOCK_MODEL = "gpt-4o-mini"


class Coordinator(Agent):
    name = "Coordinator"
    model = MOCK_MODEL
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        pass


async def make_user(name):
    from django.contrib.auth import get_user_model

    return await get_user_model().objects.acreate(email=f"{name}-{uuid.uuid4().hex[:8]}@example.com")


async def make_agent(name, *, is_public=True):
    return await AgentSettings.objects.acreate(name=name, model=MOCK_MODEL, is_public=is_public)


def mock_llm():
    return patch(
        "django_ai_sdk.agents.base.Agent.get_llm",
        return_value=MockChatGenerator(responses=["ok"]),
    )


@pytest.mark.django_db(transaction=True)
class TestRuntimeSubagents:
    async def test_by_id_and_only_those_the_user_may_view(self):
        user = await make_user("user")
        sales = await make_agent("Sales Agent")
        secret = await make_agent("Secret Agent", is_public=False)

        subagents = await resolve_subagents(
            Coordinator(), [str(sales.id), str(secret.id), str(uuid.uuid4())], user
        )

        assert [agent.name for agent in subagents] == ["Sales Agent"]
        assert subagents[0].is_runtime

    async def test_never_the_coordinator_itself(self):
        user = await make_user("user")
        sales = await make_agent("Sales Agent")
        coordinator = await resolve_subagents(Coordinator(), [str(sales.id)], user)

        subagents = await resolve_subagents(coordinator[0], [str(sales.id)], user)

        assert subagents == []

    async def test_runtime_agents_sharing_a_class_are_not_a_cycle(self):
        user = await make_user("user")
        sales = await make_agent("Sales Agent")
        legal = await make_agent("Legal Agent")

        class Crew(Coordinator):
            agents = [str(sales.id), str(legal.id)]

        with mock_llm():
            tools = await Crew().get_agent_tools(thread_id="t1", user=user)

        assert [tool.name for tool in tools] == ["sales_agent", "legal_agent"]

    async def test_the_roster_lists_the_resolved_subagents(self):
        user = await make_user("user")
        sales = await make_agent("Sales Agent")

        class Crew(Coordinator):
            agents = [str(sales.id)]

        crew = Crew()
        prompt = crew.get_system_prompt(subagents=await crew.get_subagents(user=user))

        assert "Available subagents:\n- Sales Agent" in prompt


@pytest.mark.django_db(transaction=True)
class TestDelegatedTools:
    async def test_search_their_own_knowledge_as_sources_of_the_turn(self):
        user = await make_user("user")
        await Memory.objects.acreate(name="Sales Knowledge", slug="sales", is_public=True)
        search = Tool(
            name="search_sales_knowledge",
            description="Search",
            parameters={"type": "object", "properties": {}},
            function=lambda: [],
        )

        class Sales(Coordinator):
            memories = ["sales"]
            rag_provider = MagicMock(get_tool=AsyncMock(return_value=search))

        tools = await Sales().get_delegated_tools(thread_id="t1", user=user)

        [found] = [tool for tool in tools if tool.name == "search_sales_knowledge"]
        # what it finds is registered with the coordinator's turn (collect_sources)
        assert "handler" in found.outputs_to_string

    async def test_not_knowledge_the_user_may_not_view(self):
        user = await make_user("user")
        await Memory.objects.acreate(name="Board Knowledge", slug="board", is_public=False)

        class Board(Coordinator):
            memories = ["board"]
            rag_provider = MagicMock(get_tool=AsyncMock())

        await Board().get_delegated_tools(thread_id="t1", user=user)

        Board.rag_provider.get_tool.assert_not_called()
