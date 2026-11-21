"""What retrieval found reaches oversight, and sources from earlier turns stay apart."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.components.generators.chat import MockChatGenerator
from haystack.tools import Tool

from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.adapters.citations import NumberedSource, SourceRegistry
from django_ai_sdk.agents import ToolAgent, ToolAgentConfig
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry, ThreadMemory
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.permissions import AllowAll
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from tests.factories.db import MemoryFactory, UserFactory
from tests.mocks.permissions import memory_permissions
from tests.oversight.helpers import (
    a_thread,
    call,
    chat_turns,
    make_agent,
)


@pytest.mark.django_db
class TestSeededSources:
    async def test_sources_from_earlier_turns_are_not_counted_as_found(self):
        registry = SourceRegistry()
        earlier = NumberedSource(title="Leave · §1", content="25 days", doc_id="d1", chunk_id="c1")
        registry.seed([earlier])
        registry.add(
            [NumberedSource(title="Pay · §2", content="the 25th", doc_id="d2", chunk_id="c2")]
        )
        guard = await TurnGuard.start(make_agent([], None), channel="api", input="q")

        guard.set_sources(registry.all, seeded=registry.seeded)

        assert [s.title for s in guard.context.sources] == ["Leave · §1", "Pay · §2"]
        assert [s.title for s in guard.context.fresh_sources()] == ["Pay · §2"]


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestCitations:
    async def test_citations_carry_only_masked_source_text(self):
        from haystack.dataclasses import Document

        from django_ai_sdk.adapters.citations import SourceRegistry, current_registry, from_document
        from django_ai_sdk.storage.services import aget_thread_sources
        from tests.test_citations import CitationAgent, citations

        user = await UserFactory.acreate()
        thread = await a_thread(user)
        raw = "Jan's private mail is jan@example.com and IBAN NL91ABNA0417164300"

        def search(q: str) -> str:
            registry = current_registry.get()
            assert registry is not None
            registry.add(
                [from_document(Document(id="c1", content=raw, meta={"doc_id": "d1"}))]
            )
            return raw

        tool = Tool(
            name="search",
            description="search",
            function=search,
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )

        class Cited(Agent):
            _skip_auto_register = True
            permissions = [AllowAll]
            name = "cited"
            oversight = "test-guard"  # masks personal data the way the gate does
            title_generation = False

            async def get_pipeline_adapter(self, thread_id=None, user=None):  # noqa: ANN001, ANN201
                generator = MockChatGenerator(
                    responses=[call("search", q="Jan"), "Jan can be mailed."]
                )
                config = ToolAgentConfig(model="mock", system_prompt="s", tools=[tool])
                return Stream(
                    pipeline=ToolAgent(config, generator=generator).pipeline(),
                    generator=generator,
                    storage_adapter=await self.get_storage_adapter(thread_id),
                    citation_registry=SourceRegistry(),
                )

        with patch(
            "django_ai_sdk.agents.chat.get_citation_agent", return_value=CitationAgent(citations())
        ):
            parts = await chat_turns(
                Cited(), [("user", "find Jan")], thread_id=str(thread.id), user=user
            )
        outputs = [p for p in parts if p.get("type") == "tool-output-available"]
        assert len(outputs) >= 2  # the search, and the citations card
        assert "jan@example.com" not in json.dumps(outputs)
        sources = await aget_thread_sources(str(thread.id), user=user)
        assert sources and "jan@example.com" not in json.dumps(sources, default=str)
        assert "NL91ABNA0417164300" not in json.dumps(sources, default=str)
