"""A chat says which part of the adapter build it waits on, and which parts were skipped."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest

from django_ai_sdk.agents.base import Agent
from django_ai_sdk.events import MessageStartEvent, StreamEndEvent, StreamEvent
from django_ai_sdk.integrations.base import IntegrationStatus
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.responses import stream_response


class _Adapter:
    model: str | None = None
    instructions: str | None = None
    suggestion_generator = None

    async def stream(self, messages: list[Any]) -> AsyncGenerator[StreamEvent, None]:
        yield MessageStartEvent(message_id="m1")
        yield StreamEndEvent()


async def _build() -> _Adapter:
    return _Adapter()


async def warmup_parts(*parts: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
    for part in parts:
        yield part


async def streamed(build: Any, warmup: Any = None) -> list[dict[str, Any]]:
    """The data-warmup parts a chat streams, in order."""
    response = await stream_response(build, [], VercelProtocolHandler(), warmup=warmup)
    text = b"".join([chunk async for chunk in response.streaming_content]).decode()
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and ('"data-warmup' in line or '"data-unavailable' in line)
    ]


def warmup(stage: str, detail: str) -> dict[str, Any]:
    return {
        "type": "data-warmup",
        "data": {"status": "start", "stage": stage, "detail": detail},
        "transient": True,
    }


def unavailable(stage: str, detail: str) -> dict[str, Any]:
    return {"type": "data-unavailable", "data": {"stage": stage, "detail": detail}, "transient": True}


def overall(found: list[dict[str, Any]]) -> list[str]:
    return [p["data"]["status"] for p in found if p["type"] == "data-warmup"]


class TestStreamedWarmup:
    async def test_parts_arrive_between_start_and_ready(self):
        found = await streamed(
            _build,
            warmup_parts(warmup("knowledge", "HR"), unavailable("integrations", "Zendesk")),
        )

        assert [p["type"] for p in found] == [
            "data-warmup",
            "data-warmup",
            "data-unavailable",
            "data-warmup",
            "data-warmup",
        ]
        assert overall(found) == ["start", "start", "start", "ready"]

    async def test_the_last_stage_named_is_dropped_before_the_build(self):
        found = await streamed(_build, warmup_parts(warmup("knowledge", "Engineering Knowledge")))

        # The build that follows is not loading that knowledge base any more.
        assert found[-2]["data"] == {"status": "start"}
        assert found[-1]["data"] == {"status": "ready"}

    async def test_a_failing_warmup_ends_in_failed_and_never_ready(self):
        async def broken() -> AsyncGenerator[dict[str, Any], None]:
            yield warmup("knowledge", "HR")
            raise RuntimeError("index locked")

        found = await streamed(_build, broken())

        assert overall(found) == ["start", "start", "failed"]

    async def test_a_warmup_that_names_nothing_adds_nothing(self):
        found = await streamed(_build, warmup_parts())

        assert overall(found) == ["start", "ready"]

    async def test_without_a_warmup_only_start_and_ready_are_streamed(self):
        found = await streamed(_build)

        assert overall(found) == ["start", "ready"]
        assert len(found) == 2


class _Integration:
    def __init__(
        self,
        label: str,
        status: IntegrationStatus | Exception = IntegrationStatus.ACTIVE,
        delay: float = 0,
    ):
        self.name = label.lower()
        self.label = label
        self._status = status
        self._delay = delay

    async def get_status(self, user: Any = None, agent: Any = None) -> IntegrationStatus:
        await asyncio.sleep(self._delay)
        if isinstance(self._status, Exception):
            raise self._status
        return self._status


class _Provider:
    def __init__(self) -> None:
        self.opened: list[str] = []

    async def get_rag_instance(self, agent: Any, memory_id: str | None = None) -> Any:
        self.opened.append(str(memory_id))


def make_agent(integrations: list[_Integration], provider: _Provider | None = None) -> Agent:
    class FakeAgent(Agent):
        name = "Fake"
        description = ""
        model = "gpt-fake"
        rag_provider = provider

        async def get_pipeline_adapter(self, thread_id=None, user=None):
            raise NotImplementedError

        async def _allowed_integrations(self, user=None):
            return integrations

    return FakeAgent()


class TestAgentWarmupProgress:
    async def test_every_integration_is_named_and_a_degraded_one_is_reported(self):
        agent = make_agent(
            [
                _Integration("Slow", delay=0.05),
                _Integration("Down", IntegrationStatus.DEGRADED),
            ]
        )

        parts = [p async for p in agent.warmup_progress()]

        # Both are named while both are pending; the one still pending is named after.
        assert parts == [
            warmup("integrations", "Slow, Down"),
            unavailable("integrations", "Down"),
            warmup("integrations", "Slow"),
        ]

    async def test_the_name_shrinks_to_what_is_still_being_waited_on(self):
        agent = make_agent(
            [
                _Integration("A", delay=0.01),
                _Integration("B", delay=0.03),
                _Integration("C", delay=0.05),
            ]
        )

        parts = [p async for p in agent.warmup_progress()]

        assert [p["data"]["detail"] for p in parts] == ["A, B, C", "B, C", "C"]

    async def test_an_integration_whose_check_raises_is_reported_and_the_rest_carry_on(self):
        agent = make_agent([_Integration("Broken", RuntimeError("boom")), _Integration("Fine")])

        parts = [p async for p in agent.warmup_progress()]

        assert unavailable("integrations", "Broken") in parts
        assert unavailable("integrations", "Fine") not in parts

    @pytest.mark.parametrize("status", [IntegrationStatus.DISCONNECTED, IntegrationStatus.EXPIRED])
    async def test_an_integration_that_needs_setup_is_not_reported(self, status):
        agent = make_agent([_Integration("Unconfigured", status)])

        parts = [p async for p in agent.warmup_progress()]

        assert parts == [warmup("integrations", "Unconfigured")]

    async def test_knowledge_bases_are_named_and_opened_one_after_another(self, monkeypatch):
        memories = [SimpleNamespace(id="a", name="HR"), SimpleNamespace(id="b", name="IT")]

        async def get_thread_memories(thread_id: str, user: Any = None) -> list[Any]:
            return memories

        monkeypatch.setattr(
            "django_ai_sdk.memories.services.MemoryService.get_thread_memories",
            get_thread_memories,
        )
        provider = _Provider()
        agent = make_agent([], provider)

        parts = [p async for p in agent.warmup_progress(thread_id="t1")]

        assert parts == [warmup("knowledge", "HR"), warmup("knowledge", "IT")]
        assert provider.opened == ["a", "b"]

    async def test_knowledge_is_skipped_without_a_thread(self):
        provider = _Provider()
        agent = make_agent([], provider)

        assert [p async for p in agent.warmup_progress()] == []
        assert provider.opened == []
