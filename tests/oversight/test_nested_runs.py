"""Runs inside a turn: joining it, child turns, and subagents."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from django.test import override_settings
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.tools import Tool

from django_ai_sdk.agents import Agent
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight.guard import TurnGuard, current_guard, guarding
from django_ai_sdk.oversight.models import Turn
from django_ai_sdk.permissions import AllowAll
from tests.oversight.helpers import (
    INJECTION,
    LenientPolicy,
    RunPolicy,
    agent_with,
    call,
    delegation,
    make_agent,
)


SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz12"


@pytest.fixture
def no_overlay():
    """No switches overlay, so a sync delegation can read its policy from code."""
    from django_ai_sdk.oversight import resolve
    from django_ai_sdk.oversight.resolve import OverlaySource

    before = resolve.overlay_source()
    resolve.set_overlay_source(OverlaySource())
    yield
    resolve.set_overlay_source(before)


def secret_subagent(ran: list[str], *responses: Any) -> type[Agent]:
    def send(text: str) -> str:
        ran.append(text)
        return "sent"

    tool = Tool(
        name="send",
        description="send",
        function=send,
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )

    class Sub(Agent):
        _skip_auto_register = True
        permissions = [AllowAll]
        name = "sub"
        oversight = "test-guard"  # refuses tool calls that carry a secret
        tools = [lambda **kw: tool]  # noqa: RUF012

        def get_llm(self, **kwargs: Any) -> Any:
            replies = responses or [call("send", text=SECRET), "done"]
            return MockChatGenerator(responses=list(replies))

    return Sub


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestNestedRuns:
    async def outer(self) -> TurnGuard:
        agent, _ = agent_with("x", policy=LenientPolicy)
        guard = await TurnGuard.start(agent, channel="chat", input="hi")
        assert guard.active
        return guard

    async def test_an_explicit_policy_inside_a_turn_opens_a_child_turn(self):
        agent, model = agent_with("Sure.", policy=None)
        outer = await self.outer()
        with guarding(outer), pytest.raises(Blocked):
            await agent.run([ChatMessage(role="user", content=INJECTION)], oversight=RunPolicy)
        assert not model.asked
        assert await Turn.objects.acount() == 1  # the child's own record

    async def test_a_nested_agent_with_a_policy_of_its_own_opens_a_child_turn(self):
        agent, model = agent_with("Sure.", policy=RunPolicy)
        outer = await self.outer()
        with guarding(outer), pytest.raises(Blocked):
            await agent.run([ChatMessage(role="user", content=INJECTION)])
        assert not model.asked

    async def test_the_child_turn_links_to_its_parent(self):
        from django_ai_sdk.oversight.agent_run import turn

        agent, _ = agent_with("x", policy=RunPolicy)
        outer = await self.outer()
        with guarding(outer):
            async with turn(agent, channel="chat", input="hi") as child:
                assert child.guard is not None and child.guard.parent_turn_id == outer.id
        async with turn(agent, channel="chat", input="hi") as alone:
            assert alone.guard is not None and alone.guard.parent_turn_id == ""

    async def test_the_same_policy_or_none_joins(self):
        same, _ = agent_with("Fine.", policy=LenientPolicy)
        none, _ = agent_with("Fine.", policy=None)
        outer = await self.outer()
        with guarding(outer):
            assert await same.run([ChatMessage(role="user", content="hi")]) == "Fine."
            assert await none.run([ChatMessage(role="user", content="hi")]) == "Fine."
        assert await Turn.objects.acount() == 0

    async def test_oversight_false_runs_unguarded_even_inside_a_turn(self):
        agent, model = agent_with("Sure.")
        outer = await self.outer()
        with guarding(outer):
            result = await agent.run([ChatMessage(role="user", content=INJECTION)], oversight=False)
        assert result == "Sure." and model.asked

    async def test_oversight_none_is_the_agents_own_and_a_name_replaces_it(self):
        agent, _ = agent_with("Sure.", policy=False)
        injection = [ChatMessage(role="user", content=INJECTION)]
        with override_settings(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-run"}):
            assert await agent.run(list(injection), oversight=None) == "Sure."  # opts out
            with pytest.raises(Blocked):
                await agent.run(list(injection), oversight="test-run")

    async def test_a_closed_turn_is_never_joined(self):
        """A task started inside a turn outlives it: its runs open turns of their own."""
        agent, model = agent_with("Sure.")
        outer = await TurnGuard.start(agent, channel="chat", input="hi")
        go = asyncio.Event()

        async def later() -> Any:
            await go.wait()
            assert current_guard() is None
            return await agent.run([ChatMessage(role="user", content=INJECTION)])

        with guarding(outer):
            task = asyncio.create_task(later())
        await outer.close()
        go.set()
        with pytest.raises(Blocked):
            await task
        assert not model.asked


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestSubagents:
    async def test_its_policy_checks_its_tools_under_a_laxer_coordinator(self):
        ran: list[str] = []
        lenient = await TurnGuard.start(
            make_agent([], None, policy="test-lenient"), channel="chat", input="x"
        )
        component = await delegation(secret_subagent(ran))
        with guarding(lenient):
            await component.run_async(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert ran == []

    async def test_its_policy_checks_its_tools_without_a_coordinator_policy(self):
        ran: list[str] = []
        component = await delegation(secret_subagent(ran))
        await component.run_async(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert ran == []
        assert await Turn.objects.acount() == 1

    async def test_a_refused_task_is_what_the_coordinator_reads(self):
        from django_ai_sdk.agents.subagent import subagent_response

        ran: list[str] = []
        component = await delegation(secret_subagent(ran))
        result = await component.run_async(
            messages=[HaystackChatMessage.from_user("x")], task=INJECTION
        )
        assert subagent_response(result["messages"]) == "Not like that."
        assert ran == []

    async def test_its_answer_is_finished_before_the_coordinator_reads_it(self):
        from django_ai_sdk.agents.subagent import subagent_response

        component = await delegation(secret_subagent([], "Ask jan@example.com."))
        result = await component.run_async(
            messages=[HaystackChatMessage.from_user("x")], task="who"
        )
        assert subagent_response(result["messages"]) == "Ask [EMAIL]."

    async def test_the_same_policy_joins_the_coordinators_turn(self):
        ran: list[str] = []
        guard = await TurnGuard.start(make_agent([], None), channel="chat", input="x")
        component = await delegation(secret_subagent(ran))
        with guarding(guard):
            await component.run_async(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert ran == []
        assert await Turn.objects.acount() == 0  # joined: the coordinator's turn records it

    def test_a_sync_delegation_that_needs_a_turn_is_refused(self):
        from django_ai_sdk.agents.subagent import UNGUARDABLE, subagent_response

        ran: list[str] = []
        component = asyncio.run(delegation(secret_subagent(ran)))
        result = component.run(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert subagent_response(result["messages"]) == str(UNGUARDABLE)
        assert ran == []

    @pytest.mark.usefixtures("no_overlay")
    def test_a_sync_delegation_without_oversight_runs(self):
        from django_ai_sdk.agents.subagent import UNGUARDABLE, subagent_response

        ran: list[str] = []
        sub = secret_subagent(ran)
        sub.oversight = False
        component = asyncio.run(delegation(sub))
        result = component.run(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert subagent_response(result["messages"]) != str(UNGUARDABLE)
        assert ran == [SECRET]

    @pytest.mark.usefixtures("no_overlay")
    def test_a_sync_delegation_under_the_same_policy_joins(self):
        from django_ai_sdk.agents.subagent import subagent_response

        ran: list[str] = []
        guard = asyncio.run(TurnGuard.start(make_agent([], None), channel="chat", input="x"))
        component = asyncio.run(delegation(secret_subagent(ran, "done")))
        with guarding(guard):
            result = component.run(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert subagent_response(result["messages"]) == "done"

    def test_a_sync_delegation_whose_policy_only_an_overlay_or_hook_can_tell_is_refused(self):
        from django_ai_sdk.agents.subagent import UNGUARDABLE, subagent_response
        from django_ai_sdk.oversight import resolve
        from django_ai_sdk.oversight.resolve import OverlaySource

        ran: list[str] = []
        sub = secret_subagent(ran)
        sub.oversight = False
        component = asyncio.run(delegation(sub))
        before = resolve.overlay_source()
        resolve.set_overlay_source(type("Switches", (OverlaySource,), {})())
        try:
            result = component.run(messages=[HaystackChatMessage.from_user("x")], task="send")
        finally:
            resolve.set_overlay_source(before)
        assert subagent_response(result["messages"]) == str(UNGUARDABLE)
        sub.get_oversight = lambda self, **kw: "test-guard"
        result = component.run(messages=[HaystackChatMessage.from_user("x")], task="send")
        assert subagent_response(result["messages"]) == str(UNGUARDABLE)
        assert ran == []
