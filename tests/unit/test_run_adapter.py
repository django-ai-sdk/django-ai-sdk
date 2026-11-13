"""The non-streaming `Run` adapter, used by WorkflowExecutor and title generation."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import component
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from pydantic import BaseModel

from django_ai_sdk.adapters.base import Run
from django_ai_sdk.common import ChatMessage


@component
class Scripted:
    """A chat generator that answers `reply` and counts its calls. Run runs it as a Haystack Agent."""

    def __init__(self, reply: str | HaystackChatMessage) -> None:
        self.reply = reply if isinstance(reply, HaystackChatMessage) else HaystackChatMessage.from_assistant(reply)
        self.calls = 0

    @component.output_types(replies=list[HaystackChatMessage])
    def run(self, messages: list[HaystackChatMessage], generation_kwargs: dict[str, Any] | None = None) -> dict:
        self.calls += 1
        return {"replies": [self.reply]}

    @component.output_types(replies=list[HaystackChatMessage])
    async def run_async(
        self, messages: list[HaystackChatMessage], generation_kwargs: dict[str, Any] | None = None
    ) -> dict:
        return self.run(messages, generation_kwargs)


def user_message(text="Hello"):
    return ChatMessage(id="1", role="user", content=text)


class TestWithoutTools:
    async def test_a_bare_completion_is_unchanged(self):
        generator = Scripted("hi there")

        run = Run(generator=generator)
        result = await run.run([user_message()])

        assert result == "hi there"
        assert generator.calls == 1

    async def test_structured_output_is_unchanged(self):
        class Out(BaseModel):
            value: str

        generator = Scripted('{"value": "x"}')

        run = Run(generator=generator)
        result = await run.run([user_message()], response_format=Out)

        assert result == Out(value="x")

    async def test_a_structured_call_without_reply_text_gives_none(self):
        class Out(BaseModel):
            value: str

        # E.g. the token limit spent on reasoning: a reply without text.
        reply = HaystackChatMessage.from_assistant(None, meta={"finish_reason": "length"})
        generator = Scripted(reply)

        assert await Run(generator=generator).run([user_message()], response_format=Out) is None


@pytest.mark.django_db(transaction=True)
class TestAgentRunToolsDefault:
    """Agent.run(tools=...) defaults to False: a one-shot call must not reach
    every configured integration unless it explicitly opts in."""

    async def test_a_plain_call_does_not_resolve_tools_by_default(self):
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        config = AgentSettings(name="Titler", slug=str(uuid4()), agent="test")
        await config.asave()
        agent = RuntimeAgent(config)

        generator = Scripted("A Title")

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[MagicMock()])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
        ):
            result = await agent.run([user_message("hi")], response_format=None)

        assert result == "A Title"
        tools.assert_not_awaited()

    async def test_tools_true_resolves_and_runs_them_via_the_real_agent_builder(self):
        """Agent.run(tools=True) is governed like a streamed delegation: the
        loop gets the agent's own steps and default_hooks, as build_subagent does."""
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        config = AgentSettings(name="Actor", slug=str(uuid4()), agent="test")
        await config.asave()
        agent = RuntimeAgent(config)

        generator = MagicMock()
        fake_agent = MagicMock()
        fake_agent.clone.return_value = fake_agent  # oversight attaches its hooks to a copy
        fake_agent.run_async = AsyncMock(
            return_value={"messages": [HaystackChatMessage.from_assistant("done")]}
        )
        tool = MagicMock()
        fake_agent.tools = [tool]

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[tool])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
            patch(
                "django_ai_sdk.adapters.run.Agent", return_value=fake_agent
            ) as agent_cls,
        ):
            result = await agent.run([user_message("do it")], response_format=None, tools=True)

        assert result == "done"
        tools.assert_awaited_once()
        assert agent_cls.call_args.kwargs["tools"] == [tool]
        # run_agent installs oversight: the model is offered only the tools the turn allows.
        assert fake_agent.clone.call_args.kwargs["tools"].items == [tool]
        # Governed by RuntimeAgent's own (default) max_agent_steps=6, not a
        # generic constant.
        assert agent_cls.call_args.kwargs["max_agent_steps"] == 6
        budget = [
            h
            for h in agent_cls.call_args.kwargs["hooks"]["before_tool"]
            if type(h).__name__ == "ToolCallBudgetHook"
        ]
        assert len(budget) == 1
        assert budget[0].max_tool_calls == 6

    async def test_a_non_default_max_agent_steps_is_honored(self):
        """The exact regression: a subagent with a deliberately tight budget
        (e.g. ResearchPlannerAgent's max_agent_steps=4) must keep that budget
        when run headless via tools=True, not fall back to a generic default."""
        from django_ai_sdk import Agent
        from django_ai_sdk.adapters.base import Run

        generator = MagicMock()

        class TightAgent(Agent):
            name = "Tight"
            model = "gpt-4o-mini"
            max_agent_steps = 3
            max_tool_calls = None

            async def get_run_adapter(self, thread_id=None, user=None):
                return Run(generator=generator)

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        agent = TightAgent()
        fake_agent = MagicMock()
        fake_agent.clone.return_value = fake_agent  # oversight attaches its hooks to a copy
        fake_agent.run_async = AsyncMock(return_value={"messages": []})

        with (
            patch.object(TightAgent, "get_tools", AsyncMock(return_value=[MagicMock()])),
            patch(
                "django_ai_sdk.adapters.run.Agent", return_value=fake_agent
            ) as agent_cls,
        ):
            await agent.run([user_message("hi")], response_format=None, tools=True)

        assert agent_cls.call_args.kwargs["max_agent_steps"] == 3
        # max_tool_calls=None disables the budget hook, same as build_subagent.
        hooks = agent_cls.call_args.kwargs["hooks"]["before_tool"]
        assert all(type(h).__name__ != "ToolCallBudgetHook" for h in hooks)


@pytest.mark.django_db(transaction=True)
class TestStructuredCallsDoNotResolveTools:
    """A workflow step with output_fields must not reach out to every integration,
    even if the caller passed tools=True — response_format wins (see Run.run)."""

    async def test_a_response_format_call_neither_resolves_nor_warns(self):
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        class Out(BaseModel):
            value: str

        config = AgentSettings(name="Structured", slug=str(uuid4()), agent="test")
        await config.asave()
        agent = RuntimeAgent(config)

        generator = Scripted('{"value": "x"}')

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[MagicMock()])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
            patch("django_ai_sdk.adapters.run.logger") as log,
        ):
            result = await agent.run([user_message()], response_format=Out, tools=True)

        assert result == Out(value="x")
        tools.assert_not_awaited()
        log.warning.assert_not_called()

    async def test_an_adapter_that_is_no_run_is_refused(self):
        # Oversight runs as hooks on the Run's Haystack Agent: anything else would go unchecked.
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        config = AgentSettings(name="Bare", slug=str(uuid4()), agent="test")
        await config.asave()
        agent = RuntimeAgent(config)

        bare = MagicMock(spec=["run"])
        bare.run = AsyncMock(return_value="fine")

        with (
            patch.object(RuntimeAgent, "get_run_adapter", AsyncMock(return_value=bare)),
            pytest.raises(TypeError, match="must return a Run"),
        ):
            await agent.run([user_message()])
        bare.run.assert_not_called()


class TestDefaultRunAdapter:
    async def test_an_agent_runs_on_its_own_generator_without_overriding_get_run_adapter(self):
        from haystack.components.generators.chat import MockChatGenerator

        from django_ai_sdk.agents.base import Agent
        from django_ai_sdk.permissions import AllowAll

        generator = MockChatGenerator(responses=[HaystackChatMessage.from_assistant("Hi.")])
        made = []

        def factory(**kwargs):
            made.append(kwargs)
            return generator

        class Plain(Agent):
            _skip_auto_register = True
            permissions = [AllowAll]
            name = "plain"
            model = "some-model"
            oversight = False
            llm = staticmethod(factory)

        adapter = await Plain().get_run_adapter()
        assert isinstance(adapter, Run) and adapter.generator is generator
        assert made == [{"model": "some-model"}]
        assert await Plain().run([user_message()]) == "Hi."
