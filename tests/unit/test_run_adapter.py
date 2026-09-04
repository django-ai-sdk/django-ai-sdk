"""The non-streaming `Run` adapter, used by WorkflowExecutor and title generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from pydantic import BaseModel

from django_ai_sdk.adapters.base import Run
from django_ai_sdk.common import ChatMessage


def user_message(text="Hello"):
    return ChatMessage(id="1", role="user", content=text)


class TestWithoutTools:
    async def test_a_bare_completion_is_unchanged(self):
        generator = MagicMock()
        generator.run.return_value = {"replies": [MagicMock(text="hi there")]}

        run = Run(generator=generator)
        result = await run.run([user_message()])

        assert result == "hi there"
        generator.run.assert_called_once()

    async def test_structured_output_is_unchanged(self):
        class Out(BaseModel):
            value: str

        generator = MagicMock()
        generator.run.return_value = {"replies": [MagicMock(text='{"value": "x"}')]}

        run = Run(generator=generator)
        result = await run.run([user_message()], response_format=Out)

        assert result == Out(value="x")


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

        generator = MagicMock()
        generator.run.return_value = {"replies": [MagicMock(text="A Title")]}

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[MagicMock()])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
        ):
            result = await agent.run([user_message("hi")], response_format=None)

        assert result == "A Title"
        tools.assert_not_awaited()

    async def test_tools_true_resolves_and_runs_them_via_the_real_agent_builder(self):
        """Agent.run(tools=True) must not build a shadow agent: the loop goes
        through ToolAgent.build_agent, the same path build_subagent uses for a
        streamed delegation - governed, not an ungoverned bare Agent."""
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        config = AgentSettings(name="Actor", slug=str(uuid4()), agent="test")
        await config.asave()
        agent = RuntimeAgent(config)

        generator = MagicMock()
        fake_agent = MagicMock()
        fake_agent.run_async = AsyncMock(
            return_value={"messages": [HaystackChatMessage.from_assistant("done")]}
        )
        tool = MagicMock()

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[tool])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
            patch(
                "django_ai_sdk.agents.tool_agent.HaystackAgent", return_value=fake_agent
            ) as agent_cls,
        ):
            result = await agent.run([user_message("do it")], response_format=None, tools=True)

        assert result == "done"
        tools.assert_awaited_once()
        assert agent_cls.call_args.kwargs["tools"] == [tool]
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
        fake_agent.run_async = AsyncMock(return_value={"messages": []})

        with (
            patch.object(TightAgent, "get_tools", AsyncMock(return_value=[MagicMock()])),
            patch(
                "django_ai_sdk.agents.tool_agent.HaystackAgent", return_value=fake_agent
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

        generator = MagicMock()
        generator.run.return_value = {"replies": [MagicMock(text='{"value": "x"}')]}

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[MagicMock()])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
            patch("django_ai_sdk.adapters.base.logger") as log,
        ):
            result = await agent.run([user_message()], response_format=Out, tools=True)

        assert result == Out(value="x")
        tools.assert_not_awaited()
        log.warning.assert_not_called()

    async def test_an_adapter_without_a_tools_attribute_is_left_alone(self):
        # get_run_adapter is an override point; an adapter is not required to carry tools.
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
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[MagicMock()])),
        ):
            assert await agent.run([user_message()], response_format=None, tools=True) == "fine"
