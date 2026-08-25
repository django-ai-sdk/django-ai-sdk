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


class TestWithTools:
    async def test_tools_run_through_a_haystack_agent_loop(self):
        generator = MagicMock()
        fake_agent = MagicMock()
        fake_agent.run_async = AsyncMock(
            return_value={"messages": [HaystackChatMessage.from_assistant("Two open issues.")]}
        )

        run = Run(generator=generator, tools=[MagicMock(name="linear_list_issues")])
        with patch("django_ai_sdk.adapters.base.Agent", return_value=fake_agent) as agent_cls:
            result = await run.run([user_message("What's open?")], system_prompt="Be brief.")

        assert result == "Two open issues."
        # The bare generator path must not also run.
        generator.run.assert_not_called()
        _, kwargs = agent_cls.call_args
        assert kwargs["tools"] == run.tools
        assert kwargs["system_prompt"] == "Be brief."

    async def test_tools_plus_response_format_warns_and_falls_back(self):
        """Structured output and tool use together aren't supported yet (see Run.run)."""

        class Out(BaseModel):
            value: str

        generator = MagicMock()
        generator.run.return_value = {"replies": [MagicMock(text='{"value": "x"}')]}

        run = Run(generator=generator, tools=[MagicMock()])
        with patch("django_ai_sdk.adapters.base.Agent") as agent_cls:
            with patch("django_ai_sdk.adapters.base.logger") as log:
                result = await run.run([user_message()], response_format=Out)

        assert result == Out(value="x")
        agent_cls.assert_not_called()
        # Dropping a configured capability must not be silent.
        log.warning.assert_called_once()

    async def test_the_loop_is_capped(self):
        # Nobody watches a scheduled workflow, so Haystack's default of 100 steps is a
        # bill rather than a slow reply.
        generator = MagicMock()
        fake_agent = MagicMock()
        fake_agent.run_async = AsyncMock(return_value={"messages": []})

        run = Run(generator=generator, tools=[MagicMock()])
        with patch("django_ai_sdk.adapters.base.Agent", return_value=fake_agent) as agent_cls:
            await run.run([user_message()])

        assert agent_cls.call_args.kwargs["max_agent_steps"] == 10

    async def test_the_cap_is_configurable(self):
        from django.test.utils import override_settings

        generator = MagicMock()
        fake_agent = MagicMock()
        fake_agent.run_async = AsyncMock(return_value={"messages": []})

        run = Run(generator=generator, tools=[MagicMock()])
        with override_settings(AI_SDK_RUN_MAX_TOOL_STEPS=3):
            with patch("django_ai_sdk.adapters.base.Agent", return_value=fake_agent) as agent_cls:
                await run.run([user_message()])

        assert agent_cls.call_args.kwargs["max_agent_steps"] == 3

    async def test_no_replies_returns_none(self):
        generator = MagicMock()
        fake_agent = MagicMock()
        fake_agent.run_async = AsyncMock(return_value={"messages": []})

        run = Run(generator=generator, tools=[MagicMock()])
        with patch("django_ai_sdk.adapters.base.Agent", return_value=fake_agent):
            result = await run.run([user_message()])

        assert result is None


# transaction=True so the AgentSettings row is flushed afterwards; see the note in
# tests/unit/test_memory_service.py.
@pytest.mark.django_db(transaction=True)
class TestTitleGenerationOptsOutOfTools:
    """Naming a thread must not reach every configured integration, let alone call one."""

    async def test_title_generation_neither_resolves_nor_runs_tools(self):
        from uuid import uuid4

        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent
        from django_ai_sdk.conversation.utils import generate_thread_title

        unique = str(uuid4())
        config = AgentSettings(name="Titler", slug=unique, agent="test")
        await config.asave()
        agent = RuntimeAgent(config)

        generator = MagicMock()
        generator.run.return_value = {"replies": [MagicMock(text="A Title")]}

        with (
            patch.object(RuntimeAgent, "get_tools", AsyncMock(return_value=[MagicMock()])) as tools,
            patch.object(RuntimeAgent, "get_llm", return_value=generator),
            patch("django_ai_sdk.adapters.base.Agent") as agent_cls,
        ):
            title = await generate_thread_title(agent=agent, messages=[user_message("hi")])

        assert title == "A Title"
        tools.assert_not_awaited()
        agent_cls.assert_not_called()


@pytest.mark.django_db(transaction=True)
class TestStructuredCallsDoNotResolveTools:
    """A workflow step with output_fields must not reach out to every integration."""

    async def test_a_response_format_call_neither_resolves_nor_warns(self):
        from uuid import uuid4

        from pydantic import BaseModel

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
            result = await agent.run([user_message()], response_format=Out)

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
            assert await agent.run([user_message()]) == "fine"
