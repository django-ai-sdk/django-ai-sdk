"""FakeLLM: an agent, its chat and a model judge run on canned replies."""

from __future__ import annotations

import pytest

from django_ai_sdk.agents import Agent
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.permissions import AllowAll
from django_ai_sdk.testing import FakeLLM


def agent_on(fake: FakeLLM) -> Agent:
    class Faked(Agent):
        _skip_auto_register = True
        permissions = [AllowAll]
        name = "faked"
        model = "fake"
        title_generation = False
        llm = fake

    return Faked()


def ask(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


@pytest.mark.django_db(transaction=True)
class TestFakeLLM:
    async def test_run_answers_with_the_replies_in_order_then_repeats_the_last(self):
        fake = FakeLLM("one", "two")
        agent = agent_on(fake)
        assert [await agent.run(ask(q)) for q in ("a", "b", "c")] == ["one", "two", "two"]
        assert [call[-1].text for call in fake.calls] == ["a", "b", "c"]

    async def test_with_no_replies_it_echoes(self):
        assert await agent_on(FakeLLM()).run(ask("hello")) == "hello"

    async def test_a_chat_pipeline_streams_the_reply(self):
        from django_ai_sdk.adapters.base import Stream
        from django_ai_sdk.agents.tool_agent import ToolAgent, ToolAgentConfig
        from tests.oversight.helpers import chat, text_of

        agent = agent_on(FakeLLM("On its way."))

        async def get_pipeline_adapter(thread_id=None, user=None):
            generator = agent.get_llm()
            config = ToolAgentConfig(model="fake", system_prompt="s", tools=[])
            pipeline = ToolAgent(config, generator=generator).pipeline()
            return Stream(pipeline=pipeline, generator=generator, store=False)

        agent.get_pipeline_adapter = get_pipeline_adapter  # type: ignore[method-assign]
        parts = await chat(agent, "Where is my refund?")
        assert text_of(parts) == "On its way."

    async def test_a_model_judge_runs_on_it(self):
        from django_ai_sdk.oversight import JudgeLens
        from django_ai_sdk.oversight.judges import LLMJudge
        from django_ai_sdk.oversight.judges.base import ask
        from django_ai_sdk.oversight.types import TurnContext

        fake = FakeLLM('{"value": true, "confidence": 0.9, "rationale": "r"}')
        lens = JudgeLens(name="f", question="Is it rude?")
        answer = await ask(LLMJudge(llm=fake), lens, TurnContext(input="hi"))
        assert answer is not None and (answer.value, answer.confidence) == (True, 0.9)
        assert len(fake.calls) == 1
