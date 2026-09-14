"""AgentStep: the model call as one step, and what its outcome carries."""

import pytest

from pydantic import BaseModel

from django_ai_sdk import Agent
from django_ai_sdk.workflows import AgentStep, StepContext, StepOutcome

# Distinguishes "not passed" from an explicit None, which Agent.run reads as
# "no structured output".
OMITTED = object()


class StubAgent(Agent):
    """Records the call and returns the configured result — the model call
    is the only boundary being faked.
    """

    name = "Stub Agent"
    model = "stub-model"

    result = None
    seen: list[dict] = []

    async def run(self, messages, system_prompt=None, response_format=OMITTED, **kwargs):
        StubAgent.seen.append(
            {
                "messages": messages,
                "system_prompt": system_prompt,
                "response_format": response_format,
                "user": kwargs.get("user"),
            }
        )
        return StubAgent.result


class StubStep(AgentStep):
    agent = StubAgent

    async def user_message(self, ctx: StepContext) -> str:
        return "the document's text"


@pytest.fixture(autouse=True)
def fresh_agent():
    StubAgent.seen = []
    yield
    StubAgent.seen = []


async def test_an_agent_step_returns_the_structured_result_as_its_output():
    StubAgent.result = object()

    outcome = await StubStep().run(StepContext())

    assert outcome.status == "completed"
    assert outcome.output is StubAgent.result


async def test_an_agent_step_reports_failed_when_the_agent_returns_nothing():
    StubAgent.result = None

    outcome = await StubStep().run(StepContext())

    assert outcome.status == "failed"
    assert outcome.detail == "the agent returned nothing"


async def test_an_agent_step_passes_prompt_schema_and_principal_to_the_agent():
    StubAgent.result = object()
    principal = object()

    await StubStep().run(StepContext(principal=principal))

    seen = StubAgent.seen[-1]
    assert seen["system_prompt"] == StubAgent().get_system_prompt()
    assert seen["user"] is principal
    assert seen["messages"][0].content == "the document's text"


class TestTheChatGate:
    """A code pipeline is not a permission bypass — the same gate a JSON step gets."""

    async def test_a_denied_principal_never_reaches_the_agent(self):
        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        class Gated(StubStep):
            pass

        Gated.agent = type("GatedAgent", (StubAgent,), {"permissions": [DenyAll]})
        StubAgent.result = object()

        with pytest.raises(PermissionDenied):
            await Gated().run(StepContext())

        assert StubAgent.seen == []

    async def test_a_permitted_principal_reaches_the_agent(self):
        from django_ai_sdk.permissions import AllowAll

        class Open(StubStep):
            pass

        Open.agent = type("OpenAgent", (StubAgent,), {"permissions": [AllowAll]})
        StubAgent.result = object()

        outcome = await Open().run(StepContext())

        assert outcome.status == "completed"
        assert len(StubAgent.seen) == 1


def test_an_outcome_defaults_to_completed_with_no_output():
    outcome = StepOutcome(detail="one line")

    assert outcome.status == "completed"
    assert outcome.output is None
    assert outcome.detail == "one line"


def test_a_context_reads_outputs_before_inputs():
    """A step's own output wins, so a re-run's value is the one read."""
    ctx = StepContext(inputs={"text": "stale"}, outputs={"text": "fresh"})

    assert ctx.get("text") == "fresh"
    assert ctx.get("absent", "fallback") == "fallback"


class Extracted(BaseModel):
    title: str


class TestTheSchemaReachesTheAgentUntouched:
    """A step's `schema` is the agent's `response_format`, and only when set."""

    async def test_a_declared_schema_is_passed(self):
        class WithSchema(StubStep):
            schema = Extracted

        StubAgent.result = Extracted(title="t")

        await WithSchema().run(StepContext())

        assert StubAgent.seen[-1]["response_format"] is Extracted

    async def test_no_schema_leaves_the_agents_own_response_format_alone(self):
        """Passing None would strip it: Agent.run reads None as "no format"."""
        StubAgent.result = "text"

        await StubStep().run(StepContext())

        assert StubAgent.seen[-1]["response_format"] is OMITTED
