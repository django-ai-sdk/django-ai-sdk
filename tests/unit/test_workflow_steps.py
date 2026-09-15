"""AgentStep: the model call as one step, and what its outcome carries.

One class serves both ways of naming an agent — a class a code-declared step
imports, and the id a JSON definition carries — so there is no second agent step.
"""

import pytest
from pydantic import BaseModel

from django_ai_sdk import Agent
from django_ai_sdk.workflows import AgentStep, StepOutcome, WorkflowContext

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
    name = "extract"

    async def user_message(self, ctx: WorkflowContext) -> str:
        return "the document's text"


@pytest.fixture(autouse=True)
def fresh_agent():
    StubAgent.seen = []
    yield
    StubAgent.seen = []


async def test_an_agent_step_returns_the_structured_result_as_its_output():
    StubAgent.result = "the answer"

    outcome = await StubStep().run(WorkflowContext())

    assert outcome.status == "completed"
    assert outcome.output == "the answer"


async def test_an_agent_step_reports_failed_when_the_agent_returns_nothing():
    StubAgent.result = None

    outcome = await StubStep().run(WorkflowContext())

    assert outcome.status == "failed"
    assert outcome.detail == "the agent returned nothing"


async def test_an_agent_step_passes_prompt_and_principal_to_the_agent():
    StubAgent.result = "the answer"
    principal = object()

    await StubStep().run(WorkflowContext(principal=principal))

    seen = StubAgent.seen[-1]
    assert seen["user"] is principal
    assert seen["messages"][0].content == "the document's text"


class TestTheSystemPrompt:
    """`instructions` is the override; without one the agent keeps its own."""

    async def test_no_instructions_leaves_the_agents_own_prompt_alone(self):
        StubAgent.result = "the answer"

        await StubStep().run(WorkflowContext())

        assert StubAgent.seen[-1]["system_prompt"] is None

    async def test_declared_instructions_replace_it(self):
        class Overridden(StubStep):
            instructions = "Reply as a ship's log entry."

        StubAgent.result = "the answer"

        await Overridden().run(WorkflowContext())

        assert StubAgent.seen[-1]["system_prompt"] == "Reply as a ship's log entry."


class TestTheDefaultTurn:
    """An agent step with no `user_message` sends what the steps it requires made."""

    async def test_it_renders_each_required_steps_output(self):
        step = AgentStep()
        step.agent = StubAgent
        step.name = "judge"
        step.requires = ("forecast",)
        StubAgent.result = "the answer"

        ctx = WorkflowContext(steps={"forecast": StepOutcome(output="gale force 8")})
        await step.run(ctx)

        assert StubAgent.seen[-1]["messages"][0].content == "[forecast]\ngale force 8"

    async def test_a_named_history_input_becomes_the_transcript(self):
        from django_ai_sdk.common import ChatMessage

        step = AgentStep()
        step.agent = StubAgent
        step.name = "reply"
        step.history = ("history",)
        StubAgent.result = "the answer"

        ctx = WorkflowContext(inputs={"history": [ChatMessage(role="user", content="ahoy")]})
        await step.run(ctx)

        # Nothing required, so the transcript is the whole conversation.
        assert [m.content for m in StubAgent.seen[-1]["messages"]] == ["ahoy"]

    async def test_a_history_input_is_coerced_from_the_json_it_crossed_the_queue_as(self):
        """The inline and the queued path have to send the same conversation."""
        step = AgentStep()
        step.agent = StubAgent
        step.name = "reply"
        step.history = ("history",)
        StubAgent.result = "the answer"

        ctx = WorkflowContext(inputs={"history": [{"role": "user", "content": "ahoy"}]})
        await step.run(ctx)

        assert [m.content for m in StubAgent.seen[-1]["messages"]] == ["ahoy"]

    async def test_an_input_nobody_named_is_not_mistaken_for_a_transcript(self):
        """Named, not guessed: a list of role-shaped rows is data until it is declared."""
        step = AgentStep()
        step.agent = StubAgent
        step.name = "reply"
        StubAgent.result = "the answer"

        ctx = WorkflowContext(inputs={"members": [{"role": "admin", "content": "secret"}]})
        await step.run(ctx)

        assert StubAgent.seen[-1]["messages"] == []


class TestTheChatGate:
    """A code pipeline is not a permission bypass — the same gate a JSON step gets."""

    async def test_a_denied_principal_never_reaches_the_agent(self):
        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        class Gated(StubStep):
            pass

        Gated.agent = type("GatedAgent", (StubAgent,), {"permissions": [DenyAll]})
        StubAgent.result = "the answer"

        with pytest.raises(PermissionDenied):
            await Gated().run(WorkflowContext())

        assert StubAgent.seen == []

    async def test_a_permitted_principal_reaches_the_agent(self):
        from django_ai_sdk.permissions import AllowAll

        class Open(StubStep):
            pass

        Open.agent = type("OpenAgent", (StubAgent,), {"permissions": [AllowAll]})
        StubAgent.result = "the answer"

        outcome = await Open().run(WorkflowContext())

        assert outcome.status == "completed"
        assert len(StubAgent.seen) == 1


def test_an_outcome_defaults_to_completed_with_no_output():
    outcome = StepOutcome(detail="one line")

    assert outcome.status == "completed"
    assert outcome.output is None
    assert outcome.detail == "one line"


class TestTheTwoNamespaces:
    """Inputs and step results never contend, so neither has to be renamed."""

    def test_an_input_and_a_step_may_share_a_name(self):
        ctx = WorkflowContext(
            inputs={"document": "what the caller passed"},
            steps={"document": StepOutcome(output="what the step made")},
        )

        assert ctx.input("document") == "what the caller passed"
        assert ctx.step("document") == "what the step made"

    def test_a_step_that_did_not_complete_reads_as_the_default(self):
        ctx = WorkflowContext(steps={"ocr": StepOutcome(status="failed", detail="down")})

        assert ctx.step("ocr", "fallback") == "fallback"
        assert ctx.steps["ocr"].detail == "down"

    def test_an_absent_name_reads_as_the_default(self):
        ctx = WorkflowContext()

        assert ctx.input("absent", "fallback") == "fallback"
        assert ctx.step("absent", "fallback") == "fallback"


class Extracted(BaseModel):
    title: str


class TestTheSchemaReachesTheAgentUntouched:
    """A step's `schema` is the agent's `response_format`, and only when set."""

    async def test_a_declared_schema_is_passed(self):
        class WithSchema(StubStep):
            schema = Extracted

        StubAgent.result = Extracted(title="t")

        outcome = await WithSchema().run(WorkflowContext())

        assert StubAgent.seen[-1]["response_format"] is Extracted
        # Dumped, because the outcome goes on the run's record.
        assert outcome.output == {"title": "t"}

    async def test_a_declared_schema_that_did_not_come_back_is_a_failure(self):
        class WithSchema(StubStep):
            schema = Extracted

        StubAgent.result = "just text"

        outcome = await WithSchema().run(WorkflowContext())

        assert outcome.status == "failed"
        assert outcome.detail == "the agent returned no structured output"

    async def test_no_schema_leaves_the_agents_own_response_format_alone(self):
        """Passing None would strip it: Agent.run reads None as "no format"."""
        StubAgent.result = "text"

        await StubStep().run(WorkflowContext())

        assert StubAgent.seen[-1]["response_format"] is OMITTED
