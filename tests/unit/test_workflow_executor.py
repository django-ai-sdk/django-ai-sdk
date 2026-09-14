"""
Unit tests for WorkflowExecutor — step sequencing, context injection, actions.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.workflows import StepFailed
from django_ai_sdk.workflows.executor import WorkflowExecutor
from django_ai_sdk.workflows.schemas import (
    StepField,
    WorkflowAction,
    WorkflowDefinition,
    WorkflowStep,
)


def make_agent(run_return="agent result"):
    a = MagicMock()
    a.run = AsyncMock(return_value=run_return)
    return a


def make_workflow(*steps, actions=None):
    return WorkflowDefinition(steps=list(steps), actions=actions or [])


@pytest.fixture
def executor():
    return WorkflowExecutor()


# ============================================================================
# Step execution
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestWorkflowExecutorSteps:
    async def test_single_step_returns_output(self, executor):
        agent = make_agent("hello")
        step = WorkflowStep(agent_id="a1", output_key="result")
        workflow = make_workflow(step)

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            outputs, _ = await executor.run(workflow, [])

        assert outputs == {"result": "hello"}

    async def test_two_steps_independent(self, executor):
        a1, a2 = make_agent("first"), make_agent("second")
        workflow = make_workflow(
            WorkflowStep(agent_id="a1", output_key="step1"),
            WorkflowStep(agent_id="a2", output_key="step2"),
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get",
            AsyncMock(side_effect=[a1, a2]),
        ):
            outputs, _ = await executor.run(workflow, [])

        assert outputs["step1"] == "first"
        assert outputs["step2"] == "second"

    async def test_a_required_name_reaches_the_agent_as_a_message(self, executor):
        a1 = make_agent("prior result")
        a2 = make_agent("final")
        captured = []

        async def capture_run(messages, **kwargs):
            captured.append(messages)
            return "final"

        a2.run = capture_run

        workflow = make_workflow(
            WorkflowStep(agent_id="a1", output_key="step1"),
            WorkflowStep(agent_id="a2", output_key="step2", requires=["step1"]),
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get",
            AsyncMock(side_effect=[a1, a2]),
        ):
            await executor.run(workflow, [])

        injected = captured[0]
        assert any(m.role == "user" and "prior result" in m.content for m in injected)

    async def test_a_required_name_with_no_source_is_refused_before_any_step_runs(self, executor):
        """A typo reaches the author, rather than the agent as a silent gap."""
        agent = make_agent("ok")
        workflow = make_workflow(
            WorkflowStep(agent_id="a1", output_key="result", requires=["missing_key"]),
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            pytest.raises(ImproperlyConfigured, match="missing_key"),
        ):
            await executor.run(workflow, [])

    async def test_system_prompt_override_passed(self, executor):
        agent = make_agent()
        workflow = make_workflow(
            WorkflowStep(
                agent_id="a1",
                output_key="result",
                system_prompt_override="You are a pirate.",
            )
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            await executor.run(workflow, [])

        _, kwargs = agent.run.call_args
        assert kwargs.get("system_prompt") == "You are a pirate."

    async def test_a_definition_with_no_steps_is_refused(self, executor):
        """One rule for all three doors: register, create, and execute all refuse it."""
        with pytest.raises(ImproperlyConfigured, match="at least one step"):
            await executor.run(make_workflow(), [])

    async def test_error_key_lands_in_run_outputs_and_actions(self, executor):
        from django.test import override_settings

        from django_ai_sdk.workflows.steps import Step, StepContext, StepOutcome

        class BoomStep(Step):
            async def run(self, ctx: StepContext) -> StepOutcome:
                return StepOutcome(status="failed", detail="provider 503")

        import tests.unit.test_workflow_executor as mod

        mod.BoomStepForErrorKey = BoomStep
        received: list[object] = []

        class CaptureAction:
            async def execute(self, payload, context):
                received.append(payload)

        workflow = make_workflow(
            WorkflowStep(
                type="boom",
                name="extract",
                output_key="kvk",
                on_error="continue",
                error_key="extract_err",
            ),
            actions=[WorkflowAction(type="capture", input_key="extract_err")],
        )

        with (
            override_settings(
                AI_SDK_WORKFLOW_STEPS={
                    "boom": "tests.unit.test_workflow_executor.BoomStepForErrorKey",
                }
            ),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            outputs, run = await executor.run(workflow, [])

        assert outputs["extract_err"] == {"step": "extract", "error": "provider 503"}
        assert run.outputs == outputs
        assert received == [outputs["extract_err"]]


def test_published_includes_error_key_payload():
    from django_ai_sdk.workflows.executor import _published
    from django_ai_sdk.workflows.steps import OnError, StepOutcome
    from tests.mocks.workflow import FakeStep

    steps = [
        FakeStep(
            "extract",
            provides="kvk",
            on_error=OnError.CONTINUE,
            error_key="extract_err",
        )
    ]
    outcomes = {"extract": StepOutcome(status="failed", detail="provider 503")}
    assert _published(steps, outcomes) == {
        "extract_err": {"step": "extract", "error": "provider 503"}
    }


# ============================================================================
# Structured output (output_fields)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestWorkflowExecutorStructuredOutput:
    async def test_output_fields_calls_with_response_format(self, executor):
        from pydantic import BaseModel

        dynamic_result = MagicMock(spec=BaseModel)
        dynamic_result.model_dump.return_value = {"label": "sports"}

        agent = MagicMock()
        agent.run = AsyncMock(return_value=dynamic_result)

        workflow = make_workflow(
            WorkflowStep(
                agent_id="a1",
                output_key="classification",
                output_fields={"label": StepField(type="str", description="category")},
            )
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            outputs, _ = await executor.run(workflow, [])

        assert outputs["classification"] == {"label": "sports"}
        _, kwargs = agent.run.call_args
        assert kwargs.get("response_format") is not None

    async def test_a_declared_schema_that_does_not_come_back_fails_the_step(self, executor):
        agent = make_agent("plain string, not a model")
        workflow = make_workflow(
            WorkflowStep(
                agent_id="a1",
                output_key="result",
                output_fields={"x": StepField(type="int")},
            )
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            pytest.raises(StepFailed),
        ):
            await executor.run(workflow, [])


# ============================================================================
# Actions
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db
class TestWorkflowExecutorActions:
    async def test_action_called_with_full_outputs_when_no_input_key(self, executor):
        agent = make_agent("data")
        received = []

        class CaptureAction:
            async def execute(self, payload, context):
                received.append(payload)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(agent_id="a1", output_key="result")],
            actions=[WorkflowAction(type="capture")],
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            await executor.run(workflow, [])

        assert received == [{"result": "data"}]

    async def test_action_called_with_specific_input_key(self, executor):
        agent = make_agent("step_data")
        received = []

        class CaptureAction:
            async def execute(self, payload, context):
                received.append(payload)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(agent_id="a1", output_key="summary")],
            actions=[WorkflowAction(type="capture", input_key="summary")],
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            await executor.run(workflow, [])

        assert received == ["step_data"]

    async def test_unknown_action_type_warns_and_skips(self, executor, caplog):
        agent = make_agent("data")
        workflow = WorkflowDefinition(
            steps=[WorkflowStep(agent_id="a1", output_key="result")],
            actions=[WorkflowAction(type="nonexistent")],
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch("django_ai_sdk.workflows.executor.get_action_registry", return_value={}),
        ):
            outputs, _ = await executor.run(workflow, [])

        assert outputs["result"] == "data"
        assert "nonexistent" in caplog.text

    async def test_action_input_key_missing_warns_and_skips(self, executor, caplog):
        agent = make_agent("data")
        executed = []

        class CaptureAction:
            async def execute(self, payload, context):
                executed.append(payload)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(agent_id="a1", output_key="result")],
            actions=[WorkflowAction(type="capture", input_key="does_not_exist")],
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            await executor.run(workflow, [])

        assert executed == []
        assert "does_not_exist" in caplog.text

    async def test_action_receives_context_from_the_run(self, executor):
        from tests.factories.db import UserFactory

        from django_ai_sdk.workflows.actions import ActionContext

        agent = make_agent("data")
        received: list[tuple[object, ActionContext]] = []

        class CaptureAction:
            async def execute(self, payload, context):
                received.append((payload, context))

        user = await UserFactory.acreate()
        workflow = WorkflowDefinition(
            name="ships-log",
            steps=[WorkflowStep(agent_id="a1", output_key="result")],
            actions=[WorkflowAction(type="capture")],
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            await executor.run(workflow, [], user=user)

        payload, context = received[0]
        assert payload == {"result": "data"}
        assert context.user == user
        assert context.agent_id == "a1"
        assert context.source == "ships-log"

    async def test_unnamed_workflow_source_is_the_run_id(self, executor):
        from django_ai_sdk.workflows.actions import ActionContext

        agent = make_agent("data")
        received: list[ActionContext] = []

        class CaptureAction:
            async def execute(self, payload, context):
                received.append(context)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(agent_id="a1", output_key="result")],
            actions=[WorkflowAction(type="capture")],
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            _, run = await executor.run(workflow, [])

        assert received[0].source == f"workflow:{run.id}"
        assert received[0].agent_id == "a1"


# ============================================================================
# The queued entry point
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestTheQueuedEntryPoint:
    """A recorded ending stays off the queue; an unrecorded crash reaches it.

    A stopped step and a refused claim are already on the run's own rows, so the
    task result reports only what nothing else did.
    """

    async def _run_for(self, *steps):
        from django_ai_sdk.workflows import WorkflowRun

        workflow = make_workflow(*steps)
        return await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(), inputs={"messages": []}
        )

    async def test_a_queued_run_reaches_the_workflow(self):
        from django_ai_sdk.workflows.tasks import _execute_async

        agent = make_agent("done")
        run = await self._run_for(WorkflowStep(agent_id="a1", output_key="result"))

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            await _execute_async(str(run.id))

        await run.arefresh_from_db()
        assert run.outputs == {"result": "done"}

    async def test_a_step_failure_does_not_escape_the_task(self):
        from django_ai_sdk.workflows import WorkflowRun
        from django_ai_sdk.workflows.tasks import _execute_async

        # A declared schema the agent does not return fails the step, and the
        # step's on_error stops the run.
        agent = make_agent("plain string, not a model")
        run = await self._run_for(
            WorkflowStep(
                agent_id="a1", output_key="result", output_fields={"x": StepField(type="int")}
            )
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            await _execute_async(str(run.id))

        await run.arefresh_from_db()
        assert run.status == WorkflowRun.Status.FAILED
        assert run.error

    async def test_a_claim_refusal_does_not_escape_the_task(self):
        from django_ai_sdk.workflows.steps import StepAlreadyRunning
        from django_ai_sdk.workflows.tasks import _execute_async

        agent = make_agent("done")
        run = await self._run_for(WorkflowStep(agent_id="a1", output_key="result"))

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            patch(
                "django_ai_sdk.workflows.sink.WorkflowRunStepSink.open",
                AsyncMock(side_effect=StepAlreadyRunning("result")),
            ),
        ):
            await _execute_async(str(run.id))

    async def test_a_crash_still_reaches_the_queue(self):
        from django_ai_sdk.workflows.tasks import _execute_async

        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        run = await self._run_for(WorkflowStep(agent_id="a1", output_key="result"))

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            pytest.raises(RuntimeError),
        ):
            await _execute_async(str(run.id))


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestRunState:
    """A run's inputs seed its state, persist on the row, and survive a resume."""

    async def test_inputs_seed_run_state_and_are_persisted(self, executor):
        agent = make_agent("ok")
        workflow = make_workflow(WorkflowStep(agent_id="a1", output_key="result"))

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            _outputs, run = await executor.run(workflow, inputs={"document": "doc-1"})

        await run.arefresh_from_db()
        assert run.inputs["document"] == "doc-1"
        # Chat-shaped and data-shaped runs seed one bag.
        assert run.inputs["messages"] == []

    async def test_a_step_may_require_a_run_input(self, executor):
        captured = []

        async def capture_run(messages, **kwargs):
            captured.append(messages)
            return "ok"

        agent = make_agent()
        agent.run = capture_run
        workflow = make_workflow(
            WorkflowStep(agent_id="a1", output_key="result", requires=["document"])
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            await executor.run(workflow, inputs={"document": "doc-1"})

        assert any("doc-1" in m.content for m in captured[0] if m.role == "user")

    async def test_resuming_reuses_the_rows_stored_inputs(self, executor):
        from django_ai_sdk.workflows import WorkflowRun

        agent = make_agent("ok")
        workflow = make_workflow(
            WorkflowStep(agent_id="a1", output_key="result", requires=["document"])
        )
        run = await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(),
            status=WorkflowRun.Status.PENDING,
            inputs={"document": "doc-1", "messages": []},
        )

        # The caller passes no inputs; the row supplies them.
        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get", AsyncMock(return_value=agent)
        ):
            outputs, _ = await executor.run(workflow, workflow_run=run)

        assert outputs["result"] == "ok"

    async def test_a_completed_run_short_circuits(self, executor):
        from django_ai_sdk.workflows import WorkflowRun

        workflow = make_workflow(WorkflowStep(agent_id="a1", output_key="result"))
        run = await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(),
            status=WorkflowRun.Status.COMPLETED,
            outputs={"result": "already done"},
        )

        outputs, _ = await executor.run(workflow, workflow_run=run)

        assert outputs == {"result": "already done"}
