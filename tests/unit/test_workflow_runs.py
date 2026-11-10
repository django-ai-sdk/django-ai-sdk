"""A queued run: sequencing, declared inputs, listeners, and the task entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_sdk.workflows.registry import register_workflow, reset_registry
from django_ai_sdk.workflows.schemas import (
    FieldDefinition,
    ListenerDefinition,
    StepDefinition,
    WorkflowDefinition,
)
from django_ai_sdk.workflows.steps import WorkflowStopped
from tests.mocks.workflow import AGENT_GET, run_now
from tests.mocks.workflow import fake_agent as make_agent


def make_workflow(*steps, **kwargs):
    return WorkflowDefinition(steps=list(steps), **kwargs)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestSteps:
    async def test_single_step_returns_output_keyed_by_step_name(self):
        agent = make_agent("hello")
        workflow = make_workflow(StepDefinition(name="result", agent_id="a1"))

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            outputs, _ = await run_now(workflow)

        assert outputs == {"result": "hello"}

    async def test_two_steps_independent(self):
        a1, a2 = make_agent("first"), make_agent("second")
        workflow = make_workflow(
            StepDefinition(name="step1", agent_id="a1"),
            StepDefinition(name="step2", agent_id="a2"),
        )

        with patch(AGENT_GET, AsyncMock(side_effect=[a1, a2])):
            outputs, _ = await run_now(workflow)

        assert outputs == {"step1": "first", "step2": "second"}

    async def test_a_required_step_reaches_the_agent_as_a_message(self):
        a1 = make_agent("prior result")
        a2 = make_agent("final")
        captured = []

        async def capture_run(messages, **kwargs):
            captured.append(messages)
            return "final"

        a2.run = capture_run

        workflow = make_workflow(
            StepDefinition(name="step1", agent_id="a1"),
            StepDefinition(name="step2", agent_id="a2", requires=["step1"]),
        )

        with patch(AGENT_GET, AsyncMock(side_effect=[a1, a2])):
            await run_now(workflow)

        assert any(m.role == "user" and "prior result" in m.content for m in captured[0])

    async def test_a_required_name_with_no_source_is_refused_before_any_step_runs(self):
        """A typo reaches the author, rather than the agent as a silent gap."""
        agent = make_agent("ok")
        workflow = make_workflow(StepDefinition(name="result", agent_id="a1", requires=["missing"]))

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            pytest.raises(ImproperlyConfigured, match="missing"),
        ):
            await run_now(workflow)

    async def test_a_definition_keeps_the_agents_own_system_prompt(self):
        agent = make_agent()
        workflow = make_workflow(StepDefinition(name="result", agent_id="a1"))

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            await run_now(workflow)

        _, kwargs = agent.run.call_args
        assert kwargs.get("system_prompt") is None

    async def test_a_definition_with_no_steps_is_refused(self):
        """One rule for all three doors: register_workflow, create, and execute all refuse it."""
        with pytest.raises(ImproperlyConfigured, match="no steps to run"):
            await run_now(make_workflow())

    async def test_a_continue_failure_leaves_the_step_out_of_the_outputs(self):
        """No synthetic error key: a failed step simply produced nothing."""
        from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext

        class BoomStep(Step):
            async def run(self, ctx: WorkflowContext) -> StepOutcome:
                return StepOutcome(status="failed", detail="provider 503")

        import tests.unit.test_workflow_runs as mod

        mod.BoomStepForFailure = BoomStep

        workflow = make_workflow(
            StepDefinition(type="boom", name="extract", on_error="continue"),
            StepDefinition(type="boom", name="second", on_error="continue"),
        )

        with override_settings(
            AI_SDK_WORKFLOWS={"STEPS": {"boom": "tests.unit.test_workflow_runs.BoomStepForFailure"}}
        ):
            outputs, run = await run_now(workflow)

        assert outputs == {}
        assert run.outputs == {}


def test_published_carries_only_what_completed():
    from django_ai_sdk.workflows.runs import _published
    from django_ai_sdk.workflows.steps import StepOutcome

    outcomes = {
        "ocr": StepOutcome(output="the text"),
        "extract": StepOutcome(status="failed", detail="provider 503"),
        "match": StepOutcome(status="skipped", detail="extract failed"),
    }

    assert _published(outcomes) == {"ocr": "the text"}


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestDeclaredInputs:
    """Input and output are the same mechanism, so a run's inputs are checked once."""

    async def test_declared_inputs_are_coerced_before_the_first_step(self):
        from django_ai_sdk.common import ChatMessage

        captured = []

        async def capture_run(messages, **kwargs):
            captured.append(messages)
            return "ok"

        agent = make_agent()
        agent.run = capture_run
        workflow = make_workflow(
            StepDefinition(name="reply", agent_id="a1", history=["history"]),
            input_fields={"history": FieldDefinition(type="list")},
        )

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            await run_now(workflow, inputs={"history": [{"role": "user", "content": "ahoy"}]})

        assert isinstance(captured[0][0], ChatMessage)
        assert captured[0][0].content == "ahoy"

    async def test_a_missing_declared_input_is_refused(self):
        workflow = make_workflow(
            StepDefinition(name="result", agent_id="a1"),
            input_fields={"document": FieldDefinition()},
        )

        with pytest.raises(ValueError, match="does not declare, or is missing"):
            await run_now(workflow, inputs={})

    async def test_an_undeclared_extra_is_dropped(self):
        """The schema is the contract, so a stray key never reaches a step."""
        seen = {}
        from django_ai_sdk.workflows.steps import Step, StepOutcome, WorkflowContext

        class Peek(Step):
            async def run(self, ctx: WorkflowContext) -> StepOutcome:
                seen.update(ctx.inputs)
                return StepOutcome(output="ok")

        import tests.unit.test_workflow_runs as mod

        mod.PeekStep = Peek

        workflow = make_workflow(
            StepDefinition(type="peek", name="peek"),
            input_fields={"document": FieldDefinition()},
        )

        with override_settings(
            AI_SDK_WORKFLOWS={"STEPS": {"peek": "tests.unit.test_workflow_runs.PeekStep"}}
        ):
            await run_now(workflow, inputs={"document": "doc-1", "stray": "x"})

        assert seen == {"document": "doc-1"}

    async def test_a_definition_declaring_nothing_takes_what_it_is_given(self):
        """A code-authored pipeline is not obliged to describe itself in JSON."""
        agent = make_agent("ok")
        workflow = make_workflow(StepDefinition(name="result", agent_id="a1"))

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            _outputs, run = await run_now(workflow, inputs={"anything": 1})

        await run.arefresh_from_db()
        assert run.inputs == {"anything": 1}


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestStructuredOutput:
    async def test_output_fields_calls_with_response_format(self):
        from pydantic import BaseModel

        class Classification(BaseModel):
            label: str

        agent = MagicMock()
        agent.run = AsyncMock(return_value=Classification(label="sports"))

        workflow = make_workflow(
            StepDefinition(
                name="classification",
                agent_id="a1",
                output_fields={"label": FieldDefinition(type="str", description="category")},
            )
        )

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            outputs, _ = await run_now(workflow)

        assert outputs["classification"] == {"label": "sports"}
        _, kwargs = agent.run.call_args
        assert kwargs.get("response_format") is not None

    async def test_a_declared_schema_that_does_not_come_back_fails_the_step(self):
        agent = make_agent("plain string, not a model")
        workflow = make_workflow(
            StepDefinition(
                name="result", agent_id="a1", output_fields={"x": FieldDefinition(type="int")}
            )
        )

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            pytest.raises(WorkflowStopped),
        ):
            await run_now(workflow)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestListeners:
    """A definition's listeners are built and attached; the recorder is always there."""

    async def test_a_workflow_listener_fires_for_every_step(self):
        from django_ai_sdk.workflows.listeners import RunListener
        from django_ai_sdk.workflows.models import WorkflowRun

        events: list[tuple[str, str]] = []

        class Capture(RunListener):
            async def on_run_start(self, ctx):
                events.append(("run_start", ""))

            async def on_step_end(self, ctx, step, outcome):
                events.append(("step_end", step.name))

            async def on_run_end(self, ctx, error):
                events.append(("run_end", ""))

        import tests.unit.test_workflow_runs as mod

        mod.CaptureListener = Capture
        agent = make_agent("data")
        workflow = make_workflow(
            StepDefinition(name="first", agent_id="a1"),
            StepDefinition(name="second", agent_id="a1"),
            listeners=[ListenerDefinition(type="capture")],
        )

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            override_settings(
                AI_SDK_WORKFLOWS={
                    "LISTENERS": {"capture": "tests.unit.test_workflow_runs.CaptureListener"}
                }
            ),
        ):
            await run_now(workflow)

        assert events == [
            ("run_start", ""),
            ("step_end", "first"),
            ("step_end", "second"),
            ("run_end", ""),
        ]

    async def test_a_step_listener_fires_for_that_step_alone(self):
        """The difference the two attachment points buy."""
        from django_ai_sdk.workflows.listeners import RunListener
        from django_ai_sdk.workflows.models import WorkflowRun

        events: list[str] = []

        class Capture(RunListener):
            async def on_step_end(self, ctx, step, outcome):
                events.append(step.name)

        import tests.unit.test_workflow_runs as mod

        mod.StepOnlyListener = Capture
        agent = make_agent("data")
        workflow = make_workflow(
            StepDefinition(
                name="first", agent_id="a1", listeners=[ListenerDefinition(type="capture")]
            ),
            StepDefinition(name="second", agent_id="a1"),
        )

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            override_settings(
                AI_SDK_WORKFLOWS={
                    "LISTENERS": {"capture": "tests.unit.test_workflow_runs.StepOnlyListener"}
                }
            ),
        ):
            await run_now(workflow)

        assert events == ["first"]

    async def test_an_unregistered_listener_stops_the_run_rather_than_being_skipped(self):
        """The registry is the gate; a name outside it is a configuration error."""
        agent = make_agent("data")
        workflow = make_workflow(
            StepDefinition(name="result", agent_id="a1"),
            listeners=[ListenerDefinition(type="carrier_pigeon")],
        )

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            override_settings(AI_SDK_WORKFLOWS={"LISTENERS": {}}),
            pytest.raises(ImproperlyConfigured, match="carrier_pigeon"),
        ):
            await run_now(workflow)

    async def test_a_listener_reads_the_run_and_the_step_it_was_configured_with(self):
        from tests.factories.db import UserFactory

        from django_ai_sdk.workflows.listeners import RunListener
        from django_ai_sdk.workflows.models import WorkflowRun

        seen: list[dict] = []

        class Capture(RunListener):
            @classmethod
            def validate_config(cls, config):
                pass

            async def on_run_end(self, ctx, error):
                seen.append(
                    {
                        "payload": ctx.step(self.config["step"]),
                        "user": ctx.user,
                        "workflow": ctx.workflow,
                        "run_id": ctx.run_id,
                        # The row is closed before any listener hears the run ended.
                        "status": (await WorkflowRun.objects.aget(pk=ctx.run_id)).status,
                    }
                )

        import tests.unit.test_workflow_runs as mod

        mod.CaptureConfigListener = Capture
        user = await UserFactory.acreate()
        agent = make_agent("data")
        workflow = make_workflow(
            StepDefinition(name="result", agent_id="a1"),
            name="ships-log",
            listeners=[ListenerDefinition(type="capture", config={"step": "result"})],
        )

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            override_settings(
                AI_SDK_WORKFLOWS={
                    "LISTENERS": {"capture": "tests.unit.test_workflow_runs.CaptureConfigListener"}
                }
            ),
        ):
            # Registered, so the run is keyed by its name.
            register_workflow(workflow)
            try:
                _outputs, run = await run_now(workflow, user=user)
            finally:
                reset_registry()

        assert seen[0]["payload"] == "data"
        assert seen[0]["user"] == user
        assert seen[0]["workflow"] == "ships-log"
        assert seen[0]["run_id"] == str(run.id)
        assert seen[0]["status"] == WorkflowRun.Status.COMPLETED


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestTheQueuedEntryPoint:
    """A recorded ending stays off the queue; an unrecorded crash reaches it.

    A stopped step and a refused claim are already on the run's own rows, so the
    task result reports only what nothing else did.
    """

    async def _run_for(self, *steps):
        from django_ai_sdk.workflows.models import WorkflowRun

        workflow = make_workflow(*steps)
        return await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(), inputs={}
        )

    async def test_a_queued_run_reaches_the_workflow(self):
        from django_ai_sdk.workflows.tasks import execute_workflow

        agent = make_agent("done")
        run = await self._run_for(StepDefinition(name="result", agent_id="a1"))

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            await sync_to_async(execute_workflow.call)(str(run.id))

        await run.arefresh_from_db()
        assert run.outputs == {"result": "done"}

    async def test_a_step_failure_does_not_escape_the_task(self):
        from django_ai_sdk.workflows.models import WorkflowRun
        from django_ai_sdk.workflows.tasks import execute_workflow

        # A declared schema the agent does not return fails the step, and the
        # step's on_error stops the run.
        agent = make_agent("plain string, not a model")
        run = await self._run_for(
            StepDefinition(
                name="result", agent_id="a1", output_fields={"x": FieldDefinition(type="int")}
            )
        )

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            await sync_to_async(execute_workflow.call)(str(run.id))

        await run.arefresh_from_db()
        assert run.status == WorkflowRun.Status.FAILED
        assert run.error

    async def test_a_superseded_attempt_does_not_escape_the_task(self):
        from django_ai_sdk.workflows.models import WorkflowRun
        from django_ai_sdk.workflows.steps import RunSuperseded
        from django_ai_sdk.workflows.tasks import execute_workflow

        agent = make_agent("done")
        run = await self._run_for(StepDefinition(name="result", agent_id="a1"))

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            patch(
                "django_ai_sdk.workflows.runs.RunRecorder.on_step_start",
                AsyncMock(side_effect=RunSuperseded("result")),
            ),
        ):
            await sync_to_async(execute_workflow.call)(str(run.id))

        # The attempt that replaced this one owns the run: this one must not stamp
        # failure over its progress.
        await run.arefresh_from_db()
        assert run.status == WorkflowRun.Status.RUNNING
        assert run.error == ""

    async def test_a_crash_still_reaches_the_queue(self):
        from django_ai_sdk.workflows.tasks import execute_workflow

        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        run = await self._run_for(StepDefinition(name="result", agent_id="a1"))

        with (
            patch(AGENT_GET, AsyncMock(return_value=agent)),
            pytest.raises(RuntimeError),
        ):
            await sync_to_async(execute_workflow.call)(str(run.id))


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
class TestRunState:
    """A run's inputs seed its state, persist on the row, and survive a resume."""

    async def test_inputs_are_persisted_on_the_row(self):
        agent = make_agent("ok")
        workflow = make_workflow(
            StepDefinition(name="result", agent_id="a1"),
            input_fields={"document": FieldDefinition()},
        )

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            _outputs, run = await run_now(workflow, inputs={"document": "doc-1"})

        await run.arefresh_from_db()
        assert run.inputs == {"document": "doc-1"}

    async def test_messages_are_persisted_json_safe(self):
        from django_ai_sdk.common import ChatMessage

        agent = make_agent("ok")
        workflow = make_workflow(
            StepDefinition(name="result", agent_id="a1"),
            input_fields={"history": FieldDefinition(type="list")},
        )

        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            _outputs, run = await run_now(
                workflow, inputs={"history": [ChatMessage(role="user", content="ahoy")]}
            )

        await run.arefresh_from_db()
        assert run.inputs["history"][0]["content"] == "ahoy"

    async def test_resuming_reuses_the_rows_stored_inputs(self):
        from django_ai_sdk.workflows.models import WorkflowRun

        agent = make_agent("ok")
        workflow = make_workflow(
            StepDefinition(name="result", agent_id="a1"),
            input_fields={"document": FieldDefinition()},
        )
        run = await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(),
            status=WorkflowRun.Status.PENDING,
            inputs={"document": "doc-1"},
        )

        # The caller passes no inputs; the row supplies them.
        with patch(AGENT_GET, AsyncMock(return_value=agent)):
            outputs, _ = await run_now(workflow, run=run)

        assert outputs["result"] == "ok"

    async def test_a_completed_step_is_not_re_run_on_resume(self):
        from django_ai_sdk.workflows.models import WorkflowRun, StepRun

        agent = make_agent("ok")
        workflow = make_workflow(
            StepDefinition(name="first", agent_id="a1"),
            StepDefinition(name="second", agent_id="a1", requires=["first"]),
        )
        # A failed run that was resumed: pending again, with its first step done.
        run = await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(), status=WorkflowRun.Status.PENDING
        )
        await StepRun.objects.acreate(
            run=run,
            sequence=0,
            step_name="first",
            status=StepRun.Status.COMPLETED,
            output="stored",
        )

        with patch(AGENT_GET, AsyncMock(return_value=agent)) as get:
            outputs, _ = await run_now(workflow, run=run)

        assert outputs == {"first": "stored", "second": "ok"}
        assert get.await_count == 1

    async def test_a_completed_run_short_circuits(self):
        from django_ai_sdk.workflows.models import WorkflowRun

        workflow = make_workflow(StepDefinition(name="result", agent_id="a1"))
        run = await WorkflowRun.objects.acreate(
            workflow_definition=workflow.model_dump(),
            status=WorkflowRun.Status.COMPLETED,
            outputs={"result": "already done"},
        )

        outputs, _ = await run_now(workflow, run=run)

        assert outputs == {"result": "already done"}

    async def test_a_long_step_keeps_beating_so_the_sweep_leaves_it(self):
        import asyncio

        from django_ai_sdk.workflows.models import WorkflowRun

        beats = []

        async def slow(*args, **kwargs):
            row = await WorkflowRun.objects.aget(status=WorkflowRun.Status.RUNNING)
            beats.append(row.heartbeat_at)
            await asyncio.sleep(0.2)
            row = await WorkflowRun.objects.aget(pk=row.pk)
            beats.append(row.heartbeat_at)
            return "ok"

        agent = make_agent("ok")
        agent.run = AsyncMock(side_effect=slow)
        workflow = make_workflow(StepDefinition(name="result", agent_id="a1"))
        with (
            patch("django_ai_sdk.workflows.runs.HEARTBEAT_SECONDS", 0.02),
            patch(AGENT_GET, AsyncMock(return_value=agent)),
        ):
            outputs, _ = await run_now(workflow)

        assert outputs == {"result": "ok"}
        assert beats[1] > beats[0]
