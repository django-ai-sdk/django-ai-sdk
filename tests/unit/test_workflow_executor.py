"""
Unit tests for WorkflowExecutor — step sequencing, context injection, actions.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from django_ai_sdk.workflows.executor import WorkflowExecutor


@contextmanager
def capture_logs(level="WARNING"):
    """Capture loguru log messages at or above the given level."""
    records: list[str] = []
    sink_id = logger.add(lambda msg: records.append(msg), level=level, format="{message}")
    try:
        yield records
    finally:
        logger.remove(sink_id)
from django_ai_sdk.workflows.schemas import (
    StepField,
    WorkflowAction,
    WorkflowDefinition,
    WorkflowStep,
)


def make_assistant(run_return="assistant result"):
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
class TestWorkflowExecutorSteps:
    async def test_single_step_returns_output(self, executor):
        assistant = make_assistant("hello")
        step = WorkflowStep(assistant_id="a1", output_key="result")
        workflow = make_workflow(step)

        with patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)):
            outputs = await executor.run(workflow, [])

        assert outputs == {"result": "hello"}

    async def test_two_steps_independent(self, executor):
        a1, a2 = make_assistant("first"), make_assistant("second")
        workflow = make_workflow(
            WorkflowStep(assistant_id="a1", output_key="step1"),
            WorkflowStep(assistant_id="a2", output_key="step2"),
        )

        with patch(
            "django_ai_sdk.workflows.executor.AssistantService.get",
            AsyncMock(side_effect=[a1, a2]),
        ):
            outputs = await executor.run(workflow, [])

        assert outputs["step1"] == "first"
        assert outputs["step2"] == "second"

    async def test_input_key_injects_prior_output_as_user_message(self, executor):
        a1 = make_assistant("prior result")
        a2 = make_assistant("final")
        captured = []

        async def capture_run(messages, **kwargs):
            captured.append(messages)
            return "final"

        a2.run = capture_run

        workflow = make_workflow(
            WorkflowStep(assistant_id="a1", output_key="step1"),
            WorkflowStep(assistant_id="a2", output_key="step2", input_key="step1"),
        )

        with patch(
            "django_ai_sdk.workflows.executor.AssistantService.get",
            AsyncMock(side_effect=[a1, a2]),
        ):
            await executor.run(workflow, [])

        injected = captured[0]
        assert any(
            m.role == "user" and "prior result" in m.content
            for m in injected
        )

    async def test_input_key_not_found_warns_and_uses_original_messages(self, executor):
        assistant = make_assistant("ok")
        workflow = make_workflow(
            WorkflowStep(assistant_id="a1", output_key="result", input_key="missing_key"),
        )

        with capture_logs() as records:
            with patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)):
                outputs = await executor.run(workflow, [])

        assert outputs["result"] == "ok"
        assert any("missing_key" in r for r in records)

    async def test_system_prompt_override_passed(self, executor):
        assistant = make_assistant()
        workflow = make_workflow(
            WorkflowStep(
                assistant_id="a1",
                output_key="result",
                system_prompt_override="You are a pirate.",
            )
        )

        with patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)):
            await executor.run(workflow, [])

        _, kwargs = assistant.run.call_args
        assert kwargs.get("system_prompt") == "You are a pirate."

    async def test_empty_steps_returns_empty_outputs(self, executor):
        workflow = make_workflow()
        outputs = await executor.run(workflow, [])
        assert outputs == {}


# ============================================================================
# Structured output (output_fields)
# ============================================================================


@pytest.mark.asyncio
class TestWorkflowExecutorStructuredOutput:
    async def test_output_fields_calls_with_response_format(self, executor):
        from pydantic import BaseModel

        dynamic_result = MagicMock(spec=BaseModel)
        dynamic_result.model_dump.return_value = {"label": "sports"}

        assistant = MagicMock()
        assistant.run = AsyncMock(return_value=dynamic_result)

        workflow = make_workflow(
            WorkflowStep(
                assistant_id="a1",
                output_key="classification",
                output_fields={"label": StepField(type="str", description="category")},
            )
        )

        with patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)):
            outputs = await executor.run(workflow, [])

        assert outputs["classification"] == {"label": "sports"}
        _, kwargs = assistant.run.call_args
        assert kwargs.get("response_format") is not None

    async def test_non_basemodel_result_stored_as_empty_dict(self, executor):
        assistant = make_assistant("plain string, not a model")
        workflow = make_workflow(
            WorkflowStep(
                assistant_id="a1",
                output_key="result",
                output_fields={"x": StepField(type="int")},
            )
        )

        with patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)):
            outputs = await executor.run(workflow, [])

        assert outputs["result"] == {}


# ============================================================================
# Actions
# ============================================================================


@pytest.mark.asyncio
class TestWorkflowExecutorActions:
    async def test_action_called_with_full_outputs_when_no_input_key(self, executor):
        assistant = make_assistant("data")
        received = []

        class CaptureAction:
            async def execute(self, payload):
                received.append(payload)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(assistant_id="a1", output_key="result")],
            actions=[WorkflowAction(type="capture")],
        )

        with (
            patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            await executor.run(workflow, [])

        assert received == [{"result": "data"}]

    async def test_action_called_with_specific_input_key(self, executor):
        assistant = make_assistant("step_data")
        received = []

        class CaptureAction:
            async def execute(self, payload):
                received.append(payload)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(assistant_id="a1", output_key="summary")],
            actions=[WorkflowAction(type="capture", input_key="summary")],
        )

        with (
            patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)),
            patch(
                "django_ai_sdk.workflows.executor.get_action_registry",
                return_value={"capture": CaptureAction},
            ),
        ):
            await executor.run(workflow, [])

        assert received == ["step_data"]

    async def test_unknown_action_type_warns_and_skips(self, executor):
        assistant = make_assistant("data")
        workflow = WorkflowDefinition(
            steps=[WorkflowStep(assistant_id="a1", output_key="result")],
            actions=[WorkflowAction(type="nonexistent")],
        )

        with capture_logs() as records:
            with (
                patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)),
                patch("django_ai_sdk.workflows.executor.get_action_registry", return_value={}),
            ):
                outputs = await executor.run(workflow, [])

        assert outputs["result"] == "data"
        assert any("nonexistent" in r for r in records)

    async def test_action_input_key_missing_warns_and_skips(self, executor):
        assistant = make_assistant("data")
        executed = []

        class CaptureAction:
            async def execute(self, payload):
                executed.append(payload)

        workflow = WorkflowDefinition(
            steps=[WorkflowStep(assistant_id="a1", output_key="result")],
            actions=[WorkflowAction(type="capture", input_key="does_not_exist")],
        )

        with capture_logs() as records:
            with (
                patch("django_ai_sdk.workflows.executor.AssistantService.get", AsyncMock(return_value=assistant)),
                patch(
                    "django_ai_sdk.workflows.executor.get_action_registry",
                    return_value={"capture": CaptureAction},
                ),
            ):
                await executor.run(workflow, [])

        assert executed == []
        assert any("does_not_exist" in r for r in records)
