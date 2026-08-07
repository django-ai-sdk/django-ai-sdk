"""
Unit tests for workflow Pydantic models.
"""

import pytest
from pydantic import ValidationError

from django_ai_sdk.workflows.schemas import (
    FieldType,
    StepField,
    WorkflowAction,
    WorkflowDefinition,
    WorkflowStep,
)


class TestStepField:
    def test_defaults(self):
        f = StepField()
        assert f.type == "str"
        assert f.description == ""

    def test_all_valid_types(self):
        for t in ("str", "int", "float", "bool"):
            f = StepField(type=t)
            assert f.type == t

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            StepField(type="datetime")

    def test_rejects_empty_type(self):
        with pytest.raises(ValidationError):
            StepField(type="")


class TestWorkflowStep:
    def test_minimal(self):
        s = WorkflowStep(agent_id="abc", output_key="result")
        assert s.name == ""
        assert s.input_key is None
        assert s.system_prompt_override is None
        assert s.output_fields == {}

    def test_with_name(self):
        s = WorkflowStep(name="Extract", agent_id="abc", output_key="result")
        assert s.name == "Extract"

    def test_output_fields_parsed(self):
        s = WorkflowStep(
            agent_id="abc",
            output_key="result",
            output_fields={"topic": {"type": "str", "description": "main topic"}},
        )
        assert isinstance(s.output_fields["topic"], StepField)
        assert s.output_fields["topic"].type == "str"

    def test_output_fields_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowStep(
                agent_id="abc",
                output_key="result",
                output_fields={"topic": {"type": "list"}},
            )


class TestWorkflowAction:
    def test_minimal(self):
        a = WorkflowAction(type="log")
        assert a.input_key is None

    def test_with_input_key(self):
        a = WorkflowAction(type="console_log", input_key="summary")
        assert a.input_key == "summary"


class TestWorkflowDefinition:
    def test_minimal(self):
        d = WorkflowDefinition(
            steps=[WorkflowStep(agent_id="abc", output_key="result")]
        )
        assert d.name == ""
        assert d.actions == []

    def test_round_trip_json(self):
        d = WorkflowDefinition(
            name="pipeline",
            steps=[
                WorkflowStep(
                    name="Step 1",
                    agent_id="abc-123",
                    output_key="summary",
                    output_fields={"text": StepField(type="str", description="output")},
                )
            ],
            actions=[WorkflowAction(type="console_log", input_key="summary")],
        )
        dumped = d.model_dump()
        restored = WorkflowDefinition.model_validate(dumped)
        assert restored.name == d.name
        assert restored.steps[0].name == "Step 1"
        assert restored.steps[0].output_fields["text"].type == "str"
        assert restored.actions[0].type == "console_log"

    def test_empty_steps_allowed(self):
        d = WorkflowDefinition(steps=[])
        assert d.steps == []
