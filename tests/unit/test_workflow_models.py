"""The pydantic shape of a definition: what an author may and may not write."""

import pytest
from pydantic import ValidationError

from django_ai_sdk.workflows.schemas import (
    FieldDefinition,
    HookDefinition,
    StepDefinition,
    WorkflowDefinition,
)


class TestFieldDefinition:
    def test_defaults(self):
        f = FieldDefinition()
        assert f.type == "str"
        assert f.description == ""
        assert f.required is True

    def test_every_valid_type(self):
        for t in ("str", "int", "float", "bool", "list", "dict", "messages"):
            assert FieldDefinition(type=t).type == t

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            FieldDefinition(type="datetime")

    def test_rejects_empty_type(self):
        with pytest.raises(ValidationError):
            FieldDefinition(type="")


class TestStepDefinition:
    def test_minimal(self):
        s = StepDefinition(name="result", agent_id="abc")
        assert s.type == "agent"
        assert s.requires == []
        assert s.hooks == []
        assert s.system_prompt_override is None
        assert s.output_fields == {}

    def test_the_name_is_required(self):
        """It is the key a result is filed under, so there is no default."""
        with pytest.raises(ValidationError):
            StepDefinition(name="", agent_id="abc")

    def test_output_fields_parsed(self):
        s = StepDefinition(
            name="result",
            agent_id="abc",
            output_fields={"topic": {"type": "str", "description": "main topic"}},
        )
        assert isinstance(s.output_fields["topic"], FieldDefinition)
        assert s.output_fields["topic"].type == "str"

    def test_output_fields_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            StepDefinition(name="result", agent_id="abc", output_fields={"topic": {"type": "date"}})


class TestHookDefinition:
    def test_minimal(self):
        assert HookDefinition(type="log").config == {}

    def test_with_config(self):
        """One registered class serves every definition that names it."""
        hook = HookDefinition(type="thread_message", config={"step": "summary"})
        assert hook.config == {"step": "summary"}


class TestWorkflowDefinition:
    def test_minimal(self):
        d = WorkflowDefinition(steps=[StepDefinition(name="result", agent_id="abc")])
        assert d.name == ""
        assert d.hooks == []
        assert d.input_fields == {}

    def test_round_trip_json(self):
        d = WorkflowDefinition(
            name="pipeline",
            input_fields={"history": FieldDefinition(type="messages")},
            steps=[
                StepDefinition(
                    name="summary",
                    agent_id="abc-123",
                    output_fields={"text": FieldDefinition(type="str", description="output")},
                    hooks=[HookDefinition(type="console_log")],
                )
            ],
            hooks=[HookDefinition(type="console_log", config={"step": "summary"})],
        )

        restored = WorkflowDefinition.model_validate(d.model_dump())

        assert restored.name == d.name
        assert restored.input_fields["history"].type == "messages"
        assert restored.steps[0].output_fields["text"].type == "str"
        assert restored.steps[0].hooks[0].type == "console_log"
        assert restored.hooks[0].config == {"step": "summary"}

    def test_empty_steps_allowed(self):
        """The schema permits it; register, create and execute all refuse it."""
        assert WorkflowDefinition(steps=[]).steps == []
