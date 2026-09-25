"""A definition composes agent steps and registered Step classes alike.

Inputs and outputs compile the same way — one `FieldDefinition` mechanism pointed in
two directions — and an agent step compiles to a configured `AgentStep`, not to a
class of its own.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from pydantic import BaseModel, Field, ValidationError, field_validator

import django_ai_sdk.workflows.schemas
from django_ai_sdk.workflows.definitions import (
    compile_inputs,
    compile_steps,
    get_step_registry,
    inputs_json_schema,
    model_from_fields,
)
from django_ai_sdk.workflows.runner import run_steps
from django_ai_sdk.workflows.schemas import (
    MAX_DEPTH,
    ActionDefinition,
    FieldDefinition,
    StepDefinition,
    WorkflowDefinition,
)
from django_ai_sdk.workflows.steps import AgentStep, OnError, Step, StepOutcome, WorkflowContext

REGISTERED = {"shout": "tests.unit.test_workflow_definitions.ShoutStep"}
ACTIONS = {"quiet": "tests.unit.test_workflow_definitions.QuietAction"}


class ShoutStep(Step):
    """A registered step: plain Python a runtime author may compose."""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        source = ctx.step(self.requires[0]) if self.requires else ctx.input("text", "")
        return StepOutcome(output=str(source).upper(), detail="shouted")


class QuietAction:
    pass


class TestTheSchemaHoldsTheAuthorToACoherentStep:
    """A definition that cannot run is refused where it is written."""

    def test_an_agent_step_without_an_agent_id_is_refused(self):
        with pytest.raises(ValidationError, match="agent_id"):
            StepDefinition(type="agent", name="result")

    def test_a_step_without_a_name_is_refused(self):
        """The name is the key its result is filed under, so there is no default."""
        with pytest.raises(ValidationError):
            StepDefinition(type="shout", name="")

    def test_a_registered_step_may_not_carry_an_agent_id(self):
        with pytest.raises(ValidationError, match="agent_id"):
            StepDefinition(type="shout", name="loud", agent_id="a1")

    def test_a_registered_step_needs_only_a_name(self):
        assert StepDefinition(type="shout", name="loud").type == "shout"


class TestFieldsCompileToAModel:
    """One builder for both directions, so an input is declared like an output."""

    def test_declared_fields_become_a_pydantic_model(self):
        model = model_from_fields(
            "Verdict",
            {
                "sailing": FieldDefinition(type="str", description="good | risky"),
                "windspeed_kmh": FieldDefinition(type="float"),
            },
        )

        parsed = model.model_validate({"sailing": "good", "windspeed_kmh": "12.5"})

        assert parsed.windspeed_kmh == 12.5
        assert model.model_fields["sailing"].description == "good | risky"

    def test_an_optional_field_defaults_to_none(self):
        model = model_from_fields("Inputs", {"note": FieldDefinition(required=False)})

        assert model.model_validate({}).note is None

    def test_an_optional_field_with_a_default_keeps_its_type(self):
        model = model_from_fields(
            "Inputs", {"tone": FieldDefinition(required=False, default="pirate")}
        )

        parsed = model.model_validate({})

        assert parsed.tone == "pirate"

    def test_an_enum_field_refuses_a_value_outside_it(self):
        model = model_from_fields(
            "Inputs", {"tone": FieldDefinition(type="str", enum=["pirate", "plain"])}
        )

        with pytest.raises(ValidationError):
            model.model_validate({"tone": "robot"})

    def test_a_typed_list_validates_its_items(self):
        model = model_from_fields(
            "Inputs",
            {"tags": FieldDefinition(type="list", items=FieldDefinition(type="int"))},
        )

        assert model.model_validate({"tags": [1, 2]}).tags == [1, 2]
        with pytest.raises(ValidationError):
            model.model_validate({"tags": [1, "two"]})

    def test_an_object_field_compiles_to_a_nested_model(self):
        model = model_from_fields(
            "Inputs",
            {
                "author": FieldDefinition(
                    type="object",
                    fields={"name": FieldDefinition(type="str"), "id": FieldDefinition(type="int")},
                )
            },
        )

        parsed = model.model_validate({"author": {"name": "Ann", "id": 3}})

        assert (parsed.author.name, parsed.author.id) == ("Ann", 3)
        with pytest.raises(ValidationError):
            # The nested required field is checked with the rest, not three steps in.
            model.model_validate({"author": {"name": "Ann"}})

    def test_a_declared_schema_survives_a_dump_round_trip(self):
        fields = {
            "tone": FieldDefinition(
                type="str", enum=["pirate", "plain"], required=False, default="pirate"
            ),
            "tags": FieldDefinition(type="list", items=FieldDefinition(type="str")),
        }

        restored = {
            name: FieldDefinition.model_validate(f.model_dump()) for name, f in fields.items()
        }

        assert restored == fields

    def test_a_default_is_refused_on_a_required_field(self):
        with pytest.raises(ValidationError, match="not required"):
            FieldDefinition(default="draft")

    def test_a_default_outside_the_enum_is_refused(self):
        with pytest.raises(ValidationError, match="enum"):
            FieldDefinition(type="str", enum=["a", "b"], required=False, default="c")

    def test_a_default_of_the_wrong_scalar_type_is_refused(self):
        with pytest.raises(ValidationError, match="not a int"):
            FieldDefinition(type="int", required=False, default="not-an-int")

    def test_a_bool_default_is_refused_for_an_int_field(self):
        # bool is a subclass of int in Python; the check has to be exact, not isinstance.
        with pytest.raises(ValidationError, match="not a int"):
            FieldDefinition(type="int", required=False, default=True)

    def test_an_int_default_is_refused_for_a_bool_field(self):
        with pytest.raises(ValidationError, match="not a bool"):
            FieldDefinition(type="bool", required=False, default=1)

    def test_a_default_of_the_wrong_type_is_refused_for_a_list(self):
        with pytest.raises(ValidationError, match="not a list"):
            FieldDefinition(type="list", required=False, default="not-a-list")

    def test_a_mistyped_element_in_a_list_default_is_refused(self):
        with pytest.raises(ValidationError, match=r"\[1\] is 'two', which is not a int"):
            FieldDefinition(
                type="list", items=FieldDefinition(type="int"), required=False, default=[1, "two"]
            )

    def test_a_default_of_the_wrong_type_is_refused_for_an_object(self):
        with pytest.raises(ValidationError, match="not an object"):
            FieldDefinition(
                type="object",
                fields={"name": FieldDefinition(type="str")},
                required=False,
                default="nope",
            )

    def test_an_object_default_with_an_undeclared_key_is_refused(self):
        with pytest.raises(ValidationError, match="not a declared field"):
            FieldDefinition(
                type="object",
                fields={"name": FieldDefinition(type="str")},
                required=False,
                default={"nickname": "Ann"},
            )

    def test_a_mistyped_member_in_an_object_default_is_refused(self):
        with pytest.raises(ValidationError, match=r"'age' is 'old', which is not a int"):
            FieldDefinition(
                type="object",
                fields={"age": FieldDefinition(type="int")},
                required=False,
                default={"age": "old"},
            )

    def test_items_on_something_but_a_list_are_refused(self):
        with pytest.raises(ValidationError, match="items"):
            FieldDefinition(type="str", items=FieldDefinition())

    def test_fields_on_something_but_an_object_are_refused(self):
        with pytest.raises(ValidationError, match="fields"):
            FieldDefinition(type="str", fields={"a": FieldDefinition()})

    def test_an_object_without_fields_is_refused(self):
        with pytest.raises(ValidationError, match="open JSON"):
            FieldDefinition(type="object")

    def test_a_schema_nested_past_the_depth_cap_is_refused(self):
        field = FieldDefinition(type="str")
        with pytest.raises(ValidationError, match="nests deeper"):
            for _ in range(MAX_DEPTH + 1):
                field = FieldDefinition(type="object", fields={"n": field})

    def test_an_unknown_type_is_refused_where_it_is_written(self):
        """No `file` type: a host names its own key and shapes it as str or dict."""
        with pytest.raises(ValidationError, match="'str', 'int'"):
            FieldDefinition(type="file")  # type: ignore[arg-type]

    def test_a_definition_declaring_no_inputs_compiles_to_no_model(self):
        assert (
            compile_inputs(WorkflowDefinition(steps=[StepDefinition(type="shout", name="a")]))
            is None
        )


class TestInputsAsJSONSchema:
    """One artifact serves an editor, a form and a tool — compiled from the same
    model the run itself validates against, so they cannot disagree."""

    def test_declared_fields_describe_their_shape(self):
        definition = WorkflowDefinition(
            name="digest",
            input_fields={
                "thread": FieldDefinition(type="str"),
                "tone": FieldDefinition(
                    type="str", enum=["brief", "thorough"], required=False, default="brief"
                ),
                "tags": FieldDefinition(type="list", items=FieldDefinition(type="str")),
            },
            steps=[StepDefinition(name="d", agent_id="a1")],
        )

        schema = inputs_json_schema(definition)

        assert schema["type"] == "object"
        assert schema["required"] == ["thread", "tags"]
        assert schema["properties"]["thread"] == {"title": "Thread", "type": "string"}
        assert schema["properties"]["tone"]["enum"] == ["brief", "thorough"]
        assert schema["properties"]["tone"]["default"] == "brief"
        assert schema["properties"]["tags"]["items"] == {"type": "string"}

    def test_a_definition_declaring_nothing_is_open(self):
        """The schema says "anything" rather than saying nothing."""
        definition = WorkflowDefinition(
            name="open", steps=[StepDefinition(name="d", agent_id="a1")]
        )

        assert inputs_json_schema(definition) == {"type": "object", "additionalProperties": True}


class TestAuthoringWithAPydanticModel:
    """`inputs=SomeBaseModel` writes the same fields with type annotations.

    The model is authoring, not storage: it is normalized into `input_fields`
    and never rides on the definition, so both ways of declaring have one wire
    format.
    """

    def test_a_model_normalizes_into_input_fields(self):
        class DigestInputs(BaseModel):
            thread: str
            tone: Literal["pirate", "plain"] = "pirate"
            note: str | None = None

        definition = WorkflowDefinition(
            name="digest",
            inputs=DigestInputs,
            steps=[StepDefinition(name="digest", agent_id="a1")],
        )

        assert definition.input_fields == {
            "thread": FieldDefinition(type="str"),
            "tone": FieldDefinition(
                type="str", required=False, default="pirate", enum=["pirate", "plain"]
            ),
            "note": FieldDefinition(type="str", required=False),
        }

    def test_a_nested_model_becomes_an_object_field(self):
        class Author(BaseModel):
            name: str

        class ReviewInputs(BaseModel):
            author: Author
            tags: list[str]

        definition = WorkflowDefinition(
            name="review",
            inputs=ReviewInputs,
            steps=[StepDefinition(name="r", agent_id="a1")],
        )

        assert definition.input_fields["author"] == FieldDefinition(
            type="object", fields={"name": FieldDefinition(type="str")}
        )
        assert definition.input_fields["tags"] == FieldDefinition(
            type="list", items=FieldDefinition(type="str")
        )

    def test_the_normalized_fields_compile_and_validate(self):
        class Inputs(BaseModel):
            thread: str
            tone: Literal["pirate", "plain"] = "pirate"

        definition = WorkflowDefinition(
            name="digest", inputs=Inputs, steps=[StepDefinition(name="d", agent_id="a1")]
        )

        parsed = compile_inputs(definition).model_validate({"thread": "t1"})

        assert (parsed.thread, parsed.tone) == ("t1", "pirate")

    def test_a_model_with_validators_is_refused(self):
        """A check the author wrote and the run silently skipped is the footgun."""

        class Validated(BaseModel):
            thread: str

            @field_validator("thread")
            @staticmethod
            def not_blank(value: str) -> str:
                return value

        with pytest.raises(ValidationError, match="validators"):
            WorkflowDefinition(
                name="bad", inputs=Validated, steps=[StepDefinition(name="d", agent_id="a1")]
            )

    def test_an_unmappable_annotation_is_refused_naming_the_field(self):
        class Exotic(BaseModel):
            anything: Any

        with pytest.raises(ValidationError, match="Exotic.anything"):
            WorkflowDefinition(
                name="bad", inputs=Exotic, steps=[StepDefinition(name="d", agent_id="a1")]
            )

    def test_giving_both_forms_is_refused(self):
        class Inputs(BaseModel):
            thread: str

        with pytest.raises(ValidationError, match="not both"):
            WorkflowDefinition(
                name="bad",
                inputs=Inputs,
                input_fields={"thread": FieldDefinition()},
                steps=[StepDefinition(name="d", agent_id="a1")],
            )

    def test_an_annotated_field_is_refused(self):
        """Constraints did not cross into the declared schema yet."""

        class Constrained(BaseModel):
            count: Annotated[int, Field(ge=1)]

        with pytest.raises(ValidationError, match="Constrained.count"):
            WorkflowDefinition(
                name="bad", inputs=Constrained, steps=[StepDefinition(name="d", agent_id="a1")]
            )


class TestCompiling:
    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    def test_a_registered_type_becomes_its_class(self):
        spec = WorkflowDefinition(steps=[StepDefinition(type="shout", name="loud")])

        (step,) = compile_steps(spec)

        # By name, not isinstance: import_string reaches this module under its
        # own name, which is not the one pytest collected it under.
        assert type(step).__name__ == "ShoutStep"
        assert step.name == "loud"

    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    def test_the_definition_wires_the_step_the_class_supplies_behaviour(self):
        spec = WorkflowDefinition(
            steps=[StepDefinition(type="shout", name="loudly", on_error="continue")]
        )

        (step,) = compile_steps(spec)

        assert step.name == "loudly"
        assert step.on_error is OnError.CONTINUE

    def test_an_agent_step_compiles_to_a_configured_agent_step(self):
        """No separate class for a definition's agent: one AgentStep serves both."""
        spec = WorkflowDefinition(
            steps=[
                StepDefinition(
                    name="verdict",
                    agent_id="a1",
                    history=["prompt"],
                    system_prompt_override="Be brief.",
                    output_fields={"sailing": FieldDefinition()},
                )
            ]
        )

        (step,) = compile_steps(spec)

        assert type(step) is AgentStep
        assert step.agent_id == "a1"
        assert step.agent is None
        assert step.history == ("prompt",)
        assert step.instructions == "Be brief."
        assert step.schema is not None

    def test_an_agent_step_without_history_sends_no_transcript(self):
        """History is the author's choice, not something a type implies."""
        spec = WorkflowDefinition(steps=[StepDefinition(name="verdict", agent_id="a1")])

        (step,) = compile_steps(spec)

        assert step.history == ()

    @override_settings(AI_SDK_WORKFLOW_STEPS={})
    def test_an_unregistered_type_is_refused_naming_what_is_registered(self):
        spec = WorkflowDefinition(steps=[StepDefinition(type="shout", name="loud")])

        with pytest.raises(ImproperlyConfigured, match="not registered"):
            compile_steps(spec)

    @override_settings(AI_SDK_WORKFLOW_STEPS={"broken": "nowhere.NoSuchStep"})
    def test_a_type_that_will_not_import_is_left_out_rather_than_raising(self):
        """One bad path in settings must not stop the site booting."""
        assert get_step_registry() == {}

    @override_settings(AI_SDK_WORKFLOW_ACTIONS={})
    def test_an_unregistered_step_action_is_refused(self):
        spec = WorkflowDefinition(
            steps=[StepDefinition(name="a", agent_id="a1", actions=[ActionDefinition(type="quiet")])]
        )

        with pytest.raises(ImproperlyConfigured, match="not registered"):
            compile_steps(spec)


class TestAJsonAgentStepIsGatedToo:
    """The same CHAT check as a code `AgentStep`, proved with a real permission class."""

    @staticmethod
    def _agent(*permissions):
        from unittest.mock import AsyncMock, MagicMock

        agent = MagicMock()
        agent.permissions = list(permissions)
        agent.run = AsyncMock(return_value="answered")
        return agent

    async def test_a_denied_principal_never_reaches_the_agent(self):
        from unittest.mock import AsyncMock, patch

        from django_ai_sdk.permissions import DenyAll, PermissionDenied

        agent = self._agent(DenyAll)
        (step,) = compile_steps(
            WorkflowDefinition(steps=[StepDefinition(name="result", agent_id="a1")])
        )

        with (
            patch(
                "django_ai_sdk.agents.services.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            pytest.raises(PermissionDenied),
        ):
            await step.run(WorkflowContext())

        agent.run.assert_not_called()

    async def test_a_permitted_principal_reaches_the_agent(self):
        from unittest.mock import AsyncMock, patch

        from django_ai_sdk.permissions import AllowAll

        agent = self._agent(AllowAll)
        (step,) = compile_steps(
            WorkflowDefinition(steps=[StepDefinition(name="result", agent_id="a1")])
        )

        with patch(
            "django_ai_sdk.agents.services.AgentService.get",
            AsyncMock(return_value=agent),
        ):
            outcome = await step.run(WorkflowContext())

        assert outcome.output == "answered"


class TestRunning:
    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    async def test_a_compiled_definition_runs_on_the_step_runner(self):
        spec = WorkflowDefinition(steps=[StepDefinition(type="shout", name="loud")])

        outcomes = await run_steps(compile_steps(spec), inputs={"text": "hello"})

        assert outcomes["loud"].output == "HELLO"

    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    async def test_the_run_state_carries_between_a_registered_step_and_the_next(self):
        spec = WorkflowDefinition(
            steps=[
                StepDefinition(type="shout", name="first"),
                StepDefinition(type="shout", name="second", requires=["first"]),
            ]
        )

        outcomes = await run_steps(compile_steps(spec), inputs={"text": "hi"})

        assert outcomes["second"].status == "completed"
