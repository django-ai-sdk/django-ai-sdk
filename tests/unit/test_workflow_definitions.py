"""A definition composes agent steps and registered Step classes alike.

Inputs and outputs compile the same way — one `FieldDefinition` mechanism pointed in
two directions — and an agent step compiles to a configured `AgentStep`, not to a
class of its own.
"""

from __future__ import annotations

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from pydantic import ValidationError

from django_ai_sdk.workflows import AgentStep, StepOutcome, WorkflowContext, run_steps
from django_ai_sdk.workflows.definitions import (
    compile_inputs,
    compile_steps,
    get_step_registry,
    model_from_fields,
)
from django_ai_sdk.workflows.schemas import FieldDefinition, HookDefinition, StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.steps import OnError, Step

REGISTERED = {"shout": "tests.unit.test_workflow_definitions.ShoutStep"}
HOOKS = {"quiet": "tests.unit.test_workflow_definitions.QuietHook"}


class ShoutStep(Step):
    """A registered step: plain Python a runtime author may compose."""

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        source = ctx.step(self.requires[0]) if self.requires else ctx.input("text", "")
        return StepOutcome(output=str(source).upper(), detail="shouted")


class QuietHook:
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

    def test_a_messages_field_coerces_the_json_a_run_crosses_the_queue_as(self):
        from django_ai_sdk.common import ChatMessage

        model = model_from_fields("Inputs", {"history": FieldDefinition(type="messages")})

        parsed = model.model_validate({"history": [{"role": "user", "content": "ahoy"}]})

        assert isinstance(parsed.history[0], ChatMessage)

    def test_an_unknown_type_is_refused_where_it_is_written(self):
        """No `file` type: a host names its own key and shapes it as str or dict."""
        with pytest.raises(ValidationError, match="'str', 'int'"):
            FieldDefinition(type="file")  # type: ignore[arg-type]

    def test_a_definition_declaring_no_inputs_compiles_to_no_model(self):
        assert compile_inputs(WorkflowDefinition(steps=[StepDefinition(type="shout", name="a")])) is None


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
                    system_prompt_override="Be brief.",
                    output_fields={"sailing": FieldDefinition()},
                )
            ]
        )

        (step,) = compile_steps(spec)

        assert type(step) is AgentStep
        assert step.agent_id == "a1"
        assert step.agent is None
        assert step.instructions == "Be brief."
        assert step.schema is not None

    @override_settings(AI_SDK_WORKFLOW_STEPS={})
    def test_an_unregistered_type_is_refused_naming_what_is_registered(self):
        spec = WorkflowDefinition(steps=[StepDefinition(type="shout", name="loud")])

        with pytest.raises(ImproperlyConfigured, match="not registered"):
            compile_steps(spec)

    @override_settings(AI_SDK_WORKFLOW_STEPS={"broken": "nowhere.NoSuchStep"})
    def test_a_type_that_will_not_import_is_left_out_rather_than_raising(self):
        """One bad path in settings must not stop the site booting."""
        assert get_step_registry() == {}

    @override_settings(AI_SDK_WORKFLOW_HOOKS={})
    def test_an_unregistered_step_hook_is_refused(self):
        spec = WorkflowDefinition(
            steps=[StepDefinition(name="a", agent_id="a1", hooks=[HookDefinition(type="quiet")])]
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
        (step,) = compile_steps(WorkflowDefinition(steps=[StepDefinition(name="result", agent_id="a1")]))

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
        (step,) = compile_steps(WorkflowDefinition(steps=[StepDefinition(name="result", agent_id="a1")]))

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
