"""A definition composes agent steps and registered Step classes alike."""

from __future__ import annotations

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from pydantic import ValidationError

from django_ai_sdk.workflows import StepContext, StepOutcome, run_steps
from django_ai_sdk.workflows.definitions import compile_steps, get_step_registry
from django_ai_sdk.workflows.schemas import WorkflowDefinition, WorkflowStep
from django_ai_sdk.workflows.steps import OnError, Step

REGISTERED = {"shout": "tests.unit.test_workflow_definitions.ShoutStep"}


class ShoutStep(Step):
    """A registered step: plain Python a runtime author may compose."""

    async def run(self, ctx: StepContext) -> StepOutcome:
        name = self.requires[0] if self.requires else "text"
        return StepOutcome(output=str(ctx.get(name, "")).upper(), detail="shouted")


class TestTheSchemaHoldsTheAuthorToACoherentStep:
    """A definition that cannot run is refused where it is written."""

    def test_an_agent_step_without_an_agent_id_is_refused(self):
        with pytest.raises(ValidationError, match="agent_id"):
            WorkflowStep(type="agent", output_key="result")

    def test_an_agent_step_without_an_output_key_is_refused(self):
        with pytest.raises(ValidationError, match="output_key"):
            WorkflowStep(type="agent", agent_id="a1")

    def test_a_registered_step_may_not_carry_an_agent_id(self):
        with pytest.raises(ValidationError, match="agent_id"):
            WorkflowStep(type="shout", agent_id="a1", output_key="loud")

    def test_a_registered_step_needs_neither(self):
        assert WorkflowStep(type="shout").type == "shout"


class TestCompiling:
    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    def test_a_registered_type_becomes_its_class(self):
        spec = WorkflowDefinition(
            steps=[WorkflowStep(type="shout", output_key="loud", requires=["text"])]
        )

        (step,) = compile_steps(spec)

        # By name, not isinstance: import_string reaches this module under its
        # own name, which is not the one pytest collected it under.
        assert type(step).__name__ == "ShoutStep"
        assert step.provides == "loud"
        assert step.requires == ("text",)

    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    def test_the_definition_wires_the_step_the_class_supplies_behaviour(self):
        spec = WorkflowDefinition(
            steps=[
                WorkflowStep(
                    type="shout", name="loudly", output_key="loud", on_error="continue"
                )
            ]
        )

        (step,) = compile_steps(spec)

        assert step.name == "loudly"
        assert step.on_error is OnError.CONTINUE

    @override_settings(AI_SDK_WORKFLOW_STEPS={})
    def test_an_unregistered_type_is_refused_naming_what_is_registered(self):
        spec = WorkflowDefinition(steps=[WorkflowStep(type="shout", output_key="loud")])

        with pytest.raises(ImproperlyConfigured, match="not registered"):
            compile_steps(spec)

    @override_settings(AI_SDK_WORKFLOW_STEPS={"broken": "nowhere.NoSuchStep"})
    def test_a_type_that_will_not_import_is_left_out_rather_than_raising(self):
        """One bad path in settings must not stop the site booting."""
        assert get_step_registry() == {}


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
            WorkflowDefinition(steps=[WorkflowStep(agent_id="a1", output_key="result")])
        )

        with (
            patch(
                "django_ai_sdk.workflows.definitions.AgentService.get",
                AsyncMock(return_value=agent),
            ),
            pytest.raises(PermissionDenied),
        ):
            await step.run(StepContext())

        agent.run.assert_not_called()

    async def test_a_permitted_principal_reaches_the_agent(self):
        from unittest.mock import AsyncMock, patch

        from django_ai_sdk.permissions import AllowAll

        agent = self._agent(AllowAll)
        (step,) = compile_steps(
            WorkflowDefinition(steps=[WorkflowStep(agent_id="a1", output_key="result")])
        )

        with patch(
            "django_ai_sdk.workflows.definitions.AgentService.get",
            AsyncMock(return_value=agent),
        ):
            outcome = await step.run(StepContext())

        assert outcome.output == "answered"


class TestRunning:
    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    async def test_a_compiled_definition_runs_on_the_step_runner(self):
        spec = WorkflowDefinition(
            steps=[WorkflowStep(type="shout", output_key="loud", requires=["text"])]
        )

        outcomes = await run_steps(compile_steps(spec), inputs={"text": "hello"})

        assert outcomes["loud"].output == "HELLO"

    @override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
    async def test_the_run_state_carries_between_a_registered_step_and_the_next(self):
        spec = WorkflowDefinition(
            steps=[
                WorkflowStep(type="shout", name="first", output_key="loud", requires=["text"]),
                WorkflowStep(type="shout", name="second", output_key="louder", requires=["loud"]),
            ]
        )

        outcomes = await run_steps(compile_steps(spec), inputs={"text": "hi"})

        assert outcomes["second"].status == "completed"
