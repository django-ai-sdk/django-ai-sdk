"""The workflow_key a run carries: set by the server, and the channel its steps run on."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError

from django_ai_sdk.workflows.models import Workflow, WorkflowRun
from django_ai_sdk.workflows.registry import (
    definition_of,
    register_workflow,
    reset_registry,
    validate_name,
)
from django_ai_sdk.workflows.runs import execute_run, open_run
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.steps import AgentStep, WorkflowContext


@pytest.mark.django_db(transaction=True)
class TestTheWorkflowKey:
    """A step's channel is keyed by what the server set on the run, never by the name a
    definition gives itself."""

    def payroll(self):
        return WorkflowDefinition(name="payroll", steps=[StepDefinition(name="pay", agent_id="a")])

    async def run_with(self, workflow, **opened: Any) -> str:
        run = await open_run(workflow, **opened)
        steps = AsyncMock(return_value={})
        with patch("django_ai_sdk.workflows.runs.run_steps", steps):
            await execute_run(str(run.pk))
        assert steps.call_args.kwargs["workflow"] == run.workflow_key
        return run.workflow_key

    async def test_a_definition_neither_stored_nor_registered_runs_as_adhoc(self):
        assert await self.run_with(self.payroll()) == "adhoc"

    async def test_a_registered_workflow_runs_under_its_name(self):
        register_workflow(self.payroll())
        try:
            assert await self.run_with(self.payroll()) == "payroll"
        finally:
            reset_registry()

    async def test_a_stored_workflow_runs_under_its_slug_whatever_it_calls_itself(self):
        record = await Workflow.objects.acreate(
            name="Intake", definition=self.payroll().model_dump()
        )

        assert await self.run_with(definition_of(record), record=record) == "intake"

    async def test_the_key_outlives_the_stored_workflow(self):

        record = await Workflow.objects.acreate(
            name="Intake", definition=self.payroll().model_dump()
        )
        run = await open_run(definition_of(record), record=record)
        await record.adelete()

        assert (await WorkflowRun.objects.aget(id=run.id)).workflow_key == "intake"

    async def test_no_stored_workflow_takes_the_adhoc_slug(self):
        record = await Workflow.objects.acreate(
            name="adhoc", definition=self.payroll().model_dump()
        )

        assert record.slug != "adhoc"

    async def test_the_table_refuses_the_adhoc_slug_set_by_hand(self):
        with pytest.raises(IntegrityError):
            await Workflow.objects.acreate(
                name="Sneaky", slug="adhoc", definition=self.payroll().model_dump()
            )

    def test_no_registered_workflow_takes_the_adhoc_name(self):
        with pytest.raises(ImproperlyConfigured, match="reserved"):
            validate_name("adhoc")

    def test_the_adhoc_channel(self):
        step = AgentStep()
        step.name = "pay"
        assert step.channel(WorkflowContext(workflow="")) == "workflow:adhoc:pay"
        assert step.channel(WorkflowContext(workflow="intake")) == "workflow:intake:pay"
