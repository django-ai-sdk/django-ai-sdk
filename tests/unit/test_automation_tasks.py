"""The worker side: one claimed AutomationRun, executed.

The seam under test is the one between an automation and a workflow. A schedule has
nobody typing, so the automation renders the first turn itself and supplies it under
the input the workflow declares. An input a definition does not declare is dropped
before the first step, so getting this wrong does not fail — it produces a run that
succeeded having asked the model nothing, which is why `check_automations` reports
it as W007 and why these assert on what reached the agent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.automations.tasks import run_automation
from django_ai_sdk.workflows import FieldDefinition, StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.registry import register as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows
from tests.factories.db import UserFactory

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
AGENT_ID = "663dd70f-65d2-58c4-9edc-037c9904d562"


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    reset_workflows()
    yield
    reset_registry()
    reset_workflows()


@pytest.fixture
def agent():
    """A permitted agent that records the messages it was called with."""
    from unittest.mock import MagicMock

    from django_ai_sdk.permissions import AllowAll

    stub = MagicMock()
    stub.permissions = [AllowAll]
    stub.run = AsyncMock(return_value="aye, calm seas")
    with patch("django_ai_sdk.agents.services.AgentService.get", AsyncMock(return_value=stub)):
        yield stub


def declare_workflow(name="some-workflow", *, takes_input=True):
    return register_wf(
        WorkflowDefinition(
            name=name,
            input_fields=({"messages": FieldDefinition(type="list")} if takes_input else {}),
            steps=[
                StepDefinition(
                    name="result", agent_id=AGENT_ID, history=["messages"] if takes_input else []
                )
            ],
        )
    )


def declare(name="example", **attrs):
    from django_ai_sdk.automations import Automation

    defaults = {"name": name, "cron": "0 9 * * *", "workflow": "some-workflow"}
    register(type("Example", (Automation,), {**defaults, **attrs}))


async def make_run(name="example") -> AutomationRun:
    state = await AutomationState.objects.acreate(name=name, next_run_at=NOW)
    return await AutomationRun.objects.acreate(name=name, state=state, scheduled_for=NOW)


@pytest.mark.django_db(transaction=True)
class TestTheRenderedTurnReachesTheAgent:
    async def test_the_automations_input_is_what_the_agent_is_asked(self, agent):
        declare_workflow()
        declare(input="Report the harbour since {last_run_at}.")
        run = await make_run()

        await run_automation(str(run.id))

        (messages,) = agent.run.call_args.args
        assert [m.content for m in messages] == ["Report the harbour since the beginning."]

    async def test_a_str_input_receives_the_rendered_turn_as_is(self, agent):
        """The workflow declares `messages` as a plain string; the turn lands there."""
        declare_workflow(takes_input=False)
        register_wf(
            WorkflowDefinition(
                name="some-workflow",
                input_fields={"messages": FieldDefinition(type="str")},
                steps=[StepDefinition(name="result", agent_id=AGENT_ID, history=["messages"])],
            )
        )
        declare(input="Report the harbour since {last_run_at}.")
        run = await make_run()

        await run_automation(str(run.id))

        (messages,) = agent.run.call_args.args
        assert [m.content for m in messages] == ["Report the harbour since the beginning."]

    async def test_the_run_records_what_the_step_produced(self, agent):
        declare_workflow()
        declare()
        run = await make_run()

        outputs = await run_automation(str(run.id))

        assert outputs == {"result": "aye, calm seas"}
        await run.arefresh_from_db()
        assert run.status == AutomationRun.Status.SUCCEEDED
        # The workflow run is linked, so its steps are readable from the automation.
        assert run.workflow_run_id is not None

    async def test_a_workflow_that_declares_no_input_drops_the_turn(self, agent):
        """The failure W007 exists to warn about, pinned so it stays visible."""
        declare_workflow(takes_input=False)
        declare()
        run = await make_run()

        await run_automation(str(run.id))

        (messages,) = agent.run.call_args.args
        assert messages == []


@pytest.mark.django_db(transaction=True)
class TestAMissingWorkflowSkipsRatherThanFails:
    async def test_an_unregistered_workflow_is_a_skip(self):
        declare(workflow="no-such-workflow")
        run = await make_run()

        assert await run_automation(str(run.id)) is None

        await run.arefresh_from_db()
        assert run.status == AutomationRun.Status.SKIPPED
        assert "not registered" in run.skip_reason

    async def test_a_declaration_that_is_gone_is_a_skip(self):
        """The queued task outlived the deploy that removed its automation."""
        run = await make_run("was-deleted")

        assert await run_automation(str(run.id)) is None

        await run.arefresh_from_db()
        assert run.status == AutomationRun.Status.SKIPPED
        assert run.skip_reason == "no longer declared"

    async def test_a_deleted_run_is_not_an_error(self):
        from uuid import uuid4

        assert await run_automation(str(uuid4())) is None


@pytest.mark.django_db(transaction=True)
class TestTheWorkflowIsGatedOnTheResolvedPrincipal:
    """A subscriber's run is theirs, so it may only run what they may run."""

    async def test_a_principal_who_may_not_run_the_workflow_fails_the_run(self, agent):
        from django_ai_sdk.permissions import DenyAll

        declare_workflow()
        declare()
        user = await UserFactory.acreate()
        state = await AutomationState.objects.acreate(name="example", next_run_at=NOW)
        run = await AutomationRun.objects.acreate(
            name="example", state=state, scheduled_for=NOW, user=user
        )

        with (
            patch("django_ai_sdk.permissions.get_domain_permissions", return_value=[DenyAll]),
            pytest.raises(PermissionDenied),
        ):
            await run_automation(str(run.id))

        await run.arefresh_from_db()
        assert run.status == AutomationRun.Status.FAILED
        assert "run_workflow" in run.error
        agent.run.assert_not_called()

    async def test_the_app_itself_needs_no_permission(self, agent):
        """Audience.APP resolves no principal, so there is nobody to gate on."""
        declare_workflow()
        declare()
        run = await make_run()

        await run_automation(str(run.id))

        await run.arefresh_from_db()
        assert run.status == AutomationRun.Status.SUCCEEDED
