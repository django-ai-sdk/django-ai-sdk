"""AutomationService: who may see, toggle and fire automations, and what they read back."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from django.contrib.auth.models import Permission
from django.test import override_settings
from django_ai_sdk.automations import Automation, AutomationService
from django_ai_sdk.automations.models import AutomationState
from django_ai_sdk.automations.registry import register_automation, reset_registry
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.registry import register_workflow as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows
from django_ai_sdk.workflows.schemas import FieldDefinition, StepDefinition, WorkflowDefinition
from tests.factories.db import UserFactory


async def manager():
    """A user holding the manage permission. `is_staff` alone would not do."""
    user = await UserFactory.acreate()
    permission = await Permission.objects.aget(
        content_type__app_label="django_ai_sdk_automations", codename="change_automationstate"
    )
    await user.user_permissions.aadd(permission)
    return user


AGENT_ID = "663dd70f-65d2-58c4-9edc-037c9904d562"
WORKFLOW = "some-workflow"


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    reset_workflows()
    register_wf(
        WorkflowDefinition(
            name=WORKFLOW,
            input_fields={"prompt": FieldDefinition(type="str")},
            steps=[StepDefinition(name="result", agent_id=AGENT_ID, history=["prompt"])],
        )
    )
    register_automation(
        type(
            "Example", (Automation,), {"name": "example", "cron": "0 9 * * *", "workflow": WORKFLOW}
        )
    )
    yield
    reset_registry()
    reset_workflows()


@pytest.fixture(autouse=True)
def _no_queue():
    with patch("django_ai_sdk.workflows.runs.aenqueue_run", AsyncMock()):
        yield


@pytest.mark.django_db(transaction=True)
class TestList:
    async def test_anonymous_sees_nothing(self):
        assert await AutomationService.list_for_user(None) == []

    async def test_member_sees_schedule_and_enabled_source(self):
        user = await UserFactory.acreate()
        [row] = await AutomationService.list_for_user(user)
        assert row.name == "example"
        assert row.schedule.startswith("0 9 * * *")
        assert (row.enabled, row.enabled_source) == (True, "code")
        assert row.last_run is None

    @override_settings(AI_SDK_AUTOMATIONS={"ENABLED": False})
    async def test_kill_switch_wins(self):
        user = await UserFactory.acreate()
        [row] = await AutomationService.list_for_user(user)
        assert (row.enabled, row.enabled_source) == (False, "kill-switch")


@pytest.mark.django_db(transaction=True)
class TestSetEnabled:
    async def test_a_manager_writes_the_database_layer(self):
        admin = await manager()
        row = await AutomationService.set_enabled("example", enabled=False, user=admin)
        assert (row.enabled, row.enabled_source) == (False, "db")
        assert (await AutomationState.objects.aget(name="example")).enabled is False

    async def test_member_may_not(self):
        user = await UserFactory.acreate()
        with pytest.raises(PermissionDenied):
            await AutomationService.set_enabled("example", enabled=False, user=user)

    async def test_staff_alone_may_not(self):
        # Both act as the run_as account, so it takes the explicit permission.
        staff = await UserFactory.acreate(is_staff=True)
        with pytest.raises(PermissionDenied):
            await AutomationService.set_enabled("example", enabled=False, user=staff)

    async def test_unknown_name_is_none(self):
        admin = await manager()
        assert await AutomationService.set_enabled("nope", enabled=True, user=admin) is None

    async def test_a_tick_skips_what_the_database_turned_off(self):
        from django_ai_sdk.automations.runner import tick

        admin = await manager()
        await AutomationService.set_enabled("example", enabled=False, user=admin)
        [result] = await tick()
        assert result.reason == "disabled"


@pytest.mark.django_db(transaction=True)
class TestRuns:
    async def test_run_now_dispatches_and_lists(self):
        admin = await manager()
        [dispatched] = await AutomationService.run_now("example", user=admin)

        [listed] = await AutomationService.list_runs("example", user=admin)
        assert listed.id == dispatched.id
        assert (await AutomationService.get_run(dispatched.id, user=admin)).id == dispatched.id
        [row] = await AutomationService.list_for_user(admin)
        assert row.last_run.id == dispatched.id
        assert await AutomationService.last_tick_at() is not None

    async def test_member_may_not_run(self):
        user = await UserFactory.acreate()
        with pytest.raises(PermissionDenied):
            await AutomationService.run_now("example", user=user)

    @override_settings(AI_SDK_AUTOMATIONS={"ENABLED": False})
    async def test_kill_switch_dispatches_nothing(self):
        admin = await manager()
        assert await AutomationService.run_now("example", user=admin) == []

    async def test_member_reads_only_their_own_runs(self):
        admin = await manager()
        user = await UserFactory.acreate()
        [run] = await AutomationService.run_now("example", user=admin)

        assert await AutomationService.list_runs("example", user=user) == []
        assert await AutomationService.get_run(run.id, user=user) is None

        await WorkflowRun.objects.filter(id=run.id).aupdate(user=user)
        assert [r.id for r in await AutomationService.list_runs("example", user=user)] == [run.id]

    async def test_bad_run_id_is_none(self):
        admin = await manager()
        assert await AutomationService.get_run("not-a-uuid", user=admin) is None
