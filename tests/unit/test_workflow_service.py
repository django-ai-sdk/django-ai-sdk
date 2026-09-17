"""
Unit tests for WorkflowService CRUD and run_by_id.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from django_ai_sdk.workflows.executor import WorkflowExecutor
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.services import WorkflowService


def make_definition(agent_id="asst-1", step_name="result"):
    return WorkflowDefinition(
        name="test-workflow",
        steps=[StepDefinition(name=step_name, agent_id=agent_id)],
    )


@pytest.mark.django_db
@pytest.mark.asyncio
class TestWorkflowServiceCRUD:
    async def test_create_stores_definition(self):
        definition = make_definition()
        record = await WorkflowService.create("My Workflow", definition)

        assert record.name == "My Workflow"
        assert record.definition == definition.model_dump()
        assert record.active is True

    async def test_a_signed_in_user_is_recorded_as_the_creator(self):
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        record = await WorkflowService.create("WF", make_definition(), user=user)

        assert record.created_by_id == user.pk

    async def test_anonymous_user_sets_no_creator(self):
        from django.contrib.auth.models import AnonymousUser

        record = await WorkflowService.create(
            "Anon Workflow", make_definition(), user=AnonymousUser()
        )

        assert record.created_by_id is None

    async def test_get_returns_record(self):
        definition = make_definition()
        created = await WorkflowService.create("WF", definition)
        fetched = await WorkflowService.get(str(created.id))
        assert str(fetched.id) == str(created.id)

    async def test_get_raises_for_unknown_id(self):
        from django_ai_sdk.workflows.models import WorkflowSettings

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.get(str(uuid4()))

    async def test_update_name(self):
        definition = make_definition()
        record = await WorkflowService.create("Old Name", definition)
        updated = await WorkflowService.update(str(record.id), name="New Name")
        assert updated.name == "New Name"

    async def test_update_definition(self):
        old_def = make_definition(step_name="old")
        record = await WorkflowService.create("WF", old_def)

        new_def = make_definition(step_name="new")
        updated = await WorkflowService.update(str(record.id), workflow=new_def)
        assert updated.definition["steps"][0]["name"] == "new"

    async def test_update_active_flag(self):
        definition = make_definition()
        record = await WorkflowService.create("WF", definition)
        updated = await WorkflowService.update(str(record.id), active=False)
        assert updated.active is False

    async def test_delete_removes_record(self):
        from django_ai_sdk.workflows.models import WorkflowSettings

        definition = make_definition()
        record = await WorkflowService.create("WF", definition)
        await WorkflowService.delete(str(record.id))

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.get(str(record.id))

    async def test_list_workflows_active_only(self):
        definition = make_definition()
        active = await WorkflowService.create("Active", definition)
        inactive = await WorkflowService.create("Inactive", definition)
        await WorkflowService.update(str(inactive.id), active=False)

        records = await WorkflowService.list_workflows(active_only=True)
        ids = [str(r.id) for r in records]
        assert str(active.id) in ids
        assert str(inactive.id) not in ids

    async def test_list_workflows_all(self):
        definition = make_definition()
        active = await WorkflowService.create("Active", definition)
        inactive = await WorkflowService.create("Inactive", definition)
        await WorkflowService.update(str(inactive.id), active=False)

        records = await WorkflowService.list_workflows(active_only=False)
        ids = [str(r.id) for r in records]
        assert str(active.id) in ids
        assert str(inactive.id) in ids


@pytest.mark.django_db
@pytest.mark.asyncio
class TestWorkflowServiceRunById:
    async def test_run_by_id_enqueues_task(self):
        definition = make_definition()
        record = await WorkflowService.create("WF", definition)

        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id))

        assert run.status == "pending"
        assert str(run.workflow_id) == str(record.id)

    async def test_run_by_id_raises_for_inactive(self):
        from django_ai_sdk.workflows.models import WorkflowSettings

        definition = make_definition()
        record = await WorkflowService.create("WF", definition)
        await WorkflowService.update(str(record.id), active=False)

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.run_by_id(str(record.id))

    async def test_run_by_id_raises_for_unknown_id(self):
        from django_ai_sdk.workflows.models import WorkflowSettings

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.run_by_id(str(uuid4()))


class TestWorkflowServiceListActions:
    def test_returns_registered_actions(self):
        class MyAction:
            description = "does something"

        with patch(
            "django_ai_sdk.workflows.services.get_action_registry",
            return_value={"my_action": MyAction},
        ):
            actions = WorkflowService.list_actions()

        assert actions == [{"key": "my_action", "description": "does something"}]

    def test_empty_when_no_actions_registered(self):
        with patch(
            "django_ai_sdk.workflows.services.get_action_registry",
            return_value={},
        ):
            assert WorkflowService.list_actions() == []
