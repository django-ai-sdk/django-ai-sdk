"""WorkflowService CRUD and run_by_id, as its own author sees them.

Every door is gated on a principal, so these carry a real user throughout. Who is
refused which door is `test_workflow_permissions.py`.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from django_ai_sdk.workflows.executor import WorkflowExecutor
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.services import WorkflowService


@pytest.fixture
async def author():
    from tests.factories.db import UserFactory

    return await UserFactory.acreate()


def make_definition(agent_id="asst-1", step_name="result"):
    return WorkflowDefinition(
        name="test-workflow",
        steps=[StepDefinition(name=step_name, agent_id=agent_id)],
    )


@pytest.mark.django_db
@pytest.mark.asyncio
class TestWorkflowServiceCRUD:
    async def test_create_stores_definition(self, author):
        definition = make_definition()
        record = await WorkflowService.create("My Workflow", definition, user=author)

        assert record.name == "My Workflow"
        assert record.definition == definition.model_dump()
        assert record.active is True

    async def test_a_signed_in_user_is_recorded_as_the_creator(self, author):
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        record = await WorkflowService.create("WF", make_definition(), user=user)

        assert record.created_by_id == user.pk

    async def test_get_returns_record(self, author):
        definition = make_definition()
        created = await WorkflowService.create("WF", definition, user=author)
        fetched = await WorkflowService.get(str(created.id), user=author)
        assert str(fetched.id) == str(created.id)

    async def test_get_raises_for_unknown_id(self, author):
        from django_ai_sdk.workflows.models import WorkflowSettings

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.get(str(uuid4()), user=author)

    async def test_update_name(self, author):
        definition = make_definition()
        record = await WorkflowService.create("Old Name", definition, user=author)
        updated = await WorkflowService.update(str(record.id), user=author, name="New Name")
        assert updated.name == "New Name"

    async def test_update_definition(self, author):
        old_def = make_definition(step_name="old")
        record = await WorkflowService.create("WF", old_def, user=author)

        new_def = make_definition(step_name="new")
        updated = await WorkflowService.update(str(record.id), user=author, workflow=new_def)
        assert updated.definition["steps"][0]["name"] == "new"

    async def test_update_active_flag(self, author):
        definition = make_definition()
        record = await WorkflowService.create("WF", definition, user=author)
        updated = await WorkflowService.update(str(record.id), user=author, active=False)
        assert updated.active is False

    async def test_delete_removes_record(self, author):
        from django_ai_sdk.workflows.models import WorkflowSettings

        definition = make_definition()
        record = await WorkflowService.create("WF", definition, user=author)
        await WorkflowService.delete(str(record.id), user=author)

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.get(str(record.id), user=author)

    async def test_list_workflows_active_only(self, author):
        definition = make_definition()
        active = await WorkflowService.create("Active", definition, user=author)
        inactive = await WorkflowService.create("Inactive", definition, user=author)
        await WorkflowService.update(str(inactive.id), user=author, active=False)

        records = await WorkflowService.list_workflows(user=author, active_only=True)
        ids = [str(r.id) for r in records]
        assert str(active.id) in ids
        assert str(inactive.id) not in ids

    async def test_list_workflows_all(self, author):
        definition = make_definition()
        active = await WorkflowService.create("Active", definition, user=author)
        inactive = await WorkflowService.create("Inactive", definition, user=author)
        await WorkflowService.update(str(inactive.id), user=author, active=False)

        records = await WorkflowService.list_workflows(user=author, active_only=False)
        ids = [str(r.id) for r in records]
        assert str(active.id) in ids
        assert str(inactive.id) in ids


@pytest.mark.django_db
@pytest.mark.asyncio
class TestWorkflowServiceRunById:
    async def test_run_by_id_enqueues_task(self, author):
        definition = make_definition()
        record = await WorkflowService.create("WF", definition, user=author)

        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=author)

        assert run.status == "pending"
        assert str(run.workflow_id) == str(record.id)

    async def test_run_by_id_raises_for_inactive(self, author):
        from django_ai_sdk.workflows.models import WorkflowSettings

        definition = make_definition()
        record = await WorkflowService.create("WF", definition, user=author)
        await WorkflowService.update(str(record.id), user=author, active=False)

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.run_by_id(str(record.id), user=author)

    async def test_run_by_id_raises_for_unknown_id(self, author):
        from django_ai_sdk.workflows.models import WorkflowSettings

        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.run_by_id(str(uuid4()), user=author)


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
