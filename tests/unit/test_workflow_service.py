"""WorkflowService CRUD, start and resume, as its own author sees them.

Every door is gated on a principal, so these carry a real user throughout. Who is
refused which door is `test_workflow_permissions.py`.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest

from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows.registry import definition_of
from django_ai_sdk.workflows.schemas import FieldDefinition, StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.services import WorkflowService

from tests.mocks.workflow import ENQUEUE


@pytest.fixture
async def author():
    from tests.factories.db import MANAGE_WORKFLOWS, UserFactory, agrant

    return await agrant(await UserFactory.acreate(), MANAGE_WORKFLOWS)


def make_definition(agent_id="asst-1", step_name="result"):
    return WorkflowDefinition(
        name="test-workflow",
        steps=[StepDefinition(name=step_name, agent_id=agent_id)],
    )


# transaction=True: an async ORM write is not rolled back by plain django_db.
@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestWorkflowServiceCRUD:
    async def test_create_stores_definition(self, author):
        definition = make_definition()
        record = await WorkflowService.create("My Workflow", definition, user=author)

        assert record.name == "My Workflow"
        # The slug names the stored definition, whatever name it was sent with.
        assert record.slug == "my-workflow"
        assert record.definition == {**definition.model_dump(), "name": "my-workflow"}
        assert definition_of(record).name == "my-workflow"
        assert record.active is True

    async def test_the_author_is_recorded_as_the_creator(self, author):
        record = await WorkflowService.create("WF", make_definition(), user=author)

        assert record.created_by_id == author.pk

    async def test_get_returns_record(self, author):
        definition = make_definition()
        created = await WorkflowService.create("WF", definition, user=author)
        fetched = await WorkflowService.get(str(created.id), user=author)
        assert str(fetched.id) == str(created.id)

    async def test_get_raises_for_unknown_id(self, author):
        from django_ai_sdk.workflows.models import Workflow

        with pytest.raises(Workflow.DoesNotExist):
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
        from django_ai_sdk.workflows.models import Workflow

        definition = make_definition()
        record = await WorkflowService.create("WF", definition, user=author)
        await WorkflowService.delete(str(record.id), user=author)

        with pytest.raises(Workflow.DoesNotExist):
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

    async def test_list_workflows_limit_none_returns_all(self, author):
        definition = make_definition()
        baseline = len(await WorkflowService.list_workflows(user=author, limit=None))
        for i in range(150):
            await WorkflowService.create(f"WF {i}", definition, user=author)

        assert len(await WorkflowService.list_workflows(user=author)) == 100  # default cap
        assert len(await WorkflowService.list_workflows(user=author, limit=None)) == baseline + 150

    async def test_list_runs_limit_none_returns_all(self, author):
        from django_ai_sdk.workflows.models import WorkflowRun

        definition = make_definition()
        record = await WorkflowService.create("WF", definition, user=author)
        for _ in range(60):
            await WorkflowRun.objects.acreate(workflow=record, user_id=author.pk)

        assert len(await WorkflowService.list_runs(str(record.id), user=author)) == 50
        assert (
            len(await WorkflowService.list_runs(str(record.id), user=author, limit=None)) == 60
        )


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestWorkflowServiceStart:
    async def test_a_stored_workflow_starts_by_its_slug(self, author):
        record = await WorkflowService.create("WF", make_definition(), user=author)

        with patch(ENQUEUE) as enqueue:
            run = await WorkflowService.start(record.slug, user=author)

        assert run.status == "pending"
        assert str(run.workflow_id) == str(record.id)
        assert run.workflow_key == record.slug
        enqueue.assert_called_once_with(str(run.id))

    async def test_an_inactive_workflow_does_not_start(self, author):
        from django_ai_sdk.workflows.models import Workflow

        record = await WorkflowService.create("WF", make_definition(), user=author)
        await WorkflowService.update(str(record.id), user=author, active=False)

        with pytest.raises(Workflow.DoesNotExist):
            await WorkflowService.start(record.slug, user=author)

    async def test_an_unknown_key_does_not_start(self, author):
        from django_ai_sdk.workflows.models import Workflow

        with pytest.raises(Workflow.DoesNotExist):
            await WorkflowService.start("no-such-workflow", user=author)

    async def test_a_registered_workflow_starts_by_its_name_before_a_stored_one(self, author):
        from django_ai_sdk.workflows.registry import register_workflow, reset_registry

        await WorkflowService.create("Test workflow", make_definition(), user=author)
        register_workflow(make_definition(step_name="declared"))
        try:
            with patch(ENQUEUE):
                run = await WorkflowService.start("test-workflow", user=author)
        finally:
            reset_registry()

        assert run.workflow_id is None
        assert run.workflow_key == "test-workflow"
        assert run.workflow_definition["steps"][0]["name"] == "declared"

    async def test_inputs_are_checked_when_it_starts(self, author):
        definition = WorkflowDefinition(
            name="needs-topic",
            input_fields={"topic": FieldDefinition()},
            steps=[StepDefinition(name="result", agent_id="asst-1")],
        )
        record = await WorkflowService.create("Needs topic", definition, user=author)

        with patch(ENQUEUE) as enqueue, pytest.raises(ValueError, match="missing"):
            await WorkflowService.start(record.slug, user=author, inputs={})
        enqueue.assert_not_called()

    async def test_only_the_declared_inputs_are_stored(self, author):
        definition = WorkflowDefinition(
            name="needs-topic",
            input_fields={"topic": FieldDefinition()},
            steps=[StepDefinition(name="result", agent_id="asst-1")],
        )
        record = await WorkflowService.create("Needs topic", definition, user=author)

        with patch(ENQUEUE):
            run = await WorkflowService.start(
                record.slug, user=author, inputs={"topic": "x", "secret": "s3cret"}
            )

        assert run.inputs == {"topic": "x"}

    async def test_a_stored_workflow_that_no_longer_compiles_is_a_bad_request(self, author):
        from django_ai_sdk.workflows.models import Workflow

        record = await WorkflowService.create("WF", make_definition(), user=author)
        broken = {**record.definition, "steps": [{"name": "x", "type": "gone"}]}
        await Workflow.objects.filter(pk=record.pk).aupdate(definition=broken)

        with patch(ENQUEUE) as enqueue, pytest.raises(ValueError, match="cannot run"):
            await WorkflowService.start(record.slug, user=author)
        enqueue.assert_not_called()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestWorkflowServiceResume:
    async def _run(self, author, status):
        from django_ai_sdk.workflows.models import WorkflowRun

        record = await WorkflowService.create("WF", make_definition(), user=author)
        with patch(ENQUEUE):
            run = await WorkflowService.start(record.slug, user=author)
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(status=status, error="it broke")
        return run

    @pytest.mark.parametrize("status", ["failed", "interrupted"])
    async def test_a_stopped_run_is_queued_again(self, author, status):
        run = await self._run(author, status)

        with patch(ENQUEUE) as enqueue:
            resumed = await WorkflowService.resume(str(run.id), user=author)

        assert (resumed.status, resumed.error) == ("pending", "")
        enqueue.assert_called_once_with(str(run.id))

    async def test_a_resumed_run_waits_for_a_new_queue_entry(self, author):
        """Its old task id is gone, so the sweep finds it if the new one never lands."""
        from django_ai_sdk.workflows.models import WorkflowRun

        run = await self._run(author, "failed")
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(task_id="old-task")

        with patch(ENQUEUE):
            resumed = await WorkflowService.resume(str(run.id), user=author)

        assert resumed.task_id is None

    async def test_a_run_of_a_deactivated_workflow_does_not_resume(self, author):
        run = await self._run(author, "failed")
        await WorkflowService.update(str(run.workflow_id), user=author, active=False)

        with patch(ENQUEUE) as enqueue, pytest.raises(ValueError, match="no longer runs"):
            await WorkflowService.resume(str(run.id), user=author)
        enqueue.assert_not_called()

    @pytest.mark.parametrize("status", ["pending", "running", "completed"])
    async def test_any_other_run_is_refused(self, author, status):
        run = await self._run(author, status)

        with patch(ENQUEUE) as enqueue, pytest.raises(ValueError, match="only a failed"):
            await WorkflowService.resume(str(run.id), user=author)
        enqueue.assert_not_called()

    async def test_an_unknown_run_reads_as_absent(self, author):
        from django_ai_sdk.workflows.models import WorkflowRun

        with pytest.raises(WorkflowRun.DoesNotExist):
            await WorkflowService.resume(str(uuid4()), user=author)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestWorkflowServiceInputsSchema:
    """The JSON Schema a run is composed against, read from the same model the
    worker validates with."""

    async def test_it_returns_the_declared_shape(self, author):
        definition = WorkflowDefinition(
            name="test-workflow",
            input_fields={"document": FieldDefinition(type="str")},
            steps=[StepDefinition(name="result", agent_id="asst-1")],
        )
        record = await WorkflowService.create("WF", definition, user=author)

        schema = await WorkflowService.get_inputs_schema(str(record.id), user=author)

        assert schema["properties"]["document"] == {"title": "Document", "type": "string"}
        assert schema["required"] == ["document"]

    async def test_a_definition_declaring_nothing_is_open(self, author):
        record = await WorkflowService.create("WF", make_definition(), user=author)

        assert await WorkflowService.get_inputs_schema(str(record.id), user=author) == {
            "type": "object",
            "additionalProperties": True,
        }

    async def test_an_unknown_id_reads_as_absent(self, author):
        assert await WorkflowService.get_inputs_schema(str(uuid4()), user=author) is None

    async def test_an_inactive_row_reads_as_absent(self, author):
        record = await WorkflowService.create("WF", make_definition(), user=author)
        await WorkflowService.update(str(record.id), user=author, active=False)

        assert await WorkflowService.get_inputs_schema(str(record.id), user=author) is None


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
class TestWorkflowServiceListListeners:
    async def test_returns_registered_listeners(self, author):
        class MyListener:
            description = "does something"

        with patch(
            "django_ai_sdk.workflows.services.get_listener_registry",
            return_value={"my_listener": MyListener},
        ):
            listeners = await WorkflowService.list_listeners(user=author)

        assert listeners == [{"key": "my_listener", "description": "does something"}]

    async def test_empty_when_no_listeners_registered(self, author):
        with patch(
            "django_ai_sdk.workflows.services.get_listener_registry",
            return_value={},
        ):
            assert await WorkflowService.list_listeners(user=author) == []

    async def test_it_takes_the_manage_permission(self):
        from tests.factories.db import UserFactory

        with pytest.raises(PermissionDenied):
            await WorkflowService.list_listeners(user=await UserFactory.acreate())
