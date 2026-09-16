"""WorkflowService ACL: own vs foreign, manage vs run."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows.executor import WorkflowExecutor
from django_ai_sdk.workflows.models import WorkflowSettings
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.services import WorkflowService


def definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="acl-wf",
        steps=[StepDefinition(name="result", agent_id="a")],
    )


async def make_user(label="user", **flags):
    from tests.factories.db import UserFactory

    return await UserFactory.acreate(email=f"{label}@example.com", **flags)


@pytest.mark.django_db(transaction=True)
class TestWorkflowServicePermissions:
    async def test_owner_can_get_and_update(self):
        owner = await make_user("owner")
        record = await WorkflowService.create("Mine", definition(), user=owner)
        fetched = await WorkflowService.get(str(record.id), user=owner)
        assert str(fetched.id) == str(record.id)
        updated = await WorkflowService.update(str(record.id), user=owner, name="Renamed")
        assert updated.name == "Renamed"

    async def test_foreign_user_finds_it_absent_rather_than_forbidden(self):
        """Absent on every manage door, because 403 here would confirm the id exists."""
        owner = await make_user("owner")
        other = await make_user("other")
        record = await WorkflowService.create("Mine", definition(), user=owner)
        for call in (
            WorkflowService.get(str(record.id), user=other),
            WorkflowService.update(str(record.id), user=other, name="Hijack"),
            WorkflowService.delete(str(record.id), user=other),
        ):
            with pytest.raises(WorkflowSettings.DoesNotExist):
                await call

        # The refusal was real: the row is untouched and still the owner's.
        survivor = await WorkflowService.get(str(record.id), user=owner)
        assert survivor.name == "Mine"

    async def test_staff_can_manage_others(self):
        owner = await make_user("owner")
        staff = await make_user("staff", is_staff=True)
        record = await WorkflowService.create("Mine", definition(), user=owner)
        fetched = await WorkflowService.get(str(record.id), user=staff)
        assert str(fetched.id) == str(record.id)

    async def test_list_workflows_scopes_to_owner(self):
        a = await make_user("a")
        b = await make_user("b")
        mine = await WorkflowService.create("A", definition(), user=a)
        await WorkflowService.create("B", definition(), user=b)
        rows = await WorkflowService.list_workflows(user=a)
        ids = {str(r.id) for r in rows}
        assert str(mine.id) in ids
        assert len(ids) == 1

    async def test_foreign_run_is_absent(self):
        a = await make_user("a")
        b = await make_user("b")
        record = await WorkflowService.create("A", definition(), user=a)
        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=a)
        assert await WorkflowService.get_run(str(run.id), user=a) is not None
        assert await WorkflowService.get_run(str(run.id), user=b) is None

    async def test_anonymous_cannot_create(self):
        from django.contrib.auth.models import AnonymousUser

        with pytest.raises(PermissionDenied):
            await WorkflowService.create("Nope", definition(), user=AnonymousUser())


@pytest.mark.django_db(transaction=True)
class TestRunningIsNotManaging:
    """An active definition is shared; editing one is not."""

    async def test_a_foreign_user_may_run_what_they_may_not_edit(self):
        owner = await make_user("owner")
        other = await make_user("other")
        record = await WorkflowService.create("Shared", definition(), user=owner)

        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=other)

        assert run.user_id == other.pk
        with pytest.raises(WorkflowSettings.DoesNotExist):
            await WorkflowService.update(str(record.id), user=other, name="Hijack")

    async def test_the_owner_does_not_see_a_run_someone_else_started(self):
        # A run's outputs are the principal's content, not the author's.
        owner = await make_user("owner")
        other = await make_user("other")
        record = await WorkflowService.create("Shared", definition(), user=owner)

        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=other)

        assert await WorkflowService.get_run(str(run.id), user=owner) is None
        assert await WorkflowService.list_runs(str(record.id), user=owner) == []
        assert len(await WorkflowService.list_runs(str(record.id), user=other)) == 1

    async def test_staff_read_every_run(self):
        other = await make_user("other")
        staff = await make_user("staff", is_staff=True)
        record = await WorkflowService.create("Shared", definition(), user=other)

        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=other)

        assert await WorkflowService.get_run(str(run.id), user=staff) is not None
        assert len(await WorkflowService.list_runs(str(record.id), user=staff)) == 1

    async def test_resuming_a_foreign_run_reads_as_absent(self):
        # DoesNotExist rather than PermissionDenied: a 403 confirms the id exists.
        from django_ai_sdk.workflows.models import WorkflowRun

        owner = await make_user("owner")
        other = await make_user("other")
        record = await WorkflowService.create("Shared", definition(), user=owner)
        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=owner)

        with (
            patch.object(WorkflowExecutor, "enqueue", AsyncMock()),
            pytest.raises(WorkflowRun.DoesNotExist),
        ):
            await WorkflowService.run_by_id(str(record.id), user=other, run_id=str(run.id))


@pytest.mark.django_db(transaction=True)
class TestAnAnonymousCallerReadsNothing:
    async def test_run_history_is_denied_rather_than_scoped(self):
        from django.contrib.auth.models import AnonymousUser

        owner = await make_user("owner")
        record = await WorkflowService.create("Mine", definition(), user=owner)

        with pytest.raises(PermissionDenied):
            await WorkflowService.list_runs(str(record.id), user=AnonymousUser())

    async def test_a_run_by_id_is_absent(self):
        from django.contrib.auth.models import AnonymousUser

        owner = await make_user("owner")
        record = await WorkflowService.create("Mine", definition(), user=owner)
        with patch.object(WorkflowExecutor, "enqueue", AsyncMock()):
            run = await WorkflowService.run_by_id(str(record.id), user=owner)

        assert await WorkflowService.get_run(str(run.id), user=AnonymousUser()) is None
