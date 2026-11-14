"""WorkflowService ACL: running is for any signed-in user, authoring takes a model permission,
and a run is read by the user it ran as or a holder of view_workflowrun."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows.models import WorkflowRun, Workflow
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.services import WorkflowService
from tests.factories.db import MANAGE_WORKFLOWS, VIEW_WORKFLOW_RUNS, agrant
from tests.mocks.workflow import ENQUEUE


def definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="acl-wf",
        steps=[StepDefinition(name="result", agent_id="a")],
    )


async def make_user(label="user", *perms, **flags):
    from tests.factories.db import UserFactory

    user = await UserFactory.acreate(email=f"{label}@example.com", **flags)
    return await agrant(user, *perms)


async def stored(user=None) -> Workflow:
    author = user or await make_user("author", MANAGE_WORKFLOWS)
    return await WorkflowService.create("Shared", definition(), user=author)


async def run_as(record: Workflow, user) -> WorkflowRun:
    with patch(ENQUEUE):
        return await WorkflowService.start(record.slug, user=user)


async def failed(run: WorkflowRun) -> WorkflowRun:
    await WorkflowRun.objects.filter(pk=run.pk).aupdate(status=WorkflowRun.Status.FAILED)
    return run


@pytest.mark.django_db(transaction=True)
class TestManagingTakesTheModelPermission:
    async def test_a_signed_in_user_without_it_cannot_create(self):
        plain = await make_user("plain")

        with pytest.raises(PermissionDenied):
            await WorkflowService.create("Nope", definition(), user=plain)

    async def test_staff_without_it_cannot_create(self):
        staff = await make_user("staff", is_staff=True)

        with pytest.raises(PermissionDenied):
            await WorkflowService.create("Nope", definition(), user=staff)

    async def test_a_superuser_holds_it(self):
        root = await make_user("root", is_superuser=True)

        assert await WorkflowService.create("Mine", definition(), user=root)

    async def test_a_holder_manages_every_stored_workflow(self):
        record = await stored()
        other = await make_user("other", MANAGE_WORKFLOWS)

        updated = await WorkflowService.update(str(record.id), user=other, name="Renamed")
        assert updated.name == "Renamed"
        assert [str(r.id) for r in await WorkflowService.list_workflows(user=other)] == [
            str(record.id)
        ]

    async def test_without_it_every_manage_door_reads_as_absent(self):
        """Absent rather than forbidden, because a 403 here would confirm the id exists."""
        record = await stored()
        plain = await make_user("plain")
        for call in (
            WorkflowService.get(str(record.id), user=plain),
            WorkflowService.update(str(record.id), user=plain, name="Hijack"),
            WorkflowService.delete(str(record.id), user=plain),
        ):
            with pytest.raises(Workflow.DoesNotExist):
                await call
        assert (await Workflow.objects.aget(id=record.id)).name == "Shared"

    async def test_listing_stored_workflows_is_managing_them(self):
        await stored()
        plain = await make_user("plain")

        with pytest.raises(PermissionDenied):
            await WorkflowService.list_workflows(user=plain)

    async def test_anonymous_cannot_create(self):
        from django.contrib.auth.models import AnonymousUser

        with pytest.raises(PermissionDenied):
            await WorkflowService.create("Nope", definition(), user=AnonymousUser())


@pytest.mark.django_db(transaction=True)
class TestAnUnstoredDefinition:
    """A definition that is neither stored nor registered is authored by whoever runs it."""

    async def test_a_chat_only_user_is_refused(self):
        plain = await make_user("plain")

        with (
            patch(ENQUEUE) as enqueue,
            pytest.raises(PermissionDenied),
        ):
            await WorkflowService.run(definition(), user=plain)
        enqueue.assert_not_called()
        assert await WorkflowRun.objects.acount() == 0

    async def test_a_holder_of_the_manage_permission_may_run_it(self):
        manager = await make_user("manager", MANAGE_WORKFLOWS)

        with patch(ENQUEUE):
            run = await WorkflowService.run(definition(), user=manager)

        assert run.user_id == manager.pk

    async def test_a_registered_workflow_runs_on_the_run_permission(self):
        from django_ai_sdk.workflows.registry import register_workflow, reset_registry

        plain = await make_user("plain")
        register_workflow(definition())
        try:
            with patch(ENQUEUE):
                run = await WorkflowService.run(definition(), user=plain)
            assert run.user_id == plain.pk

            # The same name with other steps is not the registered workflow.
            changed = WorkflowDefinition(
                name="acl-wf", steps=[StepDefinition(name="other", agent_id="a")]
            )
            with pytest.raises(PermissionDenied):
                await WorkflowService.run(changed, user=plain)
        finally:
            reset_registry()


@pytest.mark.django_db(transaction=True)
class TestRunningIsNotManaging:
    async def test_a_signed_in_user_may_run_what_they_may_not_edit(self):
        record = await stored()
        other = await make_user("other")

        run = await run_as(record, other)

        assert run.user_id == other.pk
        with pytest.raises(Workflow.DoesNotExist):
            await WorkflowService.update(str(record.id), user=other, name="Hijack")

    async def test_anonymous_cannot_run(self):
        from django.contrib.auth.models import AnonymousUser

        record = await stored()
        with pytest.raises(PermissionDenied):
            await run_as(record, AnonymousUser())


@pytest.mark.django_db(transaction=True)
class TestReadingRuns:
    async def test_a_run_is_read_by_the_user_it_ran_as(self):
        record = await stored()
        a = await make_user("a")
        b = await make_user("b")
        run = await run_as(record, a)

        assert await WorkflowService.get_run(str(run.id), user=a) is not None
        assert await WorkflowService.get_run(str(run.id), user=b) is None
        assert len(await WorkflowService.list_runs(str(record.id), user=a)) == 1
        assert await WorkflowService.list_runs(str(record.id), user=b) == []

    async def test_the_author_does_not_read_a_run_someone_else_started(self):
        # A run's outputs are the user's content, not the author's.
        author = await make_user("author", MANAGE_WORKFLOWS)
        record = await stored(author)
        run = await run_as(record, await make_user("other"))

        assert await WorkflowService.get_run(str(run.id), user=author) is None
        assert await WorkflowService.list_runs(str(record.id), user=author) == []

    async def test_staff_alone_read_no_one_elses_run(self):
        record = await stored()
        run = await run_as(record, await make_user("other"))
        staff = await make_user("staff", is_staff=True)

        assert await WorkflowService.get_run(str(run.id), user=staff) is None
        assert await WorkflowService.list_runs(str(record.id), user=staff) == []

    async def test_view_workflowrun_reads_every_run(self):
        record = await stored()
        run = await run_as(record, await make_user("other"))
        reader = await make_user("reader", VIEW_WORKFLOW_RUNS)

        assert await WorkflowService.get_run(str(run.id), user=reader) is not None
        assert len(await WorkflowService.list_runs(str(record.id), user=reader)) == 1

    async def test_reading_a_run_is_not_resuming_it(self):
        # DoesNotExist rather than PermissionDenied: a 403 confirms the id exists.
        record = await stored()
        run = await failed(await run_as(record, await make_user("owner")))
        reader = await make_user("reader", VIEW_WORKFLOW_RUNS)

        with patch(ENQUEUE), pytest.raises(WorkflowRun.DoesNotExist):
            await WorkflowService.resume(str(run.id), user=reader)

    async def test_resuming_a_foreign_run_reads_as_absent(self):
        record = await stored()
        run = await failed(await run_as(record, await make_user("owner")))

        with patch(ENQUEUE), pytest.raises(WorkflowRun.DoesNotExist):
            await WorkflowService.resume(str(run.id), user=await make_user("other"))

    async def test_the_user_it_ran_as_resumes_it(self):
        record = await stored()
        owner = await make_user("owner")
        run = await failed(await run_as(record, owner))

        with patch(ENQUEUE) as enqueue:
            resumed = await WorkflowService.resume(str(run.id), user=owner)

        assert resumed.status == WorkflowRun.Status.PENDING
        enqueue.assert_called_once_with(str(run.id))

    async def test_resuming_an_adhoc_run_takes_the_manage_permission_again(self):
        manager = await make_user("manager", MANAGE_WORKFLOWS)
        with patch(ENQUEUE):
            run = await failed(await WorkflowService.run(definition(), user=manager))
        # The permission was taken away after the run was started.
        await WorkflowRun.objects.filter(pk=run.pk).aupdate(user=await make_user("plain"))
        plain = await WorkflowRun.objects.select_related("user").aget(pk=run.pk)

        with patch(ENQUEUE), pytest.raises(PermissionDenied):
            await WorkflowService.resume(str(run.id), user=plain.user)


@pytest.mark.django_db(transaction=True)
class TestAnAnonymousCallerReadsNothing:
    async def test_run_history_is_denied_rather_than_scoped(self):
        from django.contrib.auth.models import AnonymousUser

        record = await stored()

        with pytest.raises(PermissionDenied):
            await WorkflowService.list_runs(str(record.id), user=AnonymousUser())

    async def test_a_run_by_id_is_absent(self):
        from django.contrib.auth.models import AnonymousUser

        record = await stored()
        run = await run_as(record, await make_user("owner"))

        assert await WorkflowService.get_run(str(run.id), user=AnonymousUser()) is None
