from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync, sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.db.models.functions import Now
from pydantic import ValidationError

from django_ai_sdk.errors import UserError
from django_ai_sdk.permissions import Operation, PermissionDomain, PermissionsMixin, user_pk
from django_ai_sdk.workflows.definitions import inputs_json_schema
from django_ai_sdk.workflows.listeners import get_listener_registry
from django_ai_sdk.workflows.registry import (
    ADHOC,
    afind,
    definition_of,
    is_registered,
    validate_definition,
)
from django_ai_sdk.workflows.runs import aenqueue_run, queue_run

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.schemas import WorkflowDefinition


def _validate_submitted(workflow: WorkflowDefinition) -> None:
    """A definition the caller sent: an illegal one is their bad request."""
    try:
        validate_definition(workflow)
    except ImproperlyConfigured as exc:
        raise UserError(str(exc)) from exc


class WorkflowService(PermissionsMixin):
    domain = PermissionDomain.WORKFLOW

    @classmethod
    async def run(
        cls,
        workflow: WorkflowDefinition,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Queue a run of `workflow` as `user`. For trusted code, never a client's JSON.

        A registered workflow needs RUN_WORKFLOW. Any other definition was written by
        the caller, so it also needs MANAGE_WORKFLOW.
        """
        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        if not is_registered(workflow):
            await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        _validate_submitted(workflow)
        return await queue_run(workflow, inputs=inputs, user=user)

    @classmethod
    async def start(
        cls,
        key: str,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Queue a run of the workflow named `key`.

        A registered workflow wins, then an active stored one by slug. Raises
        `Workflow.DoesNotExist` when neither exists or the caller may not run it.
        """
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        workflow, record = await afind(key)
        if record is not None and await cls.has_perms(
            user, Operation.RUN_WORKFLOW, obj=record, raise_on_deny=False
        ):
            try:
                workflow = definition_of(record)
            except (ImproperlyConfigured, ValidationError) as exc:
                logger.warning("Stored workflow %r cannot run: %s", key, exc)
                raise UserError(f"Workflow {key!r} cannot run; see the server log.") from exc
        if workflow is None:
            raise Workflow.DoesNotExist
        return await queue_run(workflow, inputs=inputs, user=user, record=record)

    @classmethod
    async def resume(
        cls,
        run_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        """Queue a failed or interrupted run again. It skips the steps that completed.

        The run keeps its original definition, inputs and user. A run the caller may
        not resume raises `WorkflowRun.DoesNotExist`. An ad-hoc run also needs
        MANAGE_WORKFLOW, as starting one did. Pending and running runs are refused,
        because a worker already owns them.
        """
        from django_ai_sdk.workflows.models import WorkflowRun

        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        run = await WorkflowRun.objects.filter(id=run_id).afirst()
        if run is None or not await cls.has_perms(
            user, Operation.RUN_WORKFLOW, run, raise_on_deny=False
        ):
            raise WorkflowRun.DoesNotExist
        if run.workflow_key == ADHOC:
            await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        elif await afind(run.workflow_key) == (None, None):
            # Deactivating or deleting a workflow stops its old runs too.
            raise UserError(f"Workflow {run.workflow_key!r} no longer runs.")
        reopened = await WorkflowRun.objects.filter(
            pk=run.pk, status__in=(WorkflowRun.Status.FAILED, WorkflowRun.Status.INTERRUPTED)
        ).aupdate(
            status=WorkflowRun.Status.PENDING,
            error="",
            completed_at=None,
            # A fresh queue entry: until it lands, the sweep sees a never-queued run.
            task_id=None,
            updated_at=Now(),
        )
        if not reopened:
            raise UserError(
                f"Run {run.pk} is {run.status}; only a failed or interrupted run resumes."
            )
        await run.arefresh_from_db()
        await aenqueue_run(run)
        return run

    @classmethod
    async def list_listeners(
        cls,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[dict[str, str]]:
        """The listeners a definition may name, for a UI that composes one.

        Needs MANAGE_WORKFLOW, like composing a definition.
        """
        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        return [
            {"key": key, "description": listener.description}
            for key, listener in get_listener_registry().items()
        ]

    @classmethod
    async def get_inputs_schema(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> dict[str, Any] | None:
        """The stored workflow's inputs as a JSON Schema, for whoever composes a run.

        One artifact, compiled from the same model the run validates against, so
        a form and the run cannot disagree. A row the caller may not run
        reads as absent, the same as everywhere else.
        """
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.VIEW_WORKFLOW, raise_on_deny=True)
        record = await Workflow.objects.filter(id=workflow_id, active=True).afirst()
        if record is None or not await cls.has_perms(
            user, Operation.VIEW_WORKFLOW, obj=record, raise_on_deny=False
        ):
            return None
        return inputs_json_schema(definition_of(record))

    @classmethod
    async def create(
        cls,
        name: str,
        workflow: WorkflowDefinition,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        _validate_submitted(workflow)

        record = Workflow(
            name=name,
            definition=workflow.model_dump(),
            created_by_id=user_pk(user),
        )
        await record.asave()
        return record

    @classmethod
    async def get(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """The stored definition this caller may manage, raising DoesNotExist otherwise."""
        from django_ai_sdk.workflows.models import Workflow

        record = await Workflow.objects.filter(id=workflow_id).afirst()
        if record is None or not await cls.has_perms(
            user, Operation.MANAGE_WORKFLOW, obj=record, raise_on_deny=False
        ):
            raise Workflow.DoesNotExist
        return record

    @classmethod
    async def update(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        name: str | None = None,
        workflow: WorkflowDefinition | None = None,
        active: bool | None = None,
    ) -> Any:
        record = await cls.get(workflow_id, user=user)
        if workflow is not None:
            _validate_submitted(workflow)
            record.definition = workflow.model_dump()
        if name is not None:
            record.name = name
        if active is not None:
            record.active = active
        await record.asave()
        return record

    @classmethod
    async def delete(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> None:
        from django_ai_sdk.workflows.models import Workflow

        record = await cls.get(workflow_id, user=user)
        await Workflow.objects.filter(id=record.id).adelete()

    @classmethod
    async def list_workflows(
        cls,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        active_only: bool = True,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import Workflow

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        qs = Workflow.objects.all()
        if active_only:
            qs = qs.filter(active=True)
        qs = await sync_to_async(cls.has_queryset_perms)(
            user, Operation.MANAGE_WORKFLOW, queryset=qs
        )
        return [r async for r in qs[offset : offset + limit if limit is not None else None]]

    @classmethod
    async def list_runs(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowRun

        await cls.has_perms(user, Operation.VIEW_WORKFLOW, raise_on_deny=True)
        # An unknown workflow_id filters to nothing, the same answer as a workflow
        # whose runs are all someone else's.
        qs = WorkflowRun.objects.filter(workflow_id=workflow_id).order_by("-created_at")
        qs = await sync_to_async(cls.has_queryset_perms)(user, Operation.VIEW_WORKFLOW, queryset=qs)
        return [r async for r in qs[offset : offset + limit if limit is not None else None]]

    @classmethod
    async def get_run(
        cls,
        run_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        workflow_id: str | None = None,
    ) -> Any | None:
        """One run with its steps, or None. With `workflow_id`, only a run of that workflow."""
        from django_ai_sdk.workflows.models import WorkflowRun

        runs = WorkflowRun.objects.filter(id=run_id)
        if workflow_id is not None:
            runs = runs.filter(workflow_id=workflow_id)
        run = await runs.prefetch_related("steps").afirst()
        if run is None:
            return None
        if not await cls.has_perms(user, Operation.VIEW_WORKFLOW, run, raise_on_deny=False):
            return None
        return run


# Sync wrappers, for sync views (e.g. Django REST framework).
start_workflow = async_to_sync(WorkflowService.start)
resume_workflow_run = async_to_sync(WorkflowService.resume)
list_workflow_listeners = async_to_sync(WorkflowService.list_listeners)
create_workflow = async_to_sync(WorkflowService.create)
update_workflow = async_to_sync(WorkflowService.update)
delete_workflow = async_to_sync(WorkflowService.delete)
get_workflow = async_to_sync(WorkflowService.get)
list_workflows = async_to_sync(WorkflowService.list_workflows)
list_workflow_runs = async_to_sync(WorkflowService.list_runs)
get_workflow_run = async_to_sync(WorkflowService.get_run)

__all__ = ["WorkflowService"]
