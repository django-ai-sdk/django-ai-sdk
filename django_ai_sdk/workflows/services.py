from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import Operation, PermissionDomain, PermissionsMixin, user_pk
from django_ai_sdk.workflows.actions import get_action_registry
from django_ai_sdk.workflows.definitions import inputs_json_schema
from django_ai_sdk.workflows.executor import WorkflowExecutor, open_run, validate_inputs
from django_ai_sdk.workflows.registry import validate_definition

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.schemas import WorkflowDefinition


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

        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        validate_definition(workflow)
        # Checked here rather than in the worker: a caller who supplied the wrong
        # inputs should hear about it on the call that queued the run.
        validate_inputs(workflow, inputs or {})

        run = await open_run(workflow, inputs=inputs, user=user)
        await WorkflowExecutor.enqueue(run)
        return run

    @classmethod
    async def run_by_id(
        cls,
        workflow_id: str,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        run_id: str | None = None,
    ) -> WorkflowRun:
        from django_ai_sdk.workflows.models import WorkflowRun, WorkflowSettings

        await cls.has_perms(user, Operation.RUN_WORKFLOW, raise_on_deny=True)
        record = await WorkflowSettings.objects.aget(id=workflow_id, active=True)
        await cls.has_perms(user, Operation.RUN_WORKFLOW, obj=record, raise_on_deny=True)
        workflow = record.to_workflow_definition()
        validate_definition(workflow)

        if run_id:
            # Resuming reads the run first, so a foreign one is absent rather than
            # restarted under its owner's name.
            run = await WorkflowRun.objects.filter(id=run_id, workflow_id=workflow_id).afirst()
            if run is None or not await cls.has_perms(
                user, Operation.VIEW_WORKFLOW, run, raise_on_deny=False
            ):
                raise WorkflowRun.DoesNotExist
        else:
            validate_inputs(workflow, inputs or {})
            run = await open_run(workflow, inputs=inputs, user=user, record=record)
        await WorkflowExecutor.enqueue(run)
        return run

    @staticmethod
    def list_actions() -> list[dict[str, str]]:
        """The actions a definition may name, for a UI that composes one."""
        return [
            {"key": key, "description": getattr(cls, "description", "")}
            for key, cls in get_action_registry().items()
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
        a form and the executor cannot disagree. A row the caller may not run
        reads as absent, the same as everywhere else.
        """
        from django_ai_sdk.workflows.models import WorkflowSettings

        await cls.has_perms(user, Operation.VIEW_WORKFLOW, raise_on_deny=True)
        record = await WorkflowSettings.objects.filter(id=workflow_id, active=True).afirst()
        if record is None or not await cls.has_perms(
            user, Operation.VIEW_WORKFLOW, obj=record, raise_on_deny=False
        ):
            return None
        definition = record.to_workflow_definition()
        validate_definition(definition)
        return inputs_json_schema(definition)

    @classmethod
    async def create(
        cls,
        name: str,
        workflow: WorkflowDefinition,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        from django_ai_sdk.workflows.models import WorkflowSettings

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        validate_definition(workflow)

        record = WorkflowSettings(
            name=name,
            definition=workflow.model_dump(),
            created_by_id=user_pk(user),
        )
        await record.asave()
        return record

    @classmethod
    async def _managed(
        cls,
        workflow_id: str,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> Any:
        """The stored definition this caller may manage, raising DoesNotExist otherwise."""
        from django_ai_sdk.workflows.models import WorkflowSettings

        record = await WorkflowSettings.objects.filter(id=workflow_id).afirst()
        if record is None or not await cls.has_perms(
            user, Operation.MANAGE_WORKFLOW, obj=record, raise_on_deny=False
        ):
            raise WorkflowSettings.DoesNotExist
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
        record = await cls._managed(workflow_id, user)
        if workflow is not None:
            validate_definition(workflow)
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
        from django_ai_sdk.workflows.models import WorkflowSettings

        record = await cls._managed(workflow_id, user)
        await WorkflowSettings.objects.filter(id=record.id).adelete()

    @classmethod
    async def get(
        cls,
        workflow_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        return await cls._managed(workflow_id, user)

    @classmethod
    async def list_workflows(
        cls,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        active_only: bool = True,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowSettings

        await cls.has_perms(user, Operation.MANAGE_WORKFLOW, raise_on_deny=True)
        qs = WorkflowSettings.objects.all()
        if active_only:
            qs = qs.filter(active=True)
        qs = cls.has_queryset_perms(user, Operation.MANAGE_WORKFLOW, queryset=qs)
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
        qs = cls.has_queryset_perms(user, Operation.VIEW_WORKFLOW, queryset=qs)
        return [r async for r in qs[offset : offset + limit if limit is not None else None]]

    @classmethod
    async def get_run(
        cls,
        run_id: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any | None:
        from django_ai_sdk.workflows.models import WorkflowRun

        run = await WorkflowRun.objects.filter(id=run_id).prefetch_related("steps").afirst()
        if run is None:
            return None
        if not await cls.has_perms(user, Operation.VIEW_WORKFLOW, run, raise_on_deny=False):
            return None
        return run


__all__ = ["WorkflowService"]
