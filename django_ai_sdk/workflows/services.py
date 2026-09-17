from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.permissions import user_pk
from django_ai_sdk.utils import serialize
from django_ai_sdk.workflows.actions import get_action_registry
from django_ai_sdk.workflows.executor import WorkflowExecutor, validate_inputs
from django_ai_sdk.workflows.registry import validate_definition

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.schemas import WorkflowDefinition


class WorkflowService:
    @staticmethod
    async def run(
        workflow: WorkflowDefinition,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        from django_ai_sdk.workflows.models import WorkflowRun

        validate_definition(workflow)
        # Checked here rather than in the worker: a caller who supplied the wrong
        # inputs should hear about it on the call that queued the run.
        validate_inputs(workflow, inputs or {})

        run = await WorkflowRun.objects.acreate(
            workflow=None,
            workflow_definition=workflow.model_dump(),
            status=WorkflowRun.Status.PENDING,
            inputs=serialize(inputs or {}),
            user_id=user_pk(user),
        )
        await WorkflowExecutor.enqueue(run)
        return run

    @staticmethod
    async def run_by_id(
        workflow_id: str,
        *,
        inputs: dict[str, Any] | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        run_id: str | None = None,
    ) -> WorkflowRun:
        from django_ai_sdk.workflows.models import WorkflowRun, WorkflowSettings

        record = await WorkflowSettings.objects.aget(id=workflow_id, active=True)
        workflow = record.to_workflow_definition()
        validate_definition(workflow)

        if run_id:
            run = await WorkflowRun.objects.aget(id=run_id, workflow_id=workflow_id)
        else:
            validate_inputs(workflow, inputs or {})
            run = await WorkflowRun.objects.acreate(
                workflow=record,
                workflow_definition=workflow.model_dump(),
                status=WorkflowRun.Status.PENDING,
                inputs=serialize(inputs or {}),
                user_id=user_pk(user),
            )
        await WorkflowExecutor.enqueue(run)
        return run

    @staticmethod
    def list_actions() -> list[dict[str, str]]:
        """The actions a definition may name, for a UI that composes one."""
        return [
            {"key": key, "description": getattr(cls, "description", "")}
            for key, cls in get_action_registry().items()
        ]

    @staticmethod
    async def create(
        name: str,
        workflow: WorkflowDefinition,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        from django_ai_sdk.workflows.models import WorkflowSettings

        record = WorkflowSettings(
            name=name,
            definition=workflow.model_dump(),
            created_by_id=user_pk(user),
        )
        await record.asave()
        return record

    @staticmethod
    async def update(
        workflow_id: str,
        *,
        name: str | None = None,
        workflow: WorkflowDefinition | None = None,
        active: bool | None = None,
    ) -> Any:
        from django_ai_sdk.workflows.models import WorkflowSettings

        record = await WorkflowSettings.objects.aget(id=workflow_id)
        if name is not None:
            record.name = name
        if workflow is not None:
            record.definition = workflow.model_dump()
        if active is not None:
            record.active = active
        await record.asave()
        return record

    @staticmethod
    async def delete(workflow_id: str) -> None:
        from django_ai_sdk.workflows.models import WorkflowSettings

        await WorkflowSettings.objects.filter(id=workflow_id).adelete()

    @staticmethod
    async def get(workflow_id: str) -> Any:
        from django_ai_sdk.workflows.models import WorkflowSettings

        return await WorkflowSettings.objects.aget(id=workflow_id)

    @staticmethod
    async def list_workflows(
        *, active_only: bool = True, limit: int | None = 100, offset: int = 0
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowSettings

        qs = WorkflowSettings.objects.all()
        if active_only:
            qs = qs.filter(active=True)
        return [
            r async for r in qs[offset : offset + limit if limit is not None else None]
        ]

    @staticmethod
    async def list_runs(
        workflow_id: str,
        *,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowRun

        qs = WorkflowRun.objects.filter(workflow_id=workflow_id).order_by("-created_at")
        return [
            r async for r in qs[offset : offset + limit if limit is not None else None]
        ]

    @staticmethod
    async def get_run(run_id: str) -> Any:
        from django_ai_sdk.workflows.models import WorkflowRun

        return await WorkflowRun.objects.prefetch_related("steps").aget(id=run_id)
