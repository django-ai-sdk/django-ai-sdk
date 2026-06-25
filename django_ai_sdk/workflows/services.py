from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.workflows.actions import get_action_registry
from django_ai_sdk.workflows.executor import WorkflowExecutor

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.schemas import WorkflowDefinition


def _user_id(user: AbstractBaseUser | AnonymousUser | None) -> Any:
    return user.pk if user and not getattr(user, "is_anonymous", True) else None


class WorkflowService:
    @staticmethod
    async def run(
        workflow: WorkflowDefinition,
        messages: list[ChatMessage],
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> WorkflowRun:
        from django_ai_sdk.workflows.models import WorkflowRun

        run = await WorkflowRun.objects.acreate(
            workflow=None,
            workflow_definition=workflow.model_dump(),
            status=WorkflowRun.Status.PENDING,
            input_messages=[m.model_dump() for m in messages],
            user_id=_user_id(user),
        )
        await WorkflowExecutor.enqueue(run)
        return run

    @staticmethod
    async def run_by_id(
        workflow_id: str,
        messages: list[ChatMessage],
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        run_id: str | None = None,
    ) -> WorkflowRun:
        from django_ai_sdk.workflows.models import WorkflowRun, WorkflowSettings

        record = await WorkflowSettings.objects.aget(id=workflow_id, active=True)
        workflow = record.to_workflow_definition()

        if run_id:
            run = await WorkflowRun.objects.aget(id=run_id, workflow_id=workflow_id)
        else:
            run = await WorkflowRun.objects.acreate(
                workflow=record,
                workflow_definition=workflow.model_dump(),
                status=WorkflowRun.Status.PENDING,
                input_messages=[m.model_dump() for m in messages],
                user_id=_user_id(user),
            )
        await WorkflowExecutor.enqueue(run)
        return run

    @staticmethod
    def list_actions() -> list[dict[str, str]]:
        return [
            {"key": key, "description": getattr(cls, "description", "")}
            for key, cls in get_action_registry().items()
        ]

    # --- CRUD ---

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
            created_by_id=_user_id(user),
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
        *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowSettings

        qs = WorkflowSettings.objects.all()
        if active_only:
            qs = qs.filter(active=True)
        return [r async for r in qs[offset : offset + limit]]

    # --- Run history ---

    @staticmethod
    async def list_runs(
        workflow_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowRun

        qs = WorkflowRun.objects.filter(workflow_id=workflow_id).order_by("-created_at")
        return [r async for r in qs[offset : offset + limit]]

    @staticmethod
    async def get_run(run_id: str) -> Any:
        from django_ai_sdk.workflows.models import WorkflowRun

        return await WorkflowRun.objects.prefetch_related("steps").aget(id=run_id)
