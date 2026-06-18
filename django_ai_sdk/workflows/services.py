from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.workflows.actions import get_action_registry
from django_ai_sdk.workflows.executor import WorkflowExecutor

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.workflows.schemas import WorkflowDefinition


def _parse_messages(raw: list[dict]) -> list[ChatMessage]:
    from django_ai_sdk.protocols.vercel import VercelProtocolHandler
    from django_ai_sdk.views.schemas import Message

    return VercelProtocolHandler().to_chat_messages([Message(**m) for m in raw])


class WorkflowService:
    @staticmethod
    async def run(
        workflow: WorkflowDefinition,
        messages: list[dict],
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> dict[str, Any]:
        return await WorkflowExecutor().run(workflow, _parse_messages(messages), user=user)

    @staticmethod
    async def run_by_id(
        workflow_id: str,
        messages: list[dict],
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> dict[str, Any]:
        from django_ai_sdk.workflows.models import WorkflowSettings

        record = await WorkflowSettings.objects.aget(id=workflow_id, active=True)
        workflow = record.to_workflow_definition()
        return await WorkflowExecutor().run(workflow, _parse_messages(messages), user=user)

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
            created_by_id=user.pk if user and not getattr(user, "is_anonymous", True) else None,
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
    async def list_workflows(*, active_only: bool = True) -> list[Any]:
        from django_ai_sdk.workflows.models import WorkflowSettings

        qs = WorkflowSettings.objects.all()
        if active_only:
            qs = qs.filter(active=True)
        return [r async for r in qs]
