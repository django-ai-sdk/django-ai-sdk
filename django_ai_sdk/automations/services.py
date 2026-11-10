"""Service layer for automations, for an HTTP layer to call.

The SDK ships no router. Schedules are read-only here because `cron` lives only in code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models import Max, OuterRef, Subquery

from django_ai_sdk.automations.models import AutomationState
from django_ai_sdk.automations.registry import get_automation, get_automations
from django_ai_sdk.automations.runner import dispatch_now, ensure_state, resolve_enabled
from django_ai_sdk.automations.schemas import AutomationOut, AutomationRunOut
from django_ai_sdk.permissions import Operation, PermissionDomain, PermissionsMixin, user_pk

if TYPE_CHECKING:
    from datetime import datetime

    from django.db.models import QuerySet

    from django_ai_sdk.automations.base import Automation
    from django_ai_sdk.types import UserType
    from django_ai_sdk.workflows.models import WorkflowRun


class AutomationService(PermissionsMixin):
    """Resolve, permission-check, and act on automations by name."""

    domain = PermissionDomain.AUTOMATIONS

    @classmethod
    async def list_for_user(cls, user: UserType) -> list[AutomationOut]:
        """Every automation, with live state, if `user` may view automations."""
        if not await cls.has_perms(user, Operation.VIEW_AUTOMATION, raise_on_deny=False):
            return []
        automations = get_automations()
        if not automations:
            return []
        states = {
            state.name: state
            async for state in AutomationState.objects.filter(name__in=list(automations))
        }
        latest = await cls._latest_runs(list(automations), await cls._visible_runs(user))
        return [
            cls._describe(automation, states.get(name), latest.get(name))
            for name, automation in automations.items()
        ]

    @classmethod
    async def set_enabled(cls, name: str, *, enabled: bool, user: UserType) -> AutomationOut | None:
        """Turn an automation on or off in the database. None for an unknown name."""
        automation = get_automation(name)
        if automation is None:
            return None
        await cls.has_perms(user, Operation.MANAGE_AUTOMATION)

        state = await ensure_state(automation)
        await AutomationState.objects.filter(id=state.id).aupdate(enabled=enabled)
        state.enabled = enabled
        latest = await cls._latest_runs([name], await cls._visible_runs(user))
        return cls._describe(automation, state, latest.get(name))

    @classmethod
    async def run_now(cls, name: str, *, user: UserType = None) -> list[AutomationRunOut] | None:
        """Dispatch an automation outside its schedule. None for an unknown name.

        `user` is who asked. The run itself acts as `run_as`. Returns an empty list
        when nothing was dispatched (kill switch off, or the run was skipped).
        """
        automation = get_automation(name)
        if automation is None:
            return None
        await cls.has_perms(user, Operation.MANAGE_AUTOMATION)

        run = await dispatch_now(name)
        return [cls._describe_run(run)] if run is not None else []

    @classmethod
    async def list_runs(
        cls, name: str, *, user: UserType = None, limit: int = 50, offset: int = 0
    ) -> list[AutomationRunOut]:
        """This automation's runs that `user` may read, newest first."""
        await cls.has_perms(user, Operation.VIEW_AUTOMATION)
        qs = (await cls._visible_runs(user)).filter(automation_run__automation=name)
        return [
            cls._describe_run(run)
            async for run in qs.order_by("-created_at")[offset : offset + limit]
        ]

    @classmethod
    async def get_run(cls, run_id: str, *, user: UserType = None) -> AutomationRunOut | None:
        """One automation run by id, or None.

        Someone else's run returns None, because a 403 would confirm the id exists.
        """
        from django.core.exceptions import ValidationError

        from django_ai_sdk.workflows.models import WorkflowRun

        try:
            run = await WorkflowRun.objects.filter(id=run_id, automation_run__isnull=False).afirst()
        except ValidationError:
            return None
        if run is None:
            return None
        await cls.has_perms(user, Operation.VIEW_AUTOMATION)
        mine = run.user_id is not None and run.user_id == user_pk(user)
        if not mine and not await cls.has_perms(
            user, Operation.MANAGE_AUTOMATION, raise_on_deny=False
        ):
            return None
        return cls._describe_run(run)

    @classmethod
    async def last_tick_at(cls) -> datetime | None:
        """When any automation was last dispatched, across the whole deployment.

        Use it as a dead-man's switch. The scheduler cannot report its own absence.
        """
        result = await AutomationState.objects.aaggregate(last=Max("last_dispatched_at"))
        return result["last"]

    @classmethod
    async def _visible_runs(cls, user: UserType) -> QuerySet:
        """Run output belongs to the run's account, so only a manager sees every run."""
        from django_ai_sdk.workflows.models import WorkflowRun

        runs = WorkflowRun.objects.all()
        if await cls.has_perms(user, Operation.MANAGE_AUTOMATION, raise_on_deny=False):
            return runs
        pk = user_pk(user)
        return runs.filter(user_id=pk) if pk is not None else runs.none()

    @staticmethod
    async def _latest_runs(names: list[str], runs: QuerySet) -> dict[str, WorkflowRun]:
        """The newest of `runs` per automation, in one query."""
        newest = Subquery(
            runs.filter(automation_run__automation=OuterRef("automation_run__automation"))
            .order_by("-created_at")
            .values("id")[:1]
        )
        return {
            run.automation_run.automation: run
            async for run in runs.filter(
                automation_run__automation__in=names, id=newest
            ).select_related("automation_run")
        }

    @classmethod
    def _describe(
        cls,
        automation: Automation,
        state: AutomationState | None,
        last_run: WorkflowRun | None,
    ) -> AutomationOut:
        enabled, source = resolve_enabled(automation, state)
        # Registration validated the schedule, so it builds.
        return AutomationOut(
            name=automation.name,
            label=automation.label or automation.name,
            description=automation.description,
            schedule=str(automation.get_schedule()),
            next_run_at=state.next_run_at if state else None,
            last_dispatched_at=state.last_dispatched_at if state else None,
            enabled=enabled,
            enabled_source=source,
            workflow=automation.workflow,
            run_as=automation.run_as,
            last_run=cls._describe_run(last_run) if last_run else None,
        )

    @staticmethod
    def _describe_run(run: WorkflowRun) -> AutomationRunOut:
        return AutomationRunOut(
            id=str(run.id),
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            output=run.outputs,
            error=run.error,
        )


__all__ = ["AutomationService"]
