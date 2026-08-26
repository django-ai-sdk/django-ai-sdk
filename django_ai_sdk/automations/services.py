"""Service-layer facade for automations: the seam an HTTP layer sits on.

The SDK ships no router; demo/apps/automations/views/ninja.py is a complete reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured
from django.db.models import Max, OuterRef, Subquery

from django_ai_sdk.automations.config import is_enabled
from django_ai_sdk.automations.models import AutomationRun, AutomationState
from django_ai_sdk.automations.registry import get_automation, get_automations
from django_ai_sdk.automations.schemas import AutomationOut, AutomationRunOut
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
)

if TYPE_CHECKING:
    from datetime import datetime

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.automations.base import Automation

logger = logging.getLogger(__name__)


def _pk(user: AbstractBaseUser | AnonymousUser | None) -> Any:
    """The user's primary key, or None for anonymous. Safe as a query value."""
    return user.pk if user is not None and user.is_authenticated else None


class AutomationService(PermissionsMixin):
    """Resolve, permission-check, and act on automations by name."""

    domain = PermissionDomain.AUTOMATIONS

    @classmethod
    async def list_for_user(
        cls, user: AbstractBaseUser | AnonymousUser | None
    ) -> list[AutomationOut]:
        """Every automation `user` may see, with live state."""
        automations = get_automations()
        if not automations:
            return []

        states = {
            state.name: state
            async for state in AutomationState.objects.filter(name__in=list(automations))
        }
        latest = await cls._latest_runs(list(automations))
        subscribed = await cls._subscribed_names(list(automations), user)

        rows: list[AutomationOut] = []
        for name, automation in automations.items():
            if not await automation.has_perms(user, Operation.VIEW_AUTOMATION):
                continue
            rows.append(
                cls._describe(
                    automation, states.get(name), latest.get(name), subscribed=name in subscribed
                )
            )
        return rows

    @classmethod
    async def set_enabled(
        cls, name: str, *, enabled: bool, user: AbstractBaseUser | AnonymousUser | None
    ) -> AutomationOut | None:
        """Turn an automation on or off at the database layer.

        Returns None for an unknown name, so a router answers 404 without catching.
        """
        automation = get_automation(name)
        if automation is None:
            return None
        if not await automation.has_perms(user, Operation.MANAGE_AUTOMATION):
            raise PermissionDenied(f"Not permitted to manage {name!r}")

        state = await cls._ensure_state(automation)
        await AutomationState.objects.filter(id=state.id).aupdate(enabled=enabled)
        state.enabled = enabled
        latest = await cls._latest_runs([name])
        subscribed = name in await cls._subscribed_names([name], user)
        return cls._describe(automation, state, latest.get(name), subscribed=subscribed)

    @classmethod
    async def set_subscribed(
        cls, name: str, *, user: AbstractBaseUser | AnonymousUser | None, enabled: bool
    ) -> AutomationOut | None:
        """Turn this automation on or off for `user`, and only for `user`."""
        automation = get_automation(name)
        if automation is None:
            return None
        if not await automation.has_perms(user, Operation.SUBSCRIBE_AUTOMATION):
            raise PermissionDenied(f"Not permitted to subscribe to {name!r}")

        from django_ai_sdk.automations.models import AutomationSubscription

        await AutomationSubscription.objects.aupdate_or_create(
            name=name, user=user, defaults={"enabled": enabled}
        )
        state = await AutomationState.objects.filter(name=name).afirst()
        latest = await cls._latest_runs([name])
        return cls._describe(automation, state, latest.get(name), subscribed=enabled)

    @classmethod
    async def run_now(
        cls, name: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[AutomationRun] | None:
        """Dispatch an automation immediately, or None if the name is unknown.

        `user` is who asked, not who it runs as; the audience decides that.
        """
        automation = get_automation(name)
        if automation is None:
            return None
        if not await automation.has_perms(user, Operation.RUN_AUTOMATION):
            raise PermissionDenied(f"Not permitted to run {name!r}")

        from django_ai_sdk.automations.runner import run_now as _run_now

        return await _run_now(name)

    @classmethod
    async def list_runs(
        cls,
        name: str,
        *,
        user: AbstractBaseUser | AnonymousUser | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AutomationRunOut]:
        """This automation's run history, newest first, scoped to what `user` may read.

        A run's `output` is one person's content, so only a manager sees every run.
        """
        await cls._check_may_view(name, user)
        qs = AutomationRun.objects.filter(name=name)
        if not await cls._may_read_every_run(name, user):
            # No principal owns nothing, so an app-level run is not "theirs" either.
            owner = _pk(user)
            qs = qs.filter(user_id=owner) if owner is not None else qs.none()
        return [cls._describe_run(run) async for run in qs[offset : offset + limit]]

    @classmethod
    async def get_run(
        cls, run_id: str, *, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> AutomationRunOut | None:
        """One run by id, or None, permission-checked against its automation.

        Someone else's run reads as absent: a 403 would confirm the id exists.
        """
        run = await AutomationRun.objects.filter(id=run_id).afirst()
        if run is None:
            return None
        await cls._check_may_view(run.name, user)
        owner = _pk(user)
        mine = owner is not None and run.user_id == owner
        if not mine and not await cls._may_read_every_run(run.name, user):
            return None
        return cls._describe_run(run)

    @classmethod
    async def last_tick_at(cls) -> datetime | None:
        """When any automation was last dispatched, across the whole deployment.

        A dead-man's switch: the scheduler cannot notice its own absence.
        """
        result = await AutomationState.objects.aaggregate(last=Max("last_dispatched_at"))
        return result["last"]

    @classmethod
    async def _ensure_state(cls, automation: Automation) -> AutomationState:
        """This automation's state row, created if the first tick has not run yet.

        An unusable schedule has no next occurrence to store, but the row still has to
        exist for `enabled` to live on; _describe reports the reason as `detail`.
        """
        from django.utils import timezone

        from django_ai_sdk.automations.runner import ensure_state

        now = timezone.now()
        try:
            state, _ = await ensure_state(automation, now=now)
        except ImproperlyConfigured:
            logger.warning("Automation %r has no usable schedule", automation.name)
            state, _ = await AutomationState.objects.aget_or_create(
                name=automation.name, defaults={"next_run_at": now}
            )
        return state

    @classmethod
    async def _may_read_every_run(
        cls, name: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> bool:
        """Whether `user` may read other principals' runs of this automation.

        Tied to MANAGE: VIEW only puts the automation on a settings page.
        """
        automation = get_automation(name)
        if automation is None:
            return await cls.has_perms(user, Operation.MANAGE_AUTOMATION, raise_on_deny=False)
        return await automation.has_perms(user, Operation.MANAGE_AUTOMATION)

    @classmethod
    async def _check_may_view(
        cls, name: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> None:
        """Raise PermissionDenied unless `user` may read this automation's history.

        An unresolvable name falls back to the domain default rather than skipping.
        """
        automation = get_automation(name)
        if automation is not None:
            if not await automation.has_perms(user, Operation.VIEW_AUTOMATION):
                raise PermissionDenied(f"Not permitted to view {name!r}")
            return
        await cls.has_perms(user, Operation.VIEW_AUTOMATION, raise_on_deny=True)

    # --- Presentation ---

    @classmethod
    def _describe(
        cls,
        automation: Automation,
        state: AutomationState | None,
        last_run: AutomationRun | None,
        *,
        subscribed: bool,
    ) -> AutomationOut:
        enabled, source = is_enabled(
            automation.name,
            code_default=automation.enabled,
            db_value=state.enabled if state else None,
        )
        schedule_repr, detail = cls._schedule_repr(automation, state)
        return AutomationOut(
            name=automation.name,
            label=automation.label or automation.name,
            description=automation.description,
            schedule=schedule_repr,
            next_run_at=state.next_run_at if state else None,
            enabled=enabled,
            enabled_source=source,
            workflow=automation.workflow,
            requires=list(automation.requires),
            audience=automation.audience.describe(),
            subscribed=subscribed,
            last_run=cls._describe_run(last_run) if last_run else None,
            detail=detail,
        )

    @staticmethod
    async def _subscribed_names(
        names: list[str], user: AbstractBaseUser | AnonymousUser | None
    ) -> set[str]:
        """Which of `names` the requesting user has personally enabled."""
        if _pk(user) is None:
            return set()

        from django_ai_sdk.automations.models import AutomationSubscription

        return {
            row.name
            async for row in AutomationSubscription.objects.filter(
                name__in=names, user=user, enabled=True
            )
        }

    @staticmethod
    def _schedule_repr(
        automation: Automation, state: AutomationState | None
    ) -> tuple[str, str | None]:
        """The schedule as a string, plus the reason it is unusable if it is."""
        try:
            return str(automation.get_schedule()), None
        except Exception as exc:
            return (state.schedule_repr if state else "invalid"), str(exc)

    @staticmethod
    def _describe_run(run: AutomationRun) -> AutomationRunOut:
        return AutomationRunOut(
            id=str(run.id),
            status=run.status,
            trigger=run.trigger,
            scheduled_for=run.scheduled_for,
            started_at=run.started_at,
            finished_at=run.finished_at,
            output=run.output,
            skip_reason=run.skip_reason,
            error=run.error,
            workflow_run_id=str(run.workflow_run_id) if run.workflow_run_id else None,
            dispatch_id=str(run.dispatch_id),
        )

    @staticmethod
    async def _latest_runs(names: list[str]) -> dict[str, AutomationRun]:
        """The most recent run per automation, in one query.

        The subquery picks each name's first row under the model's newest-first ordering.
        """
        newest = Subquery(AutomationRun.objects.filter(name=OuterRef("name")).values("id")[:1])
        return {
            run.name: run async for run in AutomationRun.objects.filter(name__in=names, id=newest)
        }


__all__ = ["AutomationService"]
