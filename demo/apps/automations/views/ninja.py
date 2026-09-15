"""Reference HTTP surface for automations, on django-ninja.

The SDK ships no router, so this is the copyable example of one over AutomationService.
Permission failures are answered as 403 here rather than by a global handler.
"""

from __future__ import annotations

from datetime import datetime

from django.http import HttpRequest
from django_ai_sdk.automations import AutomationOut, AutomationRunOut, AutomationService
from django_ai_sdk.automations.runner import AutomationBusy
from django_ai_sdk.permissions import PermissionDenied
from ninja import Router, Schema

from apps.agents.views.ninja import Error


class EnabledIn(Schema):
    enabled: bool


class DispatchedOut(Schema):
    """What a manual run produced: a count, since a fan-out can create hundreds of rows."""

    dispatched: int
    detail: str


router = Router()


@router.get("/", response={200: list[AutomationOut]})
async def list_automations(request: HttpRequest) -> list[AutomationOut]:
    """Every automation this user may see, with live schedule and last-run state."""
    return await AutomationService.list_for_user(request.user)


@router.patch("/{name}", response={200: AutomationOut, 403: Error, 404: Error})
async def set_enabled(
    request: HttpRequest, name: str, payload: EnabledIn
) -> AutomationOut | tuple[int, Error]:
    """Turn an automation on or off.

    Writes the database layer, which outranks everything but the global kill switch.
    """
    try:
        result = await AutomationService.set_enabled(
            name, enabled=payload.enabled, user=request.user
        )
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    if result is None:
        return 404, Error(message="Unknown automation")
    return result


@router.patch("/{name}/subscription", response={200: AutomationOut, 403: Error, 404: Error})
async def set_subscribed(
    request: HttpRequest, name: str, payload: EnabledIn
) -> AutomationOut | tuple[int, Error]:
    """Turn this automation on or off for the requesting user, and only for them."""
    try:
        result = await AutomationService.set_subscribed(
            name, enabled=payload.enabled, user=request.user
        )
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    if result is None:
        return 404, Error(message="Unknown automation")
    return result


@router.post("/{name}/run", response={200: DispatchedOut, 403: Error, 404: Error, 409: Error})
async def run_now(request: HttpRequest, name: str) -> DispatchedOut | tuple[int, Error]:
    """Dispatch immediately, outside the schedule. 409 when the lease is already held."""
    try:
        runs = await AutomationService.run_now(name, user=request.user)
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    except AutomationBusy as exc:
        return 409, Error(message=str(exc))
    if runs is None:
        return 404, Error(message="Unknown automation")
    return DispatchedOut(dispatched=len(runs), detail=f"Dispatched {len(runs)} run(s)")


@router.get("/{name}/runs", response={200: list[AutomationRunOut], 403: Error})
async def list_runs(
    request: HttpRequest, name: str, limit: int = 50, offset: int = 0
) -> list[AutomationRunOut] | tuple[int, Error]:
    """Run history, newest first, including skipped runs."""
    try:
        return await AutomationService.list_runs(
            name, user=request.user, limit=limit, offset=offset
        )
    except PermissionDenied:
        return 403, Error(message="Not permitted")


@router.get("/{name}/runs/{run_id}", response={200: AutomationRunOut, 403: Error, 404: Error})
async def get_run(
    request: HttpRequest, name: str, run_id: str
) -> AutomationRunOut | tuple[int, Error]:
    try:
        run = await AutomationService.get_run(run_id, user=request.user)
    except PermissionDenied:
        return 403, Error(message="Not permitted")
    if run is None:
        return 404, Error(message="Unknown run")
    return run


class HealthOut(Schema):
    """Whether the scheduler is still ticking. Point a dead-man's switch at this."""

    last_tick_at: datetime | None


@router.get("/health/tick", response={200: HealthOut})
async def scheduler_health(request: HttpRequest) -> HealthOut:
    """When any automation was last dispatched, across the deployment."""
    return HealthOut(last_tick_at=await AutomationService.last_tick_at())
