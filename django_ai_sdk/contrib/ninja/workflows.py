"""Workflows: stored definitions, their runs and run history.

Over HTTP only registered or stored workflows run, by key. A definition sent in a
request is never run. `WorkflowService.run` is for trusted Python code only.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from django.http import HttpRequest
from ninja import Router

from django_ai_sdk.contrib.ninja.routing import ApiRouter, Limit, Offset
from django_ai_sdk.contrib.ninja.schemas import (
    StepRunOut,
    WorkflowDecisionIn,
    WorkflowItem,
    WorkflowListenerItem,
    WorkflowRunDetailOut,
    WorkflowRunOut,
    WorkflowRunResponse,
    WorkflowTriggerItem,
    WorkflowTriggerPauseIn,
)
from django_ai_sdk.errors import NotFound
from django_ai_sdk.utils import isoformat_or_none
from django_ai_sdk.workflows.schemas import (
    WorkflowCreateRequest,
    WorkflowStartRequest,
    WorkflowUpdateRequest,
)
from django_ai_sdk.workflows.services import WorkflowService

routes = ApiRouter()


def _item(record: Any) -> WorkflowItem:
    return WorkflowItem(
        id=str(record.id),
        name=record.name,
        slug=record.slug,
        definition=record.definition,
        active=record.active,
    )


def _run_out(run: Any, **extra: Any) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "workflow_id": str(run.workflow_id) if run.workflow_id else None,
        "workflow_key": run.workflow_key,
        "status": run.status,
        "attempt": run.attempt,
        "trigger": run.trigger,
        "outputs": run.outputs,
        "error": run.error,
        "created_at": run.created_at.isoformat(),
        "started_at": isoformat_or_none(run.started_at),
        "completed_at": isoformat_or_none(run.completed_at),
        **extra,
    }


@routes.get("/workflows/listeners/", response=list[WorkflowListenerItem])
async def list_workflow_listeners(request: HttpRequest) -> Any:
    listeners = await WorkflowService.list_listeners(user=request.user)
    return [WorkflowListenerItem(**item) for item in listeners]


@routes.get("/workflows/triggers/", response=list[WorkflowTriggerItem])
async def list_workflow_triggers(request: HttpRequest) -> Any:
    items: list[WorkflowTriggerItem] = []
    for item in await WorkflowService.list_triggers(user=request.user):
        last = item.pop("last_run")
        items.append(
            WorkflowTriggerItem(**item, last_run=WorkflowRunOut(**_run_out(last)) if last else None)
        )
    return items


@routes.post("/workflows/triggers/{key}/{trigger}/pause/", response={204: None})
async def pause_workflow_trigger(
    request: HttpRequest, key: str, trigger: str, payload: WorkflowTriggerPauseIn
) -> Any:
    await WorkflowService.set_trigger_paused(key, trigger, paused=payload.paused, user=request.user)
    return 204, None


@routes.post("/workflows/triggers/{key}/{trigger}/fire/", response={202: None})
async def fire_workflow_trigger(request: HttpRequest, key: str, trigger: str) -> Any:
    await WorkflowService.fire_trigger(key, trigger, user=request.user)
    return 202, None


@routes.get("/workflows/", response=list[WorkflowItem])
async def list_workflows(
    request: HttpRequest,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Any:
    records = await WorkflowService.list_workflows(user=request.user, limit=limit, offset=offset)
    return [_item(r) for r in records]


@routes.post("/workflows/", response={201: WorkflowItem})
async def create_workflow(request: HttpRequest, payload: WorkflowCreateRequest) -> Any:
    record = await WorkflowService.create(payload.name, payload.workflow, user=request.user)
    return 201, _item(record)


@routes.get("/workflows/{workflow_id}/runs/", response=list[WorkflowRunOut])
async def list_workflow_runs(
    request: HttpRequest,
    workflow_id: str,
    limit: Limit = 50,
    offset: Offset = 0,
) -> Any:
    runs = await WorkflowService.list_runs(
        workflow_id, user=request.user, limit=limit, offset=offset
    )
    return [WorkflowRunOut(**_run_out(r)) for r in runs]


async def _run_detail(request: HttpRequest, run_id: str, workflow_id: str | None) -> Any:
    run = await WorkflowService.get_run(run_id, user=request.user, workflow_id=workflow_id)
    if run is None:
        # Absent covers both "no such run" and "not yours", by design.
        raise NotFound("Run not found")
    steps = [
        StepRunOut(
            id=str(s.id),
            sequence=s.sequence,
            step_name=s.step_name,
            output=s.output,
            status=s.status,
            error=s.error,
            detail=s.detail,
            turn_id=s.turn_id,
            started_at=isoformat_or_none(s.started_at),
            completed_at=isoformat_or_none(s.completed_at),
        )
        async for s in run.steps.all()
    ]
    return WorkflowRunDetailOut(**_run_out(run, steps=steps))


@routes.get("/workflows/runs/{run_id}/", response=WorkflowRunDetailOut)
async def get_workflow_run_by_id(request: HttpRequest, run_id: str) -> Any:
    """Get a run by id, e.g. to follow a run started by key."""
    return await _run_detail(request, run_id, None)


@routes.get("/workflows/{workflow_id}/runs/{run_id}/", response=WorkflowRunDetailOut)
async def get_workflow_run(request: HttpRequest, workflow_id: str, run_id: str) -> Any:
    return await _run_detail(request, run_id, workflow_id)


@routes.get("/workflows/{workflow_id}/", response=WorkflowItem)
async def get_workflow(request: HttpRequest, workflow_id: str) -> Any:
    return _item(await WorkflowService.get(workflow_id, user=request.user))


@routes.patch("/workflows/{workflow_id}/", response=WorkflowItem)
async def update_workflow(
    request: HttpRequest, workflow_id: str, payload: WorkflowUpdateRequest
) -> Any:
    record = await WorkflowService.update(
        workflow_id,
        user=request.user,
        name=payload.name,
        workflow=payload.workflow,
        active=payload.active,
    )
    return _item(record)


@routes.delete("/workflows/{workflow_id}/", response={204: None})
async def delete_workflow(request: HttpRequest, workflow_id: str) -> Any:
    await WorkflowService.delete(workflow_id, user=request.user)
    return 204, None


@routes.post("/workflows/{key}/run/", response={202: WorkflowRunResponse})
async def start_workflow(request: HttpRequest, key: str, payload: WorkflowStartRequest) -> Any:
    run = await WorkflowService.start(key, inputs=payload.inputs, user=request.user)
    return 202, WorkflowRunResponse(run_id=str(run.id), status=run.status)


@routes.post("/workflows/runs/{run_id}/resume/", response={202: WorkflowRunResponse})
async def resume_workflow_run(request: HttpRequest, run_id: str) -> Any:
    run = await WorkflowService.resume(run_id, user=request.user)
    return 202, WorkflowRunResponse(run_id=str(run.id), status=run.status)


@routes.post("/workflows/runs/{run_id}/cancel/", response=WorkflowRunResponse)
async def cancel_workflow_run(request: HttpRequest, run_id: str) -> Any:
    run = await WorkflowService.cancel(run_id, user=request.user)
    return WorkflowRunResponse(run_id=str(run.id), status=run.status)


@routes.post("/workflows/runs/{run_id}/steps/{step}/decision/", response=WorkflowRunResponse)
async def decide_workflow_step(
    request: HttpRequest, run_id: str, step: str, payload: WorkflowDecisionIn
) -> Any:
    """Approve or reject an approval step the run waits on. Not on a run that acts as you."""
    run = await WorkflowService.decide(
        run_id, step, approved=payload.approved, note=payload.note, user=request.user
    )
    return WorkflowRunResponse(run_id=str(run.id), status=run.status)


def get_workflows_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the workflow endpoints; mount with ``api.add_router("/", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
