"""Workflows: stored definitions, ad-hoc and stored runs, run history."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from django.http import HttpRequest
from ninja import Router, Schema

from django_ai_sdk.contrib.ninja.routing import ERRORS, Routes
from django_ai_sdk.errors import NotFound
from django_ai_sdk.views.schemas import (
    WorkflowCreateRequest,
    WorkflowRunByIdRequest,
    WorkflowRunRequest,
    WorkflowUpdateRequest,
)
from django_ai_sdk.workflows.services import WorkflowService

routes = Routes()


class WorkflowRunResponse(Schema):
    run_id: str
    status: str


class WorkflowActionItem(Schema):
    key: str
    description: str


class WorkflowRunStepOut(Schema):
    id: str
    sequence: int
    # The step's name is its key: its output is filed under it.
    step_name: str
    output: dict | None = None
    status: str
    error: str
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunOut(Schema):
    id: str
    workflow_id: str | None = None
    status: str
    outputs: dict | None = None
    error: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunDetailOut(WorkflowRunOut):
    steps: list[WorkflowRunStepOut] = []


class WorkflowItem(Schema):
    id: str
    name: str
    definition: dict
    active: bool


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _item(record: Any) -> WorkflowItem:
    return WorkflowItem(
        id=str(record.id), name=record.name, definition=record.definition, active=record.active
    )


def _run_out(run: Any, **extra: Any) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "workflow_id": str(run.workflow_id) if run.workflow_id else None,
        "status": run.status,
        "outputs": run.outputs,
        "error": run.error,
        "created_at": run.created_at.isoformat(),
        "started_at": _iso(run.started_at),
        "completed_at": _iso(run.completed_at),
        **extra,
    }


@routes.post("/workflows/run/", response={202: WorkflowRunResponse, **ERRORS})
async def run_workflow(request: HttpRequest, payload: WorkflowRunRequest) -> Any:
    run = await WorkflowService.run(payload.workflow, inputs=payload.inputs, user=request.user)
    return 202, WorkflowRunResponse(run_id=str(run.id), status=run.status)


@routes.get("/workflows/actions/", response=list[WorkflowActionItem])
def list_workflow_actions(request: HttpRequest) -> Any:
    return [WorkflowActionItem(**item) for item in WorkflowService.list_actions()]


@routes.get("/workflows/", response=list[WorkflowItem])
async def list_workflows(request: HttpRequest, limit: int = 100, offset: int = 0) -> Any:
    records = await WorkflowService.list_workflows(user=request.user, limit=limit, offset=offset)
    return [_item(r) for r in records]


@routes.post("/workflows/", response={201: WorkflowItem, **ERRORS})
async def create_workflow(request: HttpRequest, payload: WorkflowCreateRequest) -> Any:
    record = await WorkflowService.create(payload.name, payload.workflow, user=request.user)
    return 201, _item(record)


@routes.get("/workflows/{workflow_id}/runs/", response=list[WorkflowRunOut])
async def list_workflow_runs(
    request: HttpRequest, workflow_id: str, limit: int = 50, offset: int = 0
) -> Any:
    runs = await WorkflowService.list_runs(
        workflow_id, user=request.user, limit=limit, offset=offset
    )
    return [WorkflowRunOut(**_run_out(r)) for r in runs]


@routes.get("/workflows/{workflow_id}/runs/{run_id}/", response=WorkflowRunDetailOut)
async def get_workflow_run(request: HttpRequest, workflow_id: str, run_id: str) -> Any:
    run = await WorkflowService.get_run(run_id, user=request.user)
    if run is None:
        # Absent covers both "no such run" and "not yours", by design.
        raise NotFound("Run not found")
    steps = [
        WorkflowRunStepOut(
            id=str(s.id),
            sequence=s.sequence,
            step_name=s.step_name,
            output=s.output if isinstance(s.output, dict) else None,
            status=s.status,
            error=s.error,
            started_at=_iso(s.started_at),
            completed_at=_iso(s.completed_at),
        )
        async for s in run.steps.all()
    ]
    return WorkflowRunDetailOut(**_run_out(run, steps=steps))


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


@routes.delete("/workflows/{workflow_id}/", response={204: None, **ERRORS})
async def delete_workflow(request: HttpRequest, workflow_id: str) -> Any:
    await WorkflowService.delete(workflow_id, user=request.user)
    return 204, None


@routes.post("/workflows/{workflow_id}/run/", response={202: WorkflowRunResponse, **ERRORS})
async def run_workflow_by_id(
    request: HttpRequest, workflow_id: str, payload: WorkflowRunByIdRequest
) -> Any:
    run = await WorkflowService.run_by_id(
        workflow_id, inputs=payload.inputs, user=request.user, run_id=payload.run_id
    )
    return 202, WorkflowRunResponse(run_id=str(run.id), status=run.status)


def get_workflows_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the workflow endpoints; mount with ``api.add_router("/", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
