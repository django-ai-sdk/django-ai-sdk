"""The studio's own endpoints, next to the SDK's contrib routers (see demo/urls.py).

This is the extension example: anything the SDK doesn't ship (accounts, a
project-specific workflow trigger) lives in your own Router, mounted on the same API.
"""

from __future__ import annotations

from typing import Any

from django.contrib.auth import get_user_model
from django.db.models import Q
from django.http import HttpRequest
from django_ai_sdk.errors import NotFound
from django_ai_sdk.storage.services import ThreadService
from ninja import Router, Schema

router = Router()


class HealthResponse(Schema):
    status: str
    service: str


class DigestStepOut(Schema):
    name: str
    status: str
    detail: str


class DigestResponse(Schema):
    thread_id: str
    steps: list[DigestStepOut]


class UserSchema(Schema):
    id: Any
    first_name: str
    last_name: str


class UserDetailSchema(Schema):
    id: Any
    first_name: str
    last_name: str
    email: str


class UserUpdateSchema(Schema):
    first_name: str | None = None
    last_name: str | None = None


class GroupOut(Schema):
    id: int
    name: str


@router.get("/health/", response={200: HealthResponse}, operation_id="health_check")
def health_check(request: HttpRequest) -> HealthResponse:
    return HealthResponse(status="ok", service="django-ai-sdk")


@router.post("/threads/{thread_id}/digest/", response=DigestResponse, operation_id="digest_thread")
async def digest_thread(request: HttpRequest, thread_id: str) -> Any:
    """Run the declared thread-digest workflow (see apps.agents.workflows)."""
    from django_ai_sdk.workflows.executor import WorkflowExecutor
    from django_ai_sdk.workflows.registry import aget_workflow

    user = await request.auser()
    if await ThreadService.get_thread(thread_id, user=user) is None:
        raise NotFound("Thread not found")
    definition = await aget_workflow("thread-digest")
    if definition is None:
        raise NotFound("thread-digest workflow is not registered")

    # Inline, so the caller gets the outcome. WorkflowService.run() queues it.
    _outputs, run = await WorkflowExecutor().run(
        definition, inputs={"thread": thread_id}, user=user
    )
    return DigestResponse(
        thread_id=thread_id,
        steps=[
            DigestStepOut(name=row.step_name, status=row.status, detail=row.detail)
            async for row in run.steps.order_by("sequence")
        ],
    )


@router.get("/users/", response=list[UserSchema], operation_id="list_users")
def list_users(request: HttpRequest, q: str = "", limit: int = 10) -> Any:
    qs = get_user_model().objects.order_by("first_name", "last_name")
    if q.strip():
        qs = qs.filter(
            Q(first_name__icontains=q) | Q(last_name__icontains=q) | Q(email__icontains=q)
        )
    return list(qs.values("id", "first_name", "last_name")[: min(limit, 100)])


@router.get("/users/me/", response=UserDetailSchema, operation_id="get_me")
def get_me(request: HttpRequest) -> Any:
    return get_user_model().objects.get(pk=request.user.pk)


@router.patch("/users/me/", response=UserDetailSchema, operation_id="update_me")
def update_me(request: HttpRequest, payload: UserUpdateSchema) -> Any:
    user = get_user_model().objects.get(pk=request.user.pk)
    fields = payload.model_dump(exclude_none=True)
    for field, value in fields.items():
        setattr(user, field, value)
    if fields:
        user.save(update_fields=list(fields))
    return user


@router.get("/accounts/groups/", response=list[GroupOut], operation_id="search_groups")
def search_groups(request: HttpRequest, q: str = "", limit: int = 10) -> Any:
    from django.contrib.auth.models import Group

    qs = Group.objects.order_by("name")
    if q.strip():
        qs = qs.filter(name__icontains=q)
    return list(qs.values("id", "name")[: min(limit, 100)])
