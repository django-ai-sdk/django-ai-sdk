"""Policy authoring routes: define, try and withdraw policies at runtime.

`get_oversight_router()` adds them when the authoring app is installed; the oversight
manual's Review API lists the permission each operation takes.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from django_ai_sdk.contrib.ninja.routing import ApiRouter, found
from django_ai_sdk.oversight.authoring.schemas import (
    CatalogOut,
    PolicyIn,
    PolicyOut,
    TryIn,
    TryOut,
    WithdrawIn,
)
from django_ai_sdk.oversight.authoring.services import AuthoringService

routes = ApiRouter()


@routes.get("/oversight/authoring/catalog/", response=CatalogOut)
async def get_oversight_authoring_catalog(request: HttpRequest) -> Any:
    return await AuthoringService.catalog(user=request.user)


@routes.get("/oversight/authoring/policies/", response=list[PolicyOut])
async def list_oversight_authored_policies(request: HttpRequest) -> Any:
    return await AuthoringService.list_policies(user=request.user)


@routes.get("/oversight/authoring/policies/{name}/", response=PolicyOut)
async def get_oversight_authored_policy(request: HttpRequest, name: str) -> Any:
    return found(await AuthoringService.get_policy(name, user=request.user), "Policy")


# 409 when someone else saved the same policy at the same time.
@routes.put("/oversight/authoring/policies/{name}/", response=PolicyOut)
async def save_oversight_authored_policy(request: HttpRequest, name: str, payload: PolicyIn) -> Any:
    return await AuthoringService.save_policy(name, payload, user=request.user)


@routes.post("/oversight/authoring/policies/{name}/withdraw/", response=PolicyOut)
async def withdraw_oversight_authored_policy(
    request: HttpRequest, name: str, payload: WithdrawIn
) -> Any:
    return found(await AuthoringService.withdraw_policy(name, payload, user=request.user), "Policy")


@routes.post("/oversight/authoring/policies/{name}/try/", response=TryOut)
async def try_oversight_authored_policy(request: HttpRequest, name: str, payload: TryIn) -> Any:
    return await AuthoringService.try_policy(name, payload, user=request.user)
