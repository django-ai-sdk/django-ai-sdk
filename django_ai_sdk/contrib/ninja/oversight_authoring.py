"""Policy authoring routes: define, toggle, try and delete policies at runtime.

`get_oversight_router()` adds them when the authoring app is installed; the oversight
manual's Review API lists the permission each operation takes.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from django_ai_sdk.contrib.ninja.routing import ApiRouter, found
from django_ai_sdk.oversight.authoring.schemas import (
    ActiveIn,
    CatalogOut,
    LensEnabledIn,
    PolicyIn,
    PolicyOut,
    RuleModeIn,
    TryIn,
    TryOut,
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


@routes.put("/oversight/authoring/policies/{name}/", response=PolicyOut)
async def save_oversight_authored_policy(request: HttpRequest, name: str, payload: PolicyIn) -> Any:
    return await AuthoringService.save_policy(name, payload, user=request.user)


@routes.patch("/oversight/authoring/policies/{name}/", response=PolicyOut)
async def toggle_oversight_authored_policy(
    request: HttpRequest, name: str, payload: ActiveIn
) -> Any:
    policy = await AuthoringService.set_active(name, payload.active, user=request.user)
    return found(policy, "Policy")


@routes.patch("/oversight/authoring/policies/{name}/rules/{rule}/", response=PolicyOut)
async def set_oversight_authored_rule_mode(
    request: HttpRequest, name: str, rule: str, payload: RuleModeIn
) -> Any:
    policy = await AuthoringService.set_rule_mode(name, rule, payload.mode, user=request.user)
    return found(policy, "Policy")


@routes.patch("/oversight/authoring/policies/{name}/lenses/{lens}/", response=PolicyOut)
async def set_oversight_authored_lens_enabled(
    request: HttpRequest, name: str, lens: str, payload: LensEnabledIn
) -> Any:
    policy = await AuthoringService.set_lens_enabled(name, lens, payload.enabled, user=request.user)
    return found(policy, "Policy")


@routes.delete("/oversight/authoring/policies/{name}/", response={204: None})
async def delete_oversight_authored_policy(request: HttpRequest, name: str) -> Any:
    found(await AuthoringService.delete_policy(name, user=request.user) or None, "Policy")
    return 204, None


@routes.post("/oversight/authoring/policies/{name}/try/", response=TryOut)
async def try_oversight_authored_policy(request: HttpRequest, name: str, payload: TryIn) -> Any:
    return await AuthoringService.try_policy(name, payload, user=request.user)
