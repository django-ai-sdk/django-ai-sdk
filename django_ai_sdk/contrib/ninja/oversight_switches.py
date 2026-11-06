"""Runtime switch routes: set a rule's mode, set an agent's policy, list switches.

`get_oversight_router()` adds them when the switches app is installed; the oversight
manual's Review API lists the permission each operation takes.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from django_ai_sdk.contrib.ninja.routing import ApiRouter, found
from django_ai_sdk.oversight.switches.schemas import AgentPolicyIn, RuleModeIn, SwitchOut
from django_ai_sdk.oversight.switches.services import SwitchService

routes = ApiRouter()


@routes.get("/oversight/switches/", response=list[SwitchOut])
async def list_oversight_switches(request: HttpRequest) -> Any:
    return await SwitchService.list_switches(user=request.user)


# 409 when someone else switched the same rule at the same time.
@routes.post("/oversight/switches/rules/{policy}/{rule}/", response=SwitchOut)
async def set_oversight_rule_mode(
    request: HttpRequest, policy: str, rule: str, payload: RuleModeIn
) -> Any:
    switched = await SwitchService.set_rule_mode(policy, rule, payload, user=request.user)
    return found(switched, "Rule")


@routes.post("/oversight/switches/agents/{agent_id}/", response=SwitchOut)
async def set_oversight_agent_policy(
    request: HttpRequest, agent_id: str, payload: AgentPolicyIn
) -> Any:
    switched = await SwitchService.set_agent_policy(agent_id, payload, user=request.user)
    return found(switched, "Agent")
