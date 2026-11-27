from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from django.http import HttpRequest
from django.utils import timezone
from django_ai_sdk.agents.config import get_runtime_agent_class
from django_ai_sdk.agents.models import AgentSettings
from django_ai_sdk.agents.registry import registry
from django_ai_sdk.errors import NotFound
from django_ai_sdk.oversight import Maybe, Stage
from django_ai_sdk.oversight.agents import agent_key, declared_policy
from django_ai_sdk.oversight.models import LensAnswer, RuleMatch, Turn
from django_ai_sdk.oversight.permissions import METRICS, OversightAccess
from django_ai_sdk.oversight.registry import get_policies, get_resolved_policy, run_policy_loaders
from django_ai_sdk.views.schemas import ErrorResponse
from ninja import Router, Schema
from ninja.security import SessionAuth

router = Router()

MAX_ACTIVITY_DAYS = 90


def _jsonable(value: Any) -> Any:
    """Lazy translations and enums as plain strings."""
    return json.loads(json.dumps(value, default=str))


class PolicySummarySchema(Schema):
    name: str
    label: str
    description: str


class LensSchema(Schema):
    name: str
    label: str
    question: str
    kind: str
    answerers: list[str]
    detail: dict[str, Any]


class RuleSchema(Schema):
    name: str
    description: str
    mode: str
    code_mode: str
    actions: list[str]
    lenses: list[str]
    notice: str
    when: dict[str, Any]
    then: list[dict[str, Any]]
    message: str


class StageSchema(Schema):
    stage: str
    lenses: list[LensSchema]
    rules: list[RuleSchema]


class PolicySchema(PolicySummarySchema):
    stages: list[StageSchema]


class AgentSchema(Schema):
    id: str
    name: str
    policy: str
    code_policy: str
    hidden: bool


class MessageTurnSchema(Schema):
    turn_id: str


class ActivitySchema(Schema):
    since: str
    turns: int
    stages: dict[str, dict[str, int]]
    lenses: dict[str, dict[str, int]]
    rules: dict[str, dict[str, int]]


def _agents() -> list[Any]:
    """Every registered agent, then every active runtime agent."""
    configs = AgentSettings.objects.filter(active=True).order_by("name")
    return [
        *registry.all().values(),
        *(get_runtime_agent_class(config.agent)(config) for config in configs),
    ]


async def _agent_policy(agent: Any) -> str:
    """The policy the agent's turns run under; "" when unguarded or its hook fails."""
    try:
        name, _reason = await declared_policy(agent)
    except Exception:
        return ""
    return name


@router.get(
    "/agents/",
    response={200: list[AgentSchema], 403: ErrorResponse},
    auth=SessionAuth(),
    operation_id="list_oversight_agents",
)
async def list_oversight_agents(request: HttpRequest) -> Any:
    """Each agent with the policy its turns run under."""
    await OversightAccess.has_perms(request.user, METRICS)
    out = []
    for agent in await sync_to_async(_agents)():
        policy = await _agent_policy(agent)
        out.append(
            AgentSchema(
                id=agent_key(agent),
                name=str(getattr(agent, "name", "") or type(agent).__name__),
                policy=policy,
                code_policy=policy,
                hidden=bool(getattr(agent, "hidden", False)),
            )
        )
    return out


@router.get(
    "/messages/{message_id}/turn/",
    response={200: MessageTurnSchema, 403: ErrorResponse, 404: ErrorResponse},
    auth=SessionAuth(),
    operation_id="get_oversight_message_turn",
)
async def get_oversight_message_turn(request: HttpRequest, message_id: str) -> Any:
    """The turn that produced a stored answer, so the chat can open its record."""
    await OversightAccess.has_perms(request.user, METRICS)
    turns = await OversightAccess.scoped(
        request.user, METRICS, Turn.objects.filter(message_id=message_id)
    )
    turn_id = await turns.order_by("-created_at").values_list("id", flat=True).afirst()
    if turn_id is None:
        raise NotFound("No oversight record for this answer")
    return MessageTurnSchema(turn_id=str(turn_id))


def _judge_name(spec: Any) -> str:
    """A judge's name: an alias, an optional alias ("judge?"), or the judge's own."""
    if isinstance(spec, Maybe):
        return f"{spec.alias}?"
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("class", ""))
    return str(getattr(spec, "display_name", "") or type(spec).__name__)


def _lens(lens: Any) -> LensSchema:
    judges = getattr(lens, "judges", None)
    steps = judges if judges is not None else lens.steps()
    return LensSchema(
        name=lens.name,
        label=str(lens.display_name),
        question=str(getattr(lens, "question", "")),
        kind=lens.answer.kind,
        answerers=[_judge_name(spec) for spec in steps],
        detail=_jsonable(lens.describe()),
    )


def _rule(rule: Any) -> RuleSchema:
    return RuleSchema(
        name=rule.name,
        description=str(rule.description),
        mode=str(rule.mode),
        code_mode=str(rule.mode),
        actions=[action.kind for action in rule.then],
        lenses=sorted(rule.lenses()),
        notice=str(rule.notice),
        when=_jsonable(rule.when.to_dict()),
        then=_jsonable([action.to_dict() for action in rule.then]),
        message=str(rule.message or ""),
    )


@router.get(
    "/policies/",
    response={200: list[PolicySummarySchema], 403: ErrorResponse},
    auth=SessionAuth(),
    operation_id="list_oversight_policies",
)
async def list_oversight_policies(request: HttpRequest) -> Any:
    """The policies agents can run under; the SDK's packs are left out."""
    await OversightAccess.has_perms(request.user, METRICS)
    await run_policy_loaders()
    out = []
    for name in sorted(get_policies()):
        policy = await get_resolved_policy(name)
        if policy is None or policy.origin.startswith("pack:"):
            continue
        out.append(
            PolicySummarySchema(
                name=name, label=str(policy.label or name), description=str(policy.description)
            )
        )
    return out


@router.get(
    "/policies/{name}/",
    response={200: PolicySchema, 403: ErrorResponse, 404: ErrorResponse},
    auth=SessionAuth(),
    operation_id="get_oversight_policy",
)
async def get_oversight_policy(request: HttpRequest, name: str) -> Any:
    """One policy, stage by stage: the lenses that run there and the rules on them."""
    await OversightAccess.has_perms(request.user, METRICS)
    policy = await get_resolved_policy(name)
    if policy is None:
        raise NotFound("Policy not found")
    stages = []
    for stage in Stage:
        lenses = policy.active_lenses(stage)
        here = {lens.name for lens in lenses}
        rules = [rule for rule in policy.rules.values() if rule.lenses() & here]
        stages.append(
            StageSchema(
                stage=str(stage),
                lenses=[_lens(lens) for lens in lenses],
                rules=[_rule(rule) for rule in rules],
            )
        )
    return PolicySchema(
        name=name,
        label=str(policy.label or name),
        description=str(policy.description),
        stages=stages,
    )


def _found(value: Any) -> bool:
    """A lens found something when it says yes, names something or scores low."""
    if value is True:
        return True
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value < 0.5
    return False


@router.get(
    "/policies/{name}/activity/",
    response={200: ActivitySchema, 403: ErrorResponse},
    auth=SessionAuth(),
    operation_id="get_oversight_policy_activity",
)
async def get_oversight_policy_activity(request: HttpRequest, name: str, days: int = 7) -> Any:
    """What a policy did over the last days: per stage, per lens and per rule."""
    await OversightAccess.has_perms(request.user, METRICS)
    since = timezone.now() - timedelta(days=max(1, min(days, MAX_ACTIVITY_DAYS)))

    def collect() -> ActivitySchema:
        turns = OversightAccess.has_queryset_perms(
            request.user, METRICS, queryset=Turn.objects.filter(policy=name, created_at__gte=since)
        )
        answers = LensAnswer.objects.filter(turn__in=turns)
        stages = {
            row["stage"]: {"answers": row["answers"], "turns": row["turns"]}
            for row in answers.values("stage").annotate(
                answers=Count("id"), turns=Count("turn", distinct=True)
            )
        }
        lenses: dict[str, dict[str, int]] = {}
        # ponytail: found is judged in Python over the window; aggregate in SQL if it grows
        for lens, value, error in answers.values_list("lens", "value", "error"):
            entry = lenses.setdefault(lens, {"answers": 0, "found": 0, "errors": 0})
            entry["answers"] += 1
            entry["errors"] += bool(error)
            entry["found"] += not error and _found(value)
        rules = {
            row["rule"]: {"matches": row["matches"], "acted": row["acted"]}
            for row in RuleMatch.objects.filter(turn__in=turns)
            .values("rule")
            .annotate(matches=Count("id"), acted=Count("id", filter=Q(acted=True)))
        }
        return ActivitySchema(
            since=since.isoformat(),
            turns=turns.count(),
            stages=stages,
            lenses=lenses,
            rules=rules,
        )

    return await sync_to_async(collect)()
