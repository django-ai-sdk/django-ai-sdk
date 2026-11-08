"""Save, withdraw, dry-run and list authored policies.

A policy cannot be withdrawn while another authored policy extends it, a switch assigns
it to an agent, or it is DEFAULT_POLICY: turns that use it would be refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.apps import apps

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.authoring.conf import INLINE_ARGS, setting
from django_ai_sdk.oversight.authoring.definitions import (
    DefinitionError,
    allowed_inline,
    check_definition,
    is_authored,
)
from django_ai_sdk.oversight.authoring.loader import loader
from django_ai_sdk.oversight.authoring.schemas import (
    CatalogOut,
    PolicyIn,
    PolicyOut,
    StageOut,
    TryIn,
    TryOut,
    VersionOut,
    WithdrawIn,
)
from django_ai_sdk.oversight.resolve import aload_policies
from django_ai_sdk.oversight.services.access import (
    OversightAccess,
    append_change,
    require_reason,
)
from django_ai_sdk.permissions import ConflictError, Operation

if TYPE_CHECKING:
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
    from django_ai_sdk.types import UserType

# The access record's operation for a save, a withdrawal or a try.
OPERATION = "author"
VIEW = Operation.VIEW_OVERSIGHT_AUTHORING
AUTHOR = Operation.AUTHOR_OVERSIGHT


class AuthoringService(OversightAccess):
    """Manage policies that operators define at runtime."""

    @classmethod
    async def catalog(cls, *, user: UserType) -> CatalogOut:
        """List the building blocks a definition may use."""
        from django_ai_sdk.oversight.actions import ACTION_KINDS
        from django_ai_sdk.oversight.conditions import META_OPS, OPS
        from django_ai_sdk.oversight.judges.base import alias_table
        from django_ai_sdk.oversight.lenses import ANSWER_TYPES, LENS_KINDS
        from django_ai_sdk.oversight.registry import get_policies, get_validators
        from django_ai_sdk.oversight.types import Mode, Stage, Subject

        await cls.has_perms(user, VIEW)
        await aload_policies()
        return CatalogOut(
            lens_kinds=sorted(LENS_KINDS),
            answers=sorted(ANSWER_TYPES),
            actions=sorted(ACTION_KINDS),
            modes=[str(m) for m in Mode],
            operators=sorted({*OPS, *META_OPS}),
            stages=[str(s) for s in Stage],
            subjects=[str(s) for s in Subject],
            aliases=sorted(alias_table()),
            inline={name: sorted(INLINE_ARGS[name]) for name in sorted(allowed_inline())},
            validators=sorted(get_validators()),
            policies=[
                {"name": name, "label": policy.label or name, "origin": policy.origin}
                for name, policy in sorted(get_policies().items())
            ],
            must_extend=list(setting("MUST_EXTEND")),
        )

    @classmethod
    async def list_policies(cls, *, user: UserType) -> list[PolicyOut]:
        """Return the newest version of every authored policy, including withdrawn ones."""
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, VIEW)
        await aload_policies()
        return [_out(row) async for row in AuthoredPolicy.objects.newest("name").order_by("name")]

    @classmethod
    async def get_policy(cls, name: str, *, user: UserType) -> PolicyOut | None:
        """Return one authored policy with its version history, or None."""
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, VIEW)
        await aload_policies()
        rows = [row async for row in AuthoredPolicy.objects.history(name=name)]
        if not rows:
            return None
        return _out(rows[0], versions=rows)

    @classmethod
    async def save_policy(cls, name: str, change: PolicyIn, *, user: UserType) -> PolicyOut:
        """Save a new version of `name`, or create it. Raises UserError if it is invalid."""
        await cls.has_perms(user, AUTHOR)
        # Read now: the check must see parents other processes saved.
        await loader.load()
        definition = change.definition()
        try:
            await sync_to_async(check_definition)(name, definition)
        except DefinitionError as exc:
            raise UserError(f"This policy cannot be saved: {exc}") from None
        row = await _write(
            name, definition, change.reason, user, withdrawn=False, base=change.base_version
        )
        await loader.load()
        return _out(row)

    @classmethod
    async def withdraw_policy(
        cls, name: str, change: WithdrawIn, *, user: UserType
    ) -> PolicyOut | None:
        """Withdraw `name` so turns that use it are refused. None if it does not exist."""
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, AUTHOR)
        # Read now, so children and switches saved by other processes count.
        await loader.load()
        current = await AuthoredPolicy.objects.history(name=name).afirst()
        if current is None:
            return None
        if current.withdrawn:
            raise UserError(f"{name!r} is withdrawn already.")
        users = await sync_to_async(_still_used)(name)
        if users:
            raise UserError(f"{name!r} is still used: {'; '.join(users)}.")
        # What it withdraws is the version read above, so a save since is a conflict.
        base = current.version if change.base_version is None else change.base_version
        row = await _write(
            name, current.definition(), change.reason, user, withdrawn=True, base=base
        )
        await loader.load()
        return _out(row)

    @classmethod
    async def try_policy(cls, name: str, run: TryIn, *, user: UserType) -> TryOut:
        """Dry-run a draft, or the saved version, over one exchange. Nothing is recorded.

        Judges run for real, so any judge it names is called.
        """
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
        from django_ai_sdk.oversight.testing import dry_run

        await cls.has_perms(user, AUTHOR)
        await aload_policies()
        if run.definition is not None:
            definition = run.definition.definition()
        else:
            saved = await AuthoredPolicy.objects.history(name=name).afirst()
            if saved is None or saved.withdrawn:
                raise UserError(f"No authored policy {name!r} to try: send a definition.")
            definition = saved.definition()
        try:
            _, resolved = await sync_to_async(check_definition)(name, definition)
        except DefinitionError as exc:
            raise UserError(f"This policy does not check: {exc}") from None
        result = await dry_run(
            resolved, input=run.input, output=run.output, tool_calls=run.tool_calls, user=user
        )
        # A try sends operator text through real, paid judges, so it is on the record.
        await cls.record(user, OPERATION, query={"policy": name, "try": True})
        return TryOut(
            outcome=str(result.outcome),
            report_outcome=str(result.report_outcome),
            ended_at=result.ended_at,
            stages=[
                StageOut(
                    stage=str(stage.stage),
                    blocked=stage.blocked,
                    message=stage.message,
                    text=stage.text,
                    tool=stage.tool,
                    answers={k: v.model_dump(mode="json") for k, v in stage.answers.items()},
                    rules=stage.rules,
                )
                for stage in result.stages
            ],
        )


def _still_used(name: str) -> list[str]:
    """List what still depends on `name`: authored children, switches, agents that name
    it in code, and DEFAULT_POLICY."""
    from django_ai_sdk.oversight.conf import setting as oversight_setting
    from django_ai_sdk.oversight.registry import get_policies

    found = [
        f"policy {child!r} extends it"
        for child, cls in sorted(get_policies().items())
        if child != name and is_authored(cls) and name in (cls.extends or [])
    ]
    if oversight_setting("DEFAULT_POLICY") == name:
        found.append("it is DEFAULT_POLICY")
    from django_ai_sdk.agents.registry import registry
    from django_ai_sdk.oversight.agents import code_policy

    named = []
    for agent_id, agent in sorted(registry.all().items()):
        try:
            if code_policy(agent)[0] == name:
                named.append(agent_id)
        except Exception:
            continue  # An agent whose policy cannot be read here names none.
    if named:
        found.append(f"agents {named} name it in code")
    if apps.is_installed("django_ai_sdk.oversight.switches"):
        # Through the app registry: authoring does not import the switches app.
        override = apps.get_model("django_ai_sdk_oversight_switches", "Switch")
        switched = override.objects.filter(kind=override.Kind.AGENT_POLICY)
        newest = switched.newest("kind", "policy", "target").filter(value=name)
        agents = list(newest.values_list("target", flat=True))
        if agents:
            found.append(f"switched onto agents {agents}")
    return found


async def _write(
    name: str,
    definition: dict[str, Any],
    reason: str,
    user: UserType,
    *,
    withdrawn: bool,
    base: int | None = None,
) -> AuthoredPolicy:
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

    reason = require_reason(reason, "Say why.")

    def describe(before: AuthoredPolicy | None) -> tuple[dict[str, Any], dict[str, Any]]:
        if base is not None and (before.version if before else 0) != base:
            # Not retried: the caller edited a version that is no longer the newest.
            raise ConflictError(f"{name!r} was saved by someone else since version {base}.")
        change = {
            "policy": name,
            "version": before.version + 1 if before else 1,
            "before": before.version if before else None,
            "withdrawn": withdrawn,
            "reason": reason,
        }
        return change, {"type": "oversight.authoring", **change}

    # `definition` comes from PolicyIn.definition() or a row's, so it has every field.
    return await append_change(
        AuthoredPolicy,
        "authored",
        {"name": name},
        {**definition, "withdrawn": withdrawn, "reason": reason},
        user=user,
        operation=OPERATION,
        describe=describe,
        conflict="Saved by someone else at the same time; try again.",
    )


def _out(row: AuthoredPolicy, versions: list[AuthoredPolicy] | None = None) -> PolicyOut:
    from django_ai_sdk.oversight.registry import get_policy_class

    return PolicyOut(
        name=row.name,
        version=row.version,
        **row.definition(),
        withdrawn=row.withdrawn,
        loaded=loader.loaded.get(row.name) == row.version
        and is_authored(get_policy_class(row.name)),
        error="" if row.withdrawn else loader.broken.get(row.name, ""),
        reason=row.reason,
        changed_by=row.changed_by_id,
        changed_at=row.changed_at,
        versions=[
            VersionOut(
                version=v.version,
                withdrawn=v.withdrawn,
                reason=v.reason,
                changed_by=v.changed_by_id,
                changed_at=v.changed_at,
            )
            for v in versions or []
        ],
    )


__all__ = ["AuthoringService"]
