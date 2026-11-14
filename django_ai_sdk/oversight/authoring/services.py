"""Save, toggle, delete, dry-run and list authored policies.

Toggles change only what the policy itself declares; what it extends always runs.
Every write locks the row, so a save and a toggle never overwrite each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import IntegrityError, transaction

from django_ai_sdk.agents.registry import registry
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.agents import attribute_policy
from django_ai_sdk.oversight.authoring.apps import authoring_setting
from django_ai_sdk.oversight.authoring.definitions import (
    INLINE_ARGS,
    DefinitionError,
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
)
from django_ai_sdk.oversight.conf import setting as oversight_setting
from django_ai_sdk.oversight.judges.base import alias_table
from django_ai_sdk.oversight.lenses import ANSWER_TYPES, LENS_KINDS, get_validators
from django_ai_sdk.oversight.permissions import OversightAccess, reader_id
from django_ai_sdk.oversight.registry import get_policies, get_policy_class, run_policy_loaders
from django_ai_sdk.oversight.rules import ACTION_KINDS, META_OPS, OPS
from django_ai_sdk.oversight.trial import dry_run
from django_ai_sdk.oversight.types import Mode, Stage, Subject
from django_ai_sdk.permissions import ConflictError, Operation

if TYPE_CHECKING:
    from collections.abc import Callable

    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
    from django_ai_sdk.types import UserType

VIEW = Operation.VIEW_OVERSIGHT_AUTHORING
AUTHOR = Operation.AUTHOR_OVERSIGHT


class AuthoringService(OversightAccess):
    @classmethod
    async def catalog(cls, *, user: UserType) -> CatalogOut:
        await cls.has_perms(user, VIEW)
        await run_policy_loaders()
        return CatalogOut(
            lens_kinds=sorted(LENS_KINDS),
            answers=sorted(ANSWER_TYPES),
            actions=sorted(ACTION_KINDS),
            modes=[str(m) for m in Mode],
            operators=sorted({*OPS, *META_OPS}),
            stages=[str(s) for s in Stage],
            subjects=[str(s) for s in Subject],
            aliases=sorted(alias_table()),
            inline={name: sorted(args) for name, args in sorted(INLINE_ARGS.items())},
            validators=sorted(get_validators()),
            policies=[
                {"name": name, "label": policy.label or name, "origin": policy.origin}
                for name, policy in sorted(get_policies().items())
            ],
            must_extend=list(authoring_setting("MUST_EXTEND")),
        )

    @classmethod
    async def list_policies(cls, *, user: UserType) -> list[PolicyOut]:
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, VIEW)
        await run_policy_loaders()
        return [_policy_out(row) async for row in AuthoredPolicy.objects.order_by("name")]

    @classmethod
    async def get_policy(cls, name: str, *, user: UserType) -> PolicyOut | None:
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, VIEW)
        await run_policy_loaders()
        row = await AuthoredPolicy.objects.filter(name=name).afirst()
        return _policy_out(row) if row else None

    @classmethod
    async def save_policy(cls, name: str, change: PolicyIn, *, user: UserType) -> PolicyOut:
        """Create or replace `name`'s definition. UserError if invalid."""
        await cls.has_perms(user, AUTHOR)
        definition = change.model_dump()

        def replace(row: AuthoredPolicy) -> None:
            for key, value in definition.items():
                setattr(row, key, value)

        row = await sync_to_async(_save_definition)(name, replace, user, create=True)
        assert row is not None
        return _policy_out(row)

    @classmethod
    async def set_active(cls, name: str, active: bool, *, user: UserType) -> PolicyOut | None:
        return await _change_policy(name, user, lambda row: setattr(row, "active", active))

    @classmethod
    async def set_rule_mode(
        cls, name: str, rule: str, mode: Mode, *, user: UserType
    ) -> PolicyOut | None:
        """Only the policy's own rules; an inherited one raises UserError."""

        def change(row: AuthoredPolicy) -> None:
            row.rules = _set_values(row.rules, rule, "rule", {"mode": str(mode)})

        return await _change_policy(name, user, change)

    @classmethod
    async def set_lens_enabled(
        cls, name: str, lens: str, enabled: bool, *, user: UserType
    ) -> PolicyOut | None:
        def change(row: AuthoredPolicy) -> None:
            row.lenses = _set_values(row.lenses, lens, "lens", {"enabled": enabled})

        return await _change_policy(name, user, change)

    @classmethod
    async def delete_policy(cls, name: str, *, user: UserType) -> bool:
        """Refused while an agent or DEFAULT_POLICY names it: those turns would be blocked."""
        await cls.has_perms(user, AUTHOR)
        return await sync_to_async(_delete_unused_policy)(name)

    @classmethod
    async def try_policy(cls, name: str, run: TryIn, *, user: UserType) -> TryOut:
        """Dry-run a draft or the saved policy; nothing is recorded, but judges run for real."""
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, AUTHOR)
        await run_policy_loaders()
        if run.definition is not None:
            definition = run.definition.model_dump()
        else:
            saved = await AuthoredPolicy.objects.filter(name=name).afirst()
            if saved is None:
                raise UserError(f"No authored policy {name!r} to try: send a definition.")
            definition = saved.definition()
        try:
            _, resolved = await sync_to_async(check_definition)(name, definition)
        except DefinitionError as exc:
            raise UserError(f"This policy does not check: {exc}") from None
        result = await dry_run(
            resolved, input=run.input, output=run.output, tool_calls=run.tool_calls, user=user
        )
        return TryOut(
            outcome=str(result.outcome),
            report_outcome=str(result.report_outcome),
            ended_at=result.blocked_stage,
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


async def _change_policy(
    name: str, user: UserType, change: Callable[[AuthoredPolicy], None]
) -> PolicyOut | None:
    await AuthoringService.has_perms(user, AUTHOR)
    row = await sync_to_async(_save_definition)(name, change, user)
    return _policy_out(row) if row else None


def _save_definition(
    name: str,
    mutate: Callable[[AuthoredPolicy], None],
    user: UserType,
    *,
    create: bool = False,
) -> AuthoredPolicy | None:
    """Lock the row, apply `mutate`, check, save. None when missing and not `create`.

    A new name has no row to lock: when another request creates it first, ConflictError.
    """
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

    try:
        with transaction.atomic():
            row = AuthoredPolicy.objects.select_for_update().filter(name=name).first()
            if row is None and not create:
                return None
            row = row or AuthoredPolicy(name=name)
            mutate(row)
            # Read after the lock, so what other processes saved counts.
            loader.load_sync()
            try:
                check_definition(name, row.definition())
            except DefinitionError as exc:
                raise UserError(f"This policy cannot be saved: {exc}") from None
            row.updated_by_id = reader_id(user)
            row.save()
    except IntegrityError:
        raise ConflictError(f"Policy {name!r} was created at the same time. Try again.") from None
    loader.load_sync()
    return row


def _delete_unused_policy(name: str) -> bool:
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

    with transaction.atomic():
        row = AuthoredPolicy.objects.select_for_update().filter(name=name).first()
        if row is None:
            return False
        users = _dependents(name)
        if users:
            raise UserError(f"{name!r} is still used: {'; '.join(users)}.")
        row.delete()
    loader.load_sync()
    return True


def _set_values(items: list[dict[str, Any]], name: str, what: str, values: dict[str, Any]) -> list:
    if not any(item.get("name") == name for item in items):
        raise UserError(f"No {what} {name!r} of its own: a {what} it extends changes only in code.")
    return [{**item, **values} if item.get("name") == name else item for item in items]


def _dependents(name: str) -> list[str]:
    found = []
    if oversight_setting("DEFAULT_POLICY") == name:
        found.append("it is DEFAULT_POLICY")
    named = []
    for agent_id, agent in sorted(registry.all().items()):
        try:
            if attribute_policy(agent)[0] == name:
                named.append(agent_id)
        except Exception:
            continue  # An agent whose policy cannot be read here names none.
    if named:
        found.append(f"agents {named} name it in code")
    return found


def _policy_out(row: AuthoredPolicy) -> PolicyOut:
    return PolicyOut(
        name=row.name,
        **row.definition(),
        active=row.active,
        loaded=loader.loaded.get(row.name) == row.updated_at
        and is_authored(get_policy_class(row.name)),
        error=loader.broken.get(row.name, ""),
        updated_by=row.updated_by_id,
        updated_at=row.updated_at,
    )
