"""Save, toggle, delete, dry-run and list authored policies.

A save replaces the policy's definition. Toggles change only what the policy itself
declares: its rules' modes, its lenses, or the whole policy. What it extends always runs.
A policy cannot be deleted while another authored policy extends it, an agent names it
in code, or it is DEFAULT_POLICY: turns that use it would be refused.

Every write locks the row, so a save and a toggle never overwrite each other, and
rechecks the authored policies that extend it, so a change never breaks one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.db import transaction

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.authoring.conf import INLINE_ARGS, setting
from django_ai_sdk.oversight.authoring.definitions import (
    DefinitionError,
    build_policy_class,
    check_definition,
    in_force,
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
from django_ai_sdk.oversight.permissions import OversightAccess, reader_id
from django_ai_sdk.oversight.registry import drafted, get_policy_class
from django_ai_sdk.oversight.resolve import aload_policies
from django_ai_sdk.oversight.types import Mode
from django_ai_sdk.permissions import Operation

if TYPE_CHECKING:
    from collections.abc import Callable

    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.types import UserType

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
        from django_ai_sdk.oversight.types import Stage, Subject

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
            inline={name: sorted(args) for name, args in sorted(INLINE_ARGS.items())},
            validators=sorted(get_validators()),
            policies=[
                {"name": name, "label": policy.label or name, "origin": policy.origin}
                for name, policy in sorted(get_policies().items())
            ],
            must_extend=list(setting("MUST_EXTEND")),
        )

    @classmethod
    async def list_policies(cls, *, user: UserType) -> list[PolicyOut]:
        """Return every authored policy, on or off."""
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, VIEW)
        await aload_policies()
        return [_out(row) async for row in AuthoredPolicy.objects.order_by("name")]

    @classmethod
    async def get_policy(cls, name: str, *, user: UserType) -> PolicyOut | None:
        """Return one authored policy, or None."""
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

        await cls.has_perms(user, VIEW)
        await aload_policies()
        row = await AuthoredPolicy.objects.filter(name=name).afirst()
        return _out(row) if row else None

    @classmethod
    async def save_policy(cls, name: str, change: PolicyIn, *, user: UserType) -> PolicyOut:
        """Create `name`, or replace its definition. Raises UserError if it is invalid."""
        await cls.has_perms(user, AUTHOR)
        definition = change.definition()

        def replace(row: AuthoredPolicy) -> None:
            for key, value in definition.items():
                setattr(row, key, value)

        row = await sync_to_async(_write)(name, replace, user, create=True)
        assert row is not None
        return _out(row)

    @classmethod
    async def set_active(cls, name: str, active: bool, *, user: UserType) -> PolicyOut | None:
        """Turn the policy's own lenses and rules on or off. None if it does not exist."""
        return await _toggle(name, user, lambda row: setattr(row, "active", active))

    @classmethod
    async def set_rule_mode(
        cls, name: str, rule: str, mode: Mode, *, user: UserType
    ) -> PolicyOut | None:
        """Set the mode of one of the policy's own rules. None if the policy does not exist.

        A rule it inherits cannot be changed: UserError.
        """

        def change(row: AuthoredPolicy) -> None:
            row.rules = _with(row.rules, rule, "rule", {"mode": str(mode)})

        return await _toggle(name, user, change)

    @classmethod
    async def set_lens_enabled(
        cls, name: str, lens: str, enabled: bool, *, user: UserType
    ) -> PolicyOut | None:
        """Turn one of the policy's own lenses on or off. None if the policy does not exist."""

        def change(row: AuthoredPolicy) -> None:
            row.lenses = _with(row.lenses, lens, "lens", {"enabled": enabled})

        return await _toggle(name, user, change)

    @classmethod
    async def delete_policy(cls, name: str, *, user: UserType) -> bool:
        """Delete `name`, so turns that use it are refused. False if it does not exist."""
        await cls.has_perms(user, AUTHOR)
        return await sync_to_async(_delete)(name)

    @classmethod
    async def try_policy(cls, name: str, run: TryIn, *, user: UserType) -> TryOut:
        """Dry-run a draft, or the saved policy, over one exchange. Nothing is recorded.

        Judges run for real, so any judge it names is called.
        """
        from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
        from django_ai_sdk.oversight.testing import dry_run

        await cls.has_perms(user, AUTHOR)
        await aload_policies()
        if run.definition is not None:
            definition = run.definition.definition()
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


async def _toggle(
    name: str, user: UserType, change: Callable[[AuthoredPolicy], None]
) -> PolicyOut | None:
    await AuthoringService.has_perms(user, AUTHOR)
    row = await sync_to_async(_write)(name, change, user)
    return _out(row) if row else None


def _write(
    name: str,
    mutate: Callable[[AuthoredPolicy], None],
    user: UserType,
    *,
    create: bool = False,
) -> AuthoredPolicy | None:
    """Lock the row, apply `mutate`, check the result and the policies that extend it, save.

    None when the row does not exist and `create` is False. Raises UserError instead of
    saving a change that does not check. Its authored parents are locked too, so a
    concurrent delete of one either waits for this save or makes this check fail.
    """
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

    with transaction.atomic():
        row = AuthoredPolicy.objects.select_for_update().filter(name=name).first()
        if row is None and not create:
            return None
        row = row or AuthoredPolicy(name=name)
        mutate(row)
        list(AuthoredPolicy.objects.select_for_update().filter(name__in=row.extends or []))
        # Read after the locks, so what other processes saved counts.
        loader._load()
        definition = row.definition()
        try:
            check_definition(name, definition)
            draft = build_policy_class(name, in_force(definition, active=row.active))
        except DefinitionError as exc:
            raise UserError(f"This policy cannot be saved: {exc}") from None
        broken = _broken_children(name, draft)
        if broken:
            raise UserError(f"This change breaks what extends it: {'; '.join(broken)}")
        row.updated_by_id = reader_id(user)
        row.save()
    loader._load()
    return row


def _broken_children(name: str, draft: type[Policy]) -> list[str]:
    """Recheck every authored policy below `name` against `draft`, and list what fails.

    Breadth first: each child that checks stands in for itself while its own children
    are checked, so a grandchild is checked against the draft through its parent. One
    that was broken already is skipped, so it never blocks an unrelated change.
    """
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

    rows = list(AuthoredPolicy.objects.exclude(name=name))
    drafts: dict[str, type[Policy]] = {name: draft}
    checked = {name, *loader.broken}
    problems: list[str] = []
    todo = [name]
    while todo:
        parent = todo.pop(0)
        for row in rows:
            if parent not in (row.extends or []) or row.name in checked:
                continue
            checked.add(row.name)
            try:
                with drafted(drafts):
                    drafts[row.name], _ = check_definition(
                        row.name, in_force(row.definition(), active=row.active)
                    )
            except DefinitionError as exc:
                problems.append(f"{row.name!r} would not load: {exc}")
                continue
            todo.append(row.name)
    return problems


def _delete(name: str) -> bool:
    """Delete `name` unless something still uses it. False if it does not exist."""
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy

    with transaction.atomic():
        row = AuthoredPolicy.objects.select_for_update().filter(name=name).first()
        if row is None:
            return False
        users = _still_used(name)
        if users:
            raise UserError(f"{name!r} is still used: {'; '.join(users)}.")
        row.delete()
    loader._load()
    return True


def _with(items: list[dict[str, Any]], name: str, what: str, values: dict[str, Any]) -> list:
    """`items` with `values` set on the one named `name`. UserError when it is not there."""
    if not any(item.get("name") == name for item in items or []):
        raise UserError(f"No {what} {name!r} of its own: a {what} it extends changes only in code.")
    return [{**item, **values} if item.get("name") == name else item for item in items]


def _still_used(name: str) -> list[str]:
    """List what still depends on `name`: authored children, agents that name it in code,
    and DEFAULT_POLICY.

    Children are read from the table, so one that does not load counts too. A child
    saved meanwhile locks this row first (`_write`), so it waits for the delete.
    """
    from django_ai_sdk.oversight.authoring.models import AuthoredPolicy
    from django_ai_sdk.oversight.conf import setting as oversight_setting

    # Filtered here: SQLite has no JSON contains.
    found = [
        f"policy {child!r} extends it"
        for child, extends in AuthoredPolicy.objects.values_list("name", "extends")
        if child != name and name in (extends or [])
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
    return found


def _out(row: AuthoredPolicy) -> PolicyOut:
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


__all__ = ["AuthoringService"]
