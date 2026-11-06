"""System checks for the vault. docs/content/oversight-ledger.md lists the ids."""

from __future__ import annotations

from typing import Any

from django.core.checks import Error, Warning
from django.core.exceptions import ImproperlyConfigured

PREFIX = "ai_sdk.vault"
HINT = "`manage.py vault_keys` prints a new master key; see AI_SDK_VAULT['MASTER_KEYS']."


def _key_ids() -> tuple[list[str], str]:
    """The configured master key ids, current first, and why there are none."""
    from django_ai_sdk.vault.keys import provider

    try:
        ids = provider().key_ids()
    except ImproperlyConfigured as exc:
        return [], str(exc)
    return ids, "" if ids else "The vault's key provider has no key."


def check_vault(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """E012, then E001. The vault has no usable master key."""
    from django_ai_sdk.vault.conf import setting

    messages = setting.errors(PREFIX)
    if setting.broken:
        return messages
    _ids, problem = _key_ids()
    if problem:
        messages.append(Error(problem, hint=HINT, id=f"{PREFIX}.E001"))
    return messages


def check_vault_keys(app_configs: Any = None, databases: Any = None, **kwargs: Any) -> list[Any]:
    """E002. Data keys that no configured master key can unwrap."""
    from django_ai_sdk.checks import per_database
    from django_ai_sdk.vault.conf import setting
    from django_ai_sdk.vault.models import DataKey

    if setting.broken:
        return []  # E012 reports it.
    ids, problem = _key_ids()
    if problem:
        return []  # E001 reports it.

    def orphans(alias: str) -> list[str]:
        unwrappable = (
            DataKey.objects.using(alias).filter(shredded_at__isnull=True).exclude(key_id__in=ids)
        )
        return sorted(set(unwrappable.values_list("key_id", flat=True)))

    return [
        Error(
            f"Data keys in database {alias!r} are wrapped under master key ids no "
            f"AI_SDK_VAULT['MASTER_KEYS'] entry has: {', '.join(found)}. What they "
            "keep does not open.",
            hint="Put the master key with that id back in MASTER_KEYS, after the "
            "current one, then run `manage.py vault_rewrap`.",
            id=f"{PREFIX}.E002",
        )
        for alias, found in per_database(databases, orphans)
        if found
    ]


def check_vault_rotation(
    app_configs: Any = None, databases: Any = None, **kwargs: Any
) -> list[Any]:
    """W001. Data keys still wrapped under a previous master key."""
    from django_ai_sdk.checks import per_database
    from django_ai_sdk.vault.conf import setting
    from django_ai_sdk.vault.models import DataKey

    if setting.broken:
        return []
    ids, _problem = _key_ids()
    if len(ids) < 2:
        return []

    def previous(alias: str) -> int:
        keys = DataKey.objects.using(alias)
        return keys.filter(shredded_at__isnull=True, key_id__in=ids[1:]).count()

    return [
        Warning(
            f"{count} data keys in database {alias!r} are still wrapped under a "
            "previous master key.",
            hint="Run `manage.py vault_rewrap`, then drop the previous key from "
            "AI_SDK_VAULT['MASTER_KEYS'].",
            id=f"{PREFIX}.W001",
        )
        for alias, count in per_database(databases, previous)
        if count
    ]


__all__ = ["PREFIX", "check_vault", "check_vault_keys", "check_vault_rotation"]
