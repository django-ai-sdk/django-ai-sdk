"""System checks for the ledger. docs/content/oversight-ledger.md lists the ids."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.core.checks import Error, Warning
from django.core.exceptions import ImproperlyConfigured

PREFIX = "ai_sdk.oversight.ledger"
KEYS_HINT = "`manage.py ledger_keys` prints a new signing key and pseudonym secret."


def check_ledger(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """E001 and E012. Check the signing key and the settings dict."""
    from django_ai_sdk.oversight.ledger.checkpoints import private_key
    from django_ai_sdk.oversight.ledger.conf import setting

    messages = setting.errors(PREFIX)
    if setting.broken:
        return messages
    key = setting("SIGNING_KEY")
    if key and isinstance(key, str):
        try:
            private_key(key)
        except ImproperlyConfigured as exc:
            messages.append(Error(str(exc), hint=KEYS_HINT, id=f"{PREFIX}.E001"))
    return messages


def check_ledger_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """W001 and W002. Warn about unsigned checkpoints and SECRET_KEY pseudonyms."""
    from django_ai_sdk.oversight.ledger.conf import setting

    if setting.broken:
        return []  # E012 reports it.
    messages: list[Any] = []
    if not setting("SIGNING_KEY"):
        messages.append(
            Warning(
                "Ledger checkpoints are unsigned: set AI_SDK_OVERSIGHT_LEDGER['SIGNING_KEY'], "
                "and publish the verifier key where the deployment cannot change it.",
                hint=KEYS_HINT,
                id=f"{PREFIX}.W001",
            )
        )
    if not setting("PSEUDONYM_SECRET"):
        messages.append(
            Warning(
                "Pseudonyms of who asked are derived from SECRET_KEY: rotating it renames "
                "everyone on the record, and deleting a user then no longer forgets the "
                "turns under their old pseudonym. Give them their own "
                "AI_SDK_OVERSIGHT_LEDGER['PSEUDONYM_SECRET'].",
                hint=KEYS_HINT,
                id=f"{PREFIX}.W002",
            )
        )
    return messages


def check_ledger_checkpoints(
    app_configs: Any = None, databases: Any = None, **kwargs: Any
) -> list[Any]:
    """W003. Warn about entries older than CHECKPOINT_DAYS without a checkpoint."""
    from django.utils import timezone

    from django_ai_sdk.checks import per_database
    from django_ai_sdk.oversight.ledger.conf import setting
    from django_ai_sdk.oversight.ledger.models import Checkpoint, LedgerEntry

    if setting.problems():
        return []  # E012 reports it.
    days = setting("CHECKPOINT_DAYS")
    cutoff = timezone.now() - timedelta(days=days)

    def stale(alias: str) -> bool:
        newest = (
            Checkpoint.objects.using(alias)
            .order_by("-tree_size")
            .values_list("tree_size", flat=True)
            .first()
        )
        entries = LedgerEntry.objects.using(alias)
        return entries.filter(seq__gt=newest or 0, created_at__lt=cutoff).exists()

    return [
        Warning(
            f"Ledger entries in database {alias!r} are more than {days} days old and "
            "no checkpoint covers them.",
            hint="Run `manage.py ledger_checkpoint` on a schedule, and keep what it "
            "publishes where the deployment cannot change it.",
            id=f"{PREFIX}.W003",
        )
        for alias, found in per_database(databases, stale)
        if found
    ]


__all__ = ["PREFIX", "check_ledger", "check_ledger_checkpoints", "check_ledger_deploy"]
