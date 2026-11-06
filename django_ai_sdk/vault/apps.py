"""The vault app config. Registers the system checks on start."""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import Tags
from django.core.checks import register as register_check


class VaultConfig(AppConfig):
    """Opt-in vault with per-scope keys."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.vault"
    label = "django_ai_sdk_vault"
    verbose_name = "Vault"

    def ready(self) -> None:
        from django_ai_sdk.vault.checks import (
            check_vault,
            check_vault_keys,
            check_vault_rotation,
        )

        register_check(check_vault)
        register_check(check_vault_keys, Tags.database)
        register_check(check_vault_rotation, Tags.database, deploy=True)
