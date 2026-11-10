"""The ledger app config.

On start it installs LedgerCustody and registers the checks.
"""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import Tags
from django.core.checks import register as register_check


class LedgerConfig(AppConfig):
    """Opt-in oversight ledger."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight.ledger"
    label = "django_ai_sdk_oversight_ledger"
    verbose_name = "Oversight ledger"

    def ready(self) -> None:
        from django_ai_sdk.oversight.apps import extend, require_apps
        from django_ai_sdk.permissions import Operation

        require_apps(self.name, "django_ai_sdk.oversight", "django_ai_sdk.vault")
        extend(
            permissions={
                Operation.VIEW_OVERSIGHT_LEDGER: f"{self.label}.view_checkpoint",
                Operation.CHECKPOINT_OVERSIGHT: f"{self.label}.add_checkpoint",
                Operation.REVEAL_OVERSIGHT_ACTOR: f"{self.label}.reveal_actor",
            },
            routes="django_ai_sdk.contrib.ninja.oversight_ledger",
        )
        from django_ai_sdk.oversight import custody
        from django_ai_sdk.oversight.ledger.checks import (
            check_ledger,
            check_ledger_checkpoints,
            check_ledger_deploy,
        )
        from django_ai_sdk.oversight.ledger.custody import LedgerCustody

        register_check(check_ledger)
        register_check(check_ledger_deploy, deploy=True)
        register_check(check_ledger_checkpoints, Tags.database, deploy=True)
        # The one place the ledger plugs into core oversight. From here on, core stores,
        # loads and forgets turn content, and chains every row it writes, through it.
        custody.install(LedgerCustody())
