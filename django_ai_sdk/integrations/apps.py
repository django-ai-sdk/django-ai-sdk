from __future__ import annotations

import logging

from django.apps import AppConfig
from django.utils.module_loading import import_string

logger = logging.getLogger(__name__)


class IntegrationAppConfig(AppConfig):
    """Registers one integration app's `integration` (a dotted path) on ready()."""

    # False so importing this base into a subclass's apps.py leaves one default config.
    default = False

    integration: str = ""

    def ready(self) -> None:
        if not self.integration:
            logger.warning("%s has no `integration` set — nothing to register", type(self).__name__)
            return

        from django_ai_sdk.integrations.registry import register

        register(import_string(self.integration)())
