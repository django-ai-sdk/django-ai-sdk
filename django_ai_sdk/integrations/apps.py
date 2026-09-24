from __future__ import annotations

import logging

from django.apps import AppConfig
from django.utils.module_loading import import_string

logger = logging.getLogger(__name__)


class IntegrationAppConfig(AppConfig):
    """Django wiring for one integration app. Subclasses set `name` and `integration`.

    default = False, so importing this base class into a subclass's apps.py doesn't
    leave Django with two AppConfig candidates in that module. Set default = True on
    the subclass to resolve it.
    """

    default = False

    #: Dotted path to this app's Integration subclass.
    integration: str = ""

    def ready(self) -> None:
        if not self.integration:
            logger.warning("%s has no `integration` set — nothing to register", type(self).__name__)
            return

        from django_ai_sdk.integrations.registry import register

        register(import_string(self.integration)())
