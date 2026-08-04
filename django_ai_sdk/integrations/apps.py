"""Django-app wiring for an integration.

Every integration is its own Django app. Subclass this, point `integration` at the
Integration subclass, and add the app to INSTALLED_APPS:

    class ZendeskConfig(IntegrationAppConfig):
        name = "myapp.integrations.zendesk"
        integration = "myapp.integrations.zendesk.integration.ZendeskIntegration"
        default = True

INSTALLED_APPS decides whether an integration exists; AI_SDK_INTEGRATIONS (see
config.py) only configures it. ready() is just a dict write, so it's safe under
migrate/test/shell, and an unconfigured integration registers as "needs setup"
instead of failing boot.

The Integration subclass belongs in integration.py, not here — `name`/`label` mean
different things on each class (app path/label vs. registry key/display name), and
Django and the registry construct them differently. Hand-written @tool functions, if
any, live alongside it in tools.py (see weather/).
"""

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
