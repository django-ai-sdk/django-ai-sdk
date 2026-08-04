"""Django-app wiring for an integration.

Every integration is its own Django app. Subclass this, point `integration` at the
Integration subclass, and add the app to INSTALLED_APPS:

    class ZendeskConfig(IntegrationAppConfig):
        name = "myapp.integrations.zendesk"
        integration = "myapp.integrations.zendesk.integration.ZendeskIntegration"
        default = True

Being in INSTALLED_APPS is what makes an integration exist; its AI_SDK_INTEGRATIONS
entry (see config.py) is what configures it. ready() does nothing but a dict write --
no network I/O, no settings validation -- so it is safe under migrate/test/shell, and
an unconfigured integration registers as "needs setup" rather than failing boot.

Keep the Integration subclass out of apps.py: the two classes give both `name` and
`label` unrelated meanings. `name` here is this app's dotted Python path, on the
Integration it is the registry key ("github"). `label` here is the Django app label, a
machine identifier that must be unique across INSTALLED_APPS ("django_ai_sdk_github");
on the Integration it is the human display name shown in the UI ("GitHub"). Their
construction differs too: Django instantiates the AppConfig itself, positionally,
before ready() runs, while the Integration is built with a bare `()` by the registry
and directly in tests. Keeping them apart stops the two contracts colliding in one
class. The Integration subclass itself always lives in integration.py; hand-written
@tool functions (when there are any — an MCP-backed integration has none) live
alongside it in their own tools.py (see weather/).
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
    the subclass to resolve it, per
    https://docs.djangoproject.com/en/stable/ref/applications/#for-application-authors.
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
