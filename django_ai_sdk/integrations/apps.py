"""Django-app wiring for an integration.

Every integration is its own Django app. Subclass this, point `integration` at the
Integration subclass, and add the app to INSTALLED_APPS. For an integration with no
models or extra logic, define both classes directly in apps.py — Django only requires
the AppConfig to live there, not the Integration:

    class WeatherIntegration(APIIntegration):
        name = "weather"
        ...

    class WeatherConfig(IntegrationAppConfig):
        name = "myapp.integrations.weather"
        integration = f"{__name__}.WeatherIntegration"
        default = True

An integration with its own models, services, or background tasks still splits those
into their usual modules; only the Integration subclass itself is free to live
wherever's convenient.
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
            logger.warning(
                "%s has no `integration` set — nothing to register", type(self).__name__
            )
            return

        from django_ai_sdk.integrations.registry import register

        register(import_string(self.integration)())
