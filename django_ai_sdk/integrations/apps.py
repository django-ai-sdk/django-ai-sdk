"""Optional Django-app wiring for an integration.

Most integrations need nothing here: list them in ``AI_SDK_INTEGRATIONS`` and the
registry builds them on first use (see ``registry.py``).

This is the escape hatch for an integration that genuinely *is* a Django app — one
shipping its own models, migrations, or admin. Subclass it, point ``service`` at the
``IntegrationService``, and add the app to ``INSTALLED_APPS``; ``ready()`` registers
the service into the same registry the settings mapping feeds::

    class WeatherConfig(IntegrationAppConfig):
        name = "myapp.weather"
        service = "myapp.weather.services.WeatherService"

``ready()`` does nothing but a dict write — no network I/O — so it is safe under
``migrate``/``test``/``shell`` with no command blocklist. Caches populate lazily on
first use (``ResilientCache`` is stale-while-revalidate); there is deliberately no
boot-time warmup thread.
"""

from __future__ import annotations

import logging

from django.apps import AppConfig
from django.utils.module_loading import import_string

logger = logging.getLogger(__name__)


class IntegrationAppConfig(AppConfig):
    """Django wiring for one integration app. Subclasses set ``name`` and ``service``."""

    #: Dotted path to this app's ``IntegrationService`` subclass.
    service: str = ""

    def ready(self) -> None:
        if not self.service:
            logger.warning("%s has no `service` set — nothing to register", type(self).__name__)
            return

        from django_ai_sdk.integrations.registry import register

        register(import_string(self.service)())
