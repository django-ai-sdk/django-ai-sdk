"""Base AppConfig every integration app subclasses.

An integration is a Django app. Its ``AppConfig`` names the ``IntegrationService`` to
construct, and on ``ready()`` registers it into the process registry. That is all
``ready()`` does — a cheap, synchronous dict write, no network I/O — so it is safe
under ``migrate``/``test``/``shell`` with no command blocklist. Caches populate lazily
on first use (``ResilientCache`` is stale-while-revalidate); there is deliberately no
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
