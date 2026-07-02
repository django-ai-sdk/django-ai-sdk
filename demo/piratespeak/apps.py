"""Piratespeak Django app configuration.

This module demonstrates both registration methods:
1. Settings-based (recommended): Define AI_SDK_ASSISTANTS in settings.py
2. Decorator-based: Apply @auto_register to Assistant classes

Both methods work together - a class will only be registered once.
"""

from __future__ import annotations

from django.apps import AppConfig
from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class PiratespeakConfig(AppConfig):
    """Configuration for the piratespeak Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "piratespeak"
    verbose_name = "Pirate Speak AI Demo"

    def ready(self) -> None:
        """Initialize assistants when Django starts."""

        from django_ai_sdk.assistants.registry import registry

        # Setup registry (loads from settings and instantiates all assistants)
        registry.setup()

        # Log registered assistants
        for assistant_id in registry.ids():
            logger.info(f"Registered assistant: {assistant_id}")
