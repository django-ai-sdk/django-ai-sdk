"""Piratespeak Django app configuration.

Deliberately empty of assistant wiring: DjangoAISDKConfig.ready() autodiscovers
every installed app's assistants module and calls registry.setup() once, which
covers all three registration methods (AI_SDK_ASSISTANTS, @auto_register, and a
plain Assistant subclass in assistants.py). An app doesn't call setup() itself.
"""

from __future__ import annotations

from django.apps import AppConfig


class PiratespeakConfig(AppConfig):
    """Configuration for the piratespeak Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "piratespeak"
    verbose_name = "Pirate Speak AI Demo"
