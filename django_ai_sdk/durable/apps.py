"""The durable app: workflow runs, triggers and tasks on DBOS (see the package docstring)."""

from __future__ import annotations

from django.apps import AppConfig
from django.core.exceptions import ImproperlyConfigured


class DurableConfig(AppConfig):
    name = "django_ai_sdk.durable"
    label = "django_ai_sdk_durable"
    verbose_name = "Durable runs (DBOS)"

    def ready(self) -> None:
        from importlib.util import find_spec

        from django.apps import apps

        if find_spec("dbos") is None:
            raise ImproperlyConfigured(
                "'django_ai_sdk.durable' needs DBOS: install django-ai-sdk[dbos]."
            )
        if not apps.is_installed("django_ai_sdk.workflows"):
            raise ImproperlyConfigured(
                "'django_ai_sdk.durable' runs workflows: add 'django_ai_sdk.workflows' too."
            )
