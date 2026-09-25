from __future__ import annotations

from django.apps import AppConfig


class WorkflowsConfig(AppConfig):
    """SDK workflows app"""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.workflows"
    label = "django_ai_sdk_workflows"
    verbose_name = "Workflows"
