from __future__ import annotations

from django.apps import AppConfig


class WorkflowsConfig(AppConfig):
    """The workflows app — ships WorkflowSettings/WorkflowRun/WorkflowRunStep and
    their migrations.

    Code-declared workflows (`register()`, `checks.check_workflows`) need no app of
    their own; this one exists only because those models need a home. Must be in
    INSTALLED_APPS.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.workflows"
    label = "django_ai_sdk_workflows"
    verbose_name = "Django AI SDK — Workflows"
