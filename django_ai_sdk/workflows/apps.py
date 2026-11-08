"""The workflows app. On start it imports each app's `workflows.py` and registers
the workflow system checks.

Opt-in. Without it in INSTALLED_APPS there are no workflow tables or checks, and
nothing from this package is imported.
"""

from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import autodiscover_modules


class WorkflowsConfig(AppConfig):
    """Opt-in app for stored workflow definitions, queued runs and their steps."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.workflows"
    label = "django_ai_sdk_workflows"
    verbose_name = "Workflows"

    def ready(self) -> None:
        from django.apps import apps

        # Apps become ready in INSTALLED_APPS order. Steps name agents through
        # `MyAgent().agent_id`, which needs the core app ready first.
        names = [config.name for config in apps.get_app_configs()]
        if "django_ai_sdk" not in names[: names.index(self.name)]:
            raise ImproperlyConfigured(
                "List 'django_ai_sdk.workflows' after 'django_ai_sdk' in INSTALLED_APPS."
            )
        autodiscover_modules("workflows")

        from django_ai_sdk.workflows.checks import check_workflows, check_workflows_deploy

        # A rejected declaration fails no request, so only this check reports it.
        register_check(check_workflows)
        # Deploy only, like oversight's W014, because development runs tasks inline.
        register_check(check_workflows_deploy, deploy=True)
