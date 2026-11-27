from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import autodiscover_modules


class WorkflowsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.workflows"
    label = "django_ai_sdk_workflows"
    verbose_name = "Workflows"

    def ready(self) -> None:
        from django.apps import apps

        # Apps become ready in INSTALLED_APPS order.
        names = [config.name for config in apps.get_app_configs()]
        if "django_ai_sdk" not in names[: names.index(self.name)]:
            raise ImproperlyConfigured(
                "List 'django_ai_sdk.workflows' after 'django_ai_sdk' in INSTALLED_APPS."
            )
        if not apps.is_installed("django_ai_sdk.durable"):
            raise ImproperlyConfigured(
                "Workflow runs execute on DBOS: add 'django_ai_sdk.durable' to INSTALLED_APPS "
                "(needs the `dbos` extra)."
            )
        autodiscover_modules("workflows")

        from django_ai_sdk.integrations.webhooks.signals import event_received
        from django_ai_sdk.workflows.triggers import start_for_webhook

        event_received.connect(start_for_webhook, dispatch_uid="ai_sdk_workflow_event_triggers")

        from django_ai_sdk.workflows.checks import check_workflows

        register_check(check_workflows)
