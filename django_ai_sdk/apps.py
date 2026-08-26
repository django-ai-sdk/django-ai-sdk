from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.utils.module_loading import autodiscover_modules


class DjangoAISDKConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk"

    def ready(self) -> None:
        # Autodiscovers every installed app's `agents` module, so defining an
        # Agent subclass there is enough on its own — no per-app ready() hook,
        # no settings entry.
        #
        # Integrations get no equivalent autodiscovery: an agent is a thing you
        # *define*, an integration is a thing you *enable*, and INSTALLED_APPS is the
        # explicit switch for that — you can ship an integration app and leave it
        # uninstalled. Autodiscovering integrations would mean two paths writing one
        # registry at different points in startup, which is what this package replaced.
        autodiscover_modules("agents")

        from django_ai_sdk.agents.registry import registry

        registry.setup()

        # After setup(), so a step can name its agent as `MyAgent().agent_id`.
        # Workflows before automations: an automation's workflow name is checked
        # against the populated registry.
        autodiscover_modules("workflows")
        autodiscover_modules("automations")

        from django_ai_sdk.automations.checks import check_automations
        from django_ai_sdk.workflows.checks import check_workflows

        # A rejected declaration fails no request, so nothing else would surface it.
        register_check(check_workflows)
        register_check(check_automations)
