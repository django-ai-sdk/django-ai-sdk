from __future__ import annotations

from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules


class DjangoAISDKConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk"

    def ready(self) -> None:
        # Every installed app's `assistants` module, so defining an Assistant subclass
        # there is enough on its own — no per-app ready() hook, no settings entry.
        #
        # Integrations deliberately get no equivalent autodiscovery, even though the
        # asymmetry looks like an oversight. An assistant is a thing you *define*; an
        # integration is a thing you *enable*, and INSTALLED_APPS is the explicit,
        # greppable switch for that — you can ship an integration app and leave it
        # uninstalled. Autodiscovering integrations would also mean two paths writing
        # one registry at different points in startup, with precedence rules; that is
        # exactly what this package replaced.
        autodiscover_modules("assistants")

        from django_ai_sdk.assistants.registry import registry

        registry.setup()
