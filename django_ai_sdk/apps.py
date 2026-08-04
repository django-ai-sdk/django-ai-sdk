from __future__ import annotations

from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules


class DjangoAISDKConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk"

    def ready(self) -> None:
        # Autodiscovers every installed app's `assistants` module, so defining an
        # Assistant subclass there is enough on its own — no per-app ready() hook,
        # no settings entry.
        #
        # Integrations get no equivalent autodiscovery: an assistant is a thing you
        # *define*, an integration is a thing you *enable*, and INSTALLED_APPS is the
        # explicit switch for that — you can ship an integration app and leave it
        # uninstalled. Autodiscovering integrations would mean two paths writing one
        # registry at different points in startup, which is what this package replaced.
        autodiscover_modules("assistants")

        from django_ai_sdk.assistants.registry import registry

        registry.setup()
