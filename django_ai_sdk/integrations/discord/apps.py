from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class DiscordConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.discord"
    # Namespaced so the SDK never claims the global app label "discord", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_discord"
    integration = "django_ai_sdk.integrations.discord.integration.DiscordIntegration"
