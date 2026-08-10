from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class WeatherConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.weather"
    # Namespaced so the SDK never claims the global app label "weather", which a
    # host project may well want for an app of its own (as mcp/apps.py does).
    label = "django_ai_sdk_weather"
    integration = "django_ai_sdk.integrations.weather.integration.WeatherIntegration"
