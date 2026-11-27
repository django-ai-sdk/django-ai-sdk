from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class WeatherConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.weather"
    label = "django_ai_sdk_weather"
    integration = "django_ai_sdk.integrations.weather.integration.WeatherIntegration"
