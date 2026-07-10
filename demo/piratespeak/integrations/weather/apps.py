from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class WeatherConfig(IntegrationAppConfig):
    name = "piratespeak.integrations.weather"
    label = "piratespeak_weather"
    verbose_name = "Piratespeak — Weather"
    service = "piratespeak.integrations.weather.services.WeatherService"
