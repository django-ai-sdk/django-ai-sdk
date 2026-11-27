from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.weather.tools import check_weather_api, get_current_weather


class WeatherIntegration(APIIntegration):
    name = "weather"
    label = "Weather"
    tools = [get_current_weather]
    health_check = staticmethod(check_weather_api)
