"""Demo API-backed integration: current weather via Open-Meteo.

Open-Meteo (https://open-meteo.com) requires no API key, which makes this genuinely
testable with zero credentials — it demonstrates the IntegrationService contract
end-to-end (register via the app, call get_tools(), invoke the tool, report a real
get_status()) as a copy-me template for a hand-written API integration.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration

from piratespeak.integrations.weather.api import check_weather_api
from piratespeak.integrations.weather.tools import get_current_weather


class WeatherService(APIIntegration):
    """Minimal APIIntegration example — current weather, no auth required."""

    name = "weather"
    label = "Weather"
    tools = [get_current_weather]
    health_check = staticmethod(check_weather_api)
