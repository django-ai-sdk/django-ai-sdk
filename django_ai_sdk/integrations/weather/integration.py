"""Weather via Open-Meteo — the SDK's shipped example of an API-backed integration
(as opposed to github/notion/linear, which are MCP-backed).

Exercises the whole APIIntegration contract: get_tools(), invoking the tool, and a
real get_status() backed by a health probe.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.weather.tools import check_weather_api, get_current_weather


class WeatherIntegration(APIIntegration):
    """Minimal ``APIIntegration``: a ready-made tool plus a health probe.

    ``health_check`` must be a ``staticmethod`` — a bare function assigned to the
    attribute would be bound and wrongly receive ``self``.
    """

    name = "weather"
    label = "Weather"
    tools = [get_current_weather]
    health_check = staticmethod(check_weather_api)
