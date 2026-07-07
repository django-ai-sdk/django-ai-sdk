"""Demo API-backed integration: current weather via Open-Meteo.

Open-Meteo (https://open-meteo.com) requires no API key, which makes this genuinely
testable with zero credentials — the point of this example is to demonstrate the
APIIntegration contract end-to-end (register it, call get_tools(), invoke the tool,
report a real get_status()), not to be a production-grade weather tool.
"""

from __future__ import annotations

from typing import Any

import httpx
from django_ai_sdk.integrations.api.base import APIIntegration
from haystack.tools import Tool

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Amsterdam — an arbitrary, always-valid point used only to probe that the forecast
# endpoint is up, not to look up anyone's weather.
_HEALTH_CHECK_COORDS = {"latitude": 52.37, "longitude": 4.9, "current_weather": "true"}


def get_current_weather(location: str) -> dict:
    """Look up the current weather for a place name via Open-Meteo (no API key)."""
    with httpx.Client(timeout=5) as client:
        geo = client.get(_GEOCODE_URL, params={"name": location, "count": 1}).json()
        results = geo.get("results") or []
        if not results:
            return {"error": f"Could not find a location matching {location!r}"}
        place = results[0]

        forecast = client.get(
            _WEATHER_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current_weather": "true",
            },
        ).json()

    current = forecast.get("current_weather", {})
    return {
        "location": place.get("name", location),
        "country": place.get("country", ""),
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "observed_at": current.get("time"),
    }


def get_weather_tool(**kwargs: Any) -> Tool:
    return Tool(
        name="get_current_weather",
        description="Get the current weather for a place name.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, e.g. 'Rotterdam' or 'Paris, France'.",
                },
            },
            "required": ["location"],
        },
        function=get_current_weather,
    )


async def check_weather_api() -> None:
    """Health probe for get_status(): confirm the forecast endpoint responds."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(_WEATHER_URL, params=_HEALTH_CHECK_COORDS)
        response.raise_for_status()


class WeatherIntegration(APIIntegration):
    """Minimal APIIntegration example — current weather, no auth required."""

    name = "weather"
    label = "Weather"
    tools = [get_weather_tool]
    health_check = staticmethod(check_weather_api)
