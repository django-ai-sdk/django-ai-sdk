"""Example integrations for the demo app — a copy-me template.

An integration is just a class. There is no app to create, no ``INSTALLED_APPS`` entry,
no migration: name it in ``AI_SDK_INTEGRATIONS`` and the registry builds it on first use.

``WeatherIntegration`` below is the hand-written-API case (``APIIntegration``), deliberately
built on Open-Meteo, which needs no API key — so it is genuinely runnable with zero
credentials and exercises the whole contract: ``get_tools()``, invoking the tool, and a
real ``get_status()`` backed by a health probe.

For the MCP case there is nothing to write at all: point ``AI_SDK_INTEGRATIONS`` at
``django_ai_sdk.integrations.defaults.LinearIntegration`` (see the demo settings), or
subclass ``MCPIntegration`` with a url + auth for a server the SDK doesn't ship.
"""

from __future__ import annotations

import httpx
from django_ai_sdk.integrations.api.base import APIIntegration
from haystack.tools import tool

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Amsterdam — an arbitrary, always-valid point used only to probe that the forecast
# endpoint is up, not to look up anyone's weather.
_HEALTH_CHECK_COORDS = {"latitude": 52.37, "longitude": 4.9, "current_weather": "true"}


def fetch_current_weather(location: str) -> dict:
    """Geocode ``location`` and return its current weather, or an error dict."""
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


async def check_weather_api() -> None:
    """Health probe for ``get_status()``: confirm the forecast endpoint responds.

    Raising is how failure is signalled — the caller runs this through the same
    ResilientCache every integration uses, so a raise becomes DEGRADED rather than an
    exception reaching the request.
    """
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(_WEATHER_URL, params=_HEALTH_CHECK_COORDS)
        response.raise_for_status()


@tool
def get_current_weather(location: str) -> dict:
    """Get the current weather for a place name (e.g. 'Rotterdam' or 'Paris, France')."""
    return fetch_current_weather(location)


class WeatherIntegration(APIIntegration):
    """Minimal ``APIIntegration``: a ready-made tool plus a health probe.

    ``health_check`` must be a ``staticmethod`` — a bare function assigned to the
    attribute would be bound and wrongly receive ``self``.
    """

    name = "weather"
    label = "Weather"
    tools = [get_current_weather]
    health_check = staticmethod(check_weather_api)
