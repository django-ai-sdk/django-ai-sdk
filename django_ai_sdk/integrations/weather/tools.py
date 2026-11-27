from __future__ import annotations

import httpx
from haystack.tools import tool

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_HEALTH_CHECK_COORDS = {"latitude": 52.37, "longitude": 4.9, "current_weather": "true"}


def fetch_current_weather(location: str) -> dict:
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
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(_WEATHER_URL, params=_HEALTH_CHECK_COORDS)
        response.raise_for_status()


@tool
def get_current_weather(location: str) -> dict:
    """Get the current weather for a place name (e.g. 'Rotterdam' or 'Paris, France')."""
    return fetch_current_weather(location)
