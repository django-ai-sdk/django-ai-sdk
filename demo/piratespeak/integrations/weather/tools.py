"""Weather tools exposed to the assistant."""

from __future__ import annotations

from haystack.tools import tool

from piratespeak.integrations.weather.api import fetch_current_weather


@tool
def get_current_weather(location: str) -> dict:
    """Get the current weather for a place name (e.g. 'Rotterdam' or 'Paris, France')."""
    return fetch_current_weather(location)
