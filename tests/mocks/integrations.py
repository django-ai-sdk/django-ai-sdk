from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.permissions import AllowAll


def fake_tool(name: str):
    return type("T", (), {"name": name})()


class ExampleWeatherService(APIIntegration):
    name = "weather"
    label = "Weather"
    tools = [lambda **kwargs: fake_tool("get_current_weather")]


class OpenService(APIIntegration):
    name = "open"
    permissions = [AllowAll]
    tools = [lambda **kwargs: fake_tool("open_tool")]
