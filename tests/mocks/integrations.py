"""Example integration services for tests.

These live in a module of their own (rather than inside the test module) so the registry
can import them by dotted path and get the *same* class object the test compares
against — importing the test module by path would otherwise produce a second, distinct
class and break `isinstance`.
"""

from __future__ import annotations

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.permissions import AllowAll


def fake_tool(name: str):
    return type("T", (), {"name": name})()


class ExampleWeatherService(APIIntegration):
    name = "weather"
    label = "Weather"
    tools = [lambda **kwargs: fake_tool("get_current_weather")]


class UnnamedService(APIIntegration):
    """Declares no name — the registry backfills name and label from the settings key."""

    tools = []


class OpenService(APIIntegration):
    """Usable by anyone, including an anonymous/absent user.

    The INTEGRATIONS domain default requires an authenticated user, which is the right
    policy but makes it impossible to exercise tool loading without a real user. An
    integration declaring its own permissions is a supported case, so tests use it.
    """

    name = "open"
    permissions = [AllowAll]
    tools = [lambda **kwargs: fake_tool("open_tool")]
