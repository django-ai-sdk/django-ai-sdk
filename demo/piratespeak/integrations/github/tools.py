"""GitHub tool exposed to the assistant, built by a factory rather than a plain @tool.

A factory is a callable in APIIntegration.tools that's invoked per request with
only the context arguments it declares (any subset of user/assistant/thread_id,
or **kwargs for all of them) — the shape a real integration uses to bake a
*credential* into the tool it hands back, e.g. a per-user API token instead of
this demo's single deployment-wide one.
"""

from __future__ import annotations

from typing import Any

from django.conf import settings
from haystack.tools import Tool

from piratespeak.integrations.github.api import fetch_repos


def list_my_repos(**kwargs: Any) -> Tool:
    """Build the list_my_repos tool using this deployment's configured token.

    Declaring `**kwargs` (rather than nothing) documents that a factory *can*
    read `user`/`assistant`/`thread_id` here — a real per-user integration
    would read a stored token off `kwargs["user"]` instead of settings.
    """
    token = settings.AI_SDK_GITHUB.get("token", "")

    def _list_my_repos() -> list[dict]:
        """List the connected GitHub account's repositories."""
        return fetch_repos(token)

    return Tool(
        name="list_my_repos",
        description="List the connected GitHub account's repositories.",
        parameters={"type": "object", "properties": {}},
        function=_list_my_repos,
    )
