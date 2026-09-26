"""Linear, with what this demo's own Linear workspace actually contains."""

from __future__ import annotations

from django_ai_sdk.integrations.linear.integration import LinearIntegration as _LinearIntegration


class LinearIntegration(_LinearIntegration):
    hint = "Engineering issue tracker for the Pirate Speak demo project."
    default_tools = ["list_issues"]
