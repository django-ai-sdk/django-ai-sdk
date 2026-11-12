"""Let operators define whole policies at runtime; they run like policies in code.

An opt-in app; docs/content/oversight-authoring.md covers install, what a definition may
set and the routes.
"""

from __future__ import annotations

from django_ai_sdk.oversight.authoring.services import AuthoringService

__all__ = ["AuthoringService"]
