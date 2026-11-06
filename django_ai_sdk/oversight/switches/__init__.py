"""Let operators switch a rule's mode or an agent's policy without a deploy.

An opt-in app; docs/content/oversight-switches.md covers install, the floor and the routes.
"""

from __future__ import annotations

from django_ai_sdk.oversight.switches.services import SwitchService

__all__ = ["SwitchService"]
