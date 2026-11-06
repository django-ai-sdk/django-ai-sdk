"""The Automation contract: which workflow runs, when, and as whom.

A declaration only. It names a workflow instead of containing one, so the same
workflow can also serve chat and the API.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.automations.schedule import Cron

if TYPE_CHECKING:
    from datetime import datetime


class Automation:
    """One scheduled workflow. Subclass it in an app's `automations.py` and register it."""

    # Registry key. Renaming it orphans the stored state and run history.
    name: str = ""
    label: str = ""
    description: str = ""

    # 5-field cron expression with weekdays by name (mon-fri). Required.
    cron: str = ""
    # IANA zone for `cron`, so a wall-clock time stays put across DST.
    timezone: str = "UTC"

    # Name of a registered workflow. Required.
    workflow: str = ""

    # The account a run acts as, by the user model's USERNAME_FIELD. Empty means no
    # user: the run acts as the deployment itself.
    run_as: str = ""

    # The `prompt` input the default `get_inputs` sends. {user}, {last_run_at} and
    # {name} are filled in.
    input: str = "Run the {name} automation."

    # False keeps it off the schedule. `automations_tick <name>` still runs it.
    enabled: bool = True

    # Only bare `{word}`. str.format would allow `{name.__class__}` and fail on stray braces.
    _PLACEHOLDER = re.compile(r"\{(\w+)\}")

    def get_inputs(self, *, user: Any = None, last_run_at: datetime | None = None) -> dict:
        """The inputs a run starts with.

        No person is asking, so the automation writes the first turn. The workflow
        must declare every key returned here, or the run is skipped.
        """
        return {"prompt": self.render_input(user=user, last_run_at=last_run_at)}

    def get_schedule(self) -> Cron:
        """Build the schedule from `cron` and `timezone`."""
        if not self.cron:
            raise ImproperlyConfigured(f"Automation {self.name!r} must set `cron`.")
        return Cron(self.cron, tz=self.timezone)

    def render_input(self, *, user: Any = None, last_run_at: datetime | None = None) -> str:
        """Fill the input's {user}, {last_run_at} and {name} placeholders.

        Any other brace is left as written.
        """
        values = {
            "user": user.get_username() if user else "",
            "last_run_at": last_run_at.isoformat() if last_run_at else "the beginning",
            "name": self.name,
        }
        return self._PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), self.input)

    def __str__(self) -> str:
        return self.label or self.name or type(self).__name__


__all__ = ["Automation"]
