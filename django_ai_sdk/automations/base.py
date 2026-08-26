"""The Automation contract: which workflow runs, when, and as whom.

A declaration only; nothing here starts a clock. It names a workflow rather than
containing one, so one workflow serves chat, the API and a schedule.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.automations.audience import Audience
from django_ai_sdk.automations.schedule import Cron
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from datetime import datetime

    from django_ai_sdk.automations.audience import AudienceResolver


class Automation:
    """One scheduled workflow. Subclass it in an app's `automations.py` and register."""

    # Registry key. State and run history key on it, so renaming orphans them.
    name: str = ""
    label: str = ""
    description: str = ""

    # 5-field cron expression. Required.
    cron: str = ""
    # IANA zone the `cron` expression is read in, so a wall-clock time survives DST.
    timezone: str = "UTC"

    # Name of a registered workflow. Required.
    workflow: str = ""

    # The user turn this occurrence starts from. {user}, {last_run_at} and {name} are
    # substituted.
    input: str = "Run the {name} automation."

    audience: AudienceResolver = Audience.APP

    # Integration names that must be ACTIVE. A degraded one makes the run `skipped`.
    requires: ClassVar[list[str]] = []

    # A DB row or AI_SDK_AUTOMATIONS entry overrides this.
    enabled: bool = True

    # When False, a tick is skipped while a previous run holds the lease.
    allow_overlap: bool = False

    timeout: int | None = None  # seconds; falls back to AI_SDK_AUTOMATION_TIMEOUT

    # Bare `{word}` only: str.format would reach `{name.__class__}` and raise on prose.
    _PLACEHOLDER = re.compile(r"\{(\w+)\}")

    def get_schedule(self) -> Cron:
        """The effective schedule, with settings taking precedence over the class."""
        from django_ai_sdk.automations.config import get_automation_config

        config = get_automation_config(self.name)
        cron = config.get("CRON") or self.cron
        tz = config.get("TIMEZONE") or self.timezone

        if not cron:
            raise ImproperlyConfigured(f"Automation {self.name!r} must set `cron`.")
        return Cron(cron, tz=tz)

    def get_timeout(self) -> int:
        if self.timeout is not None:
            return self.timeout
        return resolve_setting("AI_SDK_AUTOMATION_TIMEOUT", 900)

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
