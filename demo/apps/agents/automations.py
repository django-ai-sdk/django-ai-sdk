"""Automation declarations for the demo application, autodiscovered on startup.

Both are disabled in demo/settings.py, so the demo makes no model calls on a timer.
"""

from __future__ import annotations

from django_ai_sdk.automations import Audience, Automation, register


@register
class DailyHarbourReport(Automation):
    """One run per user who opted in, delivered into each subscriber's own chat."""

    name = "daily-harbour-report"
    label = "Daily harbour report"
    description = "Posts the Rotterdam forecast into each subscribed user's chat."

    cron = "0 7 * * *"
    timezone = "Europe/Amsterdam"
    audience = Audience.SUBSCRIBED
    requires = ["weather"]

    workflow = "harbour-report"
    input = "Report the harbour conditions since {last_run_at}."


@register
class WeeklySailingVerdict(Automation):
    """The app itself, running a two-step workflow with a structured second step.

    An app-level run has no user to own a thread, so the result lands on the run.
    """

    name = "weekly-sailing-verdict"
    label = "Weekly sailing verdict"
    description = "Checks the forecast, then classifies it as sailing weather."

    cron = "0 7 * * 1"
    timezone = "Europe/Amsterdam"
    requires = ["weather"]

    workflow = "sailing-verdict"
    input = "Judge the sailing conditions for the week beginning {last_run_at}."
