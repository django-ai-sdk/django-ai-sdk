"""Signals of the durable app."""

from __future__ import annotations

from django.dispatch import Signal

# Sent by `ai_sdk_worker` once DBOS has launched and the cron triggers are synced,
# so the DBOS tables exist. For a host's own schedules: `sender` is the command.
worker_launched = Signal()
