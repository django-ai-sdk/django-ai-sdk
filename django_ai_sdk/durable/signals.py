from __future__ import annotations

from django.dispatch import Signal

# Sent by `ai_sdk_worker` once DBOS has launched (its tables exist).
# `sender` is the command.
worker_launched = Signal()
