"""Signals of webhook integrations."""

from __future__ import annotations

from django.dispatch import Signal

# Sent by the receive view for each admitted delivery, before it is queued, with
# `event`, the InboundEvent. The workflows app starts its event triggers on it.
event_received = Signal()
