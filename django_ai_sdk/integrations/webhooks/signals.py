from __future__ import annotations

from django.dispatch import Signal

# Sent per admitted event with `event`, the InboundEvent.
event_received = Signal()
