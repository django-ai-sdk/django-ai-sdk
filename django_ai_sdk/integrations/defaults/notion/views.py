"""Integration-specific inbound HTTP (a Ninja ``router``).

Reserved scaffold slot: define an app-specific ``router`` here and mount it if this
integration needs its own endpoints. Notion is served entirely by the generic
``/api/integrations`` surface, so none are needed.
"""

from __future__ import annotations
