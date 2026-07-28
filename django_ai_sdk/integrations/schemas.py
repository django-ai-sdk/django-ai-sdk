"""Generic, integration-kind-agnostic output schemas.

These describe integrations to the ``/api/integrations`` surface and to the
per-assistant status endpoint, regardless of whether an integration is MCP-backed or
a hand-written API wrapper.
"""

from __future__ import annotations

from pydantic import BaseModel

from django_ai_sdk.integrations.base import IntegrationStatus


class IntegrationOut(BaseModel):
    """One integration's state for the generic ``GET /api/integrations`` list.

    Capability flags let the client decide which actions to offer without knowing the
    integration's kind. ``connect_url`` is populated for integrations that connect via
    a browser redirect (e.g. MCP OAuth).
    """

    name: str
    label: str
    kind: str
    status: IntegrationStatus
    supports_connect: bool = False
    supports_test: bool = True
    #: "oauth" (follow connect_url) | None. A string rather than a bool so another
    #: flow can be added without changing this contract.
    connect_kind: str | None = None
    #: Set when the integration needs setup (e.g. a missing secret) — human-readable,
    #: shown to the user instead of a blank/confusing DISCONNECTED status.
    detail: str | None = None
    connected: bool | None = None
    connect_url: str | None = None


class AssistantIntegrationStatus(BaseModel):
    """One configured integration's status for a given assistant/user."""

    server_name: str
    label: str
    type: str
    status: IntegrationStatus
    tool_names: list[str]
