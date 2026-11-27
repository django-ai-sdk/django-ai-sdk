from __future__ import annotations

from pydantic import BaseModel

from django_ai_sdk.integrations.base import IntegrationStatus


class IntegrationOut(BaseModel):
    """One integration's state for GET /api/integrations; flags drive the client's actions."""

    name: str
    label: str
    hint: str = ""
    kind: str
    status: IntegrationStatus
    supports_connect: bool = False
    supports_test: bool = True
    #: "oauth" or None.
    connect_kind: str | None = None
    detail: str | None = None
    connected: bool | None = None


class AgentIntegrationStatus(BaseModel):
    """One configured integration's status for a given agent/user."""

    server_name: str
    label: str
    type: str
    status: IntegrationStatus
    tool_names: list[str]
