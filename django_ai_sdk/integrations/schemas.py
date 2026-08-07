"""Generic, integration-kind-agnostic output schemas.

These describe integrations to the /api/integrations surface and to the
per-agent status endpoint, regardless of whether an integration is MCP-backed or
a hand-written API wrapper.
"""

from __future__ import annotations

from pydantic import BaseModel

from django_ai_sdk.integrations.base import IntegrationStatus


class IntegrationOut(BaseModel):
    """One integration's state for the generic GET /api/integrations list.

    Capability flags let the client decide which actions to offer without knowing the
    integration's kind. When supports_connect, the client calls the host project's
    POST /{name}/connect endpoint to get the redirect URL. This list never carries
    one itself, since building it needs the request.
    """

    name: str
    label: str
    #: What this integration's data actually is, shown next to the label so a user
    #: (or an admin deciding whether to enable it for an assistant) knows what it's
    #: for, not just its name. See Integration.hint.
    hint: str = ""
    kind: str
    status: IntegrationStatus
    supports_connect: bool = False
    supports_test: bool = True
    #: "oauth" (call POST /{name}/connect, then follow its redirect_url) | None. A
    #: string rather than a bool so another flow can be added without changing this
    #: contract.
    connect_kind: str | None = None
    #: Set when the integration needs setup (e.g. a missing secret) — human-readable,
    #: shown to the user instead of a blank/confusing DISCONNECTED status.
    detail: str | None = None
    connected: bool | None = None


class AgentIntegrationStatus(BaseModel):
    """One configured integration's status for a given agent/user."""

    server_name: str
    label: str
    type: str
    status: IntegrationStatus
    tool_names: list[str]
