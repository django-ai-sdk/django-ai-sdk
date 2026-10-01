"""HTTP request payloads and the error body, shared by the SDK's views and the contrib
Ninja/DRF layers.

Success responses are the services' own ``*Out`` models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from django_ai_sdk.workflows.schemas import WorkflowDefinition


class MessagePart(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str
    text: str | None = None
    # file parts
    url: str | None = None
    media_type: str | None = Field(default=None, alias="mediaType")
    filename: str | None = None
    provider_metadata: dict | None = Field(default=None, alias="providerMetadata")


class Message(BaseModel):
    role: str
    parts: list[MessagePart]
    id: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message]
    agent_id: str | None = None
    id: str | None = None
    trigger: str | None = None


class RateMessagePayload(BaseModel):
    rating: int | None = Field(
        None, description="Rating value: 1 for good, -1 for bad, or None to unrate (optional)"
    )
    feedback: str = Field(default="", description="Optional explanation for the rating")


class ErrorResponse(BaseModel):
    """The body of an error response (see `django_ai_sdk.errors.error_response`)."""

    code: str
    message: str
    retryable: bool
    ref: str
    errors: list[dict] | None = None


# Memories


class MemoryIn(BaseModel):
    """Schema for creating a memory."""

    name: str
    slug: str = ""
    description: str = ""
    is_public: bool = True


class DocumentIn(BaseModel):
    """Schema for creating a document."""

    content: str = ""


class BulkConnectMemoriesIn(BaseModel):
    """Schema for bulk connecting memories to a thread."""

    memory_ids: list[str]


class ToggleMemoryActiveIn(BaseModel):
    """Schema for toggling memory active status."""

    active: bool


class AddMemoryUserIn(BaseModel):
    """Schema for adding a user to a memory."""

    user_id: str
    can_manage: bool = False


class UpdateMemoryUserIn(BaseModel):
    """Schema for updating a memory user."""

    can_manage: bool


class AddMemoryGroupIn(BaseModel):
    """Schema for adding a group to a memory."""

    group_id: int
    can_manage: bool = False


# Threads


class PatchThreadPayload(BaseModel):
    agent_id: str


# Runtime agents


class AgentSettingsCreateUserEntry(BaseModel):
    user_id: str
    can_manage: bool = False


class AgentSettingsCreateGroupEntry(BaseModel):
    group_id: int
    can_manage: bool = False


class AgentSettingsCreateIn(BaseModel):
    name: str
    slug: str = ""
    agent: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    tools: list[str] = []
    integrations: list[str] = []
    memories: list[str] = []
    users: list[AgentSettingsCreateUserEntry] = []
    groups: list[AgentSettingsCreateGroupEntry] = []
    suggestion_enabled: bool = False
    title_generation: bool = True
    max_history: int | None = None
    file_upload: bool = False
    vision: bool = False
    required_tools: list[str] = []


class AgentSettingsUpdateIn(BaseModel):
    name: str | None = None
    agent: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    integrations: list[str] | None = None
    memories: list[str] | None = None
    suggestion_enabled: bool | None = None
    title_generation: bool | None = None
    max_history: int | None = None
    file_upload: bool | None = None
    vision: bool | None = None
    required_tools: list[str] | None = None
    active: bool | None = None


class AddAgentUserIn(BaseModel):
    user_id: str
    can_manage: bool = False


class UpdateAgentUserIn(BaseModel):
    can_manage: bool


class AddAgentGroupIn(BaseModel):
    group_id: int
    can_manage: bool = False


# Workflows


class WorkflowRunRequest(BaseModel):
    workflow: WorkflowDefinition
    inputs: dict[str, Any] = {}


class WorkflowCreateRequest(BaseModel):
    name: str
    workflow: WorkflowDefinition


class WorkflowUpdateRequest(BaseModel):
    name: str | None = None
    workflow: WorkflowDefinition | None = None
    active: bool | None = None


class WorkflowRunByIdRequest(BaseModel):
    inputs: dict[str, Any] = {}
    run_id: str | None = None
