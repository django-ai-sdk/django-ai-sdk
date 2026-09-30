"""Response schemas of the contrib Ninja endpoints.

Request payloads are shared with the DRF layer and live in ``django_ai_sdk.views.schemas``;
services return their own ``*Out`` models, used here directly where they fit. Import from
here when you replace an endpoint and want to keep its response shape.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from ninja import Schema

from django_ai_sdk.integrations.base import IntegrationStatus
from django_ai_sdk.memories.schemas import MemoryOut
from django_ai_sdk.permissions import ObjectPermissions
from django_ai_sdk.storage.schemas import ThreadInfo
from django_ai_sdk.tracing.schemas import TraceOut

# Threads


class Success(Schema):
    success: bool
    message: str | None = None


class ThreadListItem(Schema):
    id: str
    title: str
    agent_id: str
    created_at: str
    updated_at: str
    message_count: int


class ThreadListResponse(Schema):
    threads: list[ThreadListItem]


class CreateThreadResponse(Schema):
    thread_id: str | None = None


class FeedbackResponse(Schema):
    id: str
    user_id: str | None = None
    rating: int
    feedback: str
    created_at: str | None = None


class ThreadMessage(Schema):
    id: str
    role: str
    parts: list = []
    finish_reason: str | None = None
    tool_calls: list = []
    processing_time_ms: int | None = None
    has_errors: bool = False
    usage: dict | None = None
    feedback: FeedbackResponse | None = None
    created_at: str | None = None


class ThreadDetailResponse(Schema):
    thread: ThreadInfo
    messages: list[ThreadMessage]
    permissions: ObjectPermissions = ObjectPermissions()


class ThreadFileMeta(Schema):
    file_count: int = 0
    file_memory_id: str | None = None


class DeleteAllThreadsResponse(Schema):
    success: bool
    deleted_count: int


class MessageResponse(Schema):
    id: str
    is_deleted: bool = False
    feedback: FeedbackResponse | None = None


class RunResponse(Schema):
    result: str | None = None
    thread_id: str


class ThreadTracesResponse(Schema):
    traces: list[TraceOut]


# Agents


class AgentItem(Schema):
    id: str
    name: str | None = None
    model: str | None = None
    file_upload: bool = False
    rag: bool = False


class AgentsListResponse(Schema):
    agents: list[AgentItem]


class AgentInfoResponse(Schema):
    id: str
    name: str | None = None
    model: str | None = None
    class_name: str
    description: str | None = None
    instructions: str | None = None
    file_upload: bool = False
    rag: bool = False
    permissions: ObjectPermissions = ObjectPermissions()


class Tool(Schema):
    label: str
    description: str | None = None
    children: list[Tool] = []


class IntegrationStatusOut(Schema):
    server_name: str
    label: str
    type: str
    status: str
    tool_names: list[str]


class ToolsResponse(Schema):
    tools: list[Tool]
    integrations: list[IntegrationStatusOut] = []


class AgentSettingsOut(Schema):
    id: UUID
    name: str
    slug: str
    agent: str
    model: str
    system_prompt: str
    tools: list[str]
    integrations: list[str]
    memories: list[str]
    suggestion_enabled: bool
    title_generation: bool
    max_history: int | None
    file_upload: bool
    active: bool
    created_at: datetime
    updated_at: datetime


class RuntimeAgentBaseItem(Schema):
    path: str
    name: str


class RuntimeAgentToolItem(Schema):
    key: str
    path: str


class AgentUserOut(Schema):
    user_id: str
    email: str = ""
    first_name: str = ""
    last_name: str = ""
    can_manage: bool
    created_at: str


class AgentGroupOut(Schema):
    group_id: int
    group_name: str
    can_manage: bool
    created_at: str


# Memories


class MemoryOutResponse(MemoryOut):
    permissions: ObjectPermissions = ObjectPermissions()


class SourceContentOut(Schema):
    content: str


class UploadSettingsOut(Schema):
    max_upload_size: int
    allowed_mime_types: list[str]


# Workflows


class WorkflowRunResponse(Schema):
    run_id: str
    status: str


class WorkflowActionItem(Schema):
    key: str
    description: str


class WorkflowRunStepOut(Schema):
    id: str
    sequence: int
    # The step's name is its key: its output is filed under it.
    step_name: str
    output: dict | None = None
    status: str
    error: str
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunOut(Schema):
    id: str
    workflow_id: str | None = None
    status: str
    outputs: dict | None = None
    error: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunDetailOut(WorkflowRunOut):
    steps: list[WorkflowRunStepOut] = []


class WorkflowItem(Schema):
    id: str
    name: str
    definition: dict
    active: bool


# Integrations


class DetailOut(Schema):
    detail: str


class ConnectOut(Schema):
    """Where the client should go to complete a connection (e.g. an OAuth redirect)."""

    redirect_url: str


class StatusOut(Schema):
    status: IntegrationStatus
