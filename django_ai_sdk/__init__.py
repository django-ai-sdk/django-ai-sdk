from __future__ import annotations

from django_ai_sdk.agents.base import Agent
from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType
from django_ai_sdk.artifacts.schemas import (
    ApprovalCardArtifact,
    ChainOfThoughtArtifact,
    CodeBlockArtifact,
    ConfirmationArtifact,
    DataTableArtifact,
    FileArtifact,
    ImageArtifact,
    OptionListArtifact,
    PlanArtifact,
    ProgressTrackerArtifact,
    QuestionFlowArtifact,
    SchemaDisplayArtifact,
    SnippetArtifact,
    StackTraceArtifact,
    TaskArtifact,
    TerminalArtifact,
    TestResultsArtifact,
)
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.permissions import (
    AllowAll,
    BasePermission,
    DenyAll,
    IsAdminUser,
    IsAuthenticated,
    IsOwner,
    MemoryDefaultPermission,
    Operation,
    PermissionDenied,
)
from django_ai_sdk.protocols.vercel import StreamChunk
from django_ai_sdk.receipts import with_receipt
from django_ai_sdk.responses import stream_response

try:
    from importlib.metadata import version

    __version__ = version("django-ai-sdk")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "Agent",
    "AllowAll",
    "ApprovalCardArtifact",
    "ArtifactSchema",
    "ArtifactType",
    "BasePermission",
    "ChainOfThoughtArtifact",
    "ChatMessage",
    "CodeBlockArtifact",
    "ConfirmationArtifact",
    "DataTableArtifact",
    "DenyAll",
    "FileArtifact",
    "ImageArtifact",
    "IsAdminUser",
    "IsAuthenticated",
    "IsOwner",
    "MemoryDefaultPermission",
    "Operation",
    "OptionListArtifact",
    "PermissionDenied",
    "PlanArtifact",
    "ProgressTrackerArtifact",
    "QuestionFlowArtifact",
    "SchemaDisplayArtifact",
    "SnippetArtifact",
    "StackTraceArtifact",
    "StreamChunk",
    "TaskArtifact",
    "TerminalArtifact",
    "TestResultsArtifact",
    "stream_response",
    "with_receipt",
]
