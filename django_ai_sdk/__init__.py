"""
Django AI SDK - A plug-and-play Django AI streaming SDK.
"""

from __future__ import annotations

from django_ai_sdk.artifacts import ArtifactSchema, ArtifactType
from django_ai_sdk.artifacts.schemas import (
    ApprovalCardArtifact,
    DataTableArtifact,
    OptionListArtifact,
    PlanArtifact,
    ProgressTrackerArtifact,
    QuestionFlowArtifact,
    TerminalArtifact,
)
from django_ai_sdk.assistant import Assistant
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
from django_ai_sdk.responses import stream_response

try:
    from importlib.metadata import version

    __version__ = version("django-ai-sdk")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "ArtifactSchema",
    "ArtifactType",
    "QuestionFlowArtifact",
    "OptionListArtifact",
    "DataTableArtifact",
    "ApprovalCardArtifact",
    "PlanArtifact",
    "ProgressTrackerArtifact",
    "TerminalArtifact",
    "Assistant",
    "ChatMessage",
    "StreamChunk",
    "stream_response",
    "BasePermission",
    "AllowAll",
    "DenyAll",
    "IsAuthenticated",
    "IsAdminUser",
    "IsOwner",
    "MemoryDefaultPermission",
    "PermissionDenied",
    "Operation",
]
