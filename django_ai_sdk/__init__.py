"""
Django AI SDK - A plug-and-play Django AI streaming SDK.
"""

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

__version__ = "0.1.0"
__all__ = [
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
