"""
Django AI SDK - A plug-and-play Django AI streaming SDK.
"""

from django_ai_sdk.adapters.base import BasePipelineAdapter
from django_ai_sdk.assistant import Assistant
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.frameworks.haystack import make_handoff_tool
from django_ai_sdk.protocols.vercel import StreamChunk
from django_ai_sdk.responses import stream_response

__version__ = "0.1.0"
__all__ = [
    "Assistant",
    "ChatMessage",
    "StreamChunk",
    "BasePipelineAdapter",
    "stream_response",
    "make_handoff_tool",
]
