"""
Protocol implementations for different streaming formats.

This package contains protocol-specific implementations that convert
normalized events to various streaming formats like Vercel AI SDK,
OpenAI native, etc.
"""

from __future__ import annotations

from .base import BaseProtocolHandler
from .openai import OpenAIProtocolHandler
from .vercel import VercelProtocolHandler

__all__ = [
    "BaseProtocolHandler",
    "OpenAIProtocolHandler",
    "VercelProtocolHandler",
]
