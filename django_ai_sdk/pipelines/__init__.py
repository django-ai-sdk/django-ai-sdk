"""
Pipelines module for Haystack pipeline implementations.

This module provides pipeline components for the Haystack adapter.
"""

from django_ai_sdk.pipelines.haystack import ToolAgent, ToolAgentConfig

__all__ = [
    "ToolAgent",
    "ToolAgentConfig",
]
