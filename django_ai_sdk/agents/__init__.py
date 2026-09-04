"""
Agent management package for Django AI SDK.

Provides:
- Agent base class with registration support
- AgentRegistry singleton for managing agents
- AgentInfo model for agent metadata
- AgentInfoMixin for metadata functionality
- @auto_register decorator for easy registration
- RuntimeAgent for configured agents
- AgentSettings Django model (import from django_ai_sdk.agents.models)

Registration Methods:
    1. Settings-based: Define AI_SDK_AGENTS in settings.py
       AI_SDK_AGENTS = [
           "myapp.agents.MyAgent",
       ]

    2. Decorator-based: Apply @auto_register to Agent classes
       from django_ai_sdk.agents import auto_register

       @auto_register
       class MyAgent(Agent):
           pass

Both methods can be combined, a class will only be registered once. Either way,
every app's agents.py is also autodiscovered on startup (see
DjangoAISDKConfig.ready()), so putting a class there is enough on its own.
"""

from __future__ import annotations

from .mixins import AgentInfo, AgentInfoMixin
from .registry import (
    AgentRegistrationError,
    AgentRegistry,
    auto_register,
    registry,
)
from .subagent import (
    build_subagent,
    subagent_response,
    subagent_tool_name,
)
from .tool_agent import (
    LogToolCallsHook,
    ToolAgent,
    ToolAgentConfig,
    ToolCallBudgetHook,
)

__all__ = [
    "AgentInfo",
    "AgentInfoMixin",
    "AgentRegistry",
    "AgentRegistrationError",
    "auto_register",
    "registry",
    "LogToolCallsHook",
    "ToolAgent",
    "ToolAgentConfig",
    "ToolCallBudgetHook",
    "build_subagent",
    "subagent_response",
    "subagent_tool_name",
]
