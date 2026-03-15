"""
Pirate-themed assistants for the demo application.

This package contains various pirate assistant implementations
demonstrating different AI backends and tool integrations.
"""

from .agent_swarm import AgentSwarmAssistant
from .pirate_agent import PirateAgentAssistant
from .pirate_basic import PirateBasicAssistant
from .pirate_openai import PirateOpenAIAssistant

__all__ = [
    "AgentSwarmAssistant",
    "PirateAgentAssistant",
    "PirateBasicAssistant",
    "PirateOpenAIAssistant",
]
