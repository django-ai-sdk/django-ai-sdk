"""
Pirate-themed assistants for the demo application.
"""

from .agent_swarm import AgentSwarmAssistant
from .pirate_basic import PirateBasicAssistant
from .web import DefaultWebAssistant
from .workspace import WorkspaceAssistant

__all__ = [
    "AgentSwarmAssistant",
    "DefaultWebAssistant",
    "PirateBasicAssistant",
    "WorkspaceAssistant",
]
