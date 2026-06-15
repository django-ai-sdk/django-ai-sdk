"""
Pirate-themed assistants for the demo application.
"""

from .agent_swarm import AgentSwarmAssistant
from .pirate_basic import PirateBasicAssistant
from .workspace import WorkspaceAssistant

__all__ = [
    "AgentSwarmAssistant",
    "PirateBasicAssistant",
    "WorkspaceAssistant",
]
