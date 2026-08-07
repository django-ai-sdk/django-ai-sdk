"""
Agent definitions for the demo application.

This module is auto-discovered by DjangoAISDKConfig.ready() via
autodiscover_modules("agents"), so defining an Agent subclass here
(or in any file imported by this module) is enough — no settings
entry needed, though both methods work together.
"""

from __future__ import annotations

from .agent_swarm import AgentSwarmAgent
from .pirate_basic import PirateBasicAgent
from .runtime import DefaultRuntimeAgent
from .workspace import WorkspaceAgent

__all__ = [
    "AgentSwarmAgent",
    "DefaultRuntimeAgent",
    "PirateBasicAgent",
    "WorkspaceAgent",
]
