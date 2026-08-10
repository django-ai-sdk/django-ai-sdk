from __future__ import annotations

from django_ai_sdk.agents.runtime import RuntimeAgent

__all__ = ["DefaultRuntimeAgent"]


class DefaultRuntimeAgent(RuntimeAgent):
    """Demo project's default runtime-configurable agent.

    All configuration (model, prompt, tools, MCP servers) comes from DB.
    Register in AI_SDK_RUNTIME_AGENT_BASES to make it available.
    """
