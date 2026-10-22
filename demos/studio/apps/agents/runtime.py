from __future__ import annotations

from django_ai_sdk.agents.runtime import RuntimeAgent

from .tools.memories import FileLookupMixin

__all__ = ["DefaultRuntimeAgent"]


class DefaultRuntimeAgent(FileLookupMixin, RuntimeAgent):
    """Demo project's default runtime-configurable agent.

    All configuration (model, prompt, tools, MCP servers) comes from DB.
    Register in AI_SDK_RUNTIME_AGENT_BASES to make it available.
    """
