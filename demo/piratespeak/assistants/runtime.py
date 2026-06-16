from django_ai_sdk.assistants.runtime import RuntimeAssistant

__all__ = ["DefaultRuntimeAssistant"]


class DefaultRuntimeAssistant(RuntimeAssistant):
    """Demo project's default runtime-configurable assistant.

    All configuration (model, prompt, tools, MCP servers) comes from DB.
    Register in AI_SDK_RUNTIME_ASSISTANT_BASES to make it available.
    """
