from django_ai_sdk.web_assistant import WebAssistant

__all__ = ["DefaultWebAssistant"]


class DefaultWebAssistant(WebAssistant):
    """Demo project's default web-configurable assistant.

    All configuration (model, prompt, tools, MCP servers) comes from DB.
    Register in AI_SDK_WEB_ASSISTANT_BASES to make it available.
    """
