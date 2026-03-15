"""
Assistant management package for Django AI SDK.

Provides:
- Assistant base class with registration support
- AssistantRegistry singleton for managing assistants
- AssistantInfo model for assistant metadata
- AssistantInfoMixin for metadata functionality
- @auto_register decorator for easy registration

Registration Methods:
    1. Settings-based (recommended): Define AI_SDK_ASSISTANTS in settings.py
       AI_SDK_ASSISTANTS = [
           "myapp.assistants.MyAssistant",
       ]

    2. Decorator-based: Apply @auto_register to Assistant classes
       from django_ai_sdk.assistants import auto_register

       @auto_register
       class MyAssistant(Assistant):
           pass

Both methods can be combined - a class will only be registered once.
"""

from .mixins import AssistantInfo, AssistantInfoMixin
from .registry import (
    AssistantRegistrationError,
    AssistantRegistry,
    auto_register,
    registry,
)

__all__ = [
    "AssistantInfo",
    "AssistantInfoMixin",
    "AssistantRegistry",
    "AssistantRegistrationError",
    "auto_register",
    "registry",
]
