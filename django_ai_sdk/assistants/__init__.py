"""
Assistant management package for Django AI SDK.

Provides:
- Assistant base class with registration support
- AssistantRegistry singleton for managing assistants
- AssistantInfo model for assistant metadata
- AssistantInfoMixin for metadata functionality
- @auto_register decorator for easy registration
- RuntimeAssistant for configured assistants
- AssistantSettings Django model (import from django_ai_sdk.assistants.models)

Registration Methods:
    1. Settings-based: Define AI_SDK_ASSISTANTS in settings.py
       AI_SDK_ASSISTANTS = [
           "myapp.assistants.MyAssistant",
       ]

    2. Decorator-based: Apply @auto_register to Assistant classes
       from django_ai_sdk.assistants import auto_register

       @auto_register
       class MyAssistant(Assistant):
           pass

Both methods can be combined, a class will only be registered once. Either way,
every app's assistants.py is also autodiscovered on startup (see
DjangoAISDKConfig.ready()), so putting a class there is enough on its own.
"""

from __future__ import annotations

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
