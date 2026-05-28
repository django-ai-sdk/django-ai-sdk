"""
Registry patching helpers.

The ``AssistantRegistry`` is a singleton with auto-registration side effects.
These helpers patch it at both import paths simultaneously to avoid that.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from django_ai_sdk.tests.mocks.assistant import create_assistant_mock


@contextmanager
def patch_registry(assistant=None, assistants=None, permissions=None):
    """Context manager that patches the global assistant registry.

    Both import paths (``django_ai_sdk.assistants.registry.registry`` and
    ``django_ai_sdk.assistants.services.registry``) are patched to the same
    mock, ensuring consistency.

    Yields the mock registry so callers can reconfigure it:
        with patch_registry() as reg:
            reg.get.return_value = None   # simulate "not found"

    Parameters:
        assistant:   The mock assistant to return from ``reg.get()``.
                     Default: ``create_assistant_mock()``
        assistants:  Dict mapping id -> assistant for ``reg.all()``.
                     Default: ``{assistant.id: assistant}``
        permissions: Passed to ``create_assistant_mock()`` when *assistant*
                     is not provided.
    """
    if assistant is None:
        assistant = create_assistant_mock(permissions=permissions)
    if assistants is None:
        assistants = {assistant.id: assistant}

    with patch("django_ai_sdk.assistants.registry.registry") as reg, \
         patch("django_ai_sdk.assistants.services.registry", reg):
        reg.get = MagicMock(return_value=assistant)
        reg.all = MagicMock(return_value=assistants)
        yield reg
