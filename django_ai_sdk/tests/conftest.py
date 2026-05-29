"""
Test configuration and shared fixtures for django-ai-sdk tests.
"""

import asyncio
import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator

# Django configuration for tests
pytest_plugins = ["pytest_django"]

from django_ai_sdk.rags.schemas import RagDocument


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client."""
    from unittest.mock import AsyncMock, MagicMock

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_storage_adapter():
    """Mock storage adapter."""
    from unittest.mock import AsyncMock, MagicMock

    adapter = MagicMock()
    adapter.store_chat_message = AsyncMock(return_value="msg_test_123")
    adapter.get_messages = AsyncMock(return_value=[])
    adapter.storage_callback = AsyncMock()
    return adapter


@pytest.fixture
def mock_user():
    """Mock user with predictable pk and is_authenticated."""
    from unittest.mock import MagicMock

    user = MagicMock()
    user.pk = "user-1"
    user.is_authenticated = True
    return user


@pytest.fixture
def mock_assistants_registry():
    """Patch the global assistant registry at both import paths."""
    from unittest.mock import MagicMock, patch
    from django_ai_sdk.tests.mocks.assistant import create_assistant_mock

    assistant = create_assistant_mock()
    with \
        patch("django_ai_sdk.assistants.registry.registry") as reg, \
        patch("django_ai_sdk.assistants.services.registry", reg):
        reg.get = MagicMock(return_value=assistant)
        reg.all = MagicMock(return_value={"test-assistant": assistant})
        yield reg


@pytest.fixture
def assistant_permissions(mock_assistants_registry):
    """Fixture returning a setter that changes the registry assistant's permissions.

    Usage::

        def test_something(self, assistant_permissions):
            assistant_permissions(DenyAll)        # single
            assistant_permissions(IsOwner, DenyAll)  # multiple
    """
    def set_perms(*perms):
        mock_assistants_registry.get.return_value.permissions = list(perms)
    return set_perms


@pytest.fixture
def mock_storage_adapter_registry():
    """Patch the StorageAdapterRegistry."""
    from unittest.mock import MagicMock, patch

    with patch("django_ai_sdk.storage.services.StorageAdapterRegistry") as sr:
        sr.get_all_adapters = MagicMock(return_value=[])
        yield sr


@pytest.fixture
def sample_thread_id():
    """Sample thread ID."""
    return "thread_test_12345"


@pytest.fixture
def sample_user_message():
    """Sample user message."""
    return {"role": "user", "content": "Tell me a joke"}


@pytest.fixture
def sample_assistant_response():
    """Sample assistant response."""
    return {"role": "assistant", "content": "Here's a joke for you!"}


@pytest.fixture
def rag_document_factory():
    """Factory for creating RagDocument objects."""
    def create(**kwargs):
        defaults = {
            "id": "test-doc-1",
            "content": "Test content for RAG",
            "title": "Test Document",
            "metadata": {"source": "test"},
        }
        defaults.update(kwargs)
        return RagDocument(**defaults)
    return create


@pytest.fixture
def sample_rag_documents():
    """Create a list of sample RagDocuments."""
    return [
        RagDocument(
            id=f"doc-{i}",
            content=f"Content for document {i}",
            title=f"Document {i}",
        )
        for i in range(5)
    ]
