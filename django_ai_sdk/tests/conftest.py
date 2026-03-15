"""
Test configuration and shared fixtures for django-ai-sdk tests.
"""

import asyncio
import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator

# Django configuration for tests
pytest_plugins = ["pytest_django"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_openai_client():
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
