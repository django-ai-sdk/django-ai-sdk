import asyncio
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from collections.abc import AsyncGenerator

pytest_plugins = ["pytest_django"]

from django_ai_sdk.rags.schemas import RagDocument


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def postgres_only():
    """Skip unless the tests run on PostgreSQL: SQLite serialises every writer."""
    from django.db import connection

    if connection.vendor != "postgresql":
        pytest.skip("needs PostgreSQL: set DATABASE_URL")


@pytest.fixture
def mock_storage_adapter():
    from unittest.mock import AsyncMock, MagicMock

    adapter = MagicMock()
    adapter.store_chat_message = AsyncMock(return_value="msg_test_123")
    adapter.get_messages = AsyncMock(return_value=[])
    adapter.storage_callback = AsyncMock()
    return adapter


@pytest.fixture
def mock_user():
    from unittest.mock import MagicMock

    user = MagicMock()
    user.pk = "user-1"
    user.is_authenticated = True
    return user


@pytest.fixture
def mock_agents_registry():
    from unittest.mock import MagicMock, patch
    from tests.mocks.agent import create_agent_mock

    agent = create_agent_mock()
    with (
        patch("django_ai_sdk.agents.registry.registry") as reg,
        patch("django_ai_sdk.agents.services.registry", reg),
    ):
        reg.get = MagicMock(return_value=agent)
        reg.all = MagicMock(return_value={"test-agent": agent})
        yield reg


@pytest.fixture
def agent_permissions(mock_agents_registry):
    def set_perms(*perms):
        mock_agents_registry.get.return_value.permissions = list(perms)

    return set_perms


@pytest.fixture
def mock_storage_adapter_registry():
    from unittest.mock import MagicMock, patch

    with patch("django_ai_sdk.storage.services.StorageAdapterRegistry") as sr:
        sr.get_all_adapters = MagicMock(return_value=[])
        yield sr


@pytest.fixture
def sample_thread_id():
    return "thread_test_12345"


@pytest.fixture
def sample_user_message():
    return {"role": "user", "content": "Tell me a joke"}


@pytest.fixture
def sample_agent_response():
    return {"role": "assistant", "content": "Here's a joke for you!"}


@pytest.fixture
def rag_document_factory():
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
    return [
        RagDocument(
            id=f"doc-{i}",
            content=f"Content for document {i}",
            title=f"Document {i}",
        )
        for i in range(5)
    ]


@pytest.fixture
def installs_as_is():
    installed = MagicMock(side_effect=lambda agent: agent)
    with patch("django_ai_sdk.oversight.plugin.apply_oversight", installed):
        yield installed


@pytest.fixture(autouse=True)
def inline_workflow_runs(request, monkeypatch):
    if "dbos" not in request.fixturenames:
        from django_ai_sdk.durable.testing import run_inline

        run_inline(monkeypatch)
