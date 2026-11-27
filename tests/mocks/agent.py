from unittest.mock import AsyncMock, MagicMock, patch

from django_ai_sdk.permissions import AllowAll
from django_ai_sdk.storage.memory import MemoryStorageAdapter


def create_agent_mock(
    agent_id: str = "test-agent",
    name: str = "Test Agent",
    model: str = "gpt-4",
    permissions: list | None = None,
    storage_adapter=MemoryStorageAdapter,
    **attrs,
) -> MagicMock:
    if permissions is None:
        permissions = [AllowAll]

    agent = MagicMock()
    agent.id = agent_id
    agent.name = name
    agent.model = model
    agent.storage_adapter = storage_adapter
    agent.permissions = permissions
    agent.history = AsyncMock(
        return_value=MagicMock(
            thread={"id": "thread-1", "title": "Test"}, messages=[]
        )
    )
    for k, v in attrs.items():
        setattr(agent, k, v)
    return agent


def create_mock_adapter_class(get_thread=None):
    adapter_cls = MagicMock()
    adapter_cls.__name__ = "MockAdapter"
    adapter_cls.get_thread = AsyncMock(return_value=get_thread)
    return adapter_cls


def mock_agent_memories(slugs):
    from django_ai_sdk.agents.services import registry

    mock_agent = MagicMock()
    mock_agent.memories = slugs
    return patch.object(registry, "get", return_value=mock_agent)
