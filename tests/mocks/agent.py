"""
Mock agent factory.

Agents have side effects (LLM calls, registry registration, storage I/O),
so we use controlled MagicMock instances in unit tests instead of real subclasses.
"""

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
    """Create a controlled MagicMock representing a registered agent.

    Defaults to AllowAll permissions and MemoryStorageAdapter.
    Pass extra keyword arguments to override or add attributes.
    """
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
    """Create a mock storage adapter *class* (not instance).

    The returned mock looks like a storage adapter class with
    an async ``get_thread`` classmethod.
    """
    adapter_cls = MagicMock()
    adapter_cls.__name__ = "MockAdapter"
    adapter_cls.get_thread = AsyncMock(return_value=get_thread)
    return adapter_cls


def mock_agent_memories(slugs):
    """Return a ``patch.object`` context manager that mocks
    ``registry.get`` to return an agent with the given memory slugs.

    Usage::

        with mock_agent_memories([mem1.slug]):
            result = await MemoryService.get_agent_memories("test-asst")
    """
    from django_ai_sdk.agents.services import registry

    mock_agent = MagicMock()
    mock_agent.memories = slugs
    return patch.object(registry, "get", return_value=mock_agent)
