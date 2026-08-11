---
title: Test Tooling
type: docs
weight: 115
---

The reusable test infrastructure: shared fixtures, factories, and mock builders in `tests/`.

## Shared Fixtures

`tests/conftest.py` provides session-scoped fixtures used across the suite:

```python
import pytest

@pytest.fixture
def mock_storage_adapter():
    """Mock storage adapter with async methods."""
    from unittest.mock import AsyncMock, MagicMock
    adapter = MagicMock()
    adapter.store_chat_message = AsyncMock(return_value="msg_test_123")
    adapter.get_messages = AsyncMock(return_value=[])
    adapter.storage_callback = AsyncMock()
    return adapter

@pytest.fixture
def mock_user():
    """Mock user with predictable pk and is_authenticated."""
    ...

@pytest.fixture
def mock_agents_registry():
    """Patch the global agent registry at both import paths."""
    ...
```

## Factories

Factories live in `tests/factories/`. Schema factories (`schemas.py`) use polyfactory; ORM factories (`db.py`) use factory_boy:

```python
from tests.factories.schemas import ChatMessageFactory, ThreadInfoFactory, chat_message

msg = ChatMessageFactory.build(role="user", content="Hi")
thread = ThreadInfoFactory.build(user_id="user-1", agent_id="test-agent")

payload = chat_message(role="user", text="Hello", message_id="msg-1")
# -> Message(role="user", parts=[MessagePart(type="text", text="Hello")], ...)
```

Django ORM factories live in `tests/factories/db.py`.

## Mock Builders

Reusable helpers in `tests/mocks/` keep tests declarative:

```python
from tests.mocks.agent import create_agent_mock
from tests.mocks.storage import mock_get_storage, setup_thread_adapter

# Agent that looks registered but never calls a provider
agent = create_agent_mock()

# Storage adapter class registered on the registry with a ThreadInfo
setup_thread_adapter(registry, user_id="user-1", agent_id="test-agent")

# A patched get_storage / rate_message
with mock_get_storage(method="rate_message", return_value=True):
    ...
```

```python
from tests.mocks.permissions import thread_permissions

# Patch permission resolution for a test
with thread_permissions("apps.agents.permissions.DemoThreadPermission"):
    ...
```

### Mocking the model generator

Never construct a real `OpenAIChatGenerator`. Use a stub object with the shape `Stream`/`Run` expect:

```python
class FakeGenerator:
    streaming_callback = None

    def run(self, *, messages, generation_kwargs=None):
        return {"replies": [FakeReply(text='{"title": "Test"}')]}
```

### Mocking storage adapters

```python
from unittest.mock import AsyncMock, MagicMock

storage = MagicMock()
storage.store_chat_message = AsyncMock(return_value="msg_1")
storage.get_messages = AsyncMock(return_value=[])
```

Next: [CLI](../cli/), management command implementation.
