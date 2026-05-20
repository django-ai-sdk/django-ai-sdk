---
title: Testing Guide
type: docs
weight: 105
---

Complete guide to testing your AI assistants.

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Setup](#test-setup)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [Mocking External APIs](#mocking-external-apis)
6. [Test Factories](#test-factories)
7. [Common Patterns](#common-patterns)
8. [Running Tests](#running-tests)

---

## Testing Strategy

### Test Pyramid for AI SDK

```
        /\
       /  \
      / E2E \         <- Full assistant flow
     /________\
    /          \
   / Integration \     <- Adapter + Storage + Protocol
  /________________\
 /                  \
/      Unit Tests    \  <- Individual components
/______________________\
```

![Testing Pyramid](/images/graphs/testing_pyramid.png)

### What to Test

| Component | Test Type | Focus |
|-----------|-----------|-------|
| **Assistant** | Integration | `get_pipeline_adapter()`, storage flow |
| **Adapter** | Integration | Event emission, ID generation, streaming |
| **Storage** | Integration | CRUD operations, rating, history |
| **Protocol** | Unit | Message conversion, event handling |
| **RAG** | Integration | Retrieval, caching, context injection |

### Testing Principles

1. **Mock External APIs** - Don't call OpenAI/Haystack in tests
2. **Use Memory Storage** - Fast, isolated, no database setup
3. **Test Event Flow** - Verify correct event sequence
4. **Test ID Consistency** - Same ID from generation → storage
5. **Async Everything** - All tests are async

---

## Test Setup

### Dependencies

Add to `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-django>=4.7.0",
    "pytest-mock>=3.12.0",
    "factory-boy>=3.3.0",
]
```

Install:

```bash
pip install -e ".[dev]"
```

### Configuration

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
DJANGO_SETTINGS_MODULE = "demo.settings"
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
testpaths = ["django_ai_sdk/tests"]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]
```

### Fixtures

`conftest.py`:

```python
import pytest
import pytest_asyncio

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    from unittest.mock import MagicMock, AsyncMock
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client

@pytest.fixture
def sample_thread_id():
    """Valid UUID for testing."""
    import uuid
    return str(uuid.uuid4())
```

---

## Unit Testing

### Testing ChatMessage

```python
import pytest
from django_ai_sdk.common import ChatMessage

class TestChatMessage:
    def test_create_basic_message(self):
        """Test creating a simple message."""
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.id == ""  # Not set yet
    
    def test_message_with_metadata(self):
        """Test message with all fields."""
        import uuid
        msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content="Hi there!",
            model="gpt-4o-mini",
            finish_reason="stop",
            tool_calls=[],
            sources=[],
        )
        assert msg.role == "assistant"
        assert msg.model == "gpt-4o-mini"
        assert len(msg.tool_calls) == 0
    
    def test_message_duration(self):
        """Test duration calculation."""
        import time
        msg = ChatMessage(
            role="assistant",
            content="Hello",
            started_at=time.time(),
            completed_at=time.time() + 1.5,
        )
        assert msg.duration >= 1500
```

### Testing StreamWriter

```python
import pytest
import uuid
from django_ai_sdk.common import StreamWriter, MessageChunk

class TestStreamWriter:
    def test_stream_writer_creates_message(self):
        """Test StreamWriter creates ChatMessage."""
        message_id = str(uuid.uuid4())
        writer = StreamWriter(
            adapter_type="test",
            message_id=message_id,
            model="gpt-4",
            role="assistant",
        )
        
        assert writer.message.id == message_id
        assert writer.message.role == "assistant"
        assert writer.message.model == "gpt-4"
    
    def test_add_text_chunk(self):
        """Test adding text chunks."""
        writer = StreamWriter(
            adapter_type="test",
            message_id=str(uuid.uuid4()),
        )
        
        # Add chunks
        writer.add_chunk(MessageChunk(type="text", content="Hello "))
        writer.add_chunk(MessageChunk(type="text", content="world!"))
        
        assert writer.message.content == "Hello world!"
```

---

## Integration Testing

### Testing Assistant

```python
import pytest
import pytest_asyncio
import uuid
from django_ai_sdk import Assistant
from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore
from django_ai_sdk.protocols.vercel import VercelProtocolHandler

@pytest.mark.django_db
class TestAssistant:
    @pytest_asyncio.fixture
    async def assistant(self):
        """Create test assistant."""
        
        class TestAssistant(Assistant):
            name = "test_assistant"
            model = "gpt-4o-mini"
            instructions = ["You are a test assistant"]
            protocol = VercelProtocolHandler
            storage_adapter = MemoryStorageAdapter
            
            async def get_pipeline_adapter(self, thread_id=None):
                from django_ai_sdk.adapters.openai import OpenAIAdapter
                from unittest.mock import MagicMock
                
                return OpenAIAdapter(
                    client=MagicMock(),
                    model=self.model,
                    instructions=self.get_instructions(),
                    storage_adapter=await self.get_storage_adapter(thread_id),
                )
        
        return TestAssistant()
    
    @pytest_asyncio.fixture
    async def thread_id(self):
        """Create test thread ID."""
        return str(uuid.uuid4())
    
    @pytest.mark.asyncio
    async def test_assistant_creates_storage(self, assistant, thread_id):
        """Test assistant creates storage adapter."""
        storage = await assistant.get_storage_adapter(thread_id)
        
        assert storage is not None
        assert storage.thread_id == thread_id
    
    @pytest.mark.asyncio
    async def test_message_storage_flow(self, assistant, thread_id):
        """Test complete storage flow."""
        from django_ai_sdk.common import ChatMessage
        
        storage = await assistant.get_storage_adapter(thread_id)
        
        # Create thread first
        MemoryStore.create_thread(thread_id)
        
        # Store user message
        user_msg = ChatMessage(role="user", content="Hello!")
        message_id = await storage.store_chat_message(user_msg)
        
        # Verify
        history = await storage.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Hello!"
```

### Testing Adapter

```python
import pytest
from unittest.mock import MagicMock, AsyncMock
from django_ai_sdk.adapters.openai import OpenAIAdapter
from django_ai_sdk.events import MessageStartEvent, TextChunkEvent, MessageEndEvent

@pytest.mark.asyncio
async def test_openai_adapter_generates_id(mock_openai_client):
    """Test that adapter generates UUID once."""
    
    # Mock response
    chunk = MagicMock()
    chunk.choices = [MagicMock(
        delta=MagicMock(content="Hello"),
        finish_reason=None
    )]
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].delta.tool_calls = None
    
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=async_generator([chunk])
    )
    
    # Create adapter
    adapter = OpenAIAdapter(client=mock_openai_client)
    
    # Stream
    events = []
    async for event in adapter.stream([]):
        events.append(event)
    
    # Verify single ID
    start_events = [e for e in events if isinstance(e, MessageStartEvent)]
    assert len(start_events) == 1
    
    # Verify valid UUID
    import uuid
    uuid.UUID(start_events[0].message_id)  # Should not raise

@pytest.mark.asyncio
async def test_adapter_emits_text_events(mock_openai_client):
    """Test adapter emits correct text events."""
    
    # Mock multiple chunks
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="world!"), finish_reason="stop")]),
    ]
    
    for chunk in chunks:
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.tool_calls = None
    
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=async_generator(chunks)
    )
    
    adapter = OpenAIAdapter(client=mock_openai_client)
    
    # Collect events
    text_events = []
    async for event in adapter.stream([]):
        if isinstance(event, TextChunkEvent):
            text_events.append(event)
    
    # Verify text content
    assert len(text_events) == 2
    assert text_events[0].content == "Hello "
    assert text_events[1].content == "world!"

# Helper
async def async_generator(items):
    """Convert list to async generator."""
    for item in items:
        yield item
```

### Testing Storage

```python
import pytest
import uuid
from django_ai_sdk.storage.memory import MemoryStorageAdapter, MemoryStore
from django_ai_sdk.common import ChatMessage

@pytest.mark.asyncio
async def test_storage_requires_thread_creation():
    """Test that storage requires explicit thread creation."""
    storage = MemoryStorageAdapter(thread_id=str(uuid.uuid4()))
    
    with pytest.raises(ValueError, match="not found in memory store"):
        await storage.store_chat_message(ChatMessage(role="user", content="Hello"))

@pytest.mark.asyncio
async def test_storage_rating_flow():
    """Test message rating."""
    thread_id = str(uuid.uuid4())
    
    # Setup
    MemoryStore.create_thread(thread_id)
    storage = MemoryStorageAdapter(thread_id)
    
    # Store message
    msg = ChatMessage(role="assistant", content="Hello!")
    message_id = await storage.store_chat_message(msg)
    
    # Rate as good
    success = await storage.rate_message(message_id, rating=1)
    assert success is True
    
    # Verify via MemoryStore
    stored = MemoryStore.get_messages(thread_id)
    assert stored[0].rating == 1

@pytest.mark.asyncio
async def test_conversation_history():
    """Test retrieving conversation history."""
    thread_id = str(uuid.uuid4())
    
    # Setup
    MemoryStore.create_thread(thread_id)
    storage = MemoryStorageAdapter(thread_id)
    
    # Add messages
    await storage.store_chat_message(ChatMessage(role="user", content="Q1"))
    await storage.store_chat_message(ChatMessage(role="assistant", content="A1"))
    await storage.store_chat_message(ChatMessage(role="user", content="Q2"))
    
    # Retrieve
    history = await storage.get_history()
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert history[2].role == "user"
```

---

## Mocking External APIs

### Mocking OpenAI

```python
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.fixture
def mock_openai_stream():
    """Create mock OpenAI streaming response."""
    def _create_mock(chunks_data):
        """
        chunks_data: List of dicts with 'content' and optional 'finish_reason'
        """
        chunks = []
        for data in chunks_data:
            chunk = MagicMock()
            chunk.choices = [MagicMock(
                delta=MagicMock(
                    content=data.get("content", ""),
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=data.get("finish_reason")
            )]
            chunks.append(chunk)
        
        async def mock_stream():
            for chunk in chunks:
                yield chunk
        
        return mock_stream
    
    return _create_mock

@pytest.mark.asyncio
async def test_with_mocked_openai(mock_openai_stream):
    """Test with fully mocked OpenAI."""
    from openai import AsyncOpenAI
    
    # Create mock client
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    
    # Set up streaming response
    mock_stream = mock_openai_stream([
        {"content": "Hello "},
        {"content": "world!", "finish_reason": "stop"},
    ])
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
    
    # Use in adapter
    adapter = OpenAIAdapter(client=mock_client)
    
    events = []
    async for event in adapter.stream([]):
        events.append(event)
    
    # Verify
    text_events = [e for e in events if hasattr(e, 'content') and e.event_type == "text_chunk"]
    assert len(text_events) == 2
```

### Mocking Haystack

```python
from unittest.mock import MagicMock, patch
from haystack.dataclasses import StreamingChunk

@pytest.fixture
def mock_haystack_pipeline():
    """Mock Haystack pipeline."""
    pipeline = MagicMock()
    
    # Mock run method
    async def mock_run(*args, **kwargs):
        return {
            "generator": {
                "replies": ["Hello from Haystack!"]
            }
        }
    
    pipeline.run = mock_run
    return pipeline

@pytest.mark.asyncio
async def test_haystack_adapter(mock_haystack_pipeline):
    """Test with mocked Haystack pipeline."""
    from django_ai_sdk.adapters.haystack import HaystackAdapter
    
    # Mock generator
    mock_generator = MagicMock()
    
    adapter = HaystackAdapter(
        pipeline=mock_haystack_pipeline,
        generator_component=mock_generator,
    )
    
    # Test streaming
    events = []
    async for event in adapter.stream([]):
        events.append(event)
    
    # Verify we got events
    assert len(events) > 0
```

---

## Test Factories

### Message Factory

```python
import factory
import uuid
from django_ai_sdk.common import ChatMessage

class ChatMessageFactory(factory.Factory):
    """Factory for creating ChatMessage objects."""
    
    class Meta:
        model = ChatMessage
    
    id = factory.LazyAttribute(lambda _: str(uuid.uuid4()))
    role = "assistant"
    content = factory.Faker("paragraph", nb_sentences=3)
    adapter_type = "openai"
    model = "gpt-4o-mini"
    tool_calls = []
    sources = []
    
    class Params:
        """Factory traits for different message types."""
        
        user = factory.Trait(
            role="user",
            adapter_type="",
            model="",
            content=factory.Faker("sentence", nb_words=6)
        )
        
        assistant = factory.Trait(
            role="assistant",
            content=factory.Faker("paragraph")
        )
        
        system = factory.Trait(
            role="system",
            content=factory.Faker("sentence", nb_words=10)
        )

# Usage
user_msg = ChatMessageFactory.build(user=True, content="Hello!")
assistant_msg = ChatMessageFactory.build(assistant=True)
```

---

## Common Patterns

### Pattern 1: Testing Event Sequence

```python
@pytest.mark.asyncio
async def test_event_sequence(mock_openai_client):
    """Verify correct event order."""
    
    # Setup mock...
    adapter = OpenAIAdapter(client=mock_openai_client)
    
    # Collect all events
    events = []
    async for event in adapter.stream([]):
        events.append(event)
    
    # Verify sequence
    assert events[0].event_type == "message_start"
    assert events[1].event_type in ["text_chunk", "reasoning_chunk"]
    assert events[-2].event_type == "message_end"
    assert events[-1].event_type == "stream_end"
```

### Pattern 2: Testing ID Consistency

```python
@pytest.mark.asyncio
async def test_id_consistency(mock_openai_client, mock_storage):
    """Verify same ID through entire flow."""
    
    adapter = OpenAIAdapter(
        client=mock_openai_client,
        storage_adapter=mock_storage,
    )
    
    # Get message ID from stream
    message_id = None
    async for event in adapter.stream([]):
        if event.event_type == "message_start":
            message_id = event.message_id
            break
    
    # Verify ID in storage
    history = await mock_storage.get_history()
    assert history[0].id == message_id
```

### Pattern 3: Testing Error Handling

```python
@pytest.mark.asyncio
async def test_adapter_error_handling():
    """Test adapter handles API errors."""
    
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error")
    )
    
    adapter = OpenAIAdapter(client=mock_client)
    
    # Should emit error event, not crash
    events = []
    async for event in adapter.stream([]):
        events.append(event)
    
    error_events = [e for e in events if e.event_type == "error"]
    assert len(error_events) == 1
    assert "API Error" in error_events[0].error_message
```

### Pattern 4: Testing Tool Calls

```python
@pytest.mark.asyncio
async def test_tool_call_flow(mock_openai_client):
    """Test complete tool call sequence."""
    
    # Mock tool call response
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function = MagicMock()
    tool_call.function.name = "search"
    tool_call.function.arguments = '{"query": "test"}'
    
    chunk = MagicMock()
    chunk.choices = [MagicMock(
        delta=MagicMock(content="", tool_calls=[tool_call]),
        finish_reason=None
    )]
    
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=async_generator([chunk])
    )
    
    adapter = OpenAIAdapter(client=mock_openai_client)
    
    # Verify tool events
    events = []
    async for event in adapter.stream([]):
        events.append(event)
    
    tool_start = [e for e in events if e.event_type == "tool_call_start"]
    tool_input = [e for e in events if e.event_type == "tool_input_complete"]
    
    assert len(tool_start) == 1
    assert tool_start[0].tool_name == "search"
    assert len(tool_input) == 1
    assert tool_input[0].tool_input == {"query": "test"}
```

---

## Running Tests

### Run All Tests

```bash
# With pytest
PYTHONPATH=demo pytest django_ai_sdk/tests -v

# With coverage
PYTHONPATH=demo pytest --cov=django_ai_sdk --cov-report=html

# With asyncio mode explicit
PYTHONPATH=demo pytest --asyncio-mode=auto
```

### Run Specific Test Files

```bash
# BaseAssistant tests
PYTHONPATH=demo pytest django_ai_sdk/tests/integration/test_base_assistant.py -v

# OpenAI adapter tests
PYTHONPATH=demo pytest django_ai_sdk/tests/integration/test_openai_adapter.py -v

# Storage tests
PYTHONPATH=demo pytest django_ai_sdk/tests/integration/test_storage_adapter.py -v

# Protocol handler tests
PYTHONPATH=demo pytest django_ai_sdk/tests/integration/test_protocol_handlers.py -v
```

### Run Specific Tests

```bash
# Single test
PYTHONPATH=demo pytest django_ai_sdk/tests/integration/test_base_assistant.py::TestBaseAssistant::test_assistant_creates_thread -v

# All tests in a class
PYTHONPATH=demo pytest django_ai_sdk/tests/integration/test_base_assistant.py::TestBaseAssistant -v
```

### Debug Mode

```bash
# Stop on first failure
PYTHONPATH=demo pytest -x

# Show local variables on failure
PYTHONPATH=demo pytest -v --tb=long

# Enter debugger on failure
PYTHONPATH=demo pytest --pdb
```

---

## Next Steps

- Review your test coverage: `pytest --cov=django_ai_sdk --cov-report=term-missing`
- Add tests for custom assistants
- Mock external APIs consistently
- Test error scenarios thoroughly

---

## Quick Reference

### Test File Template

```python
"""
Tests for {component}.
"""
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock

from django_ai_sdk.{module} import {Component}


class Test{Component}:
    """Test suite for {Component}."""
    
    @pytest_asyncio.fixture
    async def {fixture_name}(self):
        """Create test {component}."""
        return {Component}(...)
    
    @pytest.mark.asyncio
    async def test_{feature}(self, {fixture_name}):
        """Test {feature}."""
        # Arrange
        
        # Act
        result = await {fixture_name}.{method}()
        
        # Assert
        assert result == expected
```

### Common Assertions

```python
# Event assertions
assert event.event_type == "text_chunk"
assert event.content == "Hello"
assert isinstance(event, TextChunkEvent)

# UUID assertions
import uuid
uuid.UUID(message_id)  # Validates format

# Storage assertions
history = await storage.get_history()
assert len(history) == expected_count
assert history[0].role == "user"

# Exception assertions
with pytest.raises(ValueError, match="expected message"):
    await failing_operation()
```

---

## References

- [Architecture Guide](architecture/) - Core concepts
- [RAG Guide](rag/) - RAG testing
- [Adapters](adapters/) - Adapter patterns
- [Storage](storage/) - Storage testing
