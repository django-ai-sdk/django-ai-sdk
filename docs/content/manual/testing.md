---
title: Testing
type: docs
weight: 114
---

How the SDK test suite is structured and how to write tests for your agents.

{{< callout type="info" >}}
Fixtures, factories, and mock builders are documented on [Test Tooling](../test-tooling/).
{{< /callout >}}

## Strategy

![Testing Pyramid](/images/graphs/testing_pyramid.png)

| Component | Test type | Focus |
| --- | --- | --- |
| **Agent** | Integration | `get_pipeline_adapter()`, storage flow, `as_view()` |
| **Stream / Run** | Integration | Event emission, ID generation, tool calls |
| **Storage** | Integration | CRUD, rating, history, `ThreadService` |
| **Protocol** | Unit | Message conversion, event → part mapping |
| **RAG** | Unit + Integration | Warmup, caching, retrieval, concurrency |

Principles:

1. **Mock external services**: never call a real model provider or a real Qdrant in tests.
2. **Use MemoryStorage**: fast, isolated; DB tests opt in with `@pytest.mark.django_db`.
3. **Test event flow**: verify the event sequence of `Stream.stream()`.
4. **Test ID consistency**: the `message_id` from `MessageStartEvent` must appear in storage.
5. **Async everywhere**: `asyncio_mode = "auto"` lets pytest detect async tests.

## Setup

```bash
make setup      # demo/runtime extras only
make setup-all  # every extra, including the torch-based transformers stack
```

Pytest config (`pyproject.toml`):

```toml
[tool.pytest.ini_options]
DJANGO_SETTINGS_MODULE = "demo.settings"
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

## Running Tests

```bash
# Everything (uses the demo Django settings)
make test
# equivalent: PYTHONPATH=demo uv run pytest tests -v

# A single file
PYTHONPATH=demo uv run pytest tests/integration/test_protocol_handlers.py -v

# A single test
PYTHONPATH=demo uv run pytest tests/unit/test_thread_service.py::test_something -v

# Debug
PYTHONPATH=demo uv run pytest -x
PYTHONPATH=demo uv run pytest --pdb
```

The `PYTHONPATH=demo` prefix makes `demo.settings` importable (it lives in the `demo` directory).

## Test Structure

```
tests/
├── conftest.py          # shared fixtures, session event loop
├── factories/           # polyfactory-based model factories
│   ├── db.py            # Django ORM factories
│   └── schemas.py       # Pydantic factories (ChatMessageFactory, ...)
├── mocks/               # reusable mock builders
│   ├── agent.py         # create_agent_mock()
│   ├── integrations.py
│   ├── permissions.py   # permission patch helpers
│   ├── registry.py
│   └── storage.py       # mock storage adapters & threads
├── unit/                # component tests (no DB)
│   ├── test_haystack_bm25.py
│   ├── test_rag_provider_concurrency.py
│   ├── test_thread_service.py
│   └── ...
└── integration/         # end-to-end component tests
    ├── test_base_agent.py
    ├── test_protocol_handlers.py
    └── test_storage_adapter.py
```

## Common Patterns

### Verify the event sequence

```python
async def test_event_sequence():
    events = [e async for e in stream.stream([ChatMessage(role="user", content="Hi")])]
    assert isinstance(events[0], MessageStartEvent)
    assert isinstance(events[-2], MessageEndEvent)
    assert isinstance(events[-1], StreamEndEvent)
```

### Verify ID consistency

```python
async def test_id_consistency():
    message_ids = []
    async for event in stream.stream([ChatMessage(role="user", content="Hi")]):
        if isinstance(event, MessageStartEvent):
            message_ids.append(event.message_id)

    stored = [m for m in await storage.get_messages() if m.id == message_ids[0]]
    assert stored, "stored message shares the SSE message_id"
```

### Verify error handling

```python
async def test_stream_error():
    # Generator raises mid-stream → ErrorEvent emitted, stream still ends
    events = [e async for e in stream.stream([ChatMessage(role="user", content="Hi")])]
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert isinstance(events[-1], StreamEndEvent)
```

### RAG provider concurrency

`test_rag_provider_concurrency.py` verifies that concurrent warmups for the same memory key are serialized by the per-key lock and produce a single cached instance:

```python
async def test_concurrent_warmup_single_instance():
    provider = RAGProvider()
    results = await asyncio.gather(*[
        provider.warmup(agent, "mem-1") for _ in range(5)
    ])
    assert provider.get_cached_rag_instance(agent, "mem-1") is not None
```

### Permission-aware thread services

`ThreadService` methods raise `PermissionDenied` when the user lacks access:

```python
async def test_denied(user_without_perms):
    with pytest.raises(PermissionDenied):
        await ThreadService.get_thread(thread_id, user=user_without_perms)
```

Next: [Test Tooling](../test-tooling/), fixtures, factories, and mocks.
