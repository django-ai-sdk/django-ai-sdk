---
title: Testing
sidebar:
  order: 114
---

This page describes the structure of the SDK test suite and how to write tests for your agents.

:::note
[Test Tooling](test-tooling.md) documents the fixtures, factories and mock builders.
:::

## Strategy

::image[Testing Pyramid]{src="../images/graphs/testing_pyramid.svg" caption="The testing pyramid."}

| Component | Test type | Focus |
| --- | --- | --- |
| Agent | Integration | `get_pipeline_adapter()`, storage flow, `as_view()` |
| Stream / Run | Integration | Event emission, ID generation, tool calls |
| Storage | Integration | CRUD, rating, history, `ThreadService` |
| Protocol | Unit | Message conversion, event → part mapping |
| RAG | Unit + Integration | Warmup, caching, retrieval, concurrency |

Principles:

1. **Mock external services.** Tests never call a real model provider or a real Qdrant.
2. **Use MemoryStorage.** It is fast and isolated. DB tests opt in with `@pytest.mark.django_db`.
3. **Test the event flow.** Check the event sequence of `Stream.stream()`.
4. **Test ID consistency.** The `message_id` from `MessageStartEvent` must appear in storage.
5. **Write async tests.** With `asyncio_mode = "auto"`, pytest detects async tests.

## Setup

```bash
make setup      # runtime + demo extras only
make setup-all  # every extra, including the torch-based transformers stack
```

The pytest configuration lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
DJANGO_SETTINGS_MODULE = "tests.settings"
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["."]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

## Running tests

```bash
# Everything
make test
# equivalent: uv run pytest tests -v

# A single file
uv run pytest tests/integration/test_protocol_handlers.py -v

# A single test
uv run pytest tests/unit/test_thread_service.py::test_something -v

# Debug
uv run pytest -x
uv run pytest --pdb

# The demo projects' own tests (each demo in demos/ runs with its own settings)
make test-demos
```

`tests/settings.py` is a minimal Django project. It has the SDK apps, an in-memory SQLite
database, and `tests.testapp` with an email-only `User` that has no `username` field, like many
host projects. It never imports from `demos/`. Tests that depend on a specific
permission class set it themselves, with the helpers in `tests/mocks/permissions.py`.

Set `DATABASE_URL` to run the suite on PostgreSQL instead. Tests that race
connections against each other, for claims and locks, run only on PostgreSQL and skip on SQLite:

```bash
DATABASE_URL=postgres://postgres:pg@127.0.0.1:5432/postgres \
    uv run --with "psycopg[binary]" pytest tests
```

## Test structure

:::filetree
- tests/
  - settings.py minimal Django settings for the suite
  - urls.py
  - testapp/ email-only User model
  - conftest.py shared fixtures, session event loop
  - factories/ polyfactory-based model factories
    - db.py Django ORM factories
    - schemas.py Pydantic factories (ChatMessageFactory, ...)
  - mocks/ reusable mock builders
    - agent.py `create_agent_mock()`
    - integrations.py
    - permissions.py permission patch helpers
    - registry.py
    - storage.py mock storage adapters & threads
  - unit/ component tests (no DB)
    - `test_haystack_bm25.py`
    - `test_rag_provider_concurrency.py`
    - `test_thread_service.py`
    - …
  - integration/ end-to-end component tests
    - `test_base_agent.py`
    - `test_protocol_handlers.py`
    - `test_storage_adapter.py`
:::

## Common patterns

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

`test_rag_provider_concurrency.py` checks that the per-key lock serializes concurrent warmups for the same memory key, and that they produce a single cached instance:

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

Next: [Test Tooling](test-tooling.md), for fixtures, factories and mocks.
