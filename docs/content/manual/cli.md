---
title: CLI Implementation
type: docs
weight: 116
---

Internals of the `warmup_rag` management command. User-facing usage lives in the [CLI guide](/cli/).

File: `django_ai_sdk/management/commands/warmup_rag.py`. The SDK also ships `run_automations` (`django_ai_sdk/management/commands/run_automations.py`, documented under [Automations](/manual/automations/#the-tick)) and `refresh_integrations` (`django_ai_sdk/integrations/mcp/management/commands/refresh_integrations.py`) for MCP integrations.

## Overview

`warmup_rag` pre-builds RAG indexes for all registered agents and memories so the first chat request doesn't pay the warmup cost. It is a standard Django `BaseCommand` that runs async work via `asyncio.run()`.

## Arguments

| Option | Type | Description |
| --- | --- | --- |
| `--agent` | `str` | Only warm up a specific agent (matched by **class name**) |
| `--memory` | `str` | Only warm up a specific memory ID |
| `--force-rebuild` | flag | Rebuild indexes from scratch, deleting existing ones |

## Implementation Details

### Resolving memories

```python
from django_ai_sdk.memories.models import Memory

all_memory_ids = [str(m.id) for m in Memory.objects.all().only("id")]

if memory_filter:
    if memory_filter not in all_memory_ids:
        raise CommandError(f"Memory '{memory_filter}' not found")
    memory_ids = [memory_filter]
else:
    memory_ids = all_memory_ids
```

With no `--memory`, every `Memory` row is warmed. An unknown memory ID raises `CommandError`.

### Ensuring the registry is ready

```python
from django_ai_sdk.agents.registry import registry

try:
    agents = registry.all()
except RuntimeError:
    registry.setup(instantiate=True)
    agents = registry.all()
```

### Filtering by agent

`--agent` matches the **class name**, not the UUID:

```python
if agent_filter:
    filtered = {
        aid: inst for aid, inst in agents.items()
        if inst.__class__.__name__ == agent_filter
    }
    if not filtered:
        raise CommandError(f"Agent '{agent_filter}' not found in registry")
    agents = filtered
```

### Warming up

```python
for aid, inst in agents.items():
    if inst.rag_provider is None:
        self.stdout.write(self.style.WARNING(f"[{name}] No RAG provider configured, skipping"))
        continue

    for mid in memory_ids:
        try:
            await inst.rag_provider.warmup(inst, mid, force_rebuild=force_rebuild)
            self.stdout.write(f"  [{name}] Warmed memory {mid[:8]}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  [{name}] memory {mid[:8]} FAILED: {e}"))
            logger.exception("Warmup failed for %s memory %s", name, mid)
```

Per-memory failures are logged and don't abort the run. Agents without a `rag_provider` are skipped. Empty states (no agents, no memories) exit with a warning.

## Code Equivalent

```python
from django_ai_sdk.agents.registry import registry

agents = registry.all()
for aid, inst in agents.items():
    if inst.rag_provider is None:
        continue
    for mid in memory_ids:
        await inst.rag_provider.warmup(inst, mid, force_rebuild=False)
```

## Manual Testing

```bash
python manage.py warmup_rag --agent PirateBasicAgent --memory <memory-uuid> --force-rebuild
python manage.py warmup_rag --memory not-a-uuid   # CommandError
python manage.py warmup_rag --agent Nope          # CommandError
```
