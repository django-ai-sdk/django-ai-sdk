---
title: Command Line Interface
sidebar:
  order: 1
---

The SDK ships three management commands. `warmup_rag` pre-builds RAG pipelines so the first chat request doesn't pay the indexing cost. `reindex_memories` rebuilds every memory's index from scratch. `refresh_integrations` refreshes the tool lists of MCP integrations.

## `warmup_rag`: pre-warm RAG indexes

```bash
python manage.py warmup_rag
```

Warms up every agent that has a `rag_provider` configured, across all memories. The command skips agents without a RAG provider.

**Options:**

| Option | Description |
| --- | --- |
| `--agent <class_name>` | Only warm up a specific agent by class name, for example `PirateBasicAgent` |
| `--memory <memory_id>` | Only warm up a specific memory |
| `--force-rebuild` | Rebuild indexes from scratch, deleting existing ones |

**Examples:**

```bash
# Everything
python manage.py warmup_rag

# One agent
python manage.py warmup_rag --agent PirateBasicAgent

# One memory
python manage.py warmup_rag --memory <memory-uuid>

# Full rebuild (persistent backends like Qdrant recreate from scratch)
python manage.py warmup_rag --force-rebuild
```

## `reindex_memories`: rebuild every index

```bash
python manage.py reindex_memories                 # every memory
python manage.py reindex_memories --memory <id>   # one memory
python manage.py reindex_memories --agent SupportAgent
```

Rebuilds each memory's index once, from scratch, after a change to how documents are indexed: chunk size or overlap, `meta_fields_to_embed`, the embedding models. An index belongs to a memory, not to an agent, so the RAG config of one agent builds them all. `--agent` picks that agent and defaults to the first agent with a RAG provider. `warmup_rag --force-rebuild` instead rebuilds each memory once per agent. The command skips memories without documents. It reports a failure per memory and exits with an error.

## When to use it

- Before you deploy a new RAG setup, so indexes are ready when traffic arrives.
- After document changes, to keep retrieval current.
- As a scheduled job, for example a cron entry or `django-tasks` task that re-warms nightly.

You can also trigger the same work from code:

```python
from django_ai_sdk import Agent

agent = await AgentService.get(agent_id)
await Agent.warmup(agent, memory_id=None)            # build + cache
await Agent.reindex(agent, memory_id=None, force_rebuild=False)
Agent.clear_rag_cache(agent)
```

See the [RAG section of the Agents guide](../agents/agents.md#lifecycle) for the full lifecycle.

:::note
Contributor? The [CLI Implementation](../manual/cli.md) manual page documents how the command works internally.
:::
