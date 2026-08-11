---
title: Command Line Interface
type: docs
weight: 7
---

The SDK ships two management commands. **`warmup_rag`** pre-builds RAG pipelines so the first chat request doesn't pay the indexing cost, and **`refresh_integrations`** refreshes MCP integration tool lists.

## `warmup_rag`: Pre-warm RAG Indexes

```bash
python manage.py warmup_rag
```

Warms up every agent that has a `rag_provider` configured, across all memories. Agents without a RAG provider are skipped.

**Options:**

| Option | Description |
| --- | --- |
| `--agent <class_name>` | Only warm up a specific agent (class name, e.g. `PirateBasicAgent`) |
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

## When to Use It

- **Before deploying a new RAG setup**: so indexes are ready when traffic arrives.
- **After document changes**: to keep retrieval current.
- **As a scheduled job**: e.g. a cron entry or `django-tasks` task that re-warms nightly.

You can also trigger the same work from code:

```python
from django_ai_sdk import Agent

agent = await AgentService.get(agent_id)
await Agent.warmup(agent, memory_id=None)            # build + cache
await Agent.reindex(agent, memory_id=None, force_rebuild=False)
Agent.clear_rag_cache(agent)
```

See the [RAG section of the Agents guide](/agents/#lifecycle) for the full lifecycle.

{{< callout type="info" >}}
Contributor? The [CLI Implementation](/manual/cli/) manual page documents how the command works internally.
{{< /callout >}}
