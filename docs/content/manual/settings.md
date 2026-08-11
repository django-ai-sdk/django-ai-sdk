---
title: Settings Reference
type: docs
weight: 117
---

Every `AI_SDK_*` Django setting, with its default and purpose. Most are optional: the SDK works with sensible defaults out of the box.

Settings are read via `getattr(settings, ...)` at call time (cached where noted), so none of them require a restart of the dev server beyond Django's normal settings reload.

## Agent Registration and Runtime Agents

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_AGENTS` | `[]` | Dotted module paths to import so their `Agent` subclasses auto-register. See the [Agents guide](/docs/agents/) and [Agent Registry](/docs/manual/agent-registry/). |
| `AI_SDK_RUNTIME_AGENT_BASES` | `[]` | Base classes a DB-configured runtime agent can be built on, as dotted paths. See [Runtime-Configured Agents](/docs/views-and-routing/#runtime-configured-agents). |
| `AI_SDK_RUNTIME_AGENT_TOOLS` | `{}` | Tools selectable in runtime agent configuration: `{"key": "path.to.provider"}`. |

## Models

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_DEFAULT_MODEL` | `"gpt-4o-mini"` | Fallback model identifier for `llm_generator()` when no model is otherwise configured. |
| `AI_SDK_EXTRACTION_MODEL` | `None` | Model used by the document-extraction generator (`agents/utils.py`). Falls back to `AI_SDK_DEFAULT_MODEL` when unset. |

## RAG and Vector Stores

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_VECTOR_STORE_PATH` | `None` | Base directory for persistent vector stores. Storage configs build per-backend, per-memory paths as `{path}/{backend}/{memory_id}`; unset means in-memory stores. See [RAG Variants](/docs/manual/rag-variants/). |
| `AI_SDK_VECTOR_STORE_URL` | `None` | Qdrant server URL. When set, Qdrant storage uses `backend="server"` with collection index `memory_{memory_id}` (or `default`). |

## Files and Uploads

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_ALLOWED_FILES` | `{}` | Extra file-extension → MIME mapping used as a fallback when magic-byte detection fails (`files/processors.py`). |
| `AI_SDK_MAX_UPLOAD_SIZE` | `10 MB` | Upload size ceiling surfaced to the frontend by `get_upload_settings()`. |
| `AI_SDK_MEMORY_FILE_PIPELINE` | `None` | Dotted path (or list of paths) to a zero-argument callable returning a `FilePipeline`: the default pipeline for uploads without agent context. The first pipeline whose `accepts(file)` matches is used. See [Files](/docs/manual/files/). |
| `AI_SDK_FILE_PIPELINE_TIMEOUT` | `900` | Seconds before a background document pipeline is failed. `django_tasks` has no native timeout, so this guards hung or wedged upload processing. |

## MCP Servers

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_MCP_DISCOVERY_CACHE_TTL` | `3600` | Seconds OAuth discovery results are cached in-process. |
| `AI_SDK_MCP_DISCOVERY_TIMEOUT` | `10` | Seconds for a single OAuth discovery request. |
| `AI_SDK_MCP_ALLOWED_ISSUER_DOMAINS` | `None` | Allow-list of OAuth issuer domains (e.g. `["accounts.notion.com"]`). `None` allows any issuer; `[]` rejects all. Defense-in-depth against compromised MCP servers. |
| `AI_SDK_MCP_CLIENT_NAME` | `"MCP OAuth Client"` | Client name registered with OAuth servers during the authorization flow. |
| `AI_SDK_MCP_SERVER_LIST_CACHE_TTL` | `30` | Seconds the integrations registry caches its server list. |
| `AI_SDK_MCP_REFRESH_THRESHOLD_MINUTES` | `10` | Refresh OAuth tokens proactively when they expire within this many minutes. |
| `AI_SDK_MCP_OAUTH_SUCCESS_URL` | `"/"` | Redirect target after a successful MCP OAuth flow. |

## Integrations

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_INTEGRATIONS` | `{}` | Per-integration configuration dict, keyed by registry name, in the shape of `DATABASES`/`CACHES`. Keys are upper-cased on read. Secrets are referenced inline, e.g. `{"github": {"TOKEN": env("GITHUB_MCP_TOKEN")}}`: there is no derived `AI_SDK_GITHUB_TOKEN` variable, because GitHub Actions already injects a conflicting `GITHUB_TOKEN`. See the [Integrations guide](/docs/integrations/). |
| `AI_SDK_INTEGRATION_TIMEOUT` | `3` | Seconds per integration tool call. |
| `AI_SDK_INTEGRATION_CACHE_TTL` | `900` | Seconds tool lists and statuses are cached. |
| `AI_SDK_INTEGRATION_CB_COOLDOWN` | `60` | Seconds a failed integration stays open-circuited before retry. |

## Permissions

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_PERMISSIONS` | `{}` | Per-domain overrides: `{"memory": ["path.to.PermissionClass"], ...}`. Domains: `agent`, `thread`, `memory`, `integrations`. See [Permissions](/docs/manual/permissions/). |

## Workflows

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_WORKFLOW_ACTIONS` | `{}` | Workflow action registry: `{"console_log": "path.to.ConsoleLogAction"}`. See [Workflows](/docs/manual/workflows/). |

## Streaming and Titles

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_STREAM_CORS_ORIGIN` | `None` | Allowed origin for SSE stream responses (used alongside `django-cors-headers`). |
| `AI_SDK_SUGGESTION_TIMEOUT` | `5.0` | Seconds before suggestion generation is abandoned. |
| `AI_SDK_TITLE_SANITY_LIMIT` | `80` | Character ceiling for auto-generated thread titles. |

## Logging

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_ENABLE_LOGS` | `False` | When `True`, adds a DEBUG-level loguru handler to stderr. File logging to `logs/django_ai_sdk.log` runs regardless. |

## Provider Credentials

Not `AI_SDK_*` prefixed, but part of the same surface: the generator and RAG libraries read them directly:

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Provider API key (often injected as an environment variable). |
| `OPENAI_API_URL` | Optional custom API base URL (`rags` pipelines and `agents/utils.py` read it). |
