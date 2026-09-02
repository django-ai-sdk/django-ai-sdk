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
| `AI_SDK_AGENTS` | `[]` | Dotted module paths to import so their `Agent` subclasses auto-register. See the [Agents guide](/agents/) and [Agent Registry](/manual/agent-registry/). |
| `AI_SDK_RUNTIME_AGENT_BASES` | `[]` | Base classes a DB-configured runtime agent can be built on, as dotted paths. See [Runtime-Configured Agents](/views-and-routing/#runtime-configured-agents). |
| `AI_SDK_RUNTIME_AGENT_TOOLS` | `{}` | Tools selectable in runtime agent configuration: `{"key": "path.to.provider"}`. |

## Models

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_DEFAULT_MODEL` | `"gpt-4o-mini"` | Fallback model identifier for `llm_generator()`, and the usual value for an agent's `model` attribute. The [generator factories](/manual/generators/) never read it: an agent always passes its own model. |
| `AI_SDK_EXTRACTION_MODEL` | `None` | Model used by the document-extraction generator (`agents/utils.py`). Falls back to `AI_SDK_DEFAULT_MODEL` when unset. |

## RAG and Vector Stores

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_VECTOR_STORE_PATH` | `None` | Base directory for persistent vector stores. Storage configs build per-backend, per-memory paths as `{path}/{backend}/{memory_id}`; unset means in-memory stores. See [RAG Variants](/manual/rag-variants/). |
| `AI_SDK_VECTOR_STORE_URL` | `None` | Qdrant server URL. When set, Qdrant storage uses `backend="server"` with collection index `memory_{memory_id}` (or `default`). |

## Files and Uploads

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_ALLOWED_FILES` | `{}` | Extra file-extension → MIME mapping used as a fallback when magic-byte detection fails (`files/processors.py`). |
| `AI_SDK_MAX_UPLOAD_SIZE` | `10 MB` | Upload size ceiling surfaced to the frontend by `get_upload_settings()`. |
| `AI_SDK_MEMORY_FILE_PIPELINE` | `None` | Dotted path (or list of paths) to a zero-argument callable returning a `FilePipeline`: the default pipeline for uploads without agent context. The first pipeline whose `accepts(file)` matches is used. See [Files](/manual/files/). |
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
| `AI_SDK_INTEGRATIONS` | `{}` | Per-integration configuration dict, keyed by registry name, in the shape of `DATABASES`/`CACHES`. Keys are upper-cased on read. Secrets are referenced inline, e.g. `{"github": {"TOKEN": env("GITHUB_MCP_TOKEN")}}`: there is no derived `AI_SDK_GITHUB_TOKEN` variable, because GitHub Actions already injects a conflicting `GITHUB_TOKEN`. See the [Integrations guide](/integrations/). |
| `AI_SDK_INTEGRATION_TIMEOUT` | `3` | Seconds per integration tool call. |
| `AI_SDK_INTEGRATION_CACHE_TTL` | `900` | Seconds tool lists and statuses are cached. |
| `AI_SDK_INTEGRATION_CB_COOLDOWN` | `60` | Seconds a failed integration stays open-circuited before retry. |

## Permissions

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_PERMISSIONS` | `{}` | Per-domain overrides: `{"memory": ["path.to.PermissionClass"], ...}`. Domains: `agent`, `thread`, `memory`, `integrations`. See [Permissions](/manual/permissions/). |

## Workflows

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_WORKFLOW_ACTIONS` | `{}` | Workflow action registry: `{"console_log": "path.to.ConsoleLogAction"}`. See [Workflows](/manual/workflows/). |

## Streaming and Titles

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_STREAM_CORS_ORIGIN` | `None` | Allowed origin for SSE stream responses (used alongside `django-cors-headers`). |
| `AI_SDK_SUGGESTION_TIMEOUT` | `5.0` | Seconds before suggestion generation is abandoned. |
| `AI_SDK_TITLE_SANITY_LIMIT` | `80` | Character ceiling for auto-generated thread titles. |

## Tracing

Requires the opt-in `django_ai_sdk.tracing` app in `INSTALLED_APPS`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_TRACING_EXCLUDED_TAGS` | `[]` | Span tag keys to drop before storing, by exact key (no wildcards). Useful for static configuration that repeats identically on every run and is large: `["haystack.agent.tools", "haystack.agent.state_schema"]`. Applies to content tags too, and excluding one never affects the token columns. See [Tracing](/manual/tracing/). |

One Haystack environment variable shapes what the tracer records. Unlike the
settings above, it is read **once**, when `haystack.tracing` is first imported:
it must be a real environment variable set before the process starts, and
changing it needs a restart. Assigning it in `settings.py` does nothing.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HAYSTACK_CONTENT_TRACING_ENABLED` | `false` | When `true`, prompts, documents and replies are stored in the `tags` JSON column. Token counts and `model_name` are recorded either way. Treat the stored content as sensitive. |

## Logging

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_ENABLE_LOGS` | `False` | When `True`, adds a DEBUG-level loguru handler to stderr. File logging to `logs/django_ai_sdk.log` runs regardless. |

## Background Tasks

Document processing (`memories/tasks.py`) and workflow runs (`workflows/tasks.py`)
are enqueued with [django-tasks](https://pypi.org/project/django-tasks/), which the
SDK depends on directly. It is **not** an `AI_SDK_*` setting: the backend is Django's
`TASKS` setting, and which one you pick is a deployment decision the SDK does not
make for you.

With no `TASKS` setting at all, django-tasks defaults to
`django_tasks.backends.immediate.ImmediateBackend` - tasks run **inline, in the
calling request**. That works out of the box and needs no extra package, but it means
a document upload processes synchronously while the user waits.

For durable processing with a separate worker, install a backend that provides one.
The database backend ships separately:

```bash
pip install django-tasks-db
```

```python
INSTALLED_APPS = [
    # ...
    "django_tasks",
    "django_tasks_db",
]

TASKS = {
    "default": {
        "BACKEND": (
            "django_tasks.backends.immediate.ImmediateBackend"
            if DEBUG
            else "django_tasks_db.DatabaseBackend"
        ),
    }
}
```

Then run the worker: `./manage.py db_worker`. The demo project uses exactly this
split - immediate in `DEBUG`, the database backend otherwise.

## Provider Credentials

Not `AI_SDK_*` prefixed, but part of the same surface: the [generator
factories](/manual/generators/) read them at call time. Every key is optional - when
one is unset the factory omits it, so Haystack falls back to its own environment
variable of the same name.

| Setting | Used by | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `openai_chat`, `openai_responses_chat` | OpenAI API key. |
| `OPENAI_API_URL` | same | Custom base URL, e.g. a self-hosted OpenAI-compatible endpoint. |
| `ANTHROPIC_API_KEY` | `anthropic_chat` | Anthropic API key. `AnthropicChatGenerator` has no base-URL parameter, so there is no `ANTHROPIC_API_URL`. |
| `MISTRAL_API_KEY` | `mistral_chat` | Mistral API key. |
| `MISTRAL_API_URL` | same | Custom base URL. |
| `OPENROUTER_API_KEY` | `openrouter_chat` | OpenRouter API key. |
| `OPENROUTER_API_URL` | same | Custom base URL. |
| `OLLAMA_API_URL` | `ollama_chat` | Ollama server URL. Defaults to `http://localhost:11434`. |
| `HUGGINGFACE_API_KEY` | `huggingface_api_chat`, `transformers_chat` | Hugging Face token. |
| `HUGGINGFACE_API_URL` | `huggingface_api_chat` | Dedicated inference endpoint URL. When unset, serverless inference is used with the requested model. |
| `AZURE_OPENAI_API_KEY` | `azure_openai_chat`, `azure_openai_responses_chat` | Azure OpenAI key. |
| `AZURE_OPENAI_ENDPOINT` | same | Azure resource endpoint. |
| `AZURE_OPENAI_DEPLOYMENT` | same | Deployment name used when no model is passed. |
| `AZURE_OPENAI_API_VERSION` | `azure_openai_chat` | API version override. |
