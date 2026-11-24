---
title: Settings Reference
sidebar:
  order: 117
---

Every `AI_SDK_*` Django setting, with its default and purpose. Most are optional: the SDK works with sensible defaults out of the box.

Settings are read via `getattr(settings, ...)` at call time (cached where noted), so none of them require a restart of the dev server beyond Django's normal settings reload.

## Agent Registration and Runtime Agents

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_AGENTS` | `[]` | Dotted module paths to import so their `Agent` subclasses auto-register. See the [Agents guide](../agents.md) and [Agent Registry](agent-registry.md). |
| `AI_SDK_RUNTIME_AGENT_BASES` | `[]` | Base classes a DB-configured runtime agent can be built on, as dotted paths. See [Runtime-Configured Agents](../views-and-routing.md#runtime-configured-agents). |
| `AI_SDK_RUNTIME_AGENT_TOOLS` | `{}` | Tools selectable in runtime agent configuration: `{"key": "path.to.provider"}`. |

## Models

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_DEFAULT_MODEL` | `"gpt-4o-mini"` | Fallback model identifier for `llm_generator()`, and the usual value for an agent's `model` attribute. The [generator factories](generators.md) never read it: an agent always passes its own model. |
| `AI_SDK_EXTRACTION_MODEL` | `None` | Model used by the document-extraction generator (`agents/utils.py`). Falls back to `AI_SDK_DEFAULT_MODEL` when unset. |
| `AI_SDK_MODEL_PRICES` | `{}` | Per-million-token prices by model or model prefix: `{"gpt-4.1-mini": {"input": 0.4, "output": 1.6}}`. Fills the cost of traces, turns and judges. |

## RAG and Vector Stores

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_VECTOR_STORE_PATH` | `None` | Base directory for persistent vector stores. Storage configs build per-backend, per-memory paths as `{path}/{backend}/{memory_id}`; unset means in-memory stores. See [RAG Variants](rag-variants.md). |
| `AI_SDK_VECTOR_STORE_URL` | `None` | Qdrant server URL. When set, Qdrant storage uses `backend="server"` with collection index `memory_{memory_id}` (or `default`). Needed when uploads are processed by a task worker: the worker writes new documents into the shared server index, so chat finds them. A local or in-memory index belongs to one process. |
| `AI_SDK_EMBEDDINGS_MODEL` | `None` | Qdrant hybrid RAG: when set, dense embeddings come from this model through the OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_API_URL`) instead of local FastEmbed. Sparse (BM42) embeddings stay local. Changing it needs `reindex_memories`. |
| `AI_SDK_EMBEDDINGS_DIM` | `384` | Vector size of the Qdrant collection; must match the dense model (`Qwen/Qwen3-Embedding-8B`: `4096`). |

## Files and Uploads

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_ALLOWED_FILES` | `{}` | Extra file-extension → MIME mapping used as a fallback when magic-byte detection fails (`files/processors.py`). |
| `AI_SDK_MAX_UPLOAD_SIZE` | `10 MB` | Largest file a document or thread upload accepts (larger ones get a 400); surfaced to the frontend by `get_upload_settings()`. |
| `AI_SDK_MEMORY_FILE_PIPELINE` | `None` | Dotted path (or list of paths) to a zero-argument callable returning a `FilePipeline`: the default pipeline for uploads without agent context. The first pipeline whose `accepts(file)` matches is used. See [Files](files.md). |
| `AI_SDK_VISION_AGENT` | `None` | Dotted path to your own `Agent` subclass that answers every image question: captions for uploaded images and the `ask_image` tool. Choose its model, instructions and generator; make it `hidden = True`. Unset means images are not captioned and there is no `ask_image`. An agent sees images as pixels with `vision = True`. See [Images and vision](files.md#images-and-vision). |
| `AI_SDK_MAX_INLINE_IMAGE_BYTES` | `5 MB` | Largest image sent as pixels to a vision-capable agent; bigger ones get their description instead. `None` disables the cap. |
| `AI_SDK_IMAGE_CAPTION_LIMIT` | `1500` | Character cap on the auto-generated description an agent without vision gets for an attached image; a cut caption ends in `…`. `None`, `0` or below disables it. Override `Agent.format_image_attachment()` to change the whole hint. |
| `AI_SDK_THREAD_FILE_URL` | `None` | URL name (as for `reverse()`) of your permission-checked thread-file download view, taking `thread_id` and `doc_id`; file parts and file artifacts link to it. Unset uses the storage url, fine for public storage only. The contrib Ninja and DRF routers ship such a view (`athread_file_response` / `thread_file_response`). |
| `AI_SDK_FILE_PIPELINE_TIMEOUT` | `900` | Seconds before a background document pipeline is failed. `django_tasks` has no native timeout, so this guards hung or wedged upload processing. |
| `AI_SDK_THREAD_FILE_PRIORITY` | `10` | `django_tasks` priority (-100 to 100) for thread uploads and their retries. Memory documents stay at 0, so a positive value lets a chat upload jump a bulk memory import in the queue. It only reorders waiting tasks: a document already running finishes first. Needs a backend with `supports_priority` (the immediate and database backends have it). |
| `STORAGES["django_ai_sdk"]` | unset | A reserved `STORAGES` alias used for `EntryDocument.file` only. Define it like any other `STORAGES` entry (`BACKEND` + `OPTIONS`). Unset falls back to `STORAGES["default"]` See [Files](files.md). |
| `AI_SDK_FILE_UPLOAD_TO` | `"memories/documents/"` | `upload_to` path for `EntryDocument.file`. Unset keeps the default (no migration needed); changing it requires running `makemigrations` in the host project, same as customizing any other third-party model field. See [Files](files.md). |

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
| `AI_SDK_INTEGRATIONS` | `{}` | Per-integration configuration dict, keyed by registry name, in the shape of `DATABASES`/`CACHES`. Keys are upper-cased on read. Secrets are referenced inline, e.g. `{"github": {"TOKEN": env("GITHUB_MCP_TOKEN")}}`: there is no derived `AI_SDK_GITHUB_TOKEN` variable, because GitHub Actions already injects a conflicting `GITHUB_TOKEN`. See the [Integrations guide](../integrations.md). |
| `AI_SDK_INTEGRATION_TIMEOUT` | `3` | Seconds per integration tool call. |
| `AI_SDK_INTEGRATION_CACHE_TTL` | `900` | Seconds tool lists and statuses are cached. |
| `AI_SDK_INTEGRATION_CB_COOLDOWN` | `60` | Seconds a failed integration stays open-circuited before retry. |

## Permissions

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_PERMISSIONS` | `{}` | Per-domain overrides: `{"memory": ["path.to.PermissionClass"], ...}`. Domains: `agent`, `thread`, `memory`, `integrations`, `workflow`. See [Permissions](permissions.md). |

## Workflows

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_WORKFLOWS["STEPS"]` | `{}` | Step classes a JSON workflow may compose: `{"gather_thread": "apps.agents.steps.GatherStep"}`. Every type is the host's own — the package ships none. The registry is the gate: a type that is not listed cannot be named by an author. See [Workflows](workflows.md). |
| `AI_SDK_WORKFLOWS["LISTENERS"]` | `{}` | Run listeners a definition may attach to the workflow or to one step: `{"console_log": "apps.agents.listeners.ConsoleLogListener"}`. Every listener is the host’s own — the package ships none a definition can name. See [Workflows](workflows.md). |
| `AI_SDK_WORKFLOWS["SCHEDULES_ENABLED"]` | `True` | `False` makes every cron occurrence start nothing, such as on a staging copy of production; the schedules stay as they are. See [Triggers](workflows.md#triggers). |
| `AI_SDK_WORKFLOWS["STALE_AFTER_SECONDS"]` | `900` | How long a `running` run may go without progress before the worker checks it against DBOS, and fails it when its DBOS workflow no longer runs. See [Durable runs on DBOS](workflows.md#durable-runs-on-dbos). |

## Oversight

One dict, `AI_SDK_OVERSIGHT`, every key optional. Its keys are listed in the [Oversight manual](oversight.md#settings).

Every settings dict of the SDK (`AI_SDK_OVERSIGHT` and `AI_SDK_WORKFLOWS`) is checked on start: an unknown key or a value of the wrong kind is `<app>.E012`, which names the key and what it accepts. A value that is not a dict at all raises `ImproperlyConfigured` when the app reads it.

| Setting | Default | Purpose |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | unset | Key for oversight's optional `TypeSafe` judge (a hosted commercial service; `django-ai-sdk[typesafe]`). Without it that judge abstains. |
| `TYPESAFE_MODEL` | unset | A TypeSafe model to pin; unset leaves the choice to `typesafe-sdk`. |

## Durable runs (DBOS)

Workflows need the `django_ai_sdk.durable` app and the `dbos` extra. One dict, `AI_SDK_DBOS`, every key optional. See [Durable runs on DBOS](workflows.md#durable-runs-on-dbos).

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_DBOS["DATABASE_URL"]` | `""` | Where DBOS keeps its tables. Empty uses `DATABASES["default"]`: the same Postgres database with its connection `OPTIONS` (such as `sslmode`) and a socket `HOST`, or a file next to a SQLite one. |
| `AI_SDK_DBOS["SCHEMA"]` | `"dbos"` | The Postgres schema for DBOS's tables. |
| `AI_SDK_DBOS["APP_NAME"]` | `"django-ai-sdk"` | Who owns the workflows, queues and schedules in those tables. |
| `AI_SDK_DBOS["VERSION"]` | `"1"` | Pinned, so a deploy recovers the runs the last one left. Change it only when step code changes in a way an unfinished run cannot replay: a worker runs only its own version's runs, and warns at start about the ones left under another. See [Durable runs on DBOS](workflows.md#durable-runs-on-dbos). |
| `AI_SDK_DBOS["USE_LISTEN_NOTIFY"]` | `True` | `False` behind a pooler in transaction mode (PgBouncer), which drops `LISTEN`. Web processes may use the pooler; the worker must reach Postgres directly, because it holds a session advisory lock that transaction pooling releases. |
| `AI_SDK_DBOS["QUEUES"]` | `{"workflows": {}, "default": {"worker_concurrency": 8}}` | Queues by name, with DBOS `register_queue` options. A task's `queue_name` must be one of them. Leave `workflows` uncapped: a run waiting for an approval holds a slot for as long as it waits. |

## Webhooks

For an integration a platform pushes events into. Per-platform keys (`AGENT`, `RUN_AS`, `ALLOW_FROM`, ...) live in `AI_SDK_INTEGRATIONS[<name>]`; see [Webhooks](webhooks.md).

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_WEBHOOK_TIMEOUT` | `120` | Seconds a webhook-triggered agent run may take before the channel gets a timeout reply instead. |
| `AI_SDK_WEBHOOK_MAX_TEXT` | `4000` | Characters of the inbound question kept, so the prompt is bounded by the SDK rather than by whatever the platform accepts. |

## Streaming and Titles

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_STREAM_CORS_ORIGIN` | `None` | Allowed origin for SSE stream responses (used alongside `django-cors-headers`). |
| `AI_SDK_SUGGESTION_TIMEOUT` | `5.0` | Seconds before suggestion generation is abandoned. |
| `AI_SDK_TITLE_SANITY_LIMIT` | `80` | Character ceiling for auto-generated thread titles. |
| `AI_SDK_SUBAGENT_DIGEST_LIMIT` | `6000` | Character budget for the digest of a subagent's tool results, used when its run ended without a written report. Shared across all results; `0` or below disables truncation. See [Handoff output](agent.md#handoff-output). |
| `AI_SDK_TOOL_OUTPUT_LIMIT` | `100000` | Character cap per tool result the model gets within a run; the model is told it was cut. `ToolAgentConfig.max_tool_output_chars` overrides it per agent. `0` or below disables it. See [Error Handling](../errors.md#what-the-model-sees). |
| `AI_SDK_HISTORY_TOOL_OUTPUT_LIMIT` | `4000` | Character cap per subagent handoff result replayed into a later turn. `0` or below disables truncation. See [Subagent Delegation](agent.md#handoff-history). |

## Tracing

Requires the opt-in `django_ai_sdk.tracing` app in `INSTALLED_APPS`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_TRACING_EXCLUDED_TAGS` | `[]` | Span tag keys to drop before storing, by exact key (no wildcards). Useful for static configuration that repeats identically on every run and is large: `["haystack.agent.tools", "haystack.agent.state_schema"]`. Applies to content tags too, and excluding one never affects the token columns or the `agent_id`/`agent_name` columns. See [Tracing](tracing.md). |

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
| `AI_SDK_ENABLE_LOGS` | `False` | When `True`, adds a DEBUG-level loguru handler to stderr. File logging to `logs/django_ai_sdk.log` runs regardless. Neither writes local variables into a traceback (`diagnose` is off). |

## Background Tasks

Document processing (`memories/tasks.py`) and oversight's reflection and notifications
(`oversight/tasks.py`) are enqueued with [django-tasks](https://pypi.org/project/django-tasks/),
which the SDK depends on directly. It is **not** an `AI_SDK_*` setting: the backend is
Django's `TASKS` setting, and which one you pick is a deployment decision the SDK does
not make for you. Workflow runs are not django-tasks: they run on DBOS, in
`./manage.py ai_sdk_worker` (see [Workflows](workflows.md)).

With no `TASKS` setting at all, django-tasks defaults to
`django_tasks.backends.immediate.ImmediateBackend` - tasks run **inline, in the
calling request**. That works out of the box and needs no extra package, but it means
a document upload processes synchronously while the user waits.

For durable processing with a separate worker, install a backend that provides one.
With the workflows app installed, `DBOSTaskBackend` runs every task on the DBOS queues
that `ai_sdk_worker` already polls, so no second worker is needed:

```python
TASKS = {"default": {"BACKEND": "django_ai_sdk.durable.tasks.DBOSTaskBackend"}}
```

Declare each task's `queue_name` in `AI_SDK_DBOS["QUEUES"]`: the worker polls only
those, and a task on any other queue is never picked up.

Without the workflows app, the database backend ships separately:

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
