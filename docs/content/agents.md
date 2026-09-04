---
title: Agents
type: docs
weight: 2
---

An **agent** is the core of the Django AI SDK. It's a plain Python class that describes how your AI behaves, what it can do, and how it streams responses.

This guide covers everything about defining agents. See the [Protocols](/protocols/) guide for how `Stream` and `Run` work, and [Views and Routing](/views-and-routing/) for wiring agents to HTTP endpoints.

{{< callout type="info" >}}
Contributor? The [Developer Manual](/manual/) covers internals: [Agent](/manual/agent/), [Agent Registry](/manual/agent-registry/), [RAG](/manual/rag/).
{{< /callout >}}

## Anatomy of an Agent

```python
from django_ai_sdk import Agent
from django_ai_sdk.adapters.base import Run, Stream
from django_ai_sdk.agents import auto_register
from django_ai_sdk.common import prompt
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.db import DbStorageAdapter

@auto_register
class MyAgent(Agent):
    name = "My Bot"
    description = "A helpful assistant"
    model = "openai/gpt-oss-120b"
    instructions = prompt("You are a helpful agent.")
    protocol = VercelProtocolHandler
    storage_adapter = DbStorageAdapter
    max_history = 20
    tools = [get_today, get_memory_files]

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        # Build the Stream that powers streaming chat.
        ...

    async def get_run_adapter(self, thread_id=None, user=None):
        # Build the Run that powers non-streaming calls.
        ...
```

### Registration

Every `Agent` subclass registers itself in the registry when the module is imported. Use `AI_SDK_AGENTS` in settings to ensure your modules are loaded:

```python
# settings.py
AI_SDK_AGENTS = [
    "your_app.agents.MyAgent",
    "your_app.agents.OtherAgent",
]
```

Each agent gets a **stable UUID** derived from its module and class name, so it never changes between restarts. Resolve agents by ID with `AgentService`:

```python
from django_ai_sdk.agents.services import AgentService

agent = await AgentService.get(agent_id)
```

Mark a class `abstract = True` to use it as a shared base without registering it, or `hidden = True` to keep it out of agent listings (useful for worker-only agents).

---

## Class Configuration

These class attributes configure an agent's behavior:

| Attribute | Default | Purpose |
| --- | --- | --- |
| `name` | None | Display name |
| `description` | `None` | Short description, surfaced in agent listings |
| `model` | `None` | Model identifier passed to the generator |
| `llm` | `openai_responses_chat` | Generator factory from [`django_ai_sdk.generators`](/manual/generators/), assigned uncalled |
| `llm_kwargs` | `None` | Vendor generation parameters, e.g. `{"reasoning": {"effort": "low"}}` |
| `instructions` | `"You are a helpful agent."` | System prompt, built with `prompt()` |
| `protocol` | `VercelProtocolHandler` | Protocol handler for message conversion |
| `storage_adapter` | `MemoryStorageAdapter` | Storage used when a thread isn't found elsewhere |
| `tools` | `[]` | Class-level tool providers |
| `agents` | `[]` | Delegate-able subagent agent classes (see [Multi-Agent Swarms](#multi-agent-swarms)) |
| `max_agent_steps` | `6` | Loop limit when this agent runs as a subagent (`build_subagent`) |
| `max_tool_calls` | `6` | Tool-call cap as a subagent; `None` disables it |
| `integrations` | `[]` | Integration names whose tools reach this agent |
| `artifacts` | `[]` | `ArtifactSchema` subclasses exposed as submission tools |
| `rag_provider` | `None` | `RAGProvider` instance to enable RAG |
| `rag_document_limit` | `10000` | Intended cap on documents fetched for RAG indexing (not yet enforced) |
| `memories` | `[]` | Default connected memories |
| `permissions` | `None` | Permission classes gating agent operations; `None` or `[]` uses the `agent` domain default |
| `max_history` | `None` | Cap on messages sent to the model |
| `title_generation` | `True` | Auto-generate thread titles |
| `response_format` | `None` | Pydantic model for structured `run()` output |
| `file_upload` | `False` | Enable file upload for threads |
| `file_pipelines` | `[]` | Per-file-type processing pipelines |
| `citation_formatter_class` | `DefaultCitationFormatter` | How RAG sources are rendered to the model |
| `suggestion_generator` | `None` | Generates follow-up suggestions after a reply |
| `hidden` | `False` | Hide from registry listings |
| `abstract` | `False` | Shared base, not registered |
| `warmup_on_init` | `False` | Warm up RAG on agent initialization |

### The `prompt()` helper

Use `prompt()` to build instructions. It wraps your text in a `Prompt` and strips leading indentation, so multi-line instruction blocks stay clean:

```python
instructions = prompt("""\
    You are a helpful AI agent who always responds like a pirate.
    Use pirate language, expressions, and mannerisms in all your responses.
    - Be creative with pirate slang but keep responses helpful and informative.
""")
```

---

## Adapters: Stream and Run

An agent exposes two adapter hooks. You implement them; the SDK uses them.

- **`get_pipeline_adapter(thread_id, user)`**: returns a `Stream` for streaming chat. `Agent.as_view()` calls this.
- **`get_run_adapter(thread_id, user)`**: returns a `Run` for non-streaming calls. `Agent.run()` calls this (title generation, structured extraction, background jobs).

{{< callout type="info" >}}
A worker-only agent (`hidden = True`, never used in chat) can leave `get_pipeline_adapter()` unimplemented.
{{< /callout >}}

See the [Protocols guide](/protocols/) for the full `Stream` and `Run` API.

---

## Tools

Tools let the model call your Python functions. The SDK expects **Haystack `Tool` objects**: the model's tool schema is derived from them.

### Class-level tools

`Agent.get_tools()` runs every callable in the class-level `tools` list, passing context kwargs. Each provider may return one tool or a list:

```python
# your_app/tools.py
from haystack.tools import Tool

def get_datetime() -> dict:
    """Return the current date and time."""
    ...

def get_today(**kwargs) -> Tool:
    """Current date and time tool."""
    return Tool(
        name="get_today",
        parameters={},
        description="Get current date and time",
        function=get_datetime,
    )
```

```python
# your_app/agents.py
class MyAgent(Agent):
    ...
    tools = [get_today, get_memory_files]
```

Providers receive context about the current request:

```python
def get_memory_files(thread_id="", user_id="", **kwargs) -> Tool:
    """List files attached to the current thread."""
```

### Multi-Agent Swarms

A single agent can delegate sub-tasks to a team of **subagent agents**. Any `Agent` subclass can declare a crew with the `agents` class attribute:

```python
class PirateBoatExpertAgent(Agent):
    """Expert on pirate boats and seafaring lore."""
    name = "Pirate Boat Expert"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True          # worker-only: never listed in the registry
    tools = [pirate_boat_expert_tool]
    instructions = prompt("You are the crew's pirate boat expert. ...")


class TreasureHunterAgent(Agent):
    name = "Treasure Hunter"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True
    tools = [find_treasure_tool]
    instructions = prompt("You are the crew's treasure hunter. ...")


class PirateSwarmAgent(Agent):
    name = "Pirate Agent Swarm"
    model = settings.AI_SDK_DEFAULT_MODEL
    agents = [PirateBoatExpertAgent, TreasureHunterAgent]  # the crew
    instructions = prompt("You are a triage agent for a crew of pirate subagents. ...")
```

When the coordinator builds its tools, each subagent becomes a **native Haystack `ComponentTool`** the coordinator can call to delegate a sub-task:

- The subagent runs its **own agent loop** — its own generator, tools, and system prompt — and its final message is returned to the coordinator as the tool output. Subagents are ephemeral and stateless; the coordinator's `Stream` remains the single source of streamed *text*, but the subagent's **tool calls stream too** (see [Observability](#observability)), so sub-agent activity is visible in the UI while the report text stays single-sourced.
- The tool schema is a single `task` parameter; the coordinator is instructed to delegate with a self-contained task description.
- Subagent tool names are derived from the subagent display name (slugified), deduped with a numeric suffix on collision.
- Subagents the user lacks `VIEW_AGENT` permission on are skipped automatically.
- Cyclic crews (A delegates to B, B back to A) are detected at build time and the back-edge is skipped.
- The coordinator's `get_system_prompt()` automatically appends an `Available subagents:` roster.

Subagents are ordinary `Agent` subclasses with `hidden = True` — they keep the `name`/`description`/`model`/`tools`/`instructions` attributes, so they only need a `tools` list and an instructions prompt:

```python
class MySubagent(Agent):
    name = "My Subagent"
    description = "What it does"
    model = settings.AI_SDK_DEFAULT_MODEL
    hidden = True
    tools = [my_tool]
    instructions = prompt("You are a subagent. Complete delegated tasks concisely.")
```

Guardrails live on each `Agent` itself: subagents cap their own loop via `max_agent_steps` and `max_tool_calls` (via `ToolCallBudgetHook`), both defaulting to `6` — `max_tool_calls = None` disables the cap. The coordinator's `ToolAgentConfig` (`max_agent_steps`, `max_tool_calls`) caps its own turn the same way. See the demo `demo/apps/agents/agent_swarm.py` (pirate crew) and `demo/apps/agents/deep_research.py` (web-research worker using `duckduckgo-api-haystack` + `trafilatura`).

### Observability

`LogToolCallsHook` (a `before_tool` hook) logs every tool call an agent makes, so you can confirm delegation from the server console even when the frontend doesn't render tool events:

```python
config = ToolAgentConfig(..., hooks={"before_tool": [LogToolCallsHook()]})
```

The two swarm demos wire it into both the coordinator and (via `build_subagent`) each subagent, so you'll see lines like `Armed subagent tool 'treasure_hunter' ...`, `Tool call: treasure_hunter args={...}` (coordinator delegating), then `Tool call: find_treasure args={...}` (the subagent running its own tool). Streaming clients receive `tool_call_start` / `tool_input` / `tool_output` SSE events for the coordinator's own calls **and** for the subagent's calls: the coordinator forwards its streaming callback into subagent tools (`stream_subagent_tools`, on by default), and `SubagentStreamFilter` passes only tool-related chunks through — the subagent's raw text tokens are dropped so the report isn't streamed twice. Those forwarded chunks are persisted into the conversation history as they happen — in the order the calls actually start, so the stored history matches the streamed one and subagent tool activity survives a page reload.

Subagent loop limits are per-subagent class attributes, not just coordinator settings: `max_agent_steps` caps total loop iterations and `max_tool_calls` (via `ToolCallBudgetHook`) hard-caps tool calls, both defaulting to `6` on the subagent's own `Agent` class. The coordinator's `ToolAgentConfig` (`max_agent_steps`, `max_tool_calls`) caps its own turn the same way. See the demo `demo/apps/agents/agent_swarm.py` (pirate crew) and `demo/apps/agents/deep_research.py` (web-research worker using `duckduckgo-api-haystack` + `trafilatura`).

### Overriding `get_tools()`

For full control, override `get_tools()` on your agent. The default implementation combines:

1. Class-level `tools` providers
2. Integration tools (`integrations` list, resolved via the integrations registry)
3. Artifact submission tools (`artifacts` list)
4. Subagent agent tools (`agents` list, one `ComponentTool` per subagent)

```python
class MyAgent(Agent):
    async def get_tools(self, thread_id="", user=None):
        tools = await super().get_tools(thread_id=thread_id, user=user)
        tools.append(await self.get_custom_tool(thread_id))
        return tools
```

### Integration tools

Declare `integrations = ["linear", "weather"]` and every tool that integration exposes reaches the model. Tools are namespaced (`linear_list_issues`) so unrelated integrations never collide. Unauthorized users' tools never reach the model. See the [Integrations guide](/integrations/).

### Artifact tools

Agents can submit structured results as **artifacts**. Each `ArtifactSchema` subclass in the `artifacts` list becomes a submission tool the model can call during a turn:

```python
class MyArtifact(ArtifactSchema):
    kind = "report"
    title: str
    content: str
```

```python
class MyAgent(Agent):
    artifacts = [MyArtifact]
```

---

## Retrieval-Augmented Generation (RAG)

RAG connects documents stored in **memories** to your agent. The `RAGProvider` caches pipelines per agent and memory and exposes a per-memory retrieval tool for the model to call.

### Enabling RAG

1. Set `rag_provider = RAGProvider()` on the agent.
2. Override `get_rag_queryset()` to return the documents to index.
3. Implement `get_rag_pipeline()` (or one of the named variants) to return a RAG pipeline.
4. Link memories to threads: `get_rag_tools()` builds one tool per active memory.

```python
from django_ai_sdk.memories.models import Entry
from django_ai_sdk.rags import (
    BM25QueryExpanderRAG,
    BM25QueryExpanderRAGConfig,
    QdrantBM25HybridRAG,
    QdrantBM25HybridRAGConfig,
)
from django_ai_sdk.rags.provider import RAGProvider

class MyAgent(Agent):
    rag_provider = RAGProvider()

    async def get_rag_queryset(self, memory_id=None):
        if memory_id:
            return Entry.objects.filter(memory_id=memory_id)
        return Entry.objects.all()

    async def get_rag_pipeline(self, memory_id=None):
        documents = await self.get_rag_documents(memory_id)
        if not documents:
            return None
        return BM25QueryExpanderRAG(
            documents=documents,
            config=BM25QueryExpanderRAGConfig(top_k=5, n_expansions=4),
        )
```

### Pipeline variants

| Variant | Retrieval |
| --- | --- |
| `BM25QueryExpanderRAG` | BM25 keyword retrieval with query expansion |
| `ChromaDBQueryExpanderRAG` | Dense embedding retrieval with ChromaDB |
| `QdrantBM25HybridRAG` | Hybrid SPLADE + BGE embeddings with RRF, stored in Qdrant |

Override `get_rag_pipeline()` to return one of the variants. You can factor each into its own method (e.g. `get_rag_pipeline_qdrant()`) and call it from `get_rag_pipeline()`; the base method returns `None` (RAG disabled).

### Lifecycle

- **Warm up**: `Agent.warmup(agent, memory_id)` pre-builds and caches pipelines.
- **Reindex**: `Agent.reindex(agent, memory_id, force_rebuild)` rebuilds indexes (persistent backends like Qdrant delete and recreate).
- **Clear cache**: `Agent.clear_rag_cache(agent)` drops cached pipelines.
- **CLI**: `python manage.py warmup_rag --agent <name>` warms up from the command line (see the [CLI guide](/cli/)).

### Citations

Retrieved sources are streamed back as numbered citations. `get_citation_registry()` returns a fresh per-turn registry, and `citation_formatter_class` controls how sources are rendered for the model. The `DefaultCitationFormatter` numbers sources so the model can reference them inline.

---

## Storage

Conversations live in **threads** of **messages**. Two storage adapters ship with the SDK:

- `MemoryStorageAdapter`: in-memory, useful for tests and prototypes
- `DbStorageAdapter`: persists via the Django ORM

{{< callout type="info" >}}
`MemoryStorageAdapter` is in-memory only: use `DbStorageAdapter` whenever conversations must survive a restart.
{{< /callout >}}

`get_storage_adapter(thread_id)` locates which adapter actually holds a thread (querying all registered adapters) and returns the right instance, falling back to the agent's configured `storage_adapter`.

```python
class MyAgent(Agent):
    storage_adapter = DbStorageAdapter
```

`storage_adapter` is the attribute you configure (the same name is also used as the effective instance attribute in `__init__`). `Agent` additionally declares a `storage` class annotation for the same purpose: either way, the effective adapter defaults to `MemoryStorageAdapter` and comes from `storage_adapter` at runtime.

---

## Structured Output

`Agent.run()` returns a Pydantic model when a `response_format` is set:

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int

class MyAgent(Agent):
    response_format = Movie

    async def get_run_adapter(self, thread_id=None, user=None):
        return Run(generator=self.get_llm())

# agent.run(messages) -> Movie | str
```

Pass `response_format=None` explicitly to `run()` to disable structured output for a single call.

`run()` also takes `tools` (default `False`): pass `tools=True` and the agent's own
`get_tools()` are resolved and run to completion headless - no streaming, one call in,
one final answer out. Opt-in, not opt-out: resolving tools reaches every configured
integration, so a one-shot call (title generation, structured extraction, a workflow
step) does not pay that cost unless it explicitly asks for it. Setting a
`response_format` skips tools regardless of the flag - structured output and a tool
loop together aren't supported yet, see
[Structured output from a tool loop](/manual/generators/#structured-output-from-a-tool-loop).

This only ever applies to the SDK's own `Run` adapter - a `get_run_adapter()` override
that returns something else is left alone, rather than a tool loop being forced onto
it. See [Headless tool runs](/manual/stream-and-run/#headless-tool-runs) for the exact
call chain and the diagram: it's governed by *this* agent's own `max_agent_steps` and
`max_tool_calls`, via the same assembly a streamed subagent delegation uses - never a
generic default.

---

## Permissions

Each agent declares `permissions`: classes that gate operations like `CHAT`, `VIEW_THREAD`, and `USE_INTEGRATION`. The demo shows per-domain overrides:

```python
# settings.py
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}
```

`Agent.as_view()` and `Agent.history()` check permissions before doing anything, raising `PermissionDenied` when the user lacks access. See the [Views and Routing guide](/views-and-routing/) for wiring.

---

## Files

Set `file_upload = True` to accept file uploads on threads; leaving it `False` rejects uploads to the agent's threads with a `PermissionDenied`. Each `FilePipeline` pairs a processor (e.g. `TextFileProcessor`) with optional transforms, including LLM extraction agents, that run over the uploaded file:

```python
class MyAgent(Agent):
    file_upload = True
    file_pipelines = [
        FilePipeline(
            TextFileProcessor(),
            transforms=[DocumentExtractionTransform(ExtractionAgent())],
        ),
    ]
```

See the [Files reference](/manual/files/) for the shipped processors (Text, CSV, JSON, Docx, Pptx, Xlsx), transforms, the upload/processing lifecycle, and upload settings (`AI_SDK_MAX_UPLOAD_SIZE`, `AI_SDK_ALLOWED_FILES`, `AI_SDK_MEMORY_FILE_PIPELINE`).

---

## Suggestions

Set `suggestion_generator` to stream follow-up questions after the assistant's reply:

```python
from django_ai_sdk.suggestions import DefaultSuggestionGenerator

class MyAgent(Agent):
    suggestion_generator = DefaultSuggestionGenerator
```

---

## Reference Agents

The demo project ships seven agents in `demo/apps/agents/`, each showcasing a different pattern:

| Agent | Pattern |
| --- | --- |
| `PirateBasicAgent` | Full-featured agent: RAG, tools, integrations, citations, suggestions, files |
| `PirateSwarmAgent` | Coordinator + hidden pirate subagents (`agents = [...]`) |
| `DeepResearchAgent` | Coordinator + hidden web-research subagent |
| `DefaultRuntimeAgent` | Base for DB-configured runtime agents |
| `WorkspaceAgent` | Artifact-based workspace workflows |
| `PirateExtractionAgent` | Structured extraction used inside file pipelines |
| `ResearchPlannerAgent` | Hidden web-research worker (search + fetch tools) |

Browse them in the repository for working examples of everything in this guide.
