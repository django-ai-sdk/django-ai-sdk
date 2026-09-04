---
title: Agent
type: docs
weight: 102
---

Deep dive into the `Agent` class: what it configures and which hooks the SDK calls.

{{< callout type="info" >}}
Building an app? See the [Agents guide](/agents/). This page is the internal reference.
{{< /callout >}}

## Responsibilities

The `Agent` is the coordinator. It:

- Declares AI personality: `name`, `model`, `instructions`
- Picks a protocol handler and a storage adapter
- Provides the adapter hooks `Stream`/`Run` are built from
- Exposes lifecycle hooks for RAG warmup, citations, and suggestions

## Configuration

Key class attributes (full table in the [Agents guide](/agents/)):

| Variable | Default | Purpose |
| --- | --- | --- |
| `name` | None | Display name |
| `model` | `None` | Model identifier passed to the generator |
| `llm` | `openai_responses_chat` | Generator factory, assigned uncalled |
| `llm_kwargs` | `None` | Vendor generation parameters for that factory |
| `instructions` | built-in prompt | System prompt, built with `prompt()` |
| `protocol` | `VercelProtocolHandler` | Wire-format handler class |
| `storage_adapter` | `MemoryStorageAdapter` | Fallback storage class |
| `tools` | `[]` | Class-level tool providers |
| `integrations` | `[]` | Integration names whose tools reach this agent |
| `artifacts` | `[]` | `ArtifactSchema` subclasses exposed as submission tools |
| `rag_provider` | `None` | `RAGProvider` instance for RAG |
| `memories` | `[]` | Default connected memories |
| `max_history` | `None` | Cap on messages sent to the model |
| `response_format` | `None` | Pydantic model for structured `run()` output |
| `title_generation` | `True` | Auto-generate thread titles |
| `permissions` | `None` | Permission classes gating operations; `None` or `[]` uses the `agent` domain default |
| `warmup_on_init` | `False` | Warm up RAG on instantiation |
| `hidden` / `abstract` | `False` | Registry behavior flags |

`Agent.__init__` instantiates `self.protocol()` into `self.protocol_handler`.

## Adapter Hooks

The SDK calls two abstract hooks; you implement them:

| Hook | Returns | Used by |
| --- | --- | --- |
| `get_pipeline_adapter(thread_id, user)` | `Stream` | `Agent.as_view()` (streaming chat) |
| `get_run_adapter(thread_id, user)` | `Run` | `Agent.run()` (titles, extraction, jobs) |

```python
async def get_pipeline_adapter(self, thread_id=None, user=None):
    generator = self.get_llm()
    tool_agent = ToolAgent(
        config=ToolAgentConfig(
            model=self.get_model(),
            system_prompt=self.get_system_prompt(),
            tools=await self.get_tools(thread_id=thread_id or "", user=user),
        ),
        generator=generator,
    )
    return Stream(
        pipeline=tool_agent.pipeline(),
        generator=generator,
        storage_adapter=await self.get_storage_adapter(thread_id),
    )
```

A worker-only agent (`hidden = True`, never in chat) can leave `get_pipeline_adapter()` unimplemented.

## Storage Resolution

`get_storage_adapter(thread_id)` queries all registered adapters for the thread and returns a bound instance, falling back to the agent's `storage_adapter`. See [Storage](../storage/).

## Tool Assembly

`get_tools(thread_id, user)` combines, in order:

1. Class-level `tools` providers (each called with `thread_id`/`user` kwargs)
2. Integration tools (`integrations` list)
3. Artifact submission tools (`artifacts`)
4. Subagent agent tools (`agents` list — one `ComponentTool` per subagent)

`get_rag_tools(thread_id, user)` appends one retrieval tool per active memory when `rag_provider` is set. See [RAG](../rag/).

## Subagent Delegation

![Subagent Flow](/images/graphs/subagent_flow.png)

Declaring `agents = [SubagentA, SubagentB]` on an `Agent` turns each subagent into a delegate-able Haystack `ComponentTool`. Only plain `Agent` subclasses can appear in that list — a `RuntimeAgent` (DB-configured) never can, since its `__init__` requires an `AgentSettings` row and `agents = [...]` holds bare classes with no way to pass one in. `hidden = True` on a subagent is a convention, not an enforcement.

- `get_agent_tools(thread_id, user)` builds them, per class in `self.agents`:
  1. **Cycle guard, declaration-level.** Skip `subagent_cls is self.__class__` (no self-delegation) and any class path already seen this call.
  2. **Permission check.** `AgentService.has_perms(user, Operation.VIEW_AGENT, agent=subagent_cls)` — skip on `PermissionDenied` rather than failing the whole tool list.
  3. **Build it.** `build_subagent(subagent_cls, thread_id, user)` → `(HaystackAgent, agent_id)`, or `None` for a cyclic graph (see below).
  4. **Name the tool.** `subagent_tool_name(subagent_cls)` slugifies the display name. A collision appends a short suffix derived from the colliding class's `agent_id` (`research_planner_7f3a2c`) — guaranteed not to collide again, unlike an incrementing counter; the same pattern `get_rag_tools()` already uses for a memory-tool-name collision.
  5. **Wrap it** as a `ComponentTool` scoped to a single `task` string parameter — delegation, not remote tool invocation.
- `build_subagent(cls, thread_id, user)` (in `django_ai_sdk.agents.subagent`) builds the subagent with its own generator/tools/prompt plus a `task` user-variable. A second cycle guard here — a `ContextVar` tracking the chain of classes currently being built on this asyncio task — catches a graph cycle the declaration-level check can't (A delegates to B, B delegates to A, several hops apart). Loop limits and hooks come from `default_hooks(subagent)` (`agents/tool_agent.py`): always `LogToolCallsHook`, plus `ToolCallBudgetHook(subagent.max_tool_calls)` when that's set (default `6`; `None` disables the cap), and `max_agent_steps=subagent.max_agent_steps` (default `6`). This is the same function [`Agent.run(tools=True)`](../stream-and-run/#headless-tool-runs) uses for a headless run — a subagent class is governed identically whether it's delegated to live or run by itself.
- The subagent's response becomes the tool output via `outputs_to_string` — see [Handoff output](#handoff-output) for what that contains and how a cut-short run is recovered. The coordinator remains the single source of streamed *text*, but the subagent's tool calls/results stream to the client via `stream_subagent_tools` (default on) and are persisted into the conversation history — `SubagentStreamFilter` forwards only tool chunks, in the order they happen.
- The coordinator's `get_system_prompt()` appends an `Available subagents:` roster.

### Handoff output

The tool output the coordinator receives is built from the subagent's **whole
conversation**, not just its last message. That matters because a subagent can
run out of budget or steps *after* doing the work: reading only the last message
would hand back a "you have reached the maximum of N tool calls" notice and throw
every result away.

`subagent_response()` (`django_ai_sdk/agents/subagent.py`) returns, in order:

1. **The response** — the subagent's final written answer.
2. **Sources consulted** — one `tool(args)` line per call it made, names and
   arguments only, never bodies. Always appended, so the coordinator can cite
   even when the response forgot to.
3. **A digest** — only when there is no response: the gathered tool results
   themselves, under `"The run was cut short before a response was
   finished."` so the coordinator presents them as partial rather than as a
   finished answer. Capped by `AI_SDK_SUBAGENT_DIGEST_LIMIT` (default `6000`
   characters, `0` disables), budgeted across all results so one large page
   cannot crowd out the rest.

Before the digest is reached, `SubagentStreamFilter` makes **one tool-less LLM
call** asking the subagent to write its response from what it already
gathered. This is what makes the outcome independent of `max_agent_steps`
being generous enough to leave a spare step after the tool budget fires. It
runs only on that path, once, and a failure there falls through to the digest
rather than propagating.

`ToolCallBudgetHook` supports this by *answering* a call it refuses rather than
erasing it: the pending `assistant(tool_calls)` message stays and each call gets
a stub `tool` result. The turn remains well-formed for the chat APIs, no tool
executes, the instruction rides a message the model must read — and the client
sees the call resolve instead of a tool part stuck with no output.

### Attribution

Every stored tool call records who ran it, so a coordinator's own work is never
confused with its crew's:

| Key on the stored `tool_call` | Meaning |
| --- | --- |
| `handoff` | The coordinator delegating **to** the named subagent. |
| `agent` | A tool the named subagent called **itself** (`search_web`, …). |
| neither | An ordinary tool the coordinator called directly. |

`SubagentStreamFilter` tags each chunk it forwards with the subagent's name
(`django_ai_sdk.subagent` in `StreamingChunk.meta`); `Stream` reads that back,
together with the handoff tool names it discovers from the pipeline, and
`StreamWriter` writes both onto the tool call. Both keys are additive, so
history written before this existed still loads.

This is display attribution by name, on stored tool calls — a separate system
from the tracing span's `agent_id`/`agent_name` columns (see
[Subagents in a trace](/manual/tracing/#subagents-in-a-trace)), which exist for
stable lookups and are keyed by id precisely to avoid the name-collision this
system accepts for display.

The same marks reach the browser. The Vercel handler sends them as AI SDK
provider metadata on `tool-input-start` and `tool-input-available`, which the
client stores as `callProviderMetadata` — it survives the transition to
`output-available`, so tagging the call start labels the whole lifecycle:

```json
{ "providerMetadata": { "sdk": { "agent": "Research Planner" } } }
```

`from_chat_messages` re-emits it as `callProviderMetadata` so the label survives
a page reload. A tool the coordinator called itself carries no such key, and its
payload is unchanged.

### Handoff history

A subagent's report comes back as the tool *result* of the delegation call, and
that is what gets persisted. Replaying only the coordinator's text would hand
the next turn its own restatement of the report instead of the report, so
`Stream.get_messages` rebuilds each handoff as the assistant/tool pair it
originally was:

```
user       "Research Example Company B.V."
assistant  (tool_calls=[research_planner(task=…)])
tool       "Full report: …"          <- the subagent's findings, verbatim
user       "Which city?"
```

Only calls marked `handoff` replay. A subagent's own tool calls stay in history
for the UI and traces but are not fed back to the model — the report already
summarizes them. Each replayed result is capped by
`AI_SDK_HISTORY_TOOL_OUTPUT_LIMIT` (default `4000` characters, `0` disables) so
one large research turn cannot crowd out the context window.

Because a tool message has to follow its call directly, a handoff message is
never merged into its neighbours; `merge_messages` applies to the plain-text
runs around it.

## RAG Lifecycle Hooks

Classmethods that delegate to `rag_provider` (no-op without a provider):

| Hook | Effect |
| --- | --- |
| `MyAgent.warmup(agent, memory_id)` | Pre-build and cache pipelines |
| `MyAgent.reindex(agent, memory_id, force_rebuild)` | Drop cache entry and rebuild |
| `MyAgent.clear_rag_cache(agent)` | Drop all cached instances |

## Citations & Suggestions

| Hook | Returns |
| --- | --- |
| `get_citation_registry()` | Fresh per-turn `CitationRegistry` |
| `get_citation_formatter()` | From `citation_formatter_class` |
| `get_suggestion_generator()` | `self.suggestion_generator(agent=self)` |

## Generator

| Hook | Returns |
| --- | --- |
| `get_llm(**kwargs)` | A chat generator built by the `llm` factory (default `openai_responses_chat`), with this agent's model and `llm_kwargs` applied. See [Generators](/manual/generators/). |

## Entry Point

`as_view(protocol_messages, thread_id=None, user=None)`:

1. Checks `CHAT` permissions
2. Converts protocol messages via `self.protocol_handler.to_chat_messages()`
3. Stores the newest user message when a thread is active
4. Builds the pipeline adapter and returns a `StreamingHttpResponse`

```python
response = await agent.as_view(payload.messages, thread_id=..., user=request.user)
```

Next: [Agent Registry](../agent-registry/), registration and stable-ID resolution.
