from __future__ import annotations

import asyncio
import dataclasses
import uuid
from abc import ABC
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from django_ai_sdk.agents.mixins import AgentInfoMixin
from django_ai_sdk.agents.registry import registry
from django_ai_sdk.citations import (
    CitationFormatter,
    CitationRegistry,
    DefaultCitationFormatter,
)
from django_ai_sdk.common import ChatMessage, Prompt, prompt
from django_ai_sdk.conversation.utils import generate_thread_title, get_title_sanity_limit
from django_ai_sdk.integrations.registry import get_integrations
from django_ai_sdk.logger import get_logger
from django_ai_sdk.permissions import (
    BasePermission,
    Operation,
    check_object_permissions,
    check_permissions,
    get_agent_permissions,
)
from django_ai_sdk.prompts import build_title_generation_prompt
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.rags import queryset_to_rag_documents
from django_ai_sdk.responses import stream_response
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.storage.schemas import ThreadDetail
from django_ai_sdk.storage.services import ThreadService

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.common import Prompt
    from django_ai_sdk.files.pipeline import FilePipeline
    from django_ai_sdk.rags.schemas import RagDocument
    from django_ai_sdk.storage.base import BaseStorageAdapter
    from django_ai_sdk.suggestions import SuggestionGenerator


T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


def _namespaced(integration_name: str, tool: Any, hint: str = "") -> Any:
    """Rename tool to {integration_name}_{tool.name} and prepend `hint` (if set)
    to its description.

    Nothing stops two unrelated MCP servers from defining the same tool name
    (GitHub and Linear both have list_issues), and Haystack requires unique names
    across everything handed to one agent, so without this, enabling two
    integrations that happen to collide would fail agent construction outright.
    """
    try:
        updates: dict[str, Any] = {"name": f"{integration_name}_{tool.name}"}
        if hint:
            updates["description"] = f"{tool.description}\n\n{hint}" if tool.description else hint
        return dataclasses.replace(tool, **updates)
    except (TypeError, AttributeError):
        logger.warning(
            "Could not namespace tool %r from integration %r — left as-is, may "
            "collide with another integration's tool.",
            getattr(tool, "name", tool),
            integration_name,
        )
        return tool


class Agent(ABC, AgentInfoMixin):
    """
    Base class for AI agents in the Django AI SDK.

    Provides a consistent interface for creating agents that work
    with the streaming view system. Each agent encapsulates:
    - Name and instructions (AI personality)
    - Tools (as methods for co-location)
    - Adapter creation (backend-specific logic)
    - Protocol handler for message conversion

    This class includes the AgentInfoMixin which provides:
    - info(): Get agent metadata
    - agent_id: Stable UUID v5 ID

    Registration:

    Method 1: Settings-based (recommended)
        # In your settings.py:
        AI_SDK_AGENTS = [
            "myapp.agents.MyAgent",
        ]

        # In your AppConfig.ready():
        from django.utils.module_loading import import_string
        from django.conf import settings
        from django_ai_sdk.agents.registry import registry

        for path in resolve_setting('AI_SDK_AGENTS', []):
            import_string(path)
        registry.setup()

    Method 2: Decorator-based
        from django_ai_sdk.agents import auto_register

        @auto_register
        class MyAgent(Agent):
            name = "My Bot"
            model = "gpt-4"

    Every concrete subclass also auto-registers on definition (__init_subclass__),
    so either method above is really just what gets the module imported. An
    abstract shared base (abstract = True) is skipped regardless of how it's
    reached — see AgentRegistry.register().

    Usage:
        from django_ai_sdk.protocols.vercel import VercelProtocolHandler

        class MyAgent(Agent):
            name = "My Bot"
            model = "gpt-4"
            instructions = prompt("You are a helpful agent...")
            protocol = VercelProtocolHandler

            async def get_pipeline_adapter(self): return SomeAdapter(...)

        # Get registered agent by UUID
        from django_ai_sdk.agents.registry import registry
        agent = registry.get(agent_id)
        return await agent.as_view(protocol_messages)
    """

    # Name of agent.
    name: str

    # Short description of agent.
    description: str

    # Model identifier for LLM backend.
    model: str

    # System prompt instructions for the agent.
    instructions: Prompt = prompt("You are a helpful agent.")

    # Permission classes used to gate access to this agent's operations.
    # None or empty means no permissions are required.
    permissions: list[type[BasePermission]] | None = None

    # Default list of connected memories.
    memories: list[str] = []

    # Tools are callable from agent.
    tools: list[str] = []

    # ArtifactSchema subclasses to register as tools in stream pipelines.
    artifacts: list[type[BaseModel]] = []

    # Delegate-able subagent classes.
    agents: list[type[Agent]] = []

    # Loop limits when this agent runs as a subagent.
    max_agent_steps: int = 6
    max_tool_calls: int | None = 6

    # Protocol handler class for converting protocol messages
    protocol = None

    # Storage adapter class for persisting threads and messages
    storage: type[BaseStorageAdapter] | None = None

    # Set to an ArtifactSchema subclass to enable structured output for run() calls.
    response_format: type[BaseModel] | None = None

    # If True, hide from registry
    hidden: bool = False

    # If True, this is a shared base meant only to be subclassed.
    abstract: bool = False

    # If Agent should automatically warm up after initialization.
    warmup_on_init: bool = False

    # RAG provider: set to a RAGProvider instance to enable RAG, None disables RAG.
    rag_provider: Any = None

    # Maximum conversation history to send to LLM (None = unlimited).
    max_history: int | None = None

    # Enable file upload UI for this agent's threads.
    file_upload: bool = False

    # Declare one FilePipeline per supported file type.
    # First pipeline whose processor accepts the uploaded file is used.
    # Empty = fall back to get_default_file_pipeline() (TextFileProcessor, no LLM extraction).
    file_pipelines: list[FilePipeline] = []

    # Enable automatic thread title generation based on chat messages.
    title_generation: bool = True

    # Hard cap on documents fetched for RAG indexing (prevents OOM on large memories).
    rag_document_limit: int = 10_000

    # Citation formatter used to render retrieved documents for the LLM.
    citation_formatter_class: type[CitationFormatter] = DefaultCitationFormatter

    # Suggestion generator class for follow-up questions.
    suggestion_generator: type[SuggestionGenerator] | None = None

    # Generator factory from django_ai_sdk.generators, e.g. `openai_responses_chat`.
    # Assign the factory itself, never call it: get_llm() builds it with this
    # agent's model.
    llm: Callable[..., Any] | None = None

    # Vendor generation parameters, mapped onto Haystack's `generation_kwargs`.
    llm_kwargs: dict[str, Any] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register Agent subclasses in the registry.

        This enables settings-based registration:
        - When a class is imported via AI_SDK_AGENTS, it auto-registers
        - Works together with @auto_register decorator
        """
        super().__init_subclass__(**kwargs)
        # A plain function set as a class attribute would bind as a method and
        # receive `self` as its first argument, so keep `llm` unbound.
        llm = cls.__dict__.get("llm")
        if callable(llm) and not isinstance(llm, staticmethod):
            cls.llm = staticmethod(llm)
        # Don't register the base Agent class itself, or classes that
        # manage their own registration (e.g. RuntimeAgent).
        if cls.__name__ != "Agent" and not getattr(cls, "_skip_auto_register", False):
            registry.register(cls)

    @classmethod
    async def warmup(cls, agent: Agent, memory_id: str | None = None) -> None:
        """
        Warm up the RAG pipeline.

        Delegates to rag_provider.warmup() if configured.
        If no RAG provider is set, returns immediately.

        Args:
            agent: Instance of the agent
            memory_id: Optional memory
        """
        if agent.rag_provider is None:
            logger.debug("No RAG provider configured, skipping warmup")
            return

        await agent.rag_provider.warmup(agent, memory_id)

    @classmethod
    def clear_rag_cache(cls, agent: Agent) -> None:
        """
        Clear the RAG cache.

        Args:
            agent: Instance of the agent
        """
        if agent.rag_provider is not None:
            agent.rag_provider.clear_cache()
            logger.debug("RAG cache cleared via provider")

    @classmethod
    async def reindex(
        cls,
        agent: Agent,
        memory_id: str | None = None,
        force_rebuild: bool = False,
    ) -> Any:
        """
        Reindex the RAG pipeline for this agent.
        Delegates to rag_provider.reindex() if configured.

        Args:
            agent: Instance of the agent
            memory_id: Optional memory ID for document source
            force_rebuild: If True, forces a complete rebuild of the index.
                          For persistent storage backends (like Qdrant), this
                          will delete and recreate the index from scratch.

        Returns:
            The reindexed RAG pipeline, or None if no provider
        """
        if agent.rag_provider is None:
            logger.debug("No RAG provider configured, skipping reindex")
            return None

        return await agent.rag_provider.reindex(agent, memory_id, force_rebuild)

    def __init__(self) -> None:
        # Protocol handler setup
        self.protocol_handler = (
            self.protocol()
            if hasattr(self, "protocol") and self.protocol is not None
            else VercelProtocolHandler()
        )

        # Storage adapter setup
        self.storage_adapter = (
            self.storage_adapter
            if hasattr(self, "storage_adapter") and self.storage_adapter is not None
            else MemoryStorageAdapter
        )

        if self.warmup_on_init and self.rag_provider is not None:
            try:
                asyncio.get_running_loop().create_task(self.rag_provider.warmup(self, None))
            except RuntimeError:
                pass  # No running loop (e.g. management command) — warmup skipped

    @property
    def is_runtime(self) -> bool:
        """Return false by default, override in subclasses that are runtime agents."""
        return False

    @property
    def config(self) -> AgentSettings | None:
        """Return the AgentSettings for runtime agents"""
        return None

    async def get_storage_adapter(self, thread_id: str | None = None) -> BaseStorageAdapter | None:
        """
        Get storage adapter for the given thread.

        If thread_id is provided, finds which storage actually contains the thread
        queries all registered adapters, then returns that adapter type.
        If thread not found anywhere, falls back to agent's configured storage.
        If thread_id is None, returns None

        Args:
            thread_id: The thread ID to get storage for

        Returns:
            Storage adapter instance bound to thread_id
        """
        if thread_id is None:
            return None

        from .storage.base import StorageAdapterRegistry

        # Find storage of this thread
        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter_class.get_thread(thread_id)
            if thread:
                return adapter_class(thread_id)

        # Thread not found in any registered adapter — fall back to this
        # agent's configured storage (always set in __init__).
        return self.storage_adapter(thread_id)

    async def get_file_pipeline(self, file: object) -> FilePipeline | None:
        """Return the first FilePipeline whose processor accepts file, or None."""
        for pipeline in self.file_pipelines:
            if await pipeline.accepts(file):
                return pipeline
        return None

    def get_name(self) -> str:
        """Return the agent's display name."""
        return self.name or "Unnamed Agent"

    def get_instructions(self) -> Prompt:
        """Return formatted system instructions as a single string."""
        return self.instructions

    def get_system_prompt(self) -> str:
        """
        Alias for get_instructions()
        """
        base = self.get_instructions()
        if not self.agents:
            return base

        delgators = "\n".join(
            f"- {subagent.name}: {subagent.description or subagent.__doc__ or 'no description'}"
            for subagent in self.agents
            if subagent is not self.__class__
        )
        if not delgators:
            return base
        return f"{base}\n\nAvailable subagents:\n{delgators}"

    def get_title_generation_prompt(self) -> Prompt:
        """Return the system prompt used to generate a thread title.

        Defaults to a prompt capped at the title sanity limit, not the much
        larger `Thread.title` column `max_length` - the column width is a
        storage ceiling, not a reasonable target length for a title, and
        quoting it here would let the model "correctly" produce something
        long enough to fail `generate_thread_title`'s own sanity check.
        Override to customize tone/format.
        """
        return build_title_generation_prompt(get_title_sanity_limit())

    def get_model(self) -> str:
        """Return the model identifier."""
        return self.model or ""

    def get_citation_formatter(self) -> CitationFormatter:
        """Return the formatter used to render retrieved docs for the LLM.

        Override to inject formatter dependencies; otherwise swap formatters by
        setting the citation_formatter_class class attribute.
        """
        return self.citation_formatter_class()

    def get_citation_registry(self) -> CitationRegistry:
        """Return a fresh per-turn registry so citation indices reset between turns."""
        return CitationRegistry()

    def get_llm(self, **kwargs: Any) -> Any:
        """Build this agent's chat generator.

        Uses the `llm` factory if set, otherwise OpenAI's Responses API. Keyword
        arguments override `llm_kwargs`, and the agent always supplies the model.
        """
        from django_ai_sdk.generators import merge_generation_kwargs, openai_responses_chat

        factory = self.llm or openai_responses_chat
        if not callable(factory):
            raise TypeError(f"{self.__class__.__name__}.llm must be a generator factory.")
        if self.llm_kwargs:
            kwargs["generation_kwargs"] = merge_generation_kwargs(
                self.llm_kwargs, kwargs.get("generation_kwargs")
            )
        return factory(model=self.get_model() or None, **kwargs)

    def get_suggestion_generator(self) -> SuggestionGenerator | None:
        """Return a configured SuggestionGenerator, or None to disable.

        Uses self.suggestion_generator class attribute if set.
        Override this method in subclasses for full control.
        """
        if not self.suggestion_generator:
            return None
        return self.suggestion_generator(agent=self)

    async def get_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Build tool objects for a request. Override in subclasses for full control.

        Each callable in the class-level `tools` list is called with context kwargs
        and may return a single tool or a list. Providers can use any subset:
          def get_my_tool(thread_id="", user_id="", **kwargs): ...
          def get_my_tool(user_id="", **kwargs): ...

        """
        # class-level tools
        tools = getattr(self.__class__, "tools", [])
        result = []
        for tool in tools:
            items = tool(thread_id=thread_id, user=user)
            if isinstance(items, list):
                result.extend(items)
            else:
                result.append(items)

        # integration tools
        result.extend(await self._get_integration_tools(user, thread_id=thread_id))

        # artifact tools
        result.extend(await self.get_artifact_tools(thread_id=thread_id, user=user))

        # subagent tools
        result.extend(await self.get_agent_tools(thread_id=thread_id, user=user))

        return result

    async def get_agent_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Build one ComponentTool per subagent declared in ``self.agents``.

        Each subagent becomes a native Haystack tool the coordinator can call
        to delegate a sub-task. Guards:

        - The agent itself and duplicate classes are skipped.
        - Subagents the user lacks ``VIEW_AGENT`` permission on are skipped.
        - Tool names are derived from the subagent display name and deduped.

        The conversation is mapped into the subagent via ``inputs_from_state``,
        so it receives the same thread context as the coordinator; its final
        ``last_message`` text is returned to the coordinator as the tool output.
        """
        from haystack.tools.component_tool import ComponentTool

        from django_ai_sdk.agents.services import AgentService
        from django_ai_sdk.agents.subagent import (
            SubagentStreamFilter,
            build_subagent,
            subagent_response,
            subagent_tool_name,
        )
        from django_ai_sdk.permissions import PermissionDenied

        tools: list[Any] = []
        seen_classes: set[str] = set()
        used_names: set[str] = set()

        for subagent_cls in self.agents:
            path = f"{subagent_cls.__module__}.{subagent_cls.__qualname__}"
            if path in seen_classes or subagent_cls is self.__class__:
                continue
            seen_classes.add(path)

            try:
                await AgentService.has_perms(user, Operation.VIEW_AGENT, agent=subagent_cls)
            except PermissionDenied:
                logger.info(
                    f"Skipping subagent {subagent_cls.__name__!r}: user lacks VIEW_AGENT permission"
                )
                continue

            built = await build_subagent(subagent_cls, thread_id=thread_id, user=user)
            if built is None:
                continue
            sub_agent, agent_id = built

            name = subagent_tool_name(subagent_cls)
            if name in used_names:
                original, name = name, f"{name}_{agent_id.replace('-', '')[:6]}"
                logger.warning(f"Subagent tool name {original!r} collided, renamed to {name!r}")
            used_names.add(name)
            logger.info(
                f"Enabled subagent tool {name!r} ({subagent_cls.__name__}) on coordinator {self.__class__.__name__}"
            )

            tools.append(
                ComponentTool(
                    component=SubagentStreamFilter(
                        sub_agent, name=subagent_cls.name or name, agent_id=agent_id
                    ),
                    name=name,
                    description=(
                        subagent_cls.description
                        or f"Delegate this task to {subagent_cls.__name__}."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": (
                                    "A self-contained description of the task to delegate "
                                    "to this subagent. Copy the user's request verbatim"
                                    "do not paraphrase, autocorrect, or expand it."
                                ),
                            }
                        },
                        "required": ["task"],
                    },
                    # whole conversation, not just the last message: a subagent
                    # that ran out of budget or steps still did the work.
                    outputs_to_string={
                        "source": "messages",
                        "handler": subagent_response,
                    },
                    inputs_from_state={"messages": "messages"},
                )
            )

        return tools

    async def get_artifact_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Build artifact submission tools from the class-level `artifacts` list."""
        return [
            artifact_cls.as_tool(thread_id=thread_id, user=user)
            for artifact_cls in getattr(self.__class__, "artifacts", [])
        ]

    async def get_rag_tools(
        self,
        thread_id: str,
        *,
        citation_registry: CitationRegistry | None = None,
        citation_formatter: CitationFormatter | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Build RAG tools from active ThreadMemory links for a thread.

        Each memory with documents becomes a framework-specific tool.
        Memories with 0 documents are skipped with a warning log.

        Returns:
            List of tool objects to include in the pipeline.
        """
        if not self.rag_provider or not thread_id:
            return []

        from django_ai_sdk.memories.services import MemoryService

        tools: list[Any] = []
        memories = await MemoryService.get_thread_memories(
            thread_id,
            user=user,
        )

        used_names: set[str] = set()

        for memory in memories:
            spec = await memory.get_tool_spec()
            if spec.name in used_names:
                spec.name = f"{spec.name}_{str(memory.id).replace('-', '')[:6]}"
                logger.warning(
                    "Tool name collision for memory '{}', renamed to '{}'",
                    memory.name,
                    spec.name,
                )
            used_names.add(spec.name)

            tool = await self.rag_provider.get_tool(
                self,
                str(memory.id),
                spec=spec,
                citation_registry=citation_registry,
                citation_formatter=citation_formatter,
            )
            if tool is None:
                logger.warning(
                    "Memory '{}' has 0 documents — no RAG tool created",
                    memory.name,
                )
                continue
            tools.append(tool)

        return tools

    async def get_rag_queryset(self, memory_id: str | None = None) -> Any:
        """
        Override to return a Django QuerySet of documents for RAG.

        Subclasses should override this method to customize which documents
        are included in RAG.

        Args:
            memory_id: Optional memory ID to filter documents

        Returns:
            Django QuerySet of Document objects
        """

        # TODO: this ia to entangled with our own Entry model
        # For now this is fine, but we might want to decouple it
        # and provide a hint for overriding, but leave out the
        # implementation details.
        from django_ai_sdk.memories.models import Entry

        fields = ("id", "content", "data", "name", "memory_id")
        if memory_id:
            return Entry.objects.filter(memory_id=memory_id).only(*fields).order_by("-updated_at")
        return Entry.objects.none()

    async def get_rag_documents(self, memory_id: str | None = None) -> list[RagDocument]:
        """
        Get documents for RAG as RagDocuments.

        This method uses get_rag_queryset() to get the Django QuerySet,
        then converts it to RagDocuments using the utility function.

        Override get_rag_queryset() to customize document selection.

        Args:
            memory_id: Optional memory ID to filter documents

        Returns:
            List of RagDocuments (vendor-neutral)
        """

        logger.info(f"[get_rag_documents] memory_id={memory_id}, agent={self.__class__.__name__}")
        logger.debug(f"Fetching RAG documents for {self.__class__.__name__}, memory_id={memory_id}")

        queryset = await self.get_rag_queryset(memory_id)
        logger.info(f"[get_rag_documents] Got queryset, type={type(queryset).__name__}")

        rag_docs = await queryset_to_rag_documents(queryset, memory_id=memory_id)
        logger.info(f"[get_rag_documents] Converted to {len(rag_docs)} RagDocuments")

        logger.info(f"Fetched {len(rag_docs)} documents for RAG")
        for i, doc in enumerate(rag_docs):
            title = doc.title or "N/A"
            logger.debug(
                f"  Document {i + 1}: id={doc.id}, title='{title}', content_len={len(doc.content)}"
            )

        return rag_docs

    async def get_rag_pipeline(self, memory_id: str | None = None) -> Any:
        """
        Get RAG pipeline/adapter for retrieval-augmented generation.

        Override this method in subclasses to enable RAG functionality.

        For Haystack: Return a RAG adapter: has .get_retriever() method
        For OpenAI: Return a RAG adapter: has .retrieve() method

        Args:
            memory_id: Optional memory ID to use for document retrieval

        Returns:
            RAG adapter instance, or None if RAG is not enabled
        """
        return None

    _UNSET: Any = object()

    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | Prompt | None = None,
        response_format: type[T] | None = _UNSET,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        tools: bool = False,
    ) -> T | str | None:
        """Run LLM calls directly from adapter.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt override
            response_format: Optional Pydantic model for structured output. Pass None
                explicitly to disable structured output even if the agent has a
                default response_format set.
            thread_id: Optional thread ID (forwarded to get_run_adapter)
            user: Optional user (forwarded to get_run_adapter)
            tools: Whether to resolve the agent's tools and run a tool-calling loop,
                governed by this agent's own max_agent_steps/max_tool_calls — the
                same limits a streamed subagent delegation uses, not a generic
                default. Opt-in: resolving tools reaches every configured
                integration, so a one-shot call (title generation, structured
                extraction, ...) does not pay that cost unless it explicitly asks
                for it. Ignored when response_format is set — tools and structured
                output together are not supported yet (see Run.run).

        Returns:
            Response string, or parsed Pydantic model if response_format is set
        """
        resolved = self.response_format if response_format is self._UNSET else response_format
        adapter = await self.get_run_adapter(thread_id=thread_id, user=user)
        rendered_prompt = system_prompt if system_prompt is not None else self.get_system_prompt()

        # Only the SDK's own plain Run is ours to add a tool loop to. A custom
        # get_run_adapter() override returns something else on purpose — it is
        # left alone, calling its own .run() unchanged, rather than having a
        # tool loop forced onto it.
        from django_ai_sdk.adapters.base import Run

        if tools and resolved is None and isinstance(adapter, Run):
            return await self._run_own_tools(adapter, messages, rendered_prompt, thread_id, user)

        return await adapter.run(
            messages=messages,
            system_prompt=rendered_prompt,
            response_format=resolved,
        )

    async def _run_own_tools(
        self,
        adapter: Any,
        messages: list[ChatMessage],
        system_prompt: str,
        thread_id: str | None,
        user: AbstractBaseUser | AnonymousUser | None,
    ) -> str | None:
        """Run this agent's own tools to completion, headless (no streaming).

        Governed by this agent's own `max_agent_steps`/`max_tool_calls`/hooks —
        the same assembly `build_subagent` uses for a streamed delegation — so a
        headless one-shot run and a streamed subagent are never governed
        differently.
        """
        from django_ai_sdk.agents.tool_agent import ToolAgent, default_hooks

        haystack_agent = ToolAgent.build_agent(
            getattr(adapter, "generator", None) or self.get_llm(),
            await self.get_tools(thread_id=thread_id or "", user=user),
            system_prompt,
            max_agent_steps=self.max_agent_steps,
            hooks=default_hooks(self),
        )
        haystack_messages = adapter.get_messages(messages)
        result = await haystack_agent.run_async(messages=haystack_messages)
        replies = result.get("messages", [])
        return replies[-1].text if replies else None

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """Return a Run adapter for non-streaming tasks (title generation, etc.).

        Must be implemented by subclasses that use Agent.run() or title generation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_run_adapter().")

    #: Flat list of integration names (`["linear"]`) — every tool that integration
    #: exposes reaches this agent. Names are registry keys, i.e. the `name` on
    #: each Integration subclass. Narrowing an integration to a subset of its tools
    #: per agent is not supported: restrict it at the integration instead, via
    #: an MCP integration's `default_tools` allow-list.
    integrations: list[str] = []

    async def _get_integration_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        """Load tool objects from every integration listed in `self.integrations`.

        Resolves the names against the integrations registry (populated by each
        integration app on startup, plus any DB-declared MCP servers) and skips any
        the user isn't permitted to use, so an unauthorized integration's tools never
        reach the model. Runs the remaining integrations concurrently — each one's
        get_tools() is individually bounded.
        """
        if not self.integrations:
            return []

        async def _safe_get_tools(integration: Any) -> list[Any]:
            try:
                tools = await integration.get_tools(user, agent=self, thread_id=thread_id)
            except Exception:
                logger.exception("Failed to load tools for integration %r", integration.name)
                return []
            return [_namespaced(integration.name, tool, integration.hint) for tool in tools]

        services = (await get_integrations(self.integrations)).values()
        allowed = [s for s in services if await s.has_perms(user, Operation.USE_INTEGRATION)]
        results = await asyncio.gather(*(_safe_get_tools(i) for i in allowed))
        return [tool for tools in results for tool in tools]

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """Return the adapter used for streaming chat.

        Must be implemented by subclasses used in chat. A worker-only agent
        (hidden = True, called only via run()) can leave this unimplemented.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_pipeline_adapter()."
        )

    async def history(
        self, thread_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> ThreadDetail:
        """
        Get conversation history for a thread.

        Returns thread metadata and messages in protocol format suitable for initial load.
        This works with any storage adapter (DB, Memory, etc.).

        Args:
            thread_id: The thread ID to fetch history for
            user: Optional user for permission checking

        Returns:
            ThreadDetail containing thread metadata and protocol-formatted messages

        Raises:
            PermissionDenied: If user has no VIEW_THREAD permission
        """
        logger.debug(f"Fetching history for thread: {thread_id}")

        await check_permissions(
            user, Operation.VIEW_THREAD, get_agent_permissions(self), agent=self
        )

        storage = await self.get_storage_adapter(thread_id)

        if not storage:
            raise ValueError(f"No storage adapter found for thread: {thread_id}")

        # Get thread metadata
        thread_info = await storage.__class__.get_thread(thread_id)
        if not thread_info:
            raise ValueError(f"Thread not found: {thread_id}")

        await check_object_permissions(
            user, Operation.VIEW_THREAD, thread_info, get_agent_permissions(self), agent=self
        )

        # Get messages using the instance method
        chat_messages = await storage.get_messages()
        logger.debug(f"Retrieved {len(chat_messages)} ChatMessages, converting to protocol format")

        # Convert to protocol format
        protocol_messages = self.protocol_handler.from_chat_messages(chat_messages)

        return ThreadDetail(
            thread=thread_info,
            messages=protocol_messages,
        )

    # TODO: I liked the idea of as_view, but in the end it's more a response.
    # This might be the first API change that breaks backwards compatibility.
    # For now, keep it as a separate method with a different name.
    # But we might move to a method called to_response()
    # And we deprecate as_view() with a warning to replace with to_response()
    async def as_view(
        self,
        protocol_messages: list[Any],
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """
        Convert protocol messages to streaming HTTP response with optional storage.

        Similar to Django's as_view() pattern - converts the agent
        into a response that can be returned directly from a view.

        Args:
            protocol_messages: Raw protocol messages (e.g., Vercel Message objects)
            thread_id: Optional thread ID for conversation persistence
            user: Optional user for conversation attribution and permission checks

        Returns:
            StreamingHttpResponse ready for Django views

        Raises:
            PermissionDenied: If user has no CHAT permission for this agent/thread
        """
        logger.debug(
            f"Agent as_view called: agent={self.__class__.__name__}, "
            f"messages={len(protocol_messages) if protocol_messages else 0}, "
            f"thread_id={thread_id}, user={user}"
        )

        await check_permissions(user, Operation.CHAT, get_agent_permissions(self), agent=self)

        # Protocol handler converts to our intermediate ChatMessage format
        messages = self.protocol_handler.to_chat_messages(protocol_messages)
        logger.debug(
            f"Protocol handler converted {len(protocol_messages) if protocol_messages else 0} "
            f"protocol messages to {len(messages)} chat messages"
        )

        # Apply max_history limiting if configured
        if self.max_history and len(messages) > self.max_history:
            messages = messages[-self.max_history :]
            logger.debug(
                f"Applied max_history={self.max_history}, kept {len(messages)} most recent messages"
            )

        # Get storage adapter for thread
        storage_adapter = await self.get_storage_adapter(thread_id)

        # Store only the last user message (the new one just sent)
        # Many protocols send full conversation history, but we only want to store new messages
        if storage_adapter and thread_id:
            user_messages = []
            for message in messages:
                if message.role == "user":
                    user_messages.append(message)

            if user_messages:
                # Get the most recent user message
                last_user_message = user_messages[-1]
                logger.debug(
                    f"Storing last user message: {len(last_user_message.content)} characters"
                )
                # FIXME: we need a better way on how to create a id for the user messages
                last_user_message.id = str(uuid.uuid4())
                await storage_adapter.store_chat_message(last_user_message)
            else:
                logger.debug("No user messages found to store")

        # Create fresh adapter each time
        # RAG is cached separately via get_rag(), so adapter is not tied to it
        logger.debug("Creating pipeline adapter")
        adapter = await self.get_pipeline_adapter(thread_id=thread_id, user=user)

        # Wire suggestion generator onto the adapter
        suggestion_generator = self.get_suggestion_generator()
        if suggestion_generator:
            adapter.suggestion_generator = suggestion_generator

        # TODO: fix type error for argument, can never be None, now nested
        if thread_id:
            thread = await ThreadService.get_thread(thread_id, user=user)
            if thread:
                await check_object_permissions(
                    user, Operation.CHAT, thread, get_agent_permissions(self), agent=self
                )
            if self.title_generation and thread and not thread.title:
                title = await generate_thread_title(
                    agent=self, messages=messages, thread_id=thread_id, user=user
                )
                if title:
                    await ThreadService.update_thread(thread_id, title=title, user=user)

        logger.debug(f"Pipeline adapter created: {type(adapter).__name__}")

        logger.debug("Initiating stream response")
        return await stream_response(adapter, messages, self.protocol_handler)
