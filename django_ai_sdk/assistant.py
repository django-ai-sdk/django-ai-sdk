from __future__ import annotations

import asyncio
import dataclasses
import uuid
from abc import ABC
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from django_ai_sdk.assistants.mixins import AssistantInfoMixin
from django_ai_sdk.assistants.registry import registry
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
    AllowAll,
    BasePermission,
    Operation,
    check_object_permissions,
    check_permissions,
)
from django_ai_sdk.prompts import build_title_generation_prompt
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.rags import queryset_to_rag_documents
from django_ai_sdk.responses import stream_response
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.storage.schemas import ThreadDetail
from django_ai_sdk.storage.services import ThreadService

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import Prompt
    from django_ai_sdk.files.pipeline import FilePipeline
    from django_ai_sdk.rags.schemas import RagDocument
    from django_ai_sdk.storage.base import BaseStorageAdapter
    from django_ai_sdk.suggestions import SuggestionGenerator


T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


def _namespaced(integration_name: str, tool: Any) -> Any:
    """Rename tool to {integration_name}_{tool.name}.

    Nothing stops two unrelated MCP servers from defining the same tool name
    (GitHub and Linear both have list_issues), and Haystack requires unique names
    across everything handed to one agent, so without this, enabling two
    integrations that happen to collide would fail assistant construction outright.
    """
    try:
        return dataclasses.replace(tool, name=f"{integration_name}_{tool.name}")
    except (TypeError, AttributeError):
        logger.warning(
            "Could not namespace tool %r from integration %r — left as-is, may "
            "collide with another integration's tool.",
            getattr(tool, "name", tool),
            integration_name,
        )
        return tool


class Assistant(ABC, AssistantInfoMixin):
    """
    Base class for AI assistants in the Django AI SDK.

    Provides a consistent interface for creating assistants that work
    with the streaming view system. Each assistant encapsulates:
    - Name and instructions (AI personality)
    - Tools (as methods for co-location)
    - Adapter creation (backend-specific logic)
    - Protocol handler for message conversion

    This class includes the AssistantInfoMixin which provides:
    - info(): Get assistant metadata
    - assistant_id: Stable UUID v5 ID

    Registration:

    Method 1: Settings-based (recommended)
        # In your settings.py:
        AI_SDK_ASSISTANTS = [
            "myapp.assistants.MyAssistant",
        ]

        # In your AppConfig.ready():
        from django.utils.module_loading import import_string
        from django.conf import settings
        from django_ai_sdk.assistants.registry import registry

        for path in getattr(settings, 'AI_SDK_ASSISTANTS', []):
            import_string(path)
        registry.setup()

    Method 2: Decorator-based
        from django_ai_sdk.assistants import auto_register

        @auto_register
        class MyAssistant(Assistant):
            name = "My Bot"
            model = "gpt-4"

    Every concrete subclass also auto-registers on definition (__init_subclass__),
    so either method above is really just what gets the module imported. An
    abstract shared base (abstract = True) is skipped regardless of how it's
    reached — see AssistantRegistry.register().

    Usage:
        from django_ai_sdk.protocols.vercel import VercelProtocolHandler

        class MyAssistant(Assistant):
            name = "My Bot"
            model = "gpt-4"
            instructions = prompt("You are a helpful assistant...")
            protocol = VercelProtocolHandler

            async def get_pipeline_adapter(self): return SomeAdapter(...)

        # Get registered assistant by UUID
        from django_ai_sdk.assistants.registry import registry
        assistant = registry.get(assistant_id)
        return await assistant.as_view(protocol_messages)
    """

    name: str
    description: str
    model: str
    instructions: Prompt = prompt("You are a helpful assistant.")

    # Permission classes used to gate access to this assistant's operations.
    permissions: list[type[BasePermission]] = [AllowAll]

    # Default list of connected memories
    memories: list[str] = []

    tools: list[str] = []

    protocol = None
    storage: type[BaseStorageAdapter] | None = None

    # If True, hide from registry.visible() (used for internal assistants)
    hidden: bool = False

    # If True, this is a shared base meant only to be subclassed — it never enters the
    # registry. Checked on the class's own __dict__, so it is never inherited and each
    # concrete subclass registers normally without restating it.
    #
    # Distinct from `_skip_auto_register` (see assistants/runtime.py), which marks a
    # class that is concrete and instantiated but built on demand rather than looked up
    # by id. Both stay out of the registry, for unrelated reasons.
    abstract: bool = False

    # If Assistant should automatically warm up after initialization
    warmup_on_init: bool = False

    # RAG provider: set to a RAGProvider instance to enable RAG, None disables RAG
    rag_provider: Any = None

    # Maximum conversation history to send to LLM (None = unlimited)
    max_history: int | None = None

    # Enable file upload UI for this assistant's threads
    file_upload: bool = False

    # Declare one FilePipeline per supported file type.
    # First pipeline whose processor accepts the uploaded file is used.
    # Empty = fall back to get_default_file_pipeline() (TextFileProcessor, no LLM extraction).
    file_pipelines: list[FilePipeline] = []

    # Enable automatic thread title generation based on chat messages
    title_generation: bool = True

    # Citation formatter used to render retrieved documents for the LLM.
    citation_formatter_class: type[CitationFormatter] = DefaultCitationFormatter

    # Suggestion generator class for follow-up questions.
    suggestion_generator: type[SuggestionGenerator] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register Assistant subclasses in the registry.

        This enables settings-based registration:
        - When a class is imported via AI_SDK_ASSISTANTS, it auto-registers
        - Works together with @auto_register decorator
        """
        super().__init_subclass__(**kwargs)
        # Don't register the base Assistant class itself, or classes that
        # manage their own registration (e.g. RuntimeAssistant).
        if cls.__name__ != "Assistant" and not getattr(cls, "_skip_auto_register", False):
            registry.register(cls)

    @classmethod
    async def warmup(cls, assistant: Assistant, memory_id: str | None = None) -> None:
        """
        Warm up the RAG pipeline.

        Delegates to rag_provider.warmup() if configured.
        If no RAG provider is set, returns immediately.

        Args:
            assistant: Instance of the assistant
            memory_id: Optional memory
        """
        if assistant.rag_provider is None:
            logger.debug("No RAG provider configured, skipping warmup")
            return

        await assistant.rag_provider.warmup(assistant, memory_id)

    @classmethod
    def clear_rag_cache(cls, assistant: Assistant) -> None:
        """
        Clear the RAG cache.

        Args:
            assistant: Instance of the assistant
        """
        if assistant.rag_provider is not None:
            assistant.rag_provider.clear_cache()
            logger.debug("RAG cache cleared via provider")

    @classmethod
    async def reindex(
        cls,
        assistant: Assistant,
        memory_id: str | None = None,
        force_rebuild: bool = False,
    ) -> Any:
        """
        Reindex the RAG pipeline for this assistant.
        Delegates to rag_provider.reindex() if configured.

        Args:
            assistant: Instance of the assistant
            memory_id: Optional memory ID for document source
            force_rebuild: If True, forces a complete rebuild of the index.
                          For persistent storage backends (like Qdrant), this
                          will delete and recreate the index from scratch.

        Returns:
            The reindexed RAG pipeline, or None if no provider
        """
        if assistant.rag_provider is None:
            logger.debug("No RAG provider configured, skipping reindex")
            return None

        return await assistant.rag_provider.reindex(assistant, memory_id, force_rebuild)

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
            # Warmup RAG provider on init
            # TODO: delegate this to background task and check status.
            # for now this won't block the main thread, but it can become very slow.
            asyncio.get_event_loop().run_until_complete(self.rag_provider.warmup(self, None))

    async def get_storage_adapter(self, thread_id: str | None = None) -> BaseStorageAdapter | None:
        """
        Get storage adapter for the given thread.

        If thread_id is provided, finds which storage actually contains the thread
        queries all registered adapters, then returns that adapter type.
        If thread not found anywhere, falls back to assistant's configured storage.
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
        # assistant's configured storage (always set in __init__).
        return self.storage_adapter(thread_id)

    async def get_file_pipeline(self, file: object) -> FilePipeline | None:
        """Return the first FilePipeline whose processor accepts file, or None."""
        for pipeline in self.file_pipelines:
            if await pipeline.accepts(file):
                return pipeline
        return None

    def get_name(self) -> str:
        """Return the assistant's display name."""
        return self.name or "Unnamed Assistant"

    def get_instructions(self) -> Prompt:
        """Return formatted system instructions as a single string."""
        return self.instructions

    def get_system_prompt(self) -> str:
        """
        Alias for get_instructions()
        """
        return self.get_instructions()

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

    def get_suggestion_generator(self) -> SuggestionGenerator | None:
        """Return a configured SuggestionGenerator, or None to disable.

        Uses self.suggestion_generator class attribute if set.
        Override this method in subclasses for full control.
        """
        if not self.suggestion_generator:
            return None
        return self.suggestion_generator(assistant=self)

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
        tools = getattr(self.__class__, "tools", [])
        result = []
        for tool in tools:
            items = tool(thread_id=thread_id, user=user)
            if isinstance(items, list):
                result.extend(items)
            else:
                result.append(items)
        result.extend(await self._get_integration_tools(user, thread_id=thread_id))
        return result

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

        if memory_id:
            return Entry.objects.filter(memory_id=memory_id)
        # No memory scope → retrieve nothing. Returning every Entry in the system
        # would leak content across memories; callers always pass a memory_id.
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

        logger.info(
            f"[get_rag_documents] memory_id={memory_id}, assistant={self.__class__.__name__}"
        )
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

    async def run(
        self,
        messages: list[ChatMessage],
        system_prompt: str | Prompt | None = None,
        response_format: type[T] | None = None,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> T | str | None:
        """Run LLM calls directly from adapter.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt override
            response_format: Optional Pydantic model for structured output
            thread_id: Optional thread ID (forwarded to get_run_adapter)
            user: Optional user (forwarded to get_run_adapter)

        Returns:
            Response string, or parsed Pydantic model if response_format is set
        """
        adapter = await self.get_run_adapter(thread_id=thread_id, user=user)
        return await adapter.run(
            messages=messages,
            system_prompt=system_prompt,
            response_format=response_format,
        )

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """Return a Run adapter for non-streaming tasks (title generation, etc.).

        Must be implemented by subclasses that use Assistant.run() or title generation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_run_adapter().")

    #: Flat list of integration names (`["linear"]`) — every tool that integration
    #: exposes reaches this assistant. Names are registry keys, i.e. the `name` on
    #: each Integration subclass. Narrowing an integration to a subset of its tools
    #: per assistant is not supported: restrict it at the integration instead, via
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

        from django_ai_sdk.permissions import Operation

        async def _safe_get_tools(integration: Any) -> list[Any]:
            try:
                tools = await integration.get_tools(user, assistant=self, thread_id=thread_id)
            except Exception:
                logger.exception("Failed to load tools for integration %r", integration.name)
                return []
            return [_namespaced(integration.name, tool) for tool in tools]

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

        Must be implemented by subclasses used in chat. A worker-only assistant
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

        await check_permissions(user, Operation.VIEW_THREAD, self.permissions)

        storage = await self.get_storage_adapter(thread_id)

        if not storage:
            raise ValueError(f"No storage adapter found for thread: {thread_id}")

        # Get thread metadata
        thread_info = await storage.__class__.get_thread(thread_id)
        if not thread_info:
            raise ValueError(f"Thread not found: {thread_id}")

        await check_object_permissions(user, Operation.VIEW_THREAD, thread_info, self.permissions)

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

        Similar to Django's as_view() pattern - converts the assistant
        into a response that can be returned directly from a view.

        Args:
            protocol_messages: Raw protocol messages (e.g., Vercel Message objects)
            thread_id: Optional thread ID for conversation persistence
            user: Optional user for conversation attribution and permission checks

        Returns:
            StreamingHttpResponse ready for Django views

        Raises:
            PermissionDenied: If user has no CHAT permission for this assistant/thread
        """
        logger.debug(
            f"Assistant as_view called: assistant={self.__class__.__name__}, "
            f"messages={len(protocol_messages) if protocol_messages else 0}, "
            f"thread_id={thread_id}, user={user}"
        )

        await check_permissions(user, Operation.CHAT, self.permissions)

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
                await check_object_permissions(user, Operation.CHAT, thread, self.permissions)
            if self.title_generation and thread and not thread.title:
                title = await generate_thread_title(
                    assistant=self, messages=messages, thread_id=thread_id, user=user
                )
                if title:
                    await ThreadService.update_thread(thread_id, title=title, user=user)

        logger.debug(f"Pipeline adapter created: {type(adapter).__name__}")

        logger.debug("Initiating stream response")
        return await stream_response(adapter, messages, self.protocol_handler)
