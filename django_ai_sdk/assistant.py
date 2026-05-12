import asyncio
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from django_ai_sdk.assistants.mixins import AssistantInfoMixin
from django_ai_sdk.assistants.registry import registry
from django_ai_sdk.logger import get_logger
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.rags import queryset_to_rag_documents
from django_ai_sdk.responses import stream_response
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.storage.schemas import ThreadDetail

if TYPE_CHECKING:
    from django_ai_sdk.rags.schemas import RagDocument
    from django_ai_sdk.storage.base import BaseStorageAdapter


logger = get_logger(__name__)


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

    Usage:
        from django_ai_sdk.protocols.vercel import VercelProtocolHandler

        class MyAssistant(Assistant):
            name = "My Bot"
            model = "gpt-4"
            instructions = ["You are a helpful assistant..."]
            protocol = VercelProtocolHandler

            async def get_pipeline_adapter(self): return SomeAdapter(...)

        # Get registered assistant by UUID
        from django_ai_sdk.assistants.registry import registry
        assistant = registry.get(assistant_id)
        return await assistant.as_view(protocol_messages)
    """

    name: str | None = None
    description: str | None = None
    model: str | None = None
    instructions: list[str] | str | None = None

    # Default list of connected memories
    memories: list[str] = []

    protocol = None
    storage: type["BaseStorageAdapter"] | None = None

    # If Assistant should automatically warm up after initialization
    warmup_on_init: bool = False

    # RAG provider - set to a RAGProvider instance to enable RAG, None disables RAG
    rag_provider: Any = None

    # Maximum conversation history to send to LLM (None = unlimited)
    max_history: int | None = None

    # Enable file upload UI for this assistant's threads
    file_upload: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register Assistant subclasses in the registry.

        This enables settings-based registration:
        - When a class is imported via AI_SDK_ASSISTANTS, it auto-registers
        - Works together with @auto_register decorator
        """
        super().__init_subclass__(**kwargs)
        # Don't register the base Assistant class itself
        if cls.__name__ != "Assistant":
            registry.register(cls)

    @classmethod
    async def warmup(cls, assistant: "Assistant", memory_id: str | None = None) -> None:
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
    def clear_rag_cache(cls, assistant: "Assistant") -> None:
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
        cls, assistant: "Assistant", memory_id: str | None = None, force_rebuild: bool = False
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

        self.tools = self.get_tools()

        if self.warmup_on_init and self.rag_provider is not None:
            # Warmup RAG provider on init
            # TODO: delegate this to background task and check status.
            # for now this won't block the main thread, but it can become very slow.
            asyncio.get_event_loop().run_until_complete(self.rag_provider.warmup(self, None))

    async def get_storage_adapter(
        self, thread_id: str | None = None
    ) -> "BaseStorageAdapter | None":
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

    def get_name(self) -> str:
        """Return the assistant's display name."""
        return self.name or "Unnamed Assistant"

    def get_instructions(self) -> str:
        """Return formatted system instructions as a single string."""

        # TODO: we might want to pass on str after all.
        # Editing the list is kinda annoying
        if isinstance(self.instructions, list):
            return "\n".join(self.instructions)
        return self.instructions or ""

    def get_system_prompt(self) -> str:
        """
        Alias for get_instructions()
        """
        return self.get_instructions()

    def get_model(self) -> str:
        """Return the model identifier."""
        return self.model or ""

    def get_tools(self) -> list[Any]:
        """
        Return list of available tools.

        Override this method if your assistant has tools.
        Tools can be defined as methods on the assistant class
        for better organization and co-location.
        """
        return []

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
        return Entry.objects.all()

    async def get_rag_documents(self, memory_id: str | None = None) -> list["RagDocument"]:
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
            BaseRAGAdapter instance, or None if RAG is not enabled
        """
        return None

    # ------------------------------------------------------------------
    # Thread title generation
    #
    # Hardcoded to openai.AsyncOpenAI against the Django OPENAI_* settings.
    # That covers any OpenAI-compatible endpoint (OpenAI, vLLM, Together,
    # OpenRouter, Nebul, etc).
    #
    # TODO: should be extensible once the SDK has a unified provider (LLM)
    # abstraction. 
    # ------------------------------------------------------------------

    #: Whether thread titles are generated automatically after the first
    #: assistant response. Set False to disable entirely (no LLM call,
    #: no fallback - the application is responsible for setting titles).
    title_generation: bool = True

    #: Hard fallback used when both the LLM and the user-message fallback
    #: yield nothing. Override per assistant for branding/locale ("New chat",
    #: "Nieuw gesprek", a timestamp, etc).
    title_fallback_default: str = "New conversation"

    #: Prompt template used for thread title generation. {chat_history}
    #: is interpolated with role: content lines from the message context.
    #: Override on the subclass to drop the emoji, change examples, or
    #: localize the instructions.
    title_prompt_template: str = (
        "### Task:\n"
        "Generate a concise, 3-5 word title summarizing the chat history.\n\n"
        "### Guidelines:\n"
        "- The title should clearly represent the main theme or subject of the conversation.\n"
        "- Start the title with a single emoji that enhances understanding of the topic.\n"
        "- Write the title in the same language as the user's messages; "
        "default to English if multilingual or unclear.\n"
        "- Match the tone and register of the user (formal, casual, technical, etc.).\n"
        "- Prioritize accuracy over creativity; keep it clear and simple.\n\n"
        "### Output rules (strict):\n"
        "- Return ONLY the title. No preamble, no explanation, no commentary.\n"
        "- No markdown, no quotes, no backticks, no code fences.\n"
        "- A single line of plain text.\n\n"
        "### Examples:\n"
        "- 📉 Stock Market Trends\n"
        "- 🍪 Perfect Chocolate Chip Recipe\n"
        "- 🎮 Video Game Development Insights\n\n"
        "### Chat History:\n{chat_history}"
    )

    async def generate_thread_title(
        self, message_context: list[tuple[str, str]]
    ) -> str | None:
        """
        Call the LLM to generate a thread title. Returns None` if the LLM
        is unavailable or the response is unusable - caller falls back.

        Override to use a non-OpenAI provider, change parsing, or restructure
        the call entirely. See the TODO comment above the section for context
        on the missing provider abstraction.
        """
        if not any(role and content for role, content in message_context):
            return None

        try:
            import openai
        except ImportError:
            logger.debug("openai package not installed; skipping title generation.")
            return None

        from django.conf import settings

        api_key = getattr(settings, "OPENAI_API_KEY", None)
        base_url = getattr(settings, "OPENAI_API_URL", None) or None
        if not self.model:
            return None

        chat_history = "\n".join(
            f"{role}: {content[:200]}"
            for role, content in message_context
            if role and content
        )
        prompt = self.title_prompt_template.format(chat_history=chat_history)
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning(f"Thread title generation failed: {exc}")
            return None

        if not response.choices:
            return None
        raw = response.choices[0].message.content or ""
        title = raw.strip().strip("\"'`").strip()
        return title or None

    def _fallback_thread_title(
        self, message_context: list[tuple[str, str]]
    ) -> str:
        """
        Always-non-empty fallback used when generate_thread_title returns
        None. Default: first ~50 chars of the first user message, or
        title_fallback_default if no user message has content.
        """
        for role, content in message_context:
            if role != "user":
                continue
            text = (content or "").strip()
            if not text:
                continue
            return f"{text[:50]}…" if len(text) > 50 else text
        return self.title_fallback_default

    async def _maybe_generate_title(
        self, thread_id: str, message_context: list[tuple[str, str]]
    ) -> None:
        """
        Generate and persist a title for thread_id.

        Called from the post-store hook, which is installed only when the
        thread had no title at request start (see as_view). Always writes
        *some* title - LLM result if available, otherwise the fallback. The
        hook is therefore one-shot: after this runs, thread.title is
        non-empty and the hook is never reinstalled.
        """
        from django_ai_sdk.storage.services import ThreadService

        title = await self.generate_thread_title(message_context)
        if not title:
            title = self._fallback_thread_title(message_context)
            logger.debug(f"Using fallback title for thread {thread_id}: {title!r}")
        await ThreadService.update_thread(thread_id, title=title)
        logger.debug(f"Persisted title for thread {thread_id}: {title!r}")

    @abstractmethod
    async def get_pipeline_adapter(self, thread_id: str | None = None) -> Any:
        """
        Create and return pipeline adapter.

        Args:
            thread_id: Optional thread ID for conversation persistence.

        Returns:
            BasePipelineAdapter instance
        """
        pass

    async def history(self, thread_id: str) -> ThreadDetail:
        """
        Get conversation history for a thread.

        Returns thread metadata and messages in protocol format suitable for initial load.
        This works with any storage adapter (DB, Memory, etc.).

        Args:
            thread_id: The thread ID to fetch history for

        Returns:
            ThreadDetail containing thread metadata and protocol-formatted messages
        """
        logger.debug(f"Fetching history for thread: {thread_id}")

        storage = await self.get_storage_adapter(thread_id)

        if not storage:
            raise ValueError(f"No storage adapter found for thread: {thread_id}")

        # Get thread metadata
        thread_info = await storage.__class__.get_thread(thread_id)
        if not thread_info:
            raise ValueError(f"Thread not found: {thread_id}")

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
        user: Any = None,
    ) -> Any:
        """
        Convert protocol messages to streaming HTTP response with optional storage.

        Similar to Django's as_view() pattern - converts the assistant
        into a response that can be returned directly from a view.

        Args:
            protocol_messages: Raw protocol messages (e.g., Vercel Message objects)
            thread_id: Optional thread ID for conversation persistence
            user: Optional user for conversation attribution

        Returns:
            StreamingHttpResponse ready for Django views
        """
        logger.debug(
            f"Assistant as_view called: assistant={self.__class__.__name__}, messages={len(protocol_messages) if protocol_messages else 0}, thread_id={thread_id}, user={user}"
        )

        # Protocol handler converts to our intermediate ChatMessage format
        messages = self.protocol_handler.to_chat_messages(protocol_messages)
        logger.debug(
            f"Protocol handler converted {len(protocol_messages) if protocol_messages else 0} protocol messages to {len(messages)} chat messages"
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

        # Decide whether this request should trigger title generation. We
        # do it as a stream-completion callback rather than via a hook on
        # the storage adapter, so the assistant fully owns its lifecycle
        # and the pipeline doesn't need the storage instance threaded in.
        on_complete: Callable[[], Awaitable[None]] | None = None
        if thread_id and self.title_generation:
            from django_ai_sdk.storage.services import ThreadService

            existing_thread = await ThreadService.get_thread(thread_id)
            if existing_thread is not None and not existing_thread.title:
                _message_context: list[tuple[str, str]] = [
                    (m.role, m.content) for m in messages if m.role and m.content
                ]
                _thread_id_for_title = thread_id

                async def _generate_title_on_complete() -> None:
                    await self._maybe_generate_title(
                        _thread_id_for_title, _message_context
                    )

                on_complete = _generate_title_on_complete

        # Create fresh adapter each time
        # RAG is cached separately via get_rag(), so adapter is not tied to it
        logger.debug("Creating pipeline adapter")
        adapter = await self.get_pipeline_adapter(thread_id=thread_id)

        logger.debug(f"Pipeline adapter created: {type(adapter).__name__}")

        logger.debug("Initiating stream response")
        return await stream_response(
            adapter, messages, self.protocol_handler, on_complete=on_complete
        )
