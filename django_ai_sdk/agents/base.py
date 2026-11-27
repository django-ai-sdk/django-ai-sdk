from __future__ import annotations

import asyncio
import dataclasses
from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

from django_ai_sdk.adapters.citations import (
    SourceFormatter,
    SourceRegistry,
    SourcesFormatter,
)
from django_ai_sdk.agents.attachments import InlineFileCapability
from django_ai_sdk.agents.mixins import AgentInfoMixin
from django_ai_sdk.agents.registry import registry
from django_ai_sdk.common import (
    ChatMessage,
    Prompt,
    as_chat_messages,
    prompt,
)
from django_ai_sdk.conversation.utils import (
    get_title_sanity_limit,
)
from django_ai_sdk.errors import AiSdkError, ErrorCode, NotFound
from django_ai_sdk.integrations.base import IntegrationStatus
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
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from django_ai_sdk.storage.schemas import ThreadDetail
from django_ai_sdk.storage.services import ThreadService
from django_ai_sdk.tracing import bind as bind_trace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping, Sequence

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.adapters.citations import NumberedSource
    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.artifacts.tool_artifacts import ToolArtifact
    from django_ai_sdk.common import Prompt
    from django_ai_sdk.files.pipeline import FilePipeline
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.rags.schemas import RagDocument
    from django_ai_sdk.storage.base import BaseStorageAdapter
    from django_ai_sdk.storage.schemas import ThreadInfo


T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


def _warmup_part(stage: str, detail: str) -> dict[str, Any]:
    return {
        "type": "data-warmup",
        "data": {"status": "start", "stage": stage, "detail": detail},
        "transient": True,
    }


def _unavailable_part(stage: str, detail: str) -> dict[str, Any]:
    return {
        "type": "data-unavailable",
        "data": {"stage": stage, "detail": detail},
        "transient": True,
    }


def _claim_tool_name(name: str, used_names: set[str], suffix: str) -> str:
    if name in used_names:
        renamed = f"{name}_{suffix}"
        logger.warning(f"Tool name {name!r} collided, renamed to {renamed!r}")
        name = renamed
    used_names.add(name)
    return name


def _namespaced(integration_name: str, tool: Any, hint: str = "") -> Any:
    """Haystack needs unique tool names per agent, and two MCP servers may both define one."""
    try:
        updates: dict[str, Any] = {"name": f"{integration_name}_{tool.name}"}
        if hint:
            updates["description"] = f"{tool.description}\n\n{hint}" if tool.description else hint
        return dataclasses.replace(tool, **updates)
    except (TypeError, AttributeError):
        logger.warning(
            "Could not namespace tool {!r} from integration {!r} — left as-is, may "
            "collide with another integration's tool.",
            getattr(tool, "name", tool),
            integration_name,
        )
        return tool


class Agent(ABC, AgentInfoMixin, InlineFileCapability):
    """Concrete subclasses auto-register on definition (``abstract = True`` opts out)."""

    name: str | None = None

    description: str | None = None

    model: str

    instructions: Prompt = prompt("You are a helpful agent.")

    # None or empty means no permissions are required.
    permissions: list[type[BasePermission]] | None = None

    memories: list[str] = []

    tools: list[str] = []

    artifacts: list[type[BaseModel]] = []

    tool_artifacts: dict[str, ToolArtifact] = {}

    agents: list[type[Agent]] = []

    max_agent_steps: int = 6
    max_tool_calls: int | None = 6

    protocol = None

    storage: type[BaseStorageAdapter] | None = None

    response_format: type[BaseModel] | None = None

    hidden: bool = False

    abstract: bool = False

    warmup_on_init: bool = False

    # RAG provider: set to a RAGProvider instance to enable RAG, None disables RAG.
    rag_provider: Any = None

    # Maximum conversation history to send to LLM (None = unlimited).
    max_history: int | None = None

    file_upload: bool = False

    required_tools: list[str] = []

    # One FilePipeline per file type; the first whose processor accepts the file is used.
    # Empty falls back to get_default_file_pipeline().
    file_pipelines: list[FilePipeline] = []

    title_generation: bool = True

    rag_document_limit: int = 10_000

    citation_formatter_class: type[SourceFormatter] = SourcesFormatter

    suggestion_generator: type[SuggestionGenerator] | None = None

    # The policy for every turn, in chat and run(): a Policy subclass or a name.
    # None uses AI_SDK_OVERSIGHT["DEFAULT_POLICY"]. False turns oversight off.
    oversight: type[Policy] | str | Literal[False] | None = None

    # Assign the factory, never call it: get_llm() builds it with this agent's model.
    llm: Callable[..., Any] | None = None

    # Vendor generation parameters, mapped onto Haystack's `generation_kwargs`.
    llm_kwargs: dict[str, Any] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A plain function set as a class attribute would bind as a method.
        llm = cls.__dict__.get("llm")
        if callable(llm) and not isinstance(llm, staticmethod):
            cls.llm = staticmethod(llm)  # ty: ignore[invalid-assignment]
        from django_ai_sdk.oversight.agents import validate_oversight

        validate_oversight(cls)
        if cls.__name__ != "Agent" and not getattr(cls, "_skip_auto_register", False):
            registry.register(cls)

    @classmethod
    async def warmup(cls, agent: Agent, memory_id: str | None = None) -> None:
        if agent.rag_provider is None:
            logger.debug("No RAG provider configured, skipping warmup")
            return

        await agent.rag_provider.warmup(agent, memory_id)

    @classmethod
    def clear_rag_cache(cls, agent: Agent) -> None:
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
        """None without a provider; `force_rebuild` deletes and recreates persistent indexes."""
        if agent.rag_provider is None:
            logger.debug("No RAG provider configured, skipping reindex")
            return None

        return await agent.rag_provider.reindex(agent, memory_id, force_rebuild)

    def __init__(self) -> None:
        self.protocol_handler = (
            self.protocol()
            if hasattr(self, "protocol") and self.protocol is not None
            else VercelProtocolHandler()
        )

        self.storage_adapter = (
            getattr(self, "storage_adapter", None) or self.storage or MemoryStorageAdapter
        )

        if self.warmup_on_init and self.rag_provider is not None:
            try:
                asyncio.get_running_loop().create_task(self.rag_provider.warmup(self, None))
            except RuntimeError:
                pass

    @property
    def is_runtime(self) -> bool:
        return False

    @property
    def config(self) -> AgentSettings | None:
        return None

    async def get_storage_adapter(self, thread_id: str | None = None) -> BaseStorageAdapter | None:
        """Return the storage adapter bound to `thread_id`, or None without one."""
        if thread_id is None:
            return None

        from django_ai_sdk.storage.base import StorageAdapterRegistry

        for adapter_class in StorageAdapterRegistry.get_all_adapters():
            thread = await adapter_class.get_thread(thread_id)
            if thread:
                return adapter_class(thread_id)

        return self.storage_adapter(thread_id)

    async def get_file_pipeline(self, file: object) -> FilePipeline | None:
        """Return the first FilePipeline whose processor accepts file, or None."""
        pipelines = await self.get_file_pipelines(file)
        return pipelines[0] if pipelines else None

    async def get_file_pipelines(self, file: object) -> list[FilePipeline]:
        """Return all FilePipelines whose processor accepts file, in order."""
        return [pipeline for pipeline in self.file_pipelines if await pipeline.accepts(file)]

    def get_name(self) -> str:
        return self.name or "Unnamed Agent"

    def get_instructions(self) -> Prompt:
        return self.instructions

    def get_system_prompt(self) -> str:
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
        return build_title_generation_prompt(get_title_sanity_limit())

    def get_model(self) -> str:
        return self.model or ""

    def get_citation_formatter(self) -> SourceFormatter:
        return self.citation_formatter_class()

    def get_citation_registry(self) -> SourceRegistry:
        """Return a fresh per-turn registry of the sources an answer may cite."""
        return SourceRegistry()

    async def get_citation_sources(
        self,
        thread_id: str,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[NumberedSource]:
        """Sources every answer in the thread may cite, even when no tool found them."""
        return []

    def get_llm(self, **kwargs: Any) -> Any:
        """Uses the `llm` factory if set, otherwise OpenAI's Responses API. Keyword
        arguments override `llm_kwargs`, and the agent always supplies the model.
        """
        from django_ai_sdk.generators import (
            merge_generation_kwargs,
            openai_responses_chat,
        )

        factory = self.llm or openai_responses_chat
        if not callable(factory):
            raise TypeError(f"{self.__class__.__name__}.llm must be a generator factory.")
        if self.llm_kwargs:
            kwargs["generation_kwargs"] = merge_generation_kwargs(
                self.llm_kwargs, kwargs.get("generation_kwargs")
            )
        return factory(model=self.get_model() or None, **kwargs)

    def get_suggestion_generator(self) -> SuggestionGenerator | None:
        """Return a configured SuggestionGenerator, or None to disable."""
        if not self.suggestion_generator:
            return None
        return self.suggestion_generator(agent=self)

    def get_run_required_tools(self, messages: list[ChatMessage]) -> list[str]:
        """Names of tools the model must have called before it answers this turn."""
        return []

    async def get_oversight(
        self,
        *,
        channel: str = "default",
        user: AbstractBaseUser | AnonymousUser | None = None,
        **kwargs: Any,
    ) -> type[Policy] | str | Literal[False] | None:
        """If this raises, or names a policy that cannot load, the turn is blocked unless
        AI_SDK_OVERSIGHT["FAIL_MODE"] is "open".
        """
        return self.oversight

    async def get_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Each callable in `tools` gets context kwargs (thread_id, user_id; accept **kwargs)
        and returns a tool or a list.
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

        result.extend(await self.get_artifact_tools(thread_id=thread_id, user=user))

        result.extend(await self.get_agent_tools(thread_id=thread_id, user=user))

        result.extend(self.get_attachment_tools(thread_id))
        return result

    async def get_agent_tools(
        self,
        thread_id: str = "",
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Skips itself, duplicates and subagents the user may not VIEW_AGENT."""
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
            haystack_agent, subagent = built
            agent_id = subagent.agent_id

            name = _claim_tool_name(
                subagent_tool_name(subagent_cls), used_names, agent_id.replace("-", "")[:6]
            )
            logger.info(
                f"Enabled subagent tool {name!r} ({subagent_cls.__name__}) on coordinator {self.__class__.__name__}"
            )

            tools.append(
                ComponentTool(
                    component=SubagentStreamFilter(
                        haystack_agent,
                        name=subagent_cls.name or name,
                        subagent=subagent,
                        thread_id=thread_id,
                        user=user,
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
                                    "to this subagent. Copy the user's request verbatim, "
                                    "do not paraphrase, autocorrect, or expand it."
                                ),
                            }
                        },
                        "required": ["task"],
                    },
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
        return [
            artifact_cls.as_tool(thread_id=thread_id, user=user)
            for artifact_cls in getattr(self.__class__, "artifacts", [])
        ]

    async def get_rag_tools(
        self,
        thread_id: str,
        *,
        citation_registry: SourceRegistry | None = None,
        citation_formatter: SourceFormatter | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> list[Any]:
        """Build RAG tools from a thread's active ThreadMemory links; empty memories are skipped."""
        if not self.rag_provider or not thread_id:
            return []

        from django_ai_sdk.conversation.models import Thread
        from django_ai_sdk.memories.services import MemoryService

        tools: list[Any] = []
        memories = await MemoryService.get_thread_memories(
            thread_id,
            user=user,
        )
        file_memory_id = (
            await Thread.objects.filter(id=thread_id)
            .values_list("file_memory_id", flat=True)
            .afirst()
        )

        used_names: set[str] = set()

        for memory in memories:
            spec = await memory.get_tool_spec()

            if (
                file_memory_id
                and memory.id == file_memory_id
                and "search_uploaded_documents" not in used_names
            ):
                spec.name = "search_uploaded_documents"
                spec.description = (
                    f"Search the files the user attached to this conversation "
                    f"({spec.doc_count or 0} available). Search this first, before any "
                    f"other knowledge base, whenever the question may concern what they "
                    f"attached. That includes follow-up questions about a topic already "
                    f"raised in this chat, even when they do not say 'this document' or "
                    f"'the file I just uploaded'."
                )

            spec.name = _claim_tool_name(spec.name, used_names, str(memory.id).replace("-", "")[:6])

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
        from django_ai_sdk.memories.models import Entry

        fields = ("id", "content", "data", "name", "memory_id")
        if memory_id:
            return Entry.objects.filter(memory_id=memory_id).only(*fields).order_by("-updated_at")
        return Entry.objects.none()

    async def get_rag_documents(self, memory_id: str | None = None) -> list[RagDocument]:
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
        """Return a RAG adapter, or None. Override to enable RAG."""
        return None

    _UNSET: Any = object()

    async def run(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        system_prompt: str | Prompt | None = None,
        response_format: type[T] | None = _UNSET,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
        tools: bool = False,
        *,
        channel: str | None = None,
        oversight: type[Policy] | str | Literal[False] | None = None,
    ) -> T | str | None:
        """A masked question is masked in place in `messages`, so the caller stores that copy.

        Raises:
            Blocked: The policy blocked the question or withheld the answer; show `message`.
        """

        from django_ai_sdk.adapters.run import Run
        from django_ai_sdk.agents.tool_agent import default_hooks
        from django_ai_sdk.oversight.history import history_for, thread_storage
        from django_ai_sdk.oversight.plugin import TurnRequest, mask_stored_question

        resolved = self.response_format if response_format is self._UNSET else response_format
        rendered_prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        given = as_chat_messages(messages)

        async def admit(request: TurnRequest) -> None:
            mask_stored_question(given, request)

        with bind_trace(thread_id=thread_id):
            adapter = await self.get_run_adapter(thread_id=thread_id, user=user)
            if not isinstance(adapter, Run):
                raise TypeError(
                    f"{type(self).__name__}.get_run_adapter() must return a Run, "
                    f"not {type(adapter).__name__}."
                )
            sent, vouched = await history_for(
                self,
                await thread_storage(self, thread_id),
                given,
                channel=channel,
                user=user,
                oversight=oversight,
            )
            with_tools = tools and resolved is None
            return await adapter.run(
                sent,
                rendered_prompt,
                response_format=resolved,
                tools=await self.get_tools(thread_id=thread_id or "", user=user)
                if with_tools
                else None,
                max_agent_steps=self.max_agent_steps if with_tools else None,
                hooks=default_hooks(self),
                request=TurnRequest(
                    agent=self,
                    channel=channel,
                    user=user,
                    thread_id=thread_id,
                    oversight=oversight,
                    history_vouched=vouched,
                    on_gated=admit,
                ),
            )

    async def get_run_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        from django_ai_sdk.adapters.base import Run

        return Run(generator=self.get_llm())

    #: Integration names (registry keys); every tool of each reaches this agent.
    integrations: list[str] = []

    async def _allowed_integrations(
        self, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> list[Any]:
        if not self.integrations:
            return []
        services = (await get_integrations(self.integrations)).values()
        return [s for s in services if await s.has_perms(user, Operation.USE_INTEGRATION)]

    async def _get_integration_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        async def _safe_get_tools(integration: Any) -> list[Any]:
            try:
                tools = await integration.get_tools(user, agent=self, thread_id=thread_id)
            except Exception:
                logger.exception("Failed to load tools for integration {!r}", integration.name)
                return []
            return [_namespaced(integration.name, tool, integration.hint) for tool in tools]

        allowed = await self._allowed_integrations(user)
        results = await asyncio.gather(*(_safe_get_tools(i) for i in allowed))
        return [tool for tools in results for tool in tools]

    async def warmup_progress(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Open the slow parts of the chat build, yielding what is being waited on."""
        named = False
        if self.rag_provider and thread_id:
            from django_ai_sdk.memories.services import MemoryService

            for memory in await MemoryService.get_thread_memories(thread_id, user=user):
                named = True
                yield _warmup_part("knowledge", memory.name)
                await self.rag_provider.get_rag_instance(self, str(memory.id))

        async def load(integration: Any) -> tuple[Any, dict[str, Any] | None]:
            try:
                status = await integration.get_status(user, agent=self)
            except Exception:
                logger.exception("Failed to check integration {!r}", integration.name)
                return integration, _unavailable_part("integrations", integration.label)
            if status == IntegrationStatus.DEGRADED:
                return integration, _unavailable_part("integrations", integration.label)
            return integration, None

        waiting = await self._allowed_integrations(user)
        loads = [asyncio.ensure_future(load(i)) for i in waiting]
        if waiting:
            named = True
            yield _warmup_part("integrations", ", ".join(i.label for i in waiting))
        try:
            for finished in asyncio.as_completed(loads):
                integration, part = await finished
                waiting.remove(integration)
                if part:
                    yield part
                if waiting:
                    yield _warmup_part("integrations", ", ".join(i.label for i in waiting))
            if named:
                yield {"type": "data-warmup", "data": {"status": "start"}, "transient": True}
        finally:
            for task in loads:
                task.cancel()

    async def get_pipeline_adapter(
        self,
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        from django_ai_sdk.adapters.base import Stream
        from django_ai_sdk.agents.tool_agent import ToolAgent, ToolAgentConfig, default_hooks

        generator = self.get_llm()
        tool_agent = ToolAgent(
            config=ToolAgentConfig(
                model=self.get_model(),
                system_prompt=self.get_system_prompt(),
                tools=await self.get_tools(thread_id=thread_id or "", user=user),
                max_agent_steps=self.max_agent_steps,
                hooks=default_hooks(self),
            ),
            generator=generator,
        )
        return Stream(
            pipeline=tool_agent.pipeline(),
            generator=generator,
            storage_adapter=await self.get_storage_adapter(thread_id),
        )

    async def history(
        self, thread_id: str, user: AbstractBaseUser | AnonymousUser | None = None
    ) -> ThreadDetail:
        """Raises PermissionDenied without VIEW_THREAD."""
        logger.debug(f"Fetching history for thread: {thread_id}")

        await check_permissions(
            user, Operation.VIEW_THREAD, get_agent_permissions(self), agent=self
        )

        storage = await self.get_storage_adapter(thread_id)

        if not storage:
            raise AiSdkError(
                f"No storage adapter found for thread: {thread_id}",
                ErrorCode.CONFIGURATION_ERROR,
            )

        thread_info = await storage.__class__.get_thread(thread_id)
        if not thread_info:
            raise NotFound(f"Thread not found: {thread_id}")

        await ThreadService.has_perms(user, Operation.VIEW_THREAD, thread_info)
        await check_object_permissions(
            user,
            Operation.VIEW_THREAD,
            thread_info,
            get_agent_permissions(self),
            agent=self,
        )

        chat_messages = await storage.get_messages()
        logger.debug(f"Retrieved {len(chat_messages)} ChatMessages, converting to protocol format")

        if any(m.attachments for m in chat_messages):
            from django_ai_sdk.memories.services import MemoryService

            await MemoryService.resolve_attachments(thread_id, chat_messages, user=user, agent=self)

        protocol_messages = self.protocol_handler.from_chat_messages(chat_messages)

        return ThreadDetail(
            thread=thread_info,
            messages=protocol_messages,
        )

    async def as_view(
        self,
        protocol_messages: list[Any],
        thread_id: str | None = None,
        user: AbstractBaseUser | AnonymousUser | None = None,
    ) -> Any:
        """Raises PermissionDenied without CHAT permission for this agent/thread."""
        logger.debug(
            f"Agent as_view called: agent={self.__class__.__name__}, "
            f"messages={len(protocol_messages) if protocol_messages else 0}, "
            f"thread_id={thread_id}, user={user}"
        )

        await check_permissions(user, Operation.CHAT, get_agent_permissions(self), agent=self)

        thread = await ThreadService.get_thread(thread_id, user=user) if thread_id else None
        if thread:
            await check_object_permissions(
                user, Operation.CHAT, thread, get_agent_permissions(self), agent=self
            )

        messages = self.protocol_handler.to_chat_messages(protocol_messages)
        logger.debug(
            f"Protocol handler converted {len(protocol_messages) if protocol_messages else 0} "
            f"protocol messages to {len(messages)} chat messages"
        )

        storage_adapter = await self.get_storage_adapter(thread_id)

        await self._resolve_attachments(messages, thread, user, inline_images=self.has_vision())

        from django_ai_sdk.agents.chat import chat_turn

        return await chat_turn(self, messages, thread, thread_id, storage_adapter, user)

    async def _resolve_attachments(
        self,
        messages: list[ChatMessage],
        thread: ThreadInfo | None,
        user: AbstractBaseUser | AnonymousUser | None,
        *,
        inline_images: bool,
    ) -> None:
        if not any(m.attachments for m in messages):
            return
        if thread is None:
            for message in messages:
                message.attachments = []
            return
        from django_ai_sdk.memories.services import MemoryService

        await MemoryService.resolve_attachments(
            str(thread.id), messages, user=user, agent=self, inline_images=inline_images
        )
