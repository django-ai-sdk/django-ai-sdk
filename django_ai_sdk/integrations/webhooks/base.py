"""Base class for integrations that receive events, not only offer tools.

A WebhookIntegration is an Integration, so it shares credentials, config and the
registry entry with the outbound side. One Slack app is one integration.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from django.http import HttpResponse
from pydantic import BaseModel

from django_ai_sdk.integrations.base import Integration, IntegrationStatus

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.http import HttpRequest

    from django_ai_sdk.agents.base import Agent

logger = logging.getLogger(__name__)

# Seconds a reply to the platform may take. The run is already over by then.
REPLY_TIMEOUT_SECONDS = 10

# Dedup window for a platform that signs no timestamp. A captured request can be
# replayed once the window ends, so a week instead of the default ten minutes.
UNSIGNED_DEDUP_TTL = 7 * 24 * 60 * 60


class InboundEvent(BaseModel):
    """One message from a platform, in the same shape for every platform.

    It is queued as a dict, so keep it to what `reply()` needs.
    """

    integration: str
    event_id: str
    text: str
    conversation_id: str
    # Where to reply, such as a Slack thread_ts. Empty means the top level.
    thread_ref: str = ""
    external_user_id: str = ""
    workspace_id: str = ""
    # A per-event reply credential, for platforms that issue one. It travels
    # with the event because reply() runs in the worker.
    reply_token: str = ""


def json_body(response: Any) -> dict[str, Any]:
    """A platform response's JSON object, or {} when the body is not one."""
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


# The event being answered, set during the run. Lets a tool post to the same
# channel, such as uploading a file, without the agent passing the event along.
current_event: ContextVar[InboundEvent | None] = ContextVar("current_event", default=None)


class WebhookIntegration(Integration):
    """An integration that receives platform events and answers with an agent.

    Subclasses implement `verify`, `parse` and `reply`. `get_tools` and
    `get_status` have defaults here.
    """

    # Dotted path to the answering Agent. AI_SDK_INTEGRATIONS[name]["AGENT"] overrides it.
    agent: str = ""

    # -- inbound --
    # These are sync and run in the request, within the platform's ack window
    # (three seconds for Slack).

    @abstractmethod
    def verify(self, request: HttpRequest) -> bool:
        """Check that the request really came from the platform."""

    @abstractmethod
    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        """Turn a verified request into an event to answer.

        Return None to acknowledge and ignore it, such as the bot's own messages.
        Return an HttpResponse to answer a platform handshake directly.
        """

    def _json(self, request: HttpRequest) -> dict[str, Any] | None:
        """The request's JSON object, or None (logged) when the body is not one."""
        try:
            body = json.loads(request.body)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            logger.warning("%s sent a body that is not a JSON object", self.label)
            return None
        return body

    def ack(self, event: InboundEvent) -> HttpResponse:
        """Response sent once the event is queued, before the agent runs."""
        # Override to return a body, such as a placeholder shown to the asker.
        return HttpResponse(status=200)

    # -- de-duplication --

    #: Seconds during which a repeat of the same event id is ignored. Platforms
    #: redeliver and django_tasks is at-least-once, so this keeps one answer per
    #: event. Platforms that sign no timestamp use UNSIGNED_DEDUP_TTL, since the
    #: window is their only replay protection.
    dedup_ttl: int = 600

    # -- outbound --

    @abstractmethod
    async def reply(self, event: InboundEvent, text: str) -> None:
        """Post `text` back to the event's conversation."""

    #: One thread per thread_ref (a Slack thread, a pull request) instead of one
    #: per conversation. An event without a thread_ref then keeps no history.
    thread_per_ref: bool = False

    def thread_scope(self, event: InboundEvent) -> str | None:
        """Key for the stored conversation, or None to keep no history.

        By default one thread per platform conversation, such as a Telegram chat.
        """
        if not event.conversation_id:
            return None
        scope = f"ai-sdk:webhook:{self.name}:{event.conversation_id}"
        if not self.thread_per_ref:
            return scope
        return f"{scope}:{event.thread_ref}" if event.thread_ref else None

    # -- access --

    # Allow-list value that admits everyone, for when "anyone" is intended.
    ANY = "*"

    def allows(self, event: InboundEvent) -> bool:
        """Check that the sender and workspace may reach the agent."""
        return self._admits("ALLOW_FROM", event.external_user_id) and self._admits(
            "ALLOW_WORKSPACES", event.workspace_id
        )

    def has_audience(self) -> bool:
        """True when ALLOW_FROM is set, to a list or to ANY."""
        return bool(self._config().get("ALLOW_FROM"))

    def _admits(self, key: str, value: str) -> bool:
        # An unset list admits everyone the platform lets through.
        allowed = self._config().get(key) or []
        if not allowed or allowed == self.ANY:
            return True
        if not isinstance(allowed, (list, tuple, set, frozenset)) or not all(
            isinstance(entry, str) for entry in allowed
        ):
            logger.error(
                "AI_SDK_INTEGRATIONS[%r][%r] must be a list of platform ids or %r; "
                "refusing every event until it is one",
                self.name,
                key,
                self.ANY,
            )
            return False
        # Accept ANY inside a list too, since env.list(..., default=["*"]) gives that.
        if self.ANY in allowed:
            return True
        return value in set(allowed)

    # -- identity --

    async def aget_principal(self) -> AbstractBaseUser | None:
        """The RUN_AS user, or None to answer anonymously."""
        from django_ai_sdk.utils import aget_active_user

        identifier = self.secret("RUN_AS")
        user = await aget_active_user(identifier)
        if identifier and user is None:
            # Fall back to anonymous, the lower privilege. The agent then reaches
            # no integration.
            logger.error(
                "AI_SDK_INTEGRATIONS[%r]['RUN_AS'] = %r matches no single active user; "
                "answering anonymously",
                self.name,
                identifier,
            )
        return user

    # -- configuration --

    # A detail set by hand wins over the computed one, as on any Integration.
    _detail: str | None = None

    @property
    def detail(self) -> str | None:
        return self._detail or self.missing_config() or self._missing_agent()

    @detail.setter
    def detail(self, value: str | None) -> None:
        self._detail = value

    def missing_config(self) -> str | None:
        """Why the credentials are incomplete, or None."""
        return None

    def _missing_secrets(self, *keys: str, hint: str) -> str | None:
        """Name the unset `keys` and where to find them, or None when all are set."""
        missing = [key for key in keys if not self.secret(key)]
        if not missing:
            return None
        return (
            f"Missing {', '.join(missing)}. Set them in AI_SDK_INTEGRATIONS[{self.name!r}]: {hint}"
        )

    def _missing_agent(self) -> str | None:
        if self.agent_path():
            return None
        return (
            f"No agent configured. Set AI_SDK_INTEGRATIONS[{self.name!r}]['AGENT'] to the "
            f"dotted path of an Agent subclass, such as 'myapp.agents.SupportAgent'."
        )

    def agent_path(self) -> str:
        """The agent's dotted path. The AGENT setting wins over the class attribute."""
        return self.secret("AGENT") or self.agent

    def resolve_agent(self) -> Agent | None:
        """The answering agent, or None if the path does not resolve."""
        from django.utils.module_loading import import_string

        from django_ai_sdk.agents.registry import registry

        path = self.agent_path()
        if not path:
            return None
        try:
            agent_class = import_string(path)
        except ImportError:
            logger.exception(
                "AI_SDK_INTEGRATIONS[%r]['AGENT'] = %r cannot be imported", self.name, path
            )
            return None
        # The path may import something that is not an Agent. Only registered
        # Agent subclasses have an _agent_id, and abstract agents are never registered.
        agent_id = getattr(agent_class, "_agent_id", "")
        if not agent_id:
            logger.error(
                "%r names %r, which is not a registered Agent subclass; abstract agents "
                "and non-Agent paths cannot answer",
                self.name,
                path,
            )
            return None
        return registry.get(agent_id)

    # -- Integration contract --

    async def get_tools(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
        thread_id: str = "",
    ) -> list[Any]:
        return []

    async def get_status(
        self,
        user: AbstractBaseUser | AnonymousUser | None = None,
        agent: Agent | None = None,
    ) -> IntegrationStatus:
        return IntegrationStatus.DISCONNECTED if self.detail else IntegrationStatus.ACTIVE

    @property
    def kind(self) -> str:
        return "webhook"
