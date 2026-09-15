"""Contract for an integration that receives events, rather than only offering tools.

A WebhookIntegration is an Integration, so it shares the credential, the config slice
and the registry entry with the outbound direction: one Slack app, one row.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from django.http import HttpResponse
from pydantic import BaseModel

from django_ai_sdk.integrations.base import Integration, IntegrationStatus

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.http import HttpRequest

    from django_ai_sdk.agent import Agent

logger = logging.getLogger(__name__)


class InboundEvent(BaseModel):
    """One message pushed in by a platform, normalised across platforms.

    Crosses the queue as a dict, so it carries what `reply()` needs and nothing else.
    """

    integration: str
    event_id: str
    text: str
    conversation_id: str
    # Reply target within the conversation, e.g. a Slack thread_ts. Empty posts at
    # the top level of the conversation.
    thread_ref: str = ""
    external_user_id: str = ""
    workspace_id: str = ""
    # Credential the platform issued for answering this one event, where it uses
    # one. Crosses the queue because reply() runs in the worker.
    reply_token: str = ""


class WebhookIntegration(Integration):
    """An integration a platform pushes events into, answered by an agent.

    Subclasses implement `verify`, `parse` and `reply`; `get_tools` and `get_status`
    default here, so a receive-only integration writes neither.
    """

    # Dotted path to the Agent subclass that answers. A deployment overrides it with
    # AI_SDK_INTEGRATIONS[name]["AGENT"].
    agent: str = ""

    # -- inbound (sync: these run on the request path, inside the platform's
    # acknowledgement window, which for Slack is three seconds) --

    @abstractmethod
    def verify(self, request: HttpRequest) -> bool:
        """Whether this request really came from the platform, by signature."""

    @abstractmethod
    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        """Normalise a verified request into an event to answer.

        Returns None for anything to acknowledge and ignore, such as the bot's own
        messages, and an HttpResponse for a handshake the platform wants answered
        inline with a body of its choosing.
        """

    def ack(self, event: InboundEvent) -> HttpResponse:
        """The response that acknowledges a queued event, before the agent has run."""
        # An empty 200 is enough for a platform that only wants delivery confirmed;
        # one that shows the asker a placeholder returns its own body here.
        return HttpResponse(status=200)

    # -- outbound --

    @abstractmethod
    async def reply(self, event: InboundEvent, text: str) -> None:
        """Post text back to the conversation the event came from."""

    def thread_scope(self, event: InboundEvent) -> str | None:
        """Stable key for one persisted agent conversation, or None to stay stateless.

        Default: one SDK thread per platform conversation (Telegram chat, Discord
        channel/DM). Slack overrides this to one SDK thread per Slack thread.
        """
        if not event.conversation_id:
            return None
        return f"ai-sdk:webhook:{self.name}:{event.conversation_id}"

    # -- access --

    # An allow-list set to this admits everyone, so an integration that demands an
    # explicit audience still has a way to say "anyone" on purpose.
    ANY = "*"

    def allows(self, event: InboundEvent) -> bool:
        """Whether this sender, in this workspace, may reach the agent."""
        return self._admits("ALLOW_FROM", event.external_user_id) and self._admits(
            "ALLOW_WORKSPACES", event.workspace_id
        )

    def has_audience(self) -> bool:
        """Whether who may ask has been decided, by a list or by the ANY sentinel."""
        return bool(self._config().get("ALLOW_FROM"))

    def _admits(self, key: str, value: str) -> bool:
        # An unset list means whoever the platform lets through, which is everyone in
        # the workspace, guild or bot's direct messages.
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
        # env.list("...", default=["*"]) is the shape a deployment actually writes, so
        # the sentinel has to mean "anyone" inside a list too, not only bare.
        if self.ANY in allowed:
            return True
        return value in set(allowed)

    # -- identity --

    async def aget_principal(self) -> AbstractBaseUser | None:
        """The user a run acts as, or None to answer anonymously."""
        from django.contrib.auth import get_user_model

        identifier = self.secret("RUN_AS")
        if not identifier:
            return None

        user_model = get_user_model()
        field = user_model.USERNAME_FIELD  # ty: ignore[unresolved-attribute]
        try:
            return await user_model._default_manager.aget(**{field: identifier, "is_active": True})
        except user_model.DoesNotExist:
            # Anonymous is the lower privilege, so the answer still goes out and the
            # agent reaches no integration.
            logger.error(
                "AI_SDK_INTEGRATIONS[%r]['RUN_AS'] = %r matches no active user by %s; "
                "answering anonymously",
                self.name,
                identifier,
                field,
            )
            return None

    # -- configuration --

    @property
    def detail(self) -> str | None:
        return self.missing_config() or self._missing_agent()

    def missing_config(self) -> str | None:
        """Reason this integration's own credentials are incomplete, or None."""
        return None

    def _missing_agent(self) -> str | None:
        if self.agent_path():
            return None
        return (
            f"No agent configured. Set AI_SDK_INTEGRATIONS[{self.name!r}]['AGENT'] to the "
            f"dotted path of an Agent subclass, such as 'myapp.agents.SupportAgent'."
        )

    def agent_path(self) -> str:
        """The configured agent's dotted path, with the config slice winning."""
        return self.secret("AGENT") or self.agent

    def resolve_agent(self) -> Agent | None:
        """The agent that answers this integration's events, or None if unresolvable."""
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
        agent = registry.get(agent_class._agent_id)
        if agent is None:
            logger.error(
                "%r names agent %r, which is not registered; abstract agents cannot answer",
                self.name,
                path,
            )
        return agent

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
