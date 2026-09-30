"""The chat endpoint: a plain async Django view, so it works under any API framework.

Streaming is the one endpoint neither Ninja nor DRF helps with, so the SDK ships it
framework-free and the contrib layers mount it as-is::

    path("api/threads/<str:thread_id>/chat/", ChatView.as_view())  # the thread's agent
    path("api/chat/", ChatView.as_view(agent="support-bot"))  # no thread: stores nothing

Without a thread the conversation is not persisted; the client sends the whole history
each turn. The ``CHAT`` permission is checked either way.

Subclass and override :meth:`get_agent` to pick the agent differently, or
:meth:`error_response` to shape failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View

from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.errors import NotFound, error_response
from django_ai_sdk.views.schemas import ChatRequest

if TYPE_CHECKING:
    from django_ai_sdk import Agent
    from django_ai_sdk.types import UserType


class ChatView(View):
    http_method_names = ["post"]
    # Agent id for URLs without a thread. The payload's `agent_id` is ignored, so a
    # client can't reach another agent through a fixed-agent URL.
    agent: str | None = None

    async def post(self, request: HttpRequest, thread_id: str | None = None) -> HttpResponse:
        user = await request.auser()
        try:
            payload = ChatRequest.model_validate_json(request.body)
            agent = await self.get_agent(thread_id, user)
            return await agent.as_view(payload.messages, thread_id=thread_id, user=user)
        except Exception as exc:
            # Before the stream starts; errors inside it arrive as stream events.
            return self.error_response(exc)

    async def get_agent(self, thread_id: str | None, user: UserType) -> Agent:
        if thread_id is not None:
            return await AgentService.get_agent(thread_id, user=user)
        if self.agent is None:
            raise NotFound("ChatView without a thread needs an agent")
        return await AgentService.get(self.agent)

    def error_response(self, exc: Exception) -> HttpResponse:
        """A coded error body (see :mod:`django_ai_sdk.errors`); never the exception text."""
        status, body = error_response(exc)
        return JsonResponse(body, status=status)
