"""The chat endpoint: a plain async Django view, so it works under any API framework.

Streaming is the one endpoint neither Ninja nor DRF helps with, so the SDK ships it
framework-free and the contrib layers mount it as-is::

    path("api/threads/<str:thread_id>/chat/", ChatView.as_view())

Subclass and override :meth:`get_agent` to pick the agent differently, or
:meth:`error_response` to shape failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View

from django_ai_sdk.agents.services import AgentService
from django_ai_sdk.errors import error_response
from django_ai_sdk.views.schemas import ChatRequest

if TYPE_CHECKING:
    from django_ai_sdk import Agent
    from django_ai_sdk.types import UserType


class ChatView(View):
    http_method_names = ["post"]

    async def post(self, request: HttpRequest, thread_id: str) -> HttpResponse:
        user = await request.auser()
        try:
            payload = ChatRequest.model_validate_json(request.body)
            agent = await self.get_agent(thread_id, user)
            return await agent.as_view(payload.messages, thread_id=thread_id, user=user)
        except Exception as exc:
            # Before the stream starts; errors inside it arrive as stream events.
            return self.error_response(exc)

    async def get_agent(self, thread_id: str, user: UserType) -> Agent:
        return await AgentService.get_agent(thread_id, user=user)

    def error_response(self, exc: Exception) -> HttpResponse:
        """A coded error body (see :mod:`django_ai_sdk.errors`); never the exception text."""
        status, body = error_response(exc)
        return JsonResponse(body, status=status)
