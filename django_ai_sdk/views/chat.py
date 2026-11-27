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
    """Without a thread nothing is stored; CHAT is checked either way."""

    http_method_names = ["post"]
    agent: str | None = None

    async def post(self, request: HttpRequest, thread_id: str | None = None) -> HttpResponse:
        user = await request.auser()
        try:
            payload = ChatRequest.model_validate_json(request.body)
            agent = await self.get_agent(thread_id, user)
            return await agent.as_view(payload.messages, thread_id=thread_id, user=user)
        except Exception as exc:
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
