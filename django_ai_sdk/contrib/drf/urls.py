"""Every SDK viewset on a DefaultRouter, plus the streaming chat view.

    path("api/", include("django_ai_sdk.contrib.drf.urls"))

Want to change one resource? Register your subclass on your own router instead and
leave this module out; each viewset is importable from ``django_ai_sdk.contrib.drf``.
"""

from __future__ import annotations

from django.urls import path
from rest_framework.routers import DefaultRouter

from django_ai_sdk.contrib.drf.views import (
    AgentViewSet,
    IntegrationViewSet,
    MemoryViewSet,
    MessageViewSet,
    RuntimeAgentViewSet,
    ThreadViewSet,
    WorkflowViewSet,
)
from django_ai_sdk.views.chat import ChatView

router = DefaultRouter()
router.register("threads", ThreadViewSet, basename="ai-sdk-thread")
router.register("messages", MessageViewSet, basename="ai-sdk-message")
router.register("agents", AgentViewSet, basename="ai-sdk-agent")
router.register("runtime-agents", RuntimeAgentViewSet, basename="ai-sdk-runtime-agent")
router.register("memories", MemoryViewSet, basename="ai-sdk-memory")
router.register("workflows", WorkflowViewSet, basename="ai-sdk-workflow")
router.register("integrations", IntegrationViewSet, basename="ai-sdk-integration")

urlpatterns = [
    path("threads/<str:thread_id>/chat/", ChatView.as_view(), name="ai-sdk-chat"),
    *router.urls,
]
