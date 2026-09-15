"""The receive endpoint every webhook integration shares."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.core.cache import cache
from django.http import HttpResponse, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt

from django_ai_sdk.integrations.registry import get_integrations
from django_ai_sdk.integrations.webhooks.base import WebhookIntegration
from django_ai_sdk.integrations.webhooks.tasks import handle_inbound
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from django.http import HttpRequest

logger = logging.getLogger(__name__)

# A platform redelivers an unacknowledged event and django_tasks dispatch is
# at-least-once, so an event id is answered once within this window.
DEDUP_TTL_DEFAULT = 600

# Names already warned about being unconfigured. Once per process: the condition
# cannot change without a restart, and anything scanning the path would otherwise
# write a log line per request.
_warned_unconfigured: set[str] = set()


@csrf_exempt
async def receive(request: HttpRequest, name: str) -> HttpResponse:
    """Verify, normalise and queue one inbound platform event."""
    if request.method != "POST":
        return HttpResponseNotFound()

    integration = (await get_integrations([name])).get(name)
    # No admin-authored MCPServerConfig row can reach this endpoint: such a row builds
    # a DynamicMCPIntegration, which is not a WebhookIntegration.
    if not isinstance(integration, WebhookIntegration):
        return HttpResponseNotFound()
    if integration.detail:
        if name not in _warned_unconfigured:
            _warned_unconfigured.add(name)
            logger.warning("Refusing webhooks for %r: %s", name, integration.detail)
        return HttpResponseNotFound()

    if not integration.verify(request):
        # 401 rather than 403: Discord validates a new endpoint by sending a
        # deliberately bad signature and requires that status back.
        return HttpResponse(status=401)

    parsed = integration.parse(request)
    if isinstance(parsed, HttpResponse):
        return parsed
    if parsed is None:
        return HttpResponse(status=200)

    if not integration.allows(parsed):
        logger.info(
            "Ignoring a %r event from sender %r in workspace %r; add it to "
            "AI_SDK_INTEGRATIONS[%r]['ALLOW_FROM'] or ['ALLOW_WORKSPACES'] to answer it",
            name,
            parsed.external_user_id,
            parsed.workspace_id,
            name,
        )
        return HttpResponse(status=200)

    # A platform's own limit is far above what an agent should be handed in one turn:
    # a Slack message carries tens of thousands of characters.
    parsed.text = parsed.text[: resolve_setting("AI_SDK_WEBHOOK_MAX_TEXT", 4000)]

    key = f"ai_sdk:webhook:{name}:{parsed.event_id}"
    ttl = resolve_setting("AI_SDK_WEBHOOK_DEDUP_TTL", DEDUP_TTL_DEFAULT)
    if not await cache.aadd(key, 1, ttl):
        return HttpResponse(status=200)

    try:
        await handle_inbound.aenqueue(parsed.model_dump())
    except Exception:
        # The key is claimed before the work is durable, so a failed queue write
        # releases it and the platform's redelivery is answered rather than deduped.
        await cache.adelete(key)
        raise
    return integration.ack(parsed)


__all__ = ["DEDUP_TTL_DEFAULT", "receive"]
