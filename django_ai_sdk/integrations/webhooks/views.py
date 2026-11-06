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

# Integrations already warned about as unconfigured. Warn once per process, so
# scanners hitting the path do not flood the log.
_warned_unconfigured: set[str] = set()


@csrf_exempt
async def receive(request: HttpRequest, name: str) -> HttpResponse:
    """Verify, normalise and queue one inbound platform event."""
    if request.method != "POST":
        return HttpResponseNotFound()

    integration = (await get_integrations([name])).get(name)
    # Admin-created MCP servers are not WebhookIntegrations, so they cannot be reached here.
    if not isinstance(integration, WebhookIntegration):
        return HttpResponseNotFound()
    if integration.detail:
        if name not in _warned_unconfigured:
            _warned_unconfigured.add(name)
            logger.warning("Refusing webhooks for %r: %s", name, integration.detail)
        return HttpResponseNotFound()

    if not integration.verify(request):
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

    # Platform limits are far too high for one agent turn. Log the cut, since the
    # person in the channel cannot see it.
    max_text = resolve_setting("AI_SDK_WEBHOOK_MAX_TEXT", 4000)
    if len(parsed.text) > max_text:
        logger.info(
            "Truncated a %r event from %d to %d characters", name, len(parsed.text), max_text
        )
    parsed.text = parsed.text[:max_text]

    # Skip dedup for an event without an id, or all such events would share one key.
    key = f"ai_sdk:webhook:{name}:{parsed.event_id}" if parsed.event_id else ""
    if key and not await cache.aadd(key, 1, integration.dedup_ttl):
        return HttpResponse(status=200)

    try:
        await handle_inbound.aenqueue(parsed.model_dump())
    except Exception:
        # Release the key so the platform's redelivery is not dropped as a duplicate.
        if key:
            await cache.adelete(key)
        raise
    return integration.ack(parsed)


__all__ = ["receive"]
