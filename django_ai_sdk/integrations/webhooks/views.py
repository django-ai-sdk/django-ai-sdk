from __future__ import annotations

import logging
import time
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

    from django_ai_sdk.integrations.webhooks.base import InboundEvent

logger = logging.getLogger(__name__)

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

    if parsed is None or not parsed.text.strip():
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

    if await _over_rate_limit(name, parsed):
        logger.info(
            "Ignoring a %r event from %r: over the rate limit",
            name,
            parsed.external_user_id,
        )
        return HttpResponse(status=200)

    # Platform limits are far too high for one agent turn.
    max_text = resolve_setting("AI_SDK_WEBHOOK_MAX_TEXT", 4000)
    if len(parsed.text) > max_text:
        logger.info(
            "Truncated a %r event from %d to %d characters",
            name,
            len(parsed.text),
            max_text,
        )
    parsed.text = parsed.text[:max_text]

    # The worker drops repeats, so a crash before queueing leaves nothing claimed.
    await handle_inbound.aenqueue(parsed.model_dump())
    return integration.ack(parsed)


async def _over_rate_limit(name: str, event: InboundEvent) -> bool:
    """More than AI_SDK_WEBHOOK_RATE_LIMIT_PER_MINUTE events from one sender this minute."""
    per_minute = int(resolve_setting("AI_SDK_WEBHOOK_RATE_LIMIT_PER_MINUTE", 0) or 0)
    if per_minute <= 0:
        return False
    sender = event.external_user_id or "unknown"
    key = f"ai_sdk:webhook:rl:{name}:{sender}:{int(time.time() // 60)}"
    # Async calls: DatabaseCache may not run on the event loop.
    if await cache.aadd(key, 1, timeout=120):
        return False
    return await cache.aincr(key) > per_minute


__all__ = ["receive"]
