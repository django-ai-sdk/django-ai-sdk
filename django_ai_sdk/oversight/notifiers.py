from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import build_configured, setting
from django_ai_sdk.oversight.rules import Notify, Queue

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.guard.judging import Matched

logger = get_logger(__name__)

EVENT_NAMESPACE = uuid.UUID("5b0f6f2e-3c1d-4f43-9a55-7d3c0e0f1a11")


def event_id(turn_id: Any, rule: str, stage: str) -> str:
    return str(uuid.uuid5(EVENT_NAMESPACE, f"{turn_id}:{rule}:{stage}"))


def not_yet_raised(turn_id: Any, thread_id: str, matched: Iterable[Matched]) -> list[Matched]:
    from django_ai_sdk.oversight.models import ReviewItem

    matched = list(matched)
    queued = {
        match.rule.name: {(a.queue, match.rule.name) for a in match.acted if isinstance(a, Queue)}
        for match in matched
    }
    reviews = set().union(*queued.values())
    if not thread_id or not reviews:
        return matched
    still_open = set(
        ReviewItem.objects.filter(
            turn__thread_id=thread_id,
            status=ReviewItem.Status.OPEN,
            rule__in={rule for _, rule in reviews},
        )
        .exclude(turn_id=turn_id)
        .values_list("queue", "rule")
    )
    return [m for m in matched if not (queued[m.rule.name] and queued[m.rule.name] <= still_open)]


def notify_events(
    turn_id: Any,
    matched: Iterable[Matched],
    *,
    thread_id: str,
    agent: str,
    channel: str,
    stage: str = "",
    outcome: str | None = None,
) -> list[tuple[dict[str, Any], list[str]]]:
    notifying = [m for m in matched if any(isinstance(a, Notify) for a in m.acted)]
    events = []
    for match in not_yet_raised(turn_id, thread_id, notifying):
        for action in match.acted:
            if not isinstance(action, Notify):
                continue
            at = stage or str(match.stage)
            event = {
                "id": event_id(turn_id, match.rule.name, at),
                "type": "oversight.match",
                "turn_id": str(turn_id),
                "rule": match.rule.name,
                "stage": at,
                "severity": action.severity,
                "agent": agent,
                "channel": channel,
            }
            if outcome is not None:
                event["outcome"] = outcome
            events.append((event, list(action.to)))
    return events


class Notifier:
    """Subclass and implement `send`, which may raise (errors are logged).

    `Notify(to=[...])` selects by `name`, default the lowercased class name without "Notifier".
    """

    name: str = ""

    def __init__(self, *, name: str = "") -> None:
        self.name = name or type(self).__name__.removesuffix("Notifier").lower()

    async def send(self, event: dict[str, Any]) -> None:
        raise NotImplementedError


class LogNotifier(Notifier):
    """Write events to the SDK log at WARNING."""

    async def send(self, event: dict[str, Any]) -> None:
        logger.warning("oversight {}", json.dumps(event, default=str, sort_keys=True))


class WebhookNotifier(Notifier):
    """POST events as JSON, signed per Standard Webhooks when `secret` is set.

    A secret starting with "whsec_" is base64.
    """

    def __init__(self, url: str, *, secret: str = "", timeout: float = 5.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.secret = secret
        self.timeout = timeout

    def build_request(self, event: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(event, default=str, sort_keys=True).encode()
        msg_id = str(event["id"])
        headers = {"Content-Type": "application/json", "webhook-id": msg_id}
        if self.secret:
            timestamp = str(int(time.time()))
            signed = f"{msg_id}.{timestamp}.".encode() + body
            key = (
                base64.b64decode(self.secret.removeprefix("whsec_"))
                if self.secret.startswith("whsec_")
                else self.secret.encode()
            )
            digest = hmac.new(key, signed, hashlib.sha256).digest()
            headers["webhook-timestamp"] = timestamp
            headers["webhook-signature"] = "v1," + base64.b64encode(digest).decode()
        return body, headers

    async def send(self, event: dict[str, Any]) -> None:
        import httpx

        body, headers = self.build_request(event)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, content=body, headers=headers)
            response.raise_for_status()


def configured_notifiers(names: list[str] | None = None) -> list[Notifier]:
    notifiers: list[Notifier] = []
    for spec in setting("NOTIFIERS") or []:
        try:
            notifier = build_configured(spec, Notifier, __name__)
        except Exception:
            logger.exception("Oversight notifier {!r} could not be built", spec)
            continue
        if names and notifier.name not in names:
            continue
        notifiers.append(notifier)
    return notifiers


async def send_to_notifiers(event: dict[str, Any], *, names: list[str] | None = None) -> None:
    """Failures are logged."""
    for notifier in configured_notifiers(names):
        try:
            await notifier.send(event)
        except Exception:
            logger.exception("Oversight notifier {} failed", notifier.name)
