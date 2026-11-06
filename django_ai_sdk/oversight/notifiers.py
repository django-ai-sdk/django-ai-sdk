"""Notifiers send rule matches somewhere besides the oversight record.

    AI_SDK_OVERSIGHT = {"NOTIFIERS": [
        {"class": "django_ai_sdk.oversight.notifiers.WebhookNotifier", "name": "siem",
         "url": "https://siem.example/hook", "secret": env("SIEM_SECRET")},
    ]}

Notifiers never fail a turn. Errors are logged. Notifications are sent from the
task worker (`tasks.send_notifications`), so a slow receiver never holds a chat
stream open.

Each event has a stable `id` (see `event_id`), sent by the webhook as
`webhook-id`. A retried delivery keeps the same id, so receivers can drop repeats.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.actions import Notify

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.guard.asking import Matched

logger = get_logger(__name__)

# Event ids are derived from their content, so a repeat gets the same id.
EVENT_NAMESPACE = uuid.UUID("5b0f6f2e-3c1d-4f43-9a55-7d3c0e0f1a11")


def event_id(turn_id: Any, rule: str, stage: str) -> str:
    """Return a stable id for `rule` acting on turn `turn_id` at `stage`."""
    return str(uuid.uuid5(EVENT_NAMESPACE, f"{turn_id}:{rule}:{stage}"))


def match_events(
    turn_id: Any,
    matched: Iterable[Matched],
    *,
    agent: str,
    channel: str,
    stage: str = "",
    outcome: str | None = None,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Return (the "oversight.match" event, notifier names) for each Notify that acted.

    `stage` defaults to each match's own.
    """
    events = []
    for match in matched:
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
    """Subclass and implement `send`.

    `send` may raise, since `anotify` logs the error and carries on.

    `Notify(to=[...])` selects notifiers by `name`. It defaults to the class name,
    lowercased, without "Notifier" ("log", "webhook").
    """

    name: str = ""

    def __init__(self, *, name: str = "") -> None:
        self.name = name or type(self).__name__.removesuffix("Notifier").lower()

    async def send(self, event: dict[str, Any]) -> None:
        """Send one event."""
        raise NotImplementedError


class LogNotifier(Notifier):
    """Write events to the SDK log at WARNING, for shipping logs into a SIEM."""

    async def send(self, event: dict[str, Any]) -> None:
        logger.warning("oversight {}", json.dumps(event, default=str, sort_keys=True))


class WebhookNotifier(Notifier):
    """POST events as JSON, signed per Standard Webhooks when `secret` is set.

    webhook-signature is "v1," + base64 HMAC-SHA256 of "{webhook-id}.{webhook-timestamp}.{body}".
    The timestamp lets receivers reject replays as well as forgeries. A secret
    starting with "whsec_" is base64. webhook-id is the event's `id`.
    """

    def __init__(self, url: str, *, secret: str = "", timeout: float = 5.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.secret = secret
        self.timeout = timeout

    def _request(self, event: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(event, default=str, sort_keys=True).encode()
        msg_id = str(event["id"])
        headers = {"Content-Type": "application/json", "webhook-id": msg_id}
        if self.secret:
            timestamp = str(int(time.time()))
            signed = f"{msg_id}.{timestamp}.".encode() + body
            digest = hmac.new(_key(self.secret), signed, hashlib.sha256).digest()
            headers["webhook-timestamp"] = timestamp
            headers["webhook-signature"] = "v1," + base64.b64encode(digest).decode()
        return body, headers

    async def send(self, event: dict[str, Any]) -> None:
        import httpx

        body, headers = self._request(event)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, content=body, headers=headers)
            response.raise_for_status()


def _key(secret: str) -> bytes:
    if secret.startswith("whsec_"):
        return base64.b64decode(secret.removeprefix("whsec_"))
    return secret.encode()


def configured_notifiers(names: list[str] | None = None) -> list[Notifier]:
    """Build the NOTIFIERS setting's notifiers, or only those named in `names`."""
    from django_ai_sdk.oversight.conf import setting

    notifiers: list[Notifier] = []
    for spec in setting("NOTIFIERS") or []:
        try:
            notifier = spec if isinstance(spec, Notifier) else _build_notifier(spec)
        except Exception:
            logger.exception("Oversight notifier {!r} could not be built", spec)
            continue
        if names and notifier.name not in names:
            continue
        notifiers.append(notifier)
    return notifiers


def _build_notifier(spec: str | dict[str, Any]) -> Notifier:
    if isinstance(spec, str):
        return import_string(spec)()
    config = dict(spec)
    cls = import_string(config.pop("class"))
    return cls(**config)


async def anotify(event: dict[str, Any], *, names: list[str] | None = None) -> None:
    """Send `event` to all notifiers, or only to `names`. Failures are logged."""
    for notifier in configured_notifiers(names):
        try:
            await notifier.send(event)
        except Exception:
            logger.exception("Oversight notifier {} failed", notifier.name)


__all__ = [
    "LogNotifier",
    "Notifier",
    "WebhookNotifier",
    "anotify",
    "configured_notifiers",
    "event_id",
    "match_events",
]
