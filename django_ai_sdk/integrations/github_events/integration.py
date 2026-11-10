"""GitHub as a webhook integration: an agent reviews pull requests."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import httpx

from django_ai_sdk.integrations.github_events.verify import is_signed_by_github
from django_ai_sdk.integrations.webhooks.base import (
    REPLY_TIMEOUT_SECONDS,
    UNSIGNED_DEDUP_TTL,
    InboundEvent,
    WebhookIntegration,
)

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"

# GitHub rejects a comment body longer than this.
MAX_COMMENT_LENGTH = 65536

# A draft waits for ready_for_review; the other actions are new code to look at.
DEFAULT_ACTIONS = frozenset({"opened", "reopened", "synchronize", "ready_for_review"})

SIGNATURE_HEADER = "X-Hub-Signature-256"
EVENT_HEADER = "X-GitHub-Event"


class GitHubEventsIntegration(WebhookIntegration):
    """Run the agent on a pull request and post its answer as a PR comment.

    Point a repository or organization webhook at this integration's path
    (HTTPS, the `pull_request` event, a sha256 secret). The token must be
    allowed to comment on those repositories:

        AI_SDK_INTEGRATIONS = {"github_events": {
            "WEBHOOK_SECRET": env("GITHUB_WEBHOOK_SECRET"),
            "TOKEN": env("GITHUB_TOKEN"),
            "ALLOW_WORKSPACES": ["owner/repo"],
            "AGENT": "myapp.agents.ReviewAgent",
        }}

    The event text only names the pull request. The agent's instructions define
    the review. To read the diff, add the `github` integration to the agent.
    """

    # Not "github": that name is the MCP integration that reads the code.
    name = "github_events"
    label = "GitHub"
    dedup_ttl = UNSIGNED_DEDUP_TTL
    # One thread per pull request, so a new push continues the review.
    thread_per_ref = True

    def missing_config(self) -> str | None:
        missing = self._missing_secrets(
            "WEBHOOK_SECRET",
            "TOKEN",
            hint="the secret from the webhook settings, and a token that may comment on "
            "pull requests.",
        )
        if missing:
            return missing
        # One webhook secret can cover a whole organization, so require an
        # explicit list of repositories.
        if not self._config().get("ALLOW_WORKSPACES"):
            return (
                "No repositories. Set AI_SDK_INTEGRATIONS['github_events']"
                f"['ALLOW_WORKSPACES'] to owner/repo names, or to {self.ANY!r}."
            )
        return None

    def verify(self, request: HttpRequest) -> bool:
        return is_signed_by_github(
            self.secret("WEBHOOK_SECRET"),
            request.headers.get(SIGNATURE_HEADER, ""),
            request.body,
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        # Acknowledge other events, such as the setup ping, without answering.
        if request.headers.get(EVENT_HEADER, "") != "pull_request":
            return None
        body = self._json(request)
        if body is None:
            return None

        action = body.get("action", "")
        pr = body.get("pull_request") or {}
        if action not in self._actions() or (pr.get("draft") and action != "ready_for_review"):
            return None

        repository = (body.get("repository") or {}).get("full_name", "")
        number = pr.get("number")
        sender = (body.get("sender") or {}).get("login", "")
        if not repository or number is None:
            return None
        # Skip the bot's own pushes so it never reviews its own work.
        bot = self.secret("BOT_USERNAME")
        if bot and sender.lower() == bot.lower():
            return None

        head = (pr.get("head") or {}).get("sha", "")
        parts = [
            f"Pull request #{number} in {repository} ({action}): {pr.get('title') or ''}",
            (pr.get("body") or "").strip(),
            f"{pr.get('html_url') or ''}\nHead: {head}",
        ]
        return InboundEvent(
            integration=self.name,
            # The signature covers the body, not the delivery header, so the body
            # is the key: a captured request sent again under a new delivery id
            # is still a repeat. GitHub's own redelivery resends the same body.
            event_id=hashlib.sha256(request.body).hexdigest(),
            text="\n\n".join(part for part in parts if part),
            conversation_id=repository,
            thread_ref=str(number),
            external_user_id=sender,
            workspace_id=repository,
        )

    async def reply(self, event: InboundEvent, text: str) -> None:
        # PR comments are posted through the issues API.
        url = f"{API_BASE}/repos/{event.conversation_id}/issues/{event.thread_ref}/comments"
        async with httpx.AsyncClient(timeout=REPLY_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                json={"body": text[:MAX_COMMENT_LENGTH]},
                headers={
                    "Authorization": f"Bearer {self.secret('TOKEN')}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        if not response.is_success:
            logger.error(
                "GitHub refused a comment on %s#%s: %s %s",
                event.conversation_id,
                event.thread_ref,
                response.status_code,
                response.text,
            )

    def _actions(self) -> frozenset[str]:
        return frozenset(self._config().get("ACTIONS") or DEFAULT_ACTIONS)
