---
title: Webhooks
type: docs
weight: 124
---

A **webhook integration** points the other way from a tool integration: a platform pushes an event in, an agent answers it, and the answer is posted back in the same channel. The SDK ships `telegram`, `slack` and `github_events`; writing another is three methods.

A webhook is answered by an **agent**, not a workflow: an inbound message is a turn in a conversation, and the thread carries the history. Scheduled work has no one to reply to and runs a workflow instead; see [Automations](/manual/automations/).

## Setup

```python
# settings.py
INSTALLED_APPS += [
    "django_ai_sdk.integrations.telegram",
    "django_ai_sdk.integrations.slack",
]

AI_SDK_INTEGRATIONS = {
    "telegram": {
        "BOT_TOKEN": env("TELEGRAM_BOT_TOKEN"),
        "WEBHOOK_SECRET": env("TELEGRAM_WEBHOOK_SECRET"),
        "ALLOW_FROM": ["123456789"],
        "RUN_AS": "bot@example.com",
        "AGENT": "myapp.agents.SupportAgent",
    },
}

# urls.py
path("api/integrations/", include("django_ai_sdk.integrations.webhooks.urls")),
```

The view verifies and queues; a `django_tasks` worker runs the agent and posts the reply, so run one (`manage.py db_worker` with `django_tasks_db`). Each integration is served at `api/integrations/<name>/webhook/`.

## How an event is handled

1. **Verify.** `verify(request)` checks the platform's signature. A failure is a 401. The endpoint is `csrf_exempt` and unauthenticated, so the signature is the only gate. Only a code-registered `WebhookIntegration` is reachable: an `MCPServerConfig` row builds a different type and gets a 404.
2. **Parse.** `parse(request)` returns an `InboundEvent` to answer, `None` to acknowledge and drop (a ping, an edit, the bot's own message), or an `HttpResponse` to return verbatim (Slack's URL handshake).
3. **Admit.** `ALLOW_FROM` (sender ids) and `ALLOW_WORKSPACES` (installations) must both admit the event. A refused event still gets a 200, so the platform does not redeliver it, and is logged with the ids to add.
4. **Deduplicate.** `cache.aadd` on the event id for `dedup_ttl` seconds, so a redelivery is answered once. Use a cache every process shares in production. The key is released if queueing fails.
5. **Queue and acknowledge.** The event goes to the worker, and the platform gets `ack(event)` (an empty 200 by default).
6. **Run.** The worker loads the conversation's thread, runs the agent as `RUN_AS` with `AI_SDK_WEBHOOK_TIMEOUT`, persists the turn and calls `reply(event, text)`. A failing or slow run still replies, with an error or timeout message.

The run is `agent.run(history + [question], tools=True, user=<RUN_AS>, thread_id=...)`. The history is the thread's stored messages, cut to the agent's `max_history`.

While the agent runs, the event is available as `django_ai_sdk.integrations.webhooks.current_event` (a `ContextVar`), for a tool that answers in the channel itself, such as uploading a file to the same thread.

## Who the agent acts as

`RUN_AS` names one account, matched on the user model's `USERNAME_FIELD` and required to be active. Every admitted sender is answered as that account, so give it only what the bot needs. Unset, unknown or ambiguous runs anonymously, which reaches no integration. The account also owns the persisted thread.

A signature proves the platform sent the event, not who asked or which installation it came from: a second Slack workspace running the same app verifies cleanly. That is what the allowlists are for.

## Conversations

`thread_scope(event)` maps a conversation to a stable key; the SDK hashes it into a thread id, so every process finds the same thread. Default: one thread per conversation (a Telegram chat). Set `thread_per_ref = True` for one per `thread_ref` instead, as Slack (one per Slack thread) and GitHub (one per pull request) do. Return `None` to answer statelessly. A new thread gets the agent's `memories` linked, as the thread endpoints do.

## Writing an integration

```python
class ZendeskIntegration(WebhookIntegration):
    name = "zendesk"
    label = "Zendesk"

    def verify(self, request) -> bool: ...          # signature check, sync
    def parse(self, request) -> InboundEvent | HttpResponse | None: ...  # sync
    async def reply(self, event, text) -> None: ...  # post back
```

Register it like any integration (an `IntegrationAppConfig`). Override `ack` for a platform that wants a body, `thread_per_ref` or `thread_scope` for its threading, `dedup_ttl` for its redelivery window (`UNSIGNED_DEDUP_TTL` when it signs no timestamp), and `missing_config` to name a missing credential (the endpoint 404s while one is missing).

`InboundEvent` crosses the queue as JSON, so it carries only what `reply` needs: `integration`, `event_id`, `text`, `conversation_id`, and optionally `thread_ref`, `external_user_id`, `workspace_id` and `reply_token`.

## Config keys

Per integration, in `AI_SDK_INTEGRATIONS[<name>]`:

| Key | Purpose |
| --- | --- |
| `AGENT` | Dotted path to the answering agent. |
| `RUN_AS` | The account runs act as. |
| `ALLOW_FROM` | Platform user ids that may ask, or `"*"`. Unset admits everyone the platform delivers; a malformed value admits nobody. |
| `ALLOW_WORKSPACES` | Installations (Slack `team_id`, GitHub `owner/repo`) that may ask, or `"*"`. |

Settings: `AI_SDK_WEBHOOK_TIMEOUT` (default `120` seconds) and `AI_SDK_WEBHOOK_MAX_TEXT` (default `4000` characters of the question kept).

## Telegram

Keys: `BOT_TOKEN` (from @BotFather), `WEBHOOK_SECRET` (a value you choose), `ALLOW_FROM` (**required**: a Telegram bot is reachable by anyone who knows its name, so `"*"` has to be written down).

Register the webhook once:

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d url=https://example.com/api/integrations/telegram/webhook/ \
  -d secret_token=$TELEGRAM_WEBHOOK_SECRET \
  -d 'allowed_updates=["message"]'
```

Telegram echoes the secret in `X-Telegram-Bot-Api-Secret-Token`; `verify` compares it in constant time. Text messages are answered, a leading `@botname` is stripped, and the reply is threaded under the question. Telegram signs no timestamp, so the dedup window (a week) is also the replay bound.

## Slack

Keys: `BOT_TOKEN` and `SIGNING_SECRET` (from the Slack app). Subscribe the app to `app_mention` and `message.im` and point its Event Subscriptions URL at `.../api/integrations/slack/webhook/`; the URL handshake is answered automatically.

Signatures are Slack's v0 HMAC, refused when older than 300 seconds. Bot messages and edits are ignored. Replies go into the Slack thread that asked.

## GitHub

Keys: `WEBHOOK_SECRET`, `TOKEN` (may comment on the repositories), `ALLOW_WORKSPACES` (**required**: `owner/repo` names, since one webhook can cover an organization), optionally `ACTIONS` and `BOT_USERNAME`.

Add a repository or organization webhook for the `pull_request` event with a sha256 secret. The agent runs on a pull request that is opened, reopened, pushed to or marked ready for review (`ACTIONS` changes the list); drafts wait. The event text names the pull request; the agent's instructions say what a review is, and it needs the `github` integration to read the diff. The answer is posted as a PR comment, and a new push continues the same thread. A push by `BOT_USERNAME` is ignored.

## Known limitations

- A reply that fails to post is logged, not retried.
- Two events in one conversation can be answered out of order; there is no per-conversation lock.
- Only the question and the final answer are persisted, not the tool calls in between.
