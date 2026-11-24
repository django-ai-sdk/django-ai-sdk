---
title: Webhooks
sidebar:
  order: 124
---

A **webhook integration** points the other way from a tool integration. A platform pushes an event in, an agent answers it, and the SDK posts the answer back in the same channel. The SDK ships `telegram`, `slack` and `github_events`. Writing another takes three methods.

An **agent** answers a webhook, not a workflow. An inbound message is a turn in a conversation, and the thread carries the history. Scheduled work has no one to reply to, so it runs a workflow instead, from a [cron trigger](workflows.md#triggers). To also start a workflow from a message, report the message with `start_for_event`.

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

The view verifies and queues the event. A `django_tasks` worker runs the agent and posts the reply, so you need a worker: `manage.py ai_sdk_worker` with `DBOSTaskBackend`, or `manage.py db_worker` with `django_tasks_db`. The SDK serves each integration at `api/integrations/<name>/webhook/`.

## How an event is handled

:::steps
1. **Verify.** `verify(request)` checks the platform's signature. A failure is a 401. The endpoint is `csrf_exempt` and unauthenticated, so the signature is the only gate. Only a code-registered `WebhookIntegration` is reachable. An `MCPServerConfig` row builds a different type and gets a 404.
2. **Parse.** `parse(request)` returns one of three things. An `InboundEvent` is an event to answer. `None` acknowledges and drops the event, such as a ping, an edit or the bot's own message. An `HttpResponse` is returned verbatim, such as Slack's URL handshake.
3. **Admit.** `ALLOW_FROM` (sender ids) and `ALLOW_WORKSPACES` (installations) must both admit the event, and the sender must be within `AI_SDK_WEBHOOK_RATE_LIMIT_PER_MINUTE`. A refused event still gets a 200, so the platform does not redeliver it. The SDK logs it with the ids to add.
4. **Queue and acknowledge.** The SDK drops an event with no text left, such as a bare mention. Otherwise the event goes to the worker, and the platform gets `ack(event)`, an empty 200 by default.
5. **Deduplicate.** The worker claims the event id with `cache.aadd` for `dedup_ttl` seconds, so it drops a redelivery or a second run of the task. In production, use a cache that every process shares. The check `ai_sdk.W001` warns when the default cache is `LocMemCache` or `DummyCache`. The worker then sends `event_received` (`django_ai_sdk.integrations.webhooks.signals`). With `django_ai_sdk.workflows` installed, that signal starts the workflow event triggers on `<name>.message`. The SDK logs a failing receiver, and the agent still answers. The claim stays when the run fails, because its tools may have acted. An event therefore runs at most once, and a worker that dies mid-run leaves the event unanswered.
6. **Run.** The worker loads the conversation's thread and runs the agent as `RUN_AS` with `AI_SDK_WEBHOOK_TIMEOUT`. It persists the turn and calls `reply(event, text)`. A failing or slow run still replies, with an error or timeout message.
:::

The run is `agent.run(history + [question], tools=True, user=<RUN_AS>, thread_id=...)`. The history is the thread's stored messages, cut to the agent's `max_history`.

While the agent runs, the event is available as `django_ai_sdk.integrations.webhooks.current_event`, a `ContextVar`. A tool can use it to answer in the channel itself, for example to upload a file to the same thread.

## Who the agent acts as

`RUN_AS` names one account. The SDK matches it on the user model's `USERNAME_FIELD`, and the account must be active. The SDK answers every admitted sender as that account, so give it only what the bot needs. An unset, unknown or ambiguous `RUN_AS` runs anonymously, which reaches no integration. The account also owns the persisted thread.

A signature proves that the platform sent the event. It does not prove who asked or which installation the event came from. A second Slack workspace that runs the same app passes verification. The allowlists cover that gap.

## Conversations

`thread_scope(event)` maps a conversation to a stable key. The SDK hashes the key into a thread id, so every process finds the same thread. By default there is one thread per conversation, such as a Telegram chat. Set `thread_per_ref = True` for one thread per `thread_ref` instead. Slack does this with one thread per Slack thread, and GitHub with one per pull request. Return `None` to answer statelessly. A new thread gets the agent's `memories` linked, as the thread endpoints do.

## Writing an integration

```python
class ZendeskIntegration(WebhookIntegration):
    name = "zendesk"
    label = "Zendesk"

    def verify(self, request) -> bool: ...          # signature check, sync
    def parse(self, request) -> InboundEvent | HttpResponse | None: ...  # sync
    async def reply(self, event, text) -> None: ...  # post back
```

Register it like any integration, with an `IntegrationAppConfig`. You can override these members:

- `ack`, for a platform that wants a response body.
- `thread_per_ref` or `thread_scope`, for the platform's threading.
- `dedup_ttl`, for its redelivery window. Use `UNSIGNED_DEDUP_TTL` when the platform signs no timestamp. In that case, take the event id from what the signature covers, or a replay picks a fresh id.
- `missing_config`, to name a missing credential. The endpoint returns 404 while a credential is missing.

`InboundEvent` crosses the queue as JSON, so it carries only what `reply` needs: `integration`, `event_id`, `text`, `conversation_id`, and optionally `thread_ref`, `external_user_id`, `workspace_id` and `reply_token`.

## Config keys

Each integration takes these keys in `AI_SDK_INTEGRATIONS[<name>]`:

| Key | Purpose |
| --- | --- |
| `AGENT` | Dotted path to the answering agent. |
| `RUN_AS` | The account runs act as. |
| `ALLOW_FROM` | Platform user ids that may ask, or `"*"`. **Required** for the built-in integrations: runs act as `RUN_AS`, so who may ask is written down. A malformed value admits nobody. |
| `ALLOW_WORKSPACES` | Installations (Slack `team_id`, GitHub `owner/repo`) that may ask, or `"*"`. |

The related settings are `AI_SDK_WEBHOOK_TIMEOUT` (default `120` seconds), `AI_SDK_WEBHOOK_MAX_TEXT` (default `4000` characters of the question kept) and `AI_SDK_WEBHOOK_RATE_LIMIT_PER_MINUTE` (events per sender per minute, default `0` for no limit). The rate limit counts in the cache, so give every process the same cache.

## Telegram

Telegram takes these keys:

- `BOT_TOKEN`, from @BotFather.
- `WEBHOOK_SECRET`, a value you choose.
- `ALLOW_FROM`, **required**. Anyone who knows a Telegram bot's name can reach it, so even `"*"` has to be written down.

Register the webhook once:

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d url=https://example.com/api/integrations/telegram/webhook/ \
  -d secret_token=$TELEGRAM_WEBHOOK_SECRET \
  -d 'allowed_updates=["message"]'
```

Telegram echoes the secret in `X-Telegram-Bot-Api-Secret-Token`, and `verify` compares it in constant time. The integration answers text messages, strips a leading `@botname`, and threads the reply under the question. Telegram signs no timestamp, so the dedup window of a week is also the replay bound.

## Slack

Slack takes these keys:

- `BOT_TOKEN` and `SIGNING_SECRET`, from the Slack app.
- `ALLOW_FROM`, **required**. Use Slack member ids, or `"*"`. Every workspace member, guests included, can mention the app.

Subscribe the app to `app_mention` and `message.im`, and point its Event Subscriptions URL at `.../api/integrations/slack/webhook/`. The integration answers the URL handshake itself.

Signatures are Slack's v0 HMAC. The integration refuses a signature older than 300 seconds and ignores bot messages and edits. Replies go into the Slack thread that asked.

## GitHub

GitHub takes these keys:

- `WEBHOOK_SECRET`.
- `TOKEN`, which must be allowed to comment on the repositories.
- `ALLOW_WORKSPACES`, **required**. Use `owner/repo` names, because one webhook can cover an organization.
- `ALLOW_FROM`, **required**. Use GitHub logins, or `"*"`. On a public repository anyone can open a pull request, and the agent reads its body.
- Optionally `ACTIONS` and `BOT_USERNAME`.

Add a repository or organization webhook for the `pull_request` event with a sha256 secret. The agent runs when a pull request is opened, reopened, pushed to or marked ready for review. `ACTIONS` changes that list, and drafts wait. The event text names the pull request. The agent's instructions define a review, and the agent needs the `github` integration to read the diff. The integration posts the answer as a PR comment, and a new push continues the same thread. It ignores a push by `BOT_USERNAME`.

GitHub signs the body, but no timestamp and not the `X-GitHub-Delivery` header. The event id is therefore a digest of the body. The integration drops a captured request for the dedup window of a week, and answers it again after that.

## Known limitations

- The SDK logs a reply that fails to post, refused or unreachable, and does not retry it.
- Two events in one conversation can be answered out of order, because there is no per-conversation lock.
- The SDK persists only the question and the final answer, not the tool calls in between.
