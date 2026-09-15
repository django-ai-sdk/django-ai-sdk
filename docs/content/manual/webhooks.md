---
title: Webhooks
type: docs
weight: 124
---

Reference for `WebhookIntegration`, the contract an integration implements to receive events a platform pushes in. For the walkthrough, see the [Integrations guide](/integrations/#receiving-webhooks).

## `WebhookIntegration`

`django_ai_sdk.integrations.webhooks.WebhookIntegration`, a subclass of `Integration`. Everything on `Integration` (`name`, `label`, `hint`, `permissions`, `secret()`, `detail`) applies unchanged.

### Abstract members

| Member | Sync | Purpose |
| --- | --- | --- |
| `verify(request) -> bool` | yes | Whether the request carries a valid, current platform signature. A `False` is a 401. |
| `parse(request) -> InboundEvent \| HttpResponse \| None` | yes | Normalise a verified request. See the return table below. |
| `reply(event, text) -> None` | no | Post `text` back into the conversation `event` came from. |

`verify` and `parse` are synchronous deliberately: they run on the request path, inside the platform's acknowledgement window. Every other lifecycle method on `Integration` is async.

### `parse` return values

| Return | Result |
| --- | --- |
| `InboundEvent` | Queued for the worker; the view returns `ack(event)`. |
| `None` | The view returns 200 and nothing is queued. |
| `HttpResponse` | Returned to the platform verbatim, body included. |

### Concrete members

| Member | Default | Purpose |
| --- | --- | --- |
| `allows(event)` | the allow-lists | Whether this sender, in this workspace, reaches the agent. An unset list admits everyone, `"*"` admits everyone deliberately, and a malformed one admits nobody. |
| `has_audience()` | `ALLOW_FROM` | Whether who may ask has been decided. Used by an integration whose platform has no admission control of its own. |
| `aget_principal()` | `RUN_AS` | The user a run acts as, matched on the user model's `USERNAME_FIELD` and required to be active. Unset, unknown or deactivated all answer `None`, and the run is anonymous. |
| `thread_scope(event)` | one thread per conversation | Stable key for the persisted thread this conversation maps to, or `None` to answer statelessly. Hashed into a uuid5 thread id, so the same key resolves to the same thread in every process. `slack` overrides it to key by Slack thread. |
| `ack(event)` | empty 200 | The response sent once the event is queued. Override for a platform that wants a body while the agent thinks. |
| `agent` | `""` | Dotted path to the `Agent` subclass that answers. `AI_SDK_INTEGRATIONS[name]["AGENT"]` overrides it. |
| `missing_config()` | `None` | Reason this integration's own credentials are incomplete. Override to name the setting to change. |
| `agent_path()` | config or `agent` | The configured agent's dotted path. |
| `resolve_agent()` | registry lookup | The agent instance, or `None` when the path is unset, unimportable or unregistered. |
| `detail` | property | `missing_config()`, else a missing-agent message, else `None`. Read at call time, so `override_settings` works. |
| `get_tools()` | `[]` | A receive-only integration offers the agent nothing. |
| `get_status()` | derived | `disconnected` when `detail` is set, otherwise `active`. |
| `kind` | `"webhook"` | Category label. |

## `InboundEvent`

A pydantic model, so it survives the queue as a dict. An `HttpRequest` cannot, which is why `parse` runs in the view rather than the worker.

A queue backend persists what it is handed — `django_tasks_db` writes it to a `JSONField` — so the event carries what `reply()` needs and nothing else. The platform's original payload stays in the request.

| Field | Type | Meaning |
| --- | --- | --- |
| `integration` | `str` | Registry name, used by the worker to find the integration again. |
| `event_id` | `str` | Platform's id for this delivery. Drives de-duplication. |
| `text` | `str` | What the person said, with platform mention markup stripped. |
| `conversation_id` | `str` | Where to reply, e.g. a Slack channel id. |
| `thread_ref` | `str` | Reply target within the conversation, e.g. a Slack `thread_ts`. Empty replies at the top level. |
| `external_user_id` | `str` | The platform's id for the speaker. Not linked to a Django user. |
| `workspace_id` | `str` | The platform's tenant, e.g. a Slack `team_id`. |
| `reply_token` | `str` | Credential the platform issued for answering this event, where it uses one. Empty otherwise. |

## URL

```python
path("api/integrations/", include("django_ai_sdk.integrations.webhooks.urls")),
```

Serves `<name>/webhook/`, reverse name `integrations_webhooks:webhook`. The second URL module the SDK ships, after the MCP OAuth callback. Both exist because a platform stores the address, so it cannot be built per request the way a host's own endpoints are.

The view is `csrf_exempt` and unauthenticated: the signature `verify()` checks is the only thing standing in front of it. A failed `verify()` returns 401.

`RUN_AS` is config-driven privilege: every admitted sender is answered as that one account, so it holds only what the bot needs. An identifier matching no active user degrades to anonymous rather than refusing, because a channel with no answer reports nothing a deployer can act on. The principal also owns the thread `thread_scope` maps the conversation to, so the transcript is reachable under the ordinary thread permissions rather than orphaned.

A signature identifies the platform, not the person, and not the installation. A signing secret belongs to the *app*: a second Slack workspace or a second Discord guild running the same app produces deliveries that verify perfectly. `ALLOW_FROM` narrows the endpoint to known senders and `ALLOW_WORKSPACES` to known installations; both must admit an event for it to be answered.

An unlisted sender gets a 200, so the platform treats the event as delivered and does not redeliver it, and the sender and workspace it was refused for are logged so they can be pasted into the list.

## Security

The view resolves the name through the registry and refuses anything that is not a `WebhookIntegration` instance. An `MCPServerConfig` row builds a `DynamicMCPIntegration`, which cannot satisfy that check, so an admin-authored database row can never acquire an unauthenticated endpoint. The guard is the type, not a flag a later change can set by accident.

## Checks

| Id | Level | Raised when |
| --- | --- | --- |
| `ai_sdk.webhooks.W001` | Warning | An app declares a `WebhookIntegration`, `DEBUG` is off, and `CACHES['default']` is a local-memory or dummy backend. |
| `ai_sdk.webhooks.W002` | Warning | An agent names a receive-only integration in its `integrations` list. |

W002 catches the wiring written backwards. A webhook integration is pointed at an agent
through `AI_SDK_INTEGRATIONS[<name>]["AGENT"]`; listing its name on the agent instead
resolves, passes permissions, and contributes nothing, with no error. An integration
that receives events *and* overrides `get_tools` is not reported, because it has
something to offer. Code-defined agents only: a database-declared agent is never in the
agents registry, and a check may not read the database.

De-duplication is a `cache.aadd` on the default cache. A backend private to one process cannot refuse an event another process already claimed, so a platform's redelivery is answered once per process. The check is skipped under `DEBUG`, where `runserver` is a single process.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `AI_SDK_WEBHOOK_TIMEOUT` | `120` | Seconds an agent run may take before the reply becomes a timeout message. |
| `AI_SDK_WEBHOOK_MAX_TEXT` | `4000` | Characters of the inbound question kept. A platform's own limit is far higher. |

## Config keys

Read from `AI_SDK_INTEGRATIONS[<name>]`, upper-cased, through `secret()`.

| Key | Used by | Purpose |
| --- | --- | --- |
| `AGENT` | all | Dotted path to the answering agent. Overrides the class attribute. |
| `RUN_AS` | all | The account a run acts as, by `USERNAME_FIELD`. Unset answers anonymously, which reaches no integration. |
| `ALLOW_FROM` | all | List of platform user ids allowed to ask, or `"*"` for anyone. Unset answers everyone, except where the integration requires a choice. |
| `ALLOW_WORKSPACES` | `slack`, `discord` | List of `team_id` / `guild_id` values allowed to reach the agent, or `"*"`. Unset answers every installation. Telegram sets no workspace, so configuring it there refuses everything. |
| `BOT_TOKEN` | `slack` | Bot user OAuth token, from the Slack app's Install page. |
| `SIGNING_SECRET` | `slack` | Signing secret, from the Slack app's Basic Information page. |
| `PUBLIC_KEY` | `discord` | Ed25519 public key, from the Discord application's General Information page. Required to serve. |
| `APPLICATION_ID` | `discord` | Application id, from the same page. Addresses the follow-up webhook a reply edits, so it is required to serve. |
| `BOT_TOKEN` | `discord` | Bot token, from the Discord application's Bot page. Read only by `register_discord_command`. |
| `BOT_TOKEN` | `telegram` | Bot token, from @BotFather. |
| `WEBHOOK_SECRET` | `telegram` | A value you choose, registered with Telegram by `set_telegram_webhook` and echoed back on every delivery. |
| `ALLOW_FROM` | `telegram` | **Required.** Telegram has no admission control of its own, so the integration reports `disconnected` until this names an audience or is set to `"*"`. |

## The `slack` integration

`django_ai_sdk.integrations.slack`, app label `django_ai_sdk_slack`.

Answers `app_mention` events and `message` events with `channel_type == "im"`. An event carrying a `bot_id` or a `subtype` is ignored: the first is the agent's own reply, the second is an edit, join or file share rather than someone talking.

Signatures are Slack's v0 scheme, an HMAC-SHA256 over `v0:{timestamp}:{body}` compared with `hmac.compare_digest`. A request whose timestamp is more than 300 seconds old is refused even when the signature is valid.

Replies post to `chat.postMessage` with `thread_ts` set from `thread_ref`, so an answer lands in the thread that asked. Slack answers `200` with `ok: false` for a revoked token or a missing scope, so the response body is checked and a refusal is logged at error level.

## The `discord` integration

`django_ai_sdk.integrations.discord`, app label `django_ai_sdk_discord`. Needs the `discord` extra, which pulls in `cryptography` for Ed25519.

Answers interaction type 2, an application command, taking the first string option as the question. Type 1 is answered inline with a PONG. Every other type, including buttons, autocomplete and modal submissions, is ignored.

Signatures are Ed25519 over `{timestamp}{body}`, checked against `PUBLIC_KEY`. A request whose timestamp is more than 300 seconds old is refused even when the signature is valid, which costs nothing because an interaction token expires after fifteen minutes anyway.

`ack()` returns `{"type": 5}`, a deferred response, because Discord marks a command failed unless something answers within three seconds. `reply()` then edits that deferred message with `PATCH /webhooks/{application_id}/{token}/messages/@original`, authenticated by the interaction token in `reply_token` and addressed with `APPLICATION_ID`. No bot token is involved, and the answer is truncated to Discord's 2000-character limit.

### `register_discord_command`

```bash
manage.py register_discord_command [--name ask] [--description "Ask the agent a question."]
```

Registers one global command with a single required string option named `question`, using `APPLICATION_ID` and `BOT_TOKEN`. Discord's dashboard cannot create commands, so this is the only way to get one. Re-running it with the same name updates the existing command.

## The `telegram` integration

`django_ai_sdk.integrations.telegram`, app label `django_ai_sdk_telegram`. No extra: the whole of `verify` is a `constant_time_compare`.

Answers an update carrying a `message` with text. An `edited_message`, a `channel_post` and a callback query arrive under other keys and are ignored, as is a message whose sender is a bot and one with no text. A leading `@botname` is stripped so the agent reads the question rather than the address.

Alone among the three, this integration will not serve until `ALLOW_FROM` is set. A Slack app is installed by a workspace admin and a Discord application is added to a guild by someone who administers it; a Telegram bot is reachable by anyone who knows its name, so leaving it open has to be a decision rather than an oversight. `"*"` is how that decision is written down.

Telegram signs nothing. `set_telegram_webhook` registers `WEBHOOK_SECRET` alongside the URL, Telegram returns it in the `X-Telegram-Bot-Api-Secret-Token` header on every delivery, and `verify` compares the two in constant time. An unset secret refuses every request rather than matching a header that is also absent.

`reply()` posts to `sendMessage` with `reply_parameters` set from `thread_ref`, so the answer hangs off the message that asked and lands in the right topic in a forum group. The answer is truncated to Telegram's 4096-character limit. Telegram reports a blocked bot or a chat it was removed from as `ok: false` with a description, which is logged at error level.

### `set_telegram_webhook`

```bash
manage.py set_telegram_webhook --url https://example.com/api/integrations/telegram/webhook/ [--drop-pending]
```

Registers the URL and `WEBHOOK_SECRET` with Telegram, subscribing to `message` updates only. The URL must be HTTPS on port 443, 80, 88 or 8443, which is Telegram's restriction. `--drop-pending` discards updates that queued while no webhook was registered. Re-running it replaces the registration; a bot has one webhook.
