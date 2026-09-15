---
title: Integrations
type: docs
weight: 4
---

An **integration** connects an agent to an external service, and it points one of two ways. Most reach outward: GitHub, Linear, Notion, weather or any MCP server become tools the agent calls, and the agent opts in by listing integration names. A **webhook integration** points inward: Slack, Discord and Telegram push events to the SDK, and an agent answers in the channel. Either way it is its own Django app that registers itself on startup.

The outbound direction, which is most of them:

```
Agent.integrations = ["linear", "weather"]
        │
        ▼
get_integrations(["linear", "weather"])   ← integrations registry
        │
        ▼
┌──────────────────────┬──────────────────────────┐
│  linear (MCP-backed) │  weather (API-backed)    │
│  server tools        │  code-native haystack    │
│  discovered via MCP  │  Tool functions          │
└──────────────────────┴──────────────────────────┘
```

## How It Works

1. **Each integration is a Django app.** Add it to `INSTALLED_APPS`. On `ready()`, its `IntegrationAppConfig` constructs the `Integration` subclass and registers it in the process registry.
2. **`AI_SDK_INTEGRATIONS` configures it** (like `DATABASES` or `CACHES`): credentials and options, keyed by integration name. A missing credential degrades the integration ("needs setup") instead of crashing boot.
3. **Agents opt in** with the `integrations` class attribute — for the outbound direction. A webhook integration is wired the other way round: `AI_SDK_INTEGRATIONS[name]["AGENT"]` names the agent that answers, and the integration is never listed on it.
4. **Tools are loaded per request.** `Agent.get_tools()` resolves the listed integrations, filters by user permissions, and namespaces each tool (`linear_list_issues`) so unrelated integrations never collide.

---

## Built-in Integrations

| Integration | Type | Direction | What it does |
| --- | --- | --- | --- |
| `weather` | API-backed | outbound | `get_current_weather`: no credentials needed |
| `github` | MCP-backed | outbound | Issues, PRs, repos: connect to a GitHub MCP server |
| `linear` | MCP-backed | outbound | Issues and projects: connect to a Linear MCP server |
| `notion` | MCP-backed | outbound | Wiki and documents: connect to a Notion MCP server |
| `slack` | Webhook-backed | inbound | An agent answers mentions and DMs in Slack |
| `discord` | Webhook-backed | inbound | An agent answers a slash command in Discord |
| `telegram` | Webhook-backed | inbound | An agent answers messages sent to a Telegram bot |

`weather` needs no setup at all and works out of the box. The MCP-backed ones connect to an MCP server you configure (see below). The inbound three need an app on the platform and a worker; see [Receiving Webhooks](#receiving-webhooks).

{{< callout type="info" >}}
`weather` requires no credentials: just add it to `INSTALLED_APPS` and an agent's `integrations` list and it works.
{{< /callout >}}

---

## Enabling an Integration

Add the app and configure it:

```python
# settings.py
INSTALLED_APPS = [
    ...,
    "django_ai_sdk.integrations.weather",
    "django_ai_sdk.integrations.linear",
]

AI_SDK_INTEGRATIONS = {
    "linear": {"TOKEN": "lin_api_..."},
}
```

Then opt the agent in:

```python
class MyAgent(Agent):
    integrations = ["linear", "weather"]
```

Every tool the integration exposes reaches this agent, namespaced as `linear_...` / `weather_...`. Restrict which tools an MCP integration exposes with its `default_tools` allow-list; the API integrations declare their tools in code.

This is the outbound pattern only. Naming a webhook integration here resolves and contributes nothing, because it has no tools to give; `manage.py check` reports it as `ai_sdk.webhooks.W002` and names the setting to use instead.

---

## MCP Servers

MCP (Model Context Protocol) servers plug into agents through three configuration kinds, declared in `AI_SDK_INTEGRATIONS` under the server name:

{{< tabs >}}

{{< tab name="Static server" >}}
```python
AI_SDK_INTEGRATIONS = {
    "my_server": {
        "TYPE": "static",
        "URL": "http://localhost:8001/mcp",
        "TOOLS": ["list_issues", "get_issue"],   # optional allow-list
        "ENABLED": True,
    },
}
```
{{< /tab >}}

{{< tab name="Token-authenticated server" >}}
```python
AI_SDK_INTEGRATIONS = {
    "my_server": {
        "TYPE": "token",
        "URL": "https://mcp.example.com",
        "TOKEN": "secret-token",
    },
}
```
{{< /tab >}}

{{< tab name="OAuth server (RFC 9728)" >}}
MCP servers that advertise OAuth discovery are connected interactively. The SDK discovers the authorization and token endpoints, manages the client registration and PKCE flow, and stores the resulting token per user.

```python
AI_SDK_INTEGRATIONS = {
    "my_server": {
        "TYPE": "oauth",
        "URL": "https://mcp.example.com",
        "OAUTH_SUCCESS_URL": "/settings/integrations",
    },
}
```
{{< /tab >}}

{{< /tabs >}}

Related settings:

```python
AI_SDK_MCP_DISCOVERY_TIMEOUT = 10      # seconds
AI_SDK_MCP_DISCOVERY_CACHE_TTL = 3600  # seconds
AI_SDK_MCP_OAUTH_SUCCESS_URL = "/settings/integrations"
```

The full list (refresh thresholds, issuer allow-lists, client name, cache TTLs) is in the [Settings Reference](/manual/settings/).

### Declaring MCP servers in the database

MCP servers can also be created at runtime (from an admin UI or via the models), giving you DB-configured dynamic MCP integrations, including OAuth-connected servers whose tokens are stored per user.

### Refreshing credentials

OAuth tokens are refreshed proactively when they expire within `AI_SDK_MCP_REFRESH_THRESHOLD_MINUTES` (default 10), and the SDK ships a management command for scheduled runs:

```bash
python manage.py refresh_integrations          # all integrations
python manage.py refresh_integrations --integration linear
```

For every registered integration it calls `refresh()` (a no-op without credentials; an OAuth token refresh for the rest), then warms the cached tool list via `get_status()`, so the first message after a deploy doesn't pay for a live MCP connect. Schedule it with cron or celery beat; a single failed integration is reported without stopping the others. The command exits non-zero if any refresh fails.

---

## Resilience

Integration data flows through `ResilientCache`: a stale-while-revalidate cache with a per-key circuit breaker.

- A fresh cached value is served instantly; stale values are refreshed in the background.
- A slow or dead integration is bounded by `AI_SDK_INTEGRATION_TIMEOUT` (default 3s), then skipped for `AI_SDK_INTEGRATION_CB_COOLDOWN` (default 60s) while the breaker is open.
- Recovery is automatic: the breaker half-opens, probes once, and closes on success.

```python
# settings.py
AI_SDK_INTEGRATION_CACHE_TTL = 900       # seconds a tool list stays fresh
AI_SDK_INTEGRATION_TIMEOUT = 3           # seconds; hard bound on a cache-miss fetch
AI_SDK_INTEGRATION_CB_COOLDOWN = 60      # seconds a failing integration is skipped
```

Each integration's `get_status()` reports one of `active`, `degraded`, `expired`, or `disconnected`, so a UI can show per-integration health without guessing.

{{< callout type="info" >}}
Slow or dead integrations are bounded by `AI_SDK_INTEGRATION_TIMEOUT` (default 3s) and skipped for `AI_SDK_INTEGRATION_CB_COOLDOWN` (default 60s) while the circuit breaker is open: recovery is automatic.
{{< /callout >}}

---

## Permissions

Every integration declares `permissions`, gating `Operation.USE_INTEGRATION` (and manage operations). Before tools reach the model, `Agent._get_integration_tools()` checks the user against each integration's permissions and skips any they aren't allowed to use: an unauthorized integration's tools never reach the model.

---

## Receiving Webhooks

Most integrations reach outward: the agent calls a tool. A **webhook integration** points the other way. A platform pushes an event in, an agent answers it, and the answer is posted back. That is how `slack`, `discord` and `telegram` put an agent in a channel.

Both directions are the same object, so a Slack app has one credential, one entry in `AI_SDK_INTEGRATIONS`, and one row in an integrations list.

### Putting an agent in Slack

Create a Slack app with the `app_mentions:read`, `im:history` and `chat:write` scopes, then:

```python
# settings.py
INSTALLED_APPS += ["django_ai_sdk.integrations.slack"]

AI_SDK_INTEGRATIONS = {
    "slack": {
        "BOT_TOKEN": env("SLACK_BOT_TOKEN"),          # Install page, starts xoxb-
        "SIGNING_SECRET": env("SLACK_SIGNING_SECRET"), # Basic Information page
        "AGENT": "myapp.agents.SupportAgent",
    },
}
```

```python
# urls.py
path("api/integrations/", include("django_ai_sdk.integrations.webhooks.urls")),
```

Point the Slack app's Event Subscriptions URL at `https://example.com/api/integrations/slack/webhook/` and subscribe to `app_mention` and `message.im`. Slack verifies the URL by asking for a challenge back, which the integration answers inline.

Everyone in the workspace can then reach the agent, and so can external members of any Slack Connect channel it is invited to — an installation is not always as closed as it sounds.

Then run a worker. `@YourBot what's the weather in Amsterdam?` is answered in-thread.

{{< callout type="warning" >}}
The reply happens in a background task, so a worker (`manage.py db_worker`, or whatever runs your `django_tasks` backend) must be running. Without one Slack is acknowledged and nothing ever answers.
{{< /callout >}}

### Putting an agent in Discord

Discord pushes slash commands over HTTP and delivers plain channel messages only over a Gateway WebSocket, so `discord` answers `/ask …` rather than a mention. Its interactions are signed with Ed25519, which needs an extra:

```bash
pip install "django-ai-sdk[discord]"
```

Create a Discord application, then:

```python
# settings.py
INSTALLED_APPS += ["django_ai_sdk.integrations.discord"]

AI_SDK_INTEGRATIONS = {
    "discord": {
        "PUBLIC_KEY": env("DISCORD_PUBLIC_KEY"),          # General Information page
        "APPLICATION_ID": env("DISCORD_APPLICATION_ID"),  # General Information page
        "BOT_TOKEN": env("DISCORD_BOT_TOKEN"),            # Bot page
        "AGENT": "myapp.agents.SupportAgent",
    },
}
```

`PUBLIC_KEY` verifies a delivery and `APPLICATION_ID` addresses the reply, so both are needed to serve. `BOT_TOKEN` is read only by the registration command, because Discord has no way to create a slash command from its dashboard:

```bash
manage.py register_discord_command      # --name ask --description "..." to change it
```

Then point the application's Interactions Endpoint URL at `https://example.com/api/integrations/discord/webhook/`. Discord validates it by sending a signed PING and a deliberately mis-signed request, expecting a PONG and a 401 back; the integration answers both.

A new Discord application has **Public Bot** switched on, which lets anyone with Manage Server add it to their own guild — where its commands verify against your public key and run your agent. Turn it off under Bot in the developer portal unless you mean to distribute it, and pin the guilds you expect with `ALLOW_WORKSPACES`.

`/ask what's the weather in Amsterdam?` shows a thinking state while the agent runs, and is edited into the answer. A newly registered global command can take up to an hour to appear in every guild.

### Putting an agent in Telegram

Telegram is the simplest of the three: it signs nothing and echoes back a secret you choose, so there is no extra to install and no acknowledgement deadline to beat. Create a bot with [@BotFather](https://t.me/botfather), then:

{{< callout type="warning" >}}
A Telegram bot is reachable by anyone who knows its name, so `ALLOW_FROM` is required here. Until it names an audience — or is set to `"*"` — the integration reports `disconnected` and its webhook 404s. See [Deciding who may ask](#deciding-who-may-ask).
{{< /callout >}}

```python
# settings.py
INSTALLED_APPS += ["django_ai_sdk.integrations.telegram"]

AI_SDK_INTEGRATIONS = {
    "telegram": {
        "BOT_TOKEN": env("TELEGRAM_BOT_TOKEN"),            # @BotFather
        "WEBHOOK_SECRET": env("TELEGRAM_WEBHOOK_SECRET"),  # any value you choose
        "ALLOW_FROM": ["123456789"],                       # required; "*" for anyone
        "AGENT": "myapp.agents.SupportAgent",
    },
}
```

Telegram is told where to deliver, rather than being given a URL to poke in a dashboard:

```bash
manage.py set_telegram_webhook --url https://example.com/api/integrations/telegram/webhook/
```

The URL must be HTTPS on port 443, 80, 88 or 8443, which is Telegram's restriction rather than the SDK's. The command registers `WEBHOOK_SECRET` at the same time, and Telegram sends it back in the `X-Telegram-Bot-Api-Secret-Token` header on every delivery. That header is the whole of `verify`.

Message the bot and it answers. In a group, Telegram's default privacy mode means the bot only receives messages that address it as `@yourbot`, and the leading mention is stripped before the agent sees the question.

### Writing your own

A webhook integration is an ordinary integration app with three extra methods. `verify` and `parse` are synchronous because they run on the request path, inside the platform's acknowledgement window, which for Slack and Discord is three seconds. `reply` is async and runs in the worker.

```python
# myapp/integrations/zulip/integration.py
from django_ai_sdk.integrations import InboundEvent, WebhookIntegration


class ZulipIntegration(WebhookIntegration):
    name = "zulip"
    label = "Zulip"

    def missing_config(self):
        if not self.secret("WEBHOOK_SECRET"):
            return "Missing WEBHOOK_SECRET. Set AI_SDK_INTEGRATIONS['zulip']['WEBHOOK_SECRET']."
        return None

    def verify(self, request):
        # Whatever the platform gives you to prove it sent this: a signature, a
        # shared secret, a bearer token.
        ...

    def parse(self, request):
        # An InboundEvent to answer, None to ignore, an HttpResponse to answer inline.
        ...

    async def reply(self, event, text):
        # Post text back, however the platform wants it.
        ...
```

The three shipped integrations are the worked examples, one per shape of proof:

| Integration | `verify` | `ack` | `reply` |
| --- | --- | --- | --- |
| `slack` | HMAC-SHA256 signature | default empty 200 | posts a new threaded message |
| `discord` | Ed25519 signature | deferred response | edits the deferred message |
| `telegram` | shared secret in a header | default empty 200 | sends a message replying to the question |

The app config is the same four lines as any other integration, and the endpoint is `/<name>/webhook/` with no extra URL wiring.

`parse` has three outcomes, and between them they cover everything a platform sends:

| Return | Meaning |
| --- | --- |
| `InboundEvent` | Answer this. It is queued and the agent runs. |
| `None` | Acknowledge and ignore: the bot's own messages, edits, joins, events you do not handle. |
| `HttpResponse` | Answer inline, with a body of your choosing. This is what a URL-verification handshake needs. |

`ack` is optional too. It returns the response the platform gets once the event is queued, an empty 200 by default.

`missing_config()` is optional. Return a string naming the setting to change and the integration reports `disconnected`, its webhook returns 404, and the rest of the site is unaffected.

### What happens to an event

1. The view looks the name up in the registry and refuses anything that is not a `WebhookIntegration`. A database-backed MCP server can never acquire an unauthenticated endpoint.
2. `verify` rejects an unsigned or replayed request with 401.
3. `parse` normalises it, or ignores it, or answers it.
4. `ALLOW_FROM` and `ALLOW_WORKSPACES` decide whether this sender, in this installation, is answered, and the question is truncated to `AI_SDK_WEBHOOK_MAX_TEXT`.
5. A repeated `event_id` is dropped: platforms redeliver, and task dispatch is at-least-once.
6. The event is queued and the view returns `ack(event)`, an empty 200 unless the integration overrides it, well inside the acknowledgement window.
7. The worker resolves `RUN_AS` into the principal the run acts as, and loads the thread this conversation maps to, so the agent sees what was said before.
8. The worker runs the agent with tools, bounded by `AI_SDK_WEBHOOK_TIMEOUT` (default 120s), stores the question and the answer on that thread, and calls `reply`.

### Conversation memory

A webhook conversation maps to one persisted SDK thread, so the agent reads back what was said before instead of answering each message cold. The mapping is a uuid5 of a scope string, so the same channel resolves to the same thread across processes and restarts with no lookup table:

| Platform | One thread per |
| --- | --- |
| `slack` | Slack thread (channel + `thread_ts`); a top-level message with no `thread_ref` is answered statelessly |
| `discord` | channel or DM |
| `telegram` | chat |

Override `thread_scope(event)` to key it differently, or return `None` to stay stateless. An agent whose `storage_adapter` is unset is always stateless. History is trimmed to the agent's own `max_history`, and the thread is owned by the `RUN_AS` principal, so it is reachable under the ordinary thread permissions rather than orphaned.

### Deciding who may ask

A signature proves the platform sent the event. It says nothing about who typed it, and each question runs the agent and costs a model call. What "anyone" means depends entirely on the platform, and the three differ more than they look:

| Platform | Who can reach the bot | What stands in the way |
| --- | --- | --- |
| Slack | Workspace members, plus anyone in an external org sharing a Slack Connect channel | A workspace admin has to install the app |
| Discord | Members of every guild the application is in | Someone with Manage Server adds it — but Discord's **Public Bot** toggle is on by default, so that someone need not be you |
| Telegram | Any Telegram user who finds the bot by name | Nothing at all |

So the SDK does not give all three the same default. Slack and Discord answer everyone unless narrowed. Telegram refuses to serve until you say who may ask.

`ALLOW_FROM` narrows the audience to a list of platform user ids:

```python
# settings.py
AI_SDK_INTEGRATIONS = {
    "slack": {
        ...,
        "ALLOW_FROM": ["U024BE7LH", "U02SFDG3B"],   # Slack member ids
    },
}
```

An unlisted sender is acknowledged with a 200 and dropped, so the platform does not redeliver, and the id it was refused for is logged so it can be pasted into the list. Leaving `ALLOW_FROM` out answers everyone, which is defensible only where reaching the endpoint is already restricted — a private Slack workspace you control, say.

The ids are the platform's own: a Slack `U…` member id, a Discord user snowflake, a numeric Telegram user id. There is no account linking, so they are not Django users; see the limitations below.

`ALLOW_WORKSPACES` narrows the other axis, the installation:

```python
AI_SDK_INTEGRATIONS = {
    "slack": {..., "ALLOW_WORKSPACES": ["T024BE7LH"]},      # Slack team_id
    "discord": {..., "ALLOW_WORKSPACES": ["1109334398147125285"]},  # guild id
}
```

This matters more than it looks. A signing secret belongs to the app, not to the workspace that installed it, so a second installation — a distributed Slack app, or a Discord application someone else added to their own guild — delivers events that verify cleanly and run the agent on your budget. Only the reply fails. Pinning the installation closes that.

Both keys accept `"*"` to mean "anyone, deliberately", which is how an integration that requires an explicit choice is told to stay open. A malformed value — a bare string where a list belongs — refuses every event and logs the reason, because failing open on a typo would hand the agent to the whole workspace.

Two things this does not do. It is not a rate limit: an admitted sender can ask as fast as they can type. And it is not authentication — the ids belong to the platform, not to your Django users, so an agent behind a webhook is exactly as trustworthy as the least careful person on the list.

### Running as a service account

By default a webhook run is anonymous. `IntegrationDefaultPermission` requires an authenticated user, so every integration is filtered out of `get_tools()` and the agent reaches only its plain `tools`. That is the safe default, and for many bots it is the right one.

`RUN_AS` opts out of it by naming an account the run acts as:

```python
AI_SDK_INTEGRATIONS = {
    "slack": {..., "RUN_AS": "slack-bot@example.com"},
}
```

The value is matched against your user model's `USERNAME_FIELD`, so it is whatever you log in with — an email address, a username, an employee number. Create the account like any other:

```python
get_user_model().objects.create_user(email="slack-bot@example.com")
```

With a principal resolved, the agent's `integrations` load normally, and per-user credentials resolve against that account: connect the bot's Notion or Linear once through the OAuth flow and every question in the channel uses it.

{{< callout type="warning" >}}
The bot answers **everyone `ALLOW_FROM` admits** as this one account. Whatever it can reach, they can reach through it. Use a dedicated account that holds only what the bot needs, never a real person's login, and never an administrator.
{{< /callout >}}

An account that does not exist, or one that has been deactivated, logs an error naming the setting and the identifier, and the run continues anonymously — the channel still gets an answer, with no tools behind it.

### Limitations

- **A webhook run is anonymous unless `RUN_AS` is set**, and an anonymous run reaches no integration at all. See [Running as a service account](#running-as-a-service-account). `RUN_AS` is a service account, not identity: there is no account linking, so the agent never knows which person asked.
- **The turn is stored as two writes, not a transaction.** A storage adapter is not necessarily a database, so there is no `atomic()` spanning the question and the answer. A failure between them logs loudly and leaves a question with no answer in history.
- **No streaming.** The person sees nothing until the answer is complete. A run that fails or times out replies with a short apology rather than silence.
- **A worker is required.** Nothing answers without one.
- **Discord answers a command, not a mention.** Channel messages arrive only over a Gateway WebSocket, which is a persistent connection rather than a webhook. A Discord reply is capped at 2000 characters and must land inside the interaction's fifteen-minute window.
- **Telegram answers what its privacy mode lets through.** In a group the bot receives only messages addressing it by name unless privacy mode is turned off in @BotFather, and a reply is capped at 4096 characters.
- **A Discord interaction token travels through the queue** in `reply_token`, so it sits wherever `django_tasks` keeps payloads until that row is cleaned up. It authenticates the reply and expires with the interaction, and it is the only credential an `InboundEvent` carries.
- **There is no rate limit.** `ALLOW_FROM` and `ALLOW_WORKSPACES` decide *who*, never *how often*: an admitted sender can ask as fast as they can type, and each question runs the agent. Put the endpoint behind your own throttling middleware if that matters.
- **De-duplication is cache-backed**, not a database row, so a cache flush inside the ten-minute window can let one duplicate reply through, and it needs a cache every process shares. A local-memory default answers a redelivery once per process; `ai_sdk.webhooks.W001` reports that at deploy time.

---

## Writing Your Own Integration

An API-backed integration is the simplest kind. Create an app with two pieces:

### 1. The integration class (`integration.py`)

```python
# myapp/integrations/zendesk/integration.py
from django_ai_sdk.integrations.api.base import APIIntegration

class ZendeskIntegration(APIIntegration):
    name = "zendesk"
    label = "Zendesk"
    tools = [search_tickets]               # haystack Tool objects or @tool functions
    health_check = staticmethod(check_zendesk_api)

    async def get_status(self, user=None, agent=None):
        ...
```

- `tools`: entries are either a `haystack.Tool` instance or a function decorated with `@tool` (schema inferred from its signature).
- `health_check`: optional async, no-arg callable that raises on failure; drives `get_status()`.
- Credentials come from `self.secret("TOKEN")`, reading `AI_SDK_INTEGRATIONS["zendesk"]["TOKEN"]`. Check the result and set `self.detail` rather than raising, so a missing credential degrades gracefully.

### 2. The app config (`apps.py`)

```python
# myapp/integrations/zendesk/apps.py
from django_ai_sdk.integrations.apps import IntegrationAppConfig

class ZendeskConfig(IntegrationAppConfig):
    default = True
    name = "myapp.integrations.zendesk"
    integration = "myapp.integrations.zendesk.integration.ZendeskIntegration"
```

Add `"myapp.integrations.zendesk"` to `INSTALLED_APPS`, configure it under `AI_SDK_INTEGRATIONS["zendesk"]`, and list `"zendesk"` in any agent's `integrations`. For an MCP-backed integration instead, subclass `MCPIntegration` and point `config` at an `MCPIntegrationConfig`.

See `django_ai_sdk/integrations/weather/` for a complete minimal example.
