---
title: Integrations
type: docs
weight: 4
---

**Integrations** give agents tools from external services: GitHub, Linear, Notion, weather, or any MCP server. Each integration is its own Django app that registers itself on startup; an agent opts in by listing integration names.

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
3. **Agents opt in** with the `integrations` class attribute.
4. **Tools are loaded per request.** `Agent.get_tools()` resolves the listed integrations, filters by user permissions, and namespaces each tool (`linear_list_issues`) so unrelated integrations never collide.

---

## Built-in Integrations

| Integration | Type | Tools |
| --- | --- | --- |
| `weather` | API-backed | `get_current_weather`: no credentials needed |
| `github` | MCP-backed | Issues, PRs, repos: connect to a GitHub MCP server |
| `linear` | MCP-backed | Issues and projects: connect to a Linear MCP server |
| `notion` | MCP-backed | Wiki and documents: connect to a Notion MCP server |

`weather` needs no setup at all and works out of the box. The MCP-backed ones connect to an MCP server you configure (see below).

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
