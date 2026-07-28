# Integrations

An **integration** is a pluggable source of tools for an assistant: an MCP server, a
hand-written API client, or anything else that can produce Haystack tools. They all
implement one contract, so the assistant, the HTTP surface and the status UI never
branch on what kind of thing they're talking to.

## Enabling one

Integrations are declared in settings, by name:

```python
AI_SDK_INTEGRATIONS = {
    "notion": "django_ai_sdk.integrations.defaults.NotionService",
    "linear": "django_ai_sdk.integrations.defaults.LinearService",
    "weather": "myapp.integrations.WeatherService",
}

# Per-integration params/credentials, read on construction.
AI_SDK_NOTION = {"tools": ["notion-search"]}
AI_SDK_LINEAR = {"token": env("LINEAR_API_KEY", default="")}
```

Then give an assistant access to them by name:

```python
class SupportAssistant(Assistant):
    integrations = ["notion", "weather"]
```

To expose the HTTP surface, include both URLconfs — plain Django views, no
`django-ninja` or other API framework required:

```python
INSTALLED_APPS = [..., "django_ai_sdk.integrations.mcp"]

urlpatterns += [
    path("api/integrations/", include("django_ai_sdk.integrations.urls")),
    path("api/integrations/", include("django_ai_sdk.integrations.mcp.urls")),
]
```

Only the `mcp` toolkit app goes in `INSTALLED_APPS` — it owns the OAuth token models and
their migrations. Integrations themselves are **not** apps.

## Writing one

### A hand-written API

Subclass `APIIntegration`. Each entry in `tools` is either a ready-made Haystack tool or
a factory called per request:

```python
from django_ai_sdk.integrations.api.base import APIIntegration
from haystack.tools import tool

@tool
def get_current_weather(location: str) -> dict:
    """Get the current weather for a place name."""
    return fetch_current_weather(location)

async def check_weather_api() -> None:
    """Raise if the upstream API is unhealthy."""
    ...

class WeatherService(APIIntegration):
    name = "weather"
    tools = [get_current_weather]
    health_check = staticmethod(check_weather_api)   # staticmethod, or it gets `self`
```

A factory receives only the arguments it declares — any subset of `user`, `assistant`,
`thread_id`, or all of them via `**kwargs`:

```python
def issue_tool(user):
    return build_authed_tool(token=user.linear_token)
```

Without a `health_check`, `get_status()` reports ACTIVE unconditionally. With one, it
runs through the same cache and breaker as everything else, so a down API shows as
DEGRADED rather than a false ACTIVE.

### An MCP server

Subclass `MCPIntegrationService` with a URL and an auth style. Every attribute is
overridable per deployment via the `AI_SDK_<NAME>` settings slice, so a project needing
a different URL or tool allow-list doesn't need its own subclass:

```python
class NotionService(MCPIntegrationService):
    name = "notion"
    url = "https://mcp.notion.com/mcp"
    auth = "oauth"                  # "static" | "token" | "oauth"
    default_tools = ["notion-search"]
```

A missing required secret never crashes boot: the integration registers, reports
DISCONNECTED, and explains itself through `detail`.

### Something else entirely

Implement `Integration` directly. Only `get_tools()` and `get_status()` are
abstract; the connection lifecycle methods have sensible no-op defaults.

## Status values

| Status | Meaning |
|---|---|
| `active` | The last real attempt succeeded. |
| `degraded` | Recent failures. Retries automatically — the breaker half-opens after a cooldown and closes on success. |
| `expired` | Was connected, but the credential (e.g. an OAuth token) lapsed and can't refresh. Needs a reconnect. |
| `disconnected` | Never connected, or not configured yet (see `detail`). |

`active` always means "the last real attempt succeeded", never "we never checked" —
`get_status()` forces an attempt when the cache has nothing on file.

## Resilience

This is part of the contract, not something each integration reimplements. Tool lists go
through `ResilientCache`:

- **Warm hit** → served immediately.
- **Stale hit** (past 80% of the TTL) → the cached value is served immediately while a
  refresh runs in the background. Nobody waits for it.
- **Cold miss** → one live fetch, hard-bounded by `AI_SDK_INTEGRATION_TIMEOUT`. On
  failure it degrades to *no tools from this integration*, never an error to the caller.
- **Repeated failures** → a per-key circuit breaker opens and fetches are skipped for
  `AI_SDK_INTEGRATION_CB_COOLDOWN`, then it half-opens and probes once. Recovery is
  automatic; there is no terminal state.

```python
AI_SDK_INTEGRATION_CACHE_TTL = 900   # seconds a discovered tool list stays fresh
AI_SDK_INTEGRATION_TIMEOUT = 3       # hard bound on a cache-miss fetch
AI_SDK_INTEGRATION_CB_COOLDOWN = 60  # seconds a failing integration is skipped
```

So the worst case an integration can add to a chat turn is one `AI_SDK_INTEGRATION_TIMEOUT`,
and only on a genuinely cold key. Integrations also load concurrently, and each one's
failure is isolated: a broken integration never costs another its tools.

Run `manage.py refresh_integrations` on a schedule. It rotates OAuth tokens nearing
expiry and warms the tool cache, so a user's first message after a deploy doesn't pay a
live connect. (Per-user OAuth keys can't be pre-warmed — only shared static/token
servers benefit.)

## Permissions

Integrations use the `INTEGRATIONS` permission domain, with two operations:
`USE_INTEGRATION` (may an assistant load its tools for this user) and
`MANAGE_INTEGRATION` (may this user connect/disconnect it). The default requires an
authenticated user — so a system or anonymous context loads no integration tools at all.

Override globally or per integration:

```python
AI_SDK_PERMISSIONS = {"integrations": ["myapp.permissions.StaffOnly"]}

class WeatherService(APIIntegration):
    permissions = [AllowAll]
```

Permissions are checked *before* tools are handed to the model, not after.

## HTTP surface

| Endpoint | Purpose |
|---|---|
| `GET /api/integrations/` | Every integration the user may use, with status and capability flags. |
| `POST /api/integrations/{name}/connect` | Begin connecting; returns `{"redirect_url": ...}` for OAuth. |
| `POST /api/integrations/{name}/disconnect` | Drop the user's stored credential. |
| `POST /api/integrations/{name}/reconnect` | Force a fresh attempt now; returns the real outcome. |
| `GET /api/integrations/oauth/{name}/callback/` | OAuth return; exchanges the code and stores the token. |

Clients decide what to offer from the capability flags (`supports_connect`,
`connect_kind`, `supports_test`) rather than from `kind`, so a new integration kind needs
no client change.

For OAuth: the client calls `POST /{name}/connect` and navigates the browser to the
`redirect_url` it gets back — there is no dedicated "start" URL. Only the callback has
to live at a fixed URL, since the identity provider is the one redirecting the browser
there; building the authorization URL is plain business logic on
`Integration.connect()`, reachable through the same generic endpoint every
integration kind uses.

## Not included (yet)

- **Admin-managed MCP servers.** Servers are declared in code today. The registry lookup
  is already `async` and MCP config is already built by a pure function from parameters,
  so a database-backed second source can be added without touching callers.
- **Per-assistant tool subsets.** `Assistant.integrations` is a list of names; all of an
  integration's tools are exposed. It can widen to `{name: [tool, ...]}` later without
  invalidating rows written in the list form.
- **Per-user credential entry.** Only OAuth has a per-user connect flow; token auth uses
  one shared deployment secret.
