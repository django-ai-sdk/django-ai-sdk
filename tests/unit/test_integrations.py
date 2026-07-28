"""Tests for the integrations layer.

Three things are under test, in order of how much they'd hurt if wrong:

1. ``ResilientCache`` — the latency and failure guarantees the whole layer rests on.
   A dead integration must cost a bounded wait once and ~nothing after, and must be
   visible as DEGRADED rather than silently contributing no tools.
2. The registry and the ``IntegrationService`` contract — a broken integration is
   isolated, integrations load concurrently, and a third-party backend plugs in with
   no special-casing.
3. The MCP OAuth flow — start, callback, code exchange, refresh, and the concurrent
   refresh race. This is shared infrastructure several consumers rely on, so it is
   covered here rather than only in whichever project happens to use it.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.base import (
    IntegrationService,
    IntegrationStatus,
    ResilientCache,
)
from django_ai_sdk.integrations.mcp.loader import MCPIntegration
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)
from django_ai_sdk.integrations.registry import (
    get_all_integrations,
    get_integrations,
    register,
    reset_registry,
)
from django_ai_sdk.permissions import AllowAll

from tests.mocks.integrations import ExampleWeatherService, UnnamedService


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global and caches instances for their breaker state, so
    every test starts and ends with an empty one."""
    reset_registry()
    yield
    reset_registry()


def _mock_transport(token_response: dict, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=token_response)

    return httpx.MockTransport(handler)


def _patch_oauth_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    """Inject a fake token-endpoint transport into every AsyncOAuth2Client this package
    builds, whether called from loader.refresh_oauth_token directly or from
    services.exchange_token (which imports build_oauth_client locally, so patching the
    loader's copy covers both)."""
    from django_ai_sdk.integrations.mcp import loader as loader_module

    original = loader_module.build_oauth_client

    def wrapper(client_id, client_secret, **kwargs):
        kwargs.setdefault("transport", transport)
        return original(client_id, client_secret, **kwargs)

    monkeypatch.setattr(loader_module, "build_oauth_client", wrapper)


def _patch_discovery(monkeypatch, token_endpoint: str = "https://auth.example.com/token") -> None:
    """Stub OAuth metadata discovery.

    Patches every module that binds ``discover``, not just its home module: `services`
    and `oauth_views` import the name directly, so rebinding only
    ``discovery.discover`` would leave them calling the real one — which does DNS-based
    SSRF checks and would reject a test hostname outright.
    """
    import django_ai_sdk.integrations.mcp.discovery as discovery_module
    from django_ai_sdk.integrations.mcp import oauth_views
    from django_ai_sdk.integrations.mcp import services as services_module

    async def fake_discover(*args, **kwargs):
        return discovery_module.OAuthDiscovery(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint=token_endpoint,
        )

    monkeypatch.setattr(discovery_module, "discover", fake_discover)
    monkeypatch.setattr(services_module, "discover", fake_discover)
    monkeypatch.setattr(oauth_views, "discover", fake_discover)


def _fake_tool(name: str):
    return type("T", (), {"name": name})()


class TestResilientCache:
    async def test_cache_hit_returns_immediately(self):
        cache = ResilientCache(ttl=60, timeout=5)
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return ["tool"]

        first = await cache.get("k", fetch)
        second = await cache.get("k", fetch)

        assert first == ["tool"]
        assert second == ["tool"]
        assert calls == 1  # second call was a cache hit, no re-fetch

    async def test_stale_entry_served_immediately_while_refreshing_in_background(self):
        # ttl=1.0 -> background refresh kicks in at early_ttl=0.8; hard expiry at 1.0.
        cache = ResilientCache(ttl=1.0, timeout=5)
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.3)  # slow refresh — must not block the caller
            return [f"tool-{calls}"]

        first = await cache.get("k", fetch)
        assert first == ["tool-1"]

        await asyncio.sleep(0.9)  # age now in [early_ttl=0.8, ttl=1.0): stale window

        start = time.monotonic()
        second = await cache.get("k", fetch)
        elapsed = time.monotonic() - start

        assert second == ["tool-1"]  # stale value served immediately, not the new one
        assert elapsed < 0.1  # nowhere near the 0.3s refresh — never blocked on it

        await asyncio.sleep(0.5)  # let the background refresh finish
        third = await cache.get("k", fetch)
        assert third == ["tool-2"]  # now warmed by the background refresh

    async def test_cache_miss_bounded_by_timeout(self):
        cache = ResilientCache(ttl=60, timeout=0.05)

        async def hangs_forever():
            await asyncio.sleep(10)
            return ["never"]

        start = time.monotonic()
        result = await cache.get("k", hangs_forever)
        elapsed = time.monotonic() - start

        assert result == []  # degrades to empty rather than hanging
        assert elapsed < 0.5  # bounded by `timeout`, not the 10s fetch

    async def test_circuit_breaker_opens_after_repeated_failures_and_stops_retrying(self):
        cache = ResilientCache(ttl=60, timeout=1)
        calls = 0

        async def always_fails():
            nonlocal calls
            calls += 1
            raise RuntimeError("dead server")

        # Each failure is truthfully reported as DEGRADED while the breaker is closed.
        for _ in range(3):
            await cache.get("k", always_fails)
            assert cache.status_for("k") == IntegrationStatus.DEGRADED

        # Breaker is now open — further gets short-circuit to empty without a live fetch.
        result = await cache.get("k", always_fails)
        assert result == []
        assert calls == 3  # the 4th get() didn't attempt another live fetch
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

    async def test_breaker_open_transition_logs_exactly_once(self, caplog):
        """The 'just opened' warning fires once, on the transition — not on every
        subsequent short-circuited call, which would otherwise spam the logs for as
        long as the integration stays down."""
        cache = ResilientCache(ttl=60, timeout=1)

        async def always_fails():
            raise RuntimeError("dead server")

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                await cache.get("k", always_fails)
            # Breaker is now open. Further short-circuited calls must not add another
            # "just opened" warning.
            for _ in range(3):
                await cache.get("k", always_fails)

        assert caplog.text.count("circuit breaker just opened") == 1

    async def test_recovery_after_open_breaker_logs_once(self, caplog):
        cache = ResilientCache(ttl=60, timeout=1, cb_cooldown=0.05)

        async def always_fails():
            raise RuntimeError("dead server")

        async def succeeds():
            return ["ok"]

        for _ in range(3):
            await cache.get("k", always_fails)
        await asyncio.sleep(0.1)  # let the cooldown elapse so it half-opens

        with caplog.at_level(logging.INFO):
            result = await cache.get("k", succeeds)

        assert result == ["ok"]
        assert caplog.text.count("recovered") == 1

    async def test_single_failure_immediately_reports_degraded_not_active(self):
        """A wrong/invalid token must never show as ACTIVE — status reflects the last
        real attempt, not an optimistic default, even before the breaker trips."""
        cache = ResilientCache(ttl=60, timeout=1)

        async def wrong_token():
            raise RuntimeError("401 Unauthorized")

        await cache.get("k", wrong_token)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

    def test_never_attempted_key_raises_instead_of_guessing(self):
        """status_for() can't distinguish "never checked" from "healthy" on its own,
        so it refuses to guess: callers (e.g. MCPIntegration.get_status()) must force
        a real attempt via get() first, or this raises rather than reporting a false
        ACTIVE for an integration nothing has actually checked yet."""
        cache = ResilientCache(ttl=60, timeout=1)
        with pytest.raises(RuntimeError, match="never-checked"):
            cache.status_for("never-checked")

    async def test_invalidate_resets_a_degraded_key(self):
        cache = ResilientCache(ttl=60, timeout=1)

        async def fails():
            raise RuntimeError("dead")

        async def succeeds():
            return ["ok"]

        await cache.get("k", fails)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

        await cache.invalidate("k")  # the manual "reconnect" / retry-now action
        result = await cache.get("k", succeeds)
        assert result == ["ok"]
        assert cache.status_for("k") == IntegrationStatus.ACTIVE

    async def test_invalidate_clears_a_cached_success_value(self):
        """invalidate() ("retry now") must drop a previously cached *successful* value,
        not just reset the breaker — otherwise reconnect keeps serving stale."""
        cache = ResilientCache(ttl=60, timeout=1)
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return [f"tools-{calls}"]

        assert await cache.get("k", fetch) == ["tools-1"]
        assert await cache.get("k", fetch) == ["tools-1"]  # cache hit, no re-fetch
        assert calls == 1

        await cache.invalidate("k")
        assert await cache.get("k", fetch) == ["tools-2"]  # cleared -> real re-fetch
        assert calls == 2

    async def test_invalidate_clears_the_whole_instance_not_just_one_key(self):
        """Documents a deliberate trade: invalidate() drops every key in this cache,
        not only the one named. Each instance holds a single integration's keys, so the
        cost is that other users of the *same* integration pay one bounded re-fetch —
        accepted in exchange for not coupling to cashews' internal key layout."""
        cache = ResilientCache(ttl=60, timeout=1)
        calls = {"a": 0, "b": 0}

        async def fetch_a():
            calls["a"] += 1
            return ["a"]

        async def fetch_b():
            calls["b"] += 1
            return ["b"]

        await cache.get("user-a", fetch_a)
        await cache.get("user-b", fetch_b)
        assert calls == {"a": 1, "b": 1}

        await cache.invalidate("user-a")

        await cache.get("user-a", fetch_a)
        await cache.get("user-b", fetch_b)
        assert calls == {"a": 2, "b": 2}  # user-b was dropped too


class TestResilientCacheRecovery:
    """The breaker recovers on its own via half-open probing after the cooldown — there
    is no terminal give-up state."""

    async def test_open_breaker_half_opens_and_recovers_after_cooldown(self):
        cache = ResilientCache(ttl=60, timeout=1, cb_cooldown=0.2)
        state = {"fail": True}

        async def fetch():
            if state["fail"]:
                raise RuntimeError("dead")
            return ["ok"]

        for _ in range(3):
            await cache.get("k", fetch)
        assert await cache.get("k", fetch) == []  # breaker open

        # The server comes back; after the cooldown the breaker half-opens and, within
        # a few probes, closes again — no manual reconnect required.
        state["fail"] = False
        recovered = False
        for _ in range(20):
            await asyncio.sleep(0.1)
            if await cache.get("k", fetch) == ["ok"]:
                recovered = True
                break
        assert recovered
        assert cache.status_for("k") == IntegrationStatus.ACTIVE

    async def test_isolated_per_key(self):
        """One dead key's breaker must never trip another's."""
        cache = ResilientCache(ttl=60, timeout=1)

        async def fails():
            raise RuntimeError("dead")

        async def succeeds():
            return ["ok"]

        for _ in range(4):
            await cache.get("dead", fails)
        assert cache.status_for("dead") == IntegrationStatus.DEGRADED

        assert await cache.get("healthy", succeeds) == ["ok"]
        assert cache.status_for("healthy") == IntegrationStatus.ACTIVE


class TestMCPIntegrationCacheKeys:
    def test_static_config_key_ignores_user(self):
        integration = MCPIntegration(
            "linear", StaticMCPIntegrationConfig(url="https://example.com/mcp")
        )
        user_a = type("U", (), {"pk": 1})()
        user_b = type("U", (), {"pk": 2})()

        assert integration._cache_key(user_a) == "linear"
        assert integration._cache_key(user_b) == "linear"
        assert integration._cache_key(None) == "linear"

    def test_oauth_config_key_is_per_user(self):
        """Each user has their own token, so their discovered tool lists must not share
        a cache entry."""
        integration = MCPIntegration(
            "notion", OAuthMCPIntegrationConfig(url="https://example.com/mcp")
        )
        user_a = type("U", (), {"pk": 1})()
        user_b = type("U", (), {"pk": 2})()

        assert integration._cache_key(user_a) == "notion:1"
        assert integration._cache_key(user_b) == "notion:2"
        assert integration._cache_key(None) is None  # no user — nothing to key on


class TestIntegrationDisplayMetadata:
    """``kind``/``get_tool_names()`` are called per integration by the status endpoints,
    so they must stay cheap — no live I/O when config already answers the question."""

    async def test_mcp_integration_kind_reflects_config_type(self):
        token = MCPIntegration("linear", TokenMCPIntegrationConfig(url="https://x", token="t"))
        oauth = MCPIntegration("notion", OAuthMCPIntegrationConfig(url="https://x"))

        assert token.kind == "token"
        assert oauth.kind == "oauth"

    async def test_mcp_integration_tool_names_reads_config_without_connecting(self, monkeypatch):
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def explodes_if_called(*args, **kwargs):
            raise AssertionError("get_tool_names() must not trigger a live connect")

        monkeypatch.setattr(loader_module, "_connect", explodes_if_called)

        integration = MCPIntegration(
            "linear",
            TokenMCPIntegrationConfig(url="https://x", token="t", tools=["list_issues"]),
        )

        assert await integration.get_tool_names() == ["list_issues"]

    async def test_mcp_integration_tool_names_falls_back_to_discovery_with_no_allowlist(
        self, monkeypatch
    ):
        """With auto-discovery (no `tools` configured) there is no static name list to
        read, so this must fall back to the real, cached discovered tools rather than
        reporting none — otherwise those tools are invisible to the UI's Integrations
        grouping and leak into the flat Tools list instead."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        calls = 0

        async def fake_connect(*args, **kwargs):
            nonlocal calls
            calls += 1
            return [_fake_tool("list_issues"), _fake_tool("get_issue")]

        monkeypatch.setattr(loader_module, "_connect", fake_connect)

        integration = MCPIntegration(
            "linear", TokenMCPIntegrationConfig(url="https://x", token="t")
        )

        assert await integration.get_tool_names() == ["list_issues", "get_issue"]
        assert calls == 1

    async def test_mcp_integration_tool_names_reuses_cache_already_primed_by_get_status(
        self, monkeypatch
    ):
        """get_integration_status() calls get_status() then get_tool_names() per
        integration — the fallback above must not double the live-fetch cost when
        get_status() already warmed the same cache key moments earlier."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        calls = 0

        async def fake_connect(*args, **kwargs):
            nonlocal calls
            calls += 1
            return [_fake_tool("list_issues")]

        monkeypatch.setattr(loader_module, "_connect", fake_connect)

        integration = MCPIntegration(
            "linear", TokenMCPIntegrationConfig(url="https://x", token="t")
        )

        await integration.get_status()
        assert await integration.get_tool_names() == ["list_issues"]
        assert calls == 1  # second call was a cache hit, not a second live fetch

    async def test_api_integration_kind_defaults_to_api(self):
        class DummyIntegration(APIIntegration):
            name = "dummy"
            tools = []

        assert DummyIntegration().kind == "api"

    async def test_api_integration_tool_names_derives_from_get_tools(self):
        class DummyIntegration(APIIntegration):
            name = "dummy"
            tools = [lambda **kwargs: _fake_tool("dummy_tool")]

        assert await DummyIntegration().get_tool_names() == ["dummy_tool"]

    async def test_api_integration_factory_receives_only_declared_kwargs(self):
        """A tool factory may declare any subset of user/assistant/thread_id (or
        **kwargs) — it's called with only what it accepts, never a spurious TypeError."""

        class DummyIntegration(APIIntegration):
            name = "dummy"
            tools = [
                lambda user: _fake_tool("only_user"),
                lambda user, thread_id: _fake_tool("user_thread"),
                lambda **kw: _fake_tool("kwargs_all" if "assistant" in kw else "kwargs_bad"),
            ]

        names = [t.name for t in await DummyIntegration().get_tools(user="u", thread_id="t")]
        assert names == ["only_user", "user_thread", "kwargs_all"]


class TestMCPIntegrationGetStatus:
    async def test_wrong_static_token_reports_degraded_on_first_check(self, monkeypatch):
        """A token integration whose credential is simply wrong must not show as
        connected just because get_status() was never exercised by a chat turn yet."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def fails(*args, **kwargs):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(loader_module, "_connect", fails)

        integration = MCPIntegration(
            "linear-wrong-token",
            TokenMCPIntegrationConfig(url="https://example.com/mcp", token="bad"),
        )

        assert await integration.get_status() == IntegrationStatus.DEGRADED

    async def test_correct_static_token_reports_active_on_first_check(self, monkeypatch):
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def succeeds(*args, **kwargs):
            return ["tool"]

        monkeypatch.setattr(loader_module, "_connect", succeeds)

        integration = MCPIntegration(
            "linear-correct-token",
            TokenMCPIntegrationConfig(url="https://example.com/mcp", token="good"),
        )

        assert await integration.get_status() == IntegrationStatus.ACTIVE

    async def test_reconnect_recovers_a_degraded_integration(self, monkeypatch):
        """reconnect() is an immediate retry-now: once the underlying problem is fixed
        it clears the cached failure and re-probes without waiting out the cooldown."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def fails(*args, **kwargs):
            raise RuntimeError("dead")

        monkeypatch.setattr(loader_module, "_connect", fails)

        integration = MCPIntegration(
            "linear-broken",
            TokenMCPIntegrationConfig(url="https://example.com/mcp", token="bad"),
        )
        assert await integration.get_status() == IntegrationStatus.DEGRADED

        async def succeeds(*args, **kwargs):
            return ["tool"]

        monkeypatch.setattr(loader_module, "_connect", succeeds)
        await integration.reconnect()
        assert await integration.get_status() == IntegrationStatus.ACTIVE

    async def test_missing_secret_reports_disconnected_and_contributes_no_tools(self):
        """A token server with no token configured must not crash at construction —
        it registers, explains itself via `detail`, and stays out of the way."""
        from django_ai_sdk.integrations.mcp.loader import build_mcp_config_safe

        config, needs_setup = build_mcp_config_safe(
            auth="token", url="https://example.com/mcp", label="Linear", tools=[], token=""
        )
        integration = MCPIntegration(
            "linear", config, needs_setup=needs_setup, intended_kind="token"
        )

        assert needs_setup  # a human-readable reason, not an exception
        assert integration.detail == needs_setup
        assert await integration.get_status() == IntegrationStatus.DISCONNECTED
        assert await integration.get_tools() == []
        assert await integration.get_tool_names() == []
        # The failed build leaves a *static* placeholder config behind; the reported kind
        # must still describe what the deployer is configuring, not the placeholder.
        assert integration.kind == "token"


class TestAPIIntegrationGetStatus:
    """A hand-written API integration must report real health too — a down backend
    shows up as DEGRADED, not a hardcoded ACTIVE."""

    async def test_no_health_check_is_always_active(self):
        class NoHealthCheckIntegration(APIIntegration):
            name = "no-health-check"
            tools = []

        assert await NoHealthCheckIntegration().get_status() == IntegrationStatus.ACTIVE

    async def test_failing_health_check_reports_degraded(self):
        async def fails():
            raise RuntimeError("503 Service Unavailable")

        class FlakyIntegration(APIIntegration):
            name = "flaky"
            tools = []
            health_check = staticmethod(fails)

        assert await FlakyIntegration().get_status() == IntegrationStatus.DEGRADED

    async def test_succeeding_health_check_reports_active(self):
        async def succeeds():
            return None

        class HealthyIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "healthy"
            tools = []
            health_check = staticmethod(succeeds)

        assert await HealthyIntegration().get_status() == IntegrationStatus.ACTIVE

    async def test_reconnect_recovers_a_degraded_health_check(self):
        state = {"fail": True}

        async def probe():
            if state["fail"]:
                raise RuntimeError("dead")

        class RecoverableIntegration(APIIntegration):
            name = "recoverable"
            tools = []
            health_check = staticmethod(probe)

        integration = RecoverableIntegration()
        assert await integration.get_status() == IntegrationStatus.DEGRADED

        state["fail"] = False
        await integration.reconnect()  # retry now, don't wait out the cooldown
        assert await integration.get_status() == IntegrationStatus.ACTIVE


class TestRegistry:
    """Integrations are declared as ``{name: dotted.path}``; a bad entry must degrade
    to "that one is unavailable" rather than breaking every other integration."""

    async def test_dotted_path_is_imported_and_instantiated(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"weather": "tests.mocks.integrations.ExampleWeatherService"}

        integrations = await get_all_integrations()

        assert list(integrations) == ["weather"]
        assert isinstance(integrations["weather"], ExampleWeatherService)

    async def test_instance_entries_are_used_as_is(self, settings):
        instance = ExampleWeatherService()
        settings.AI_SDK_INTEGRATIONS = {"weather": instance}

        assert (await get_all_integrations())["weather"] is instance

    async def test_service_is_built_once_and_reused(self, settings):
        """Each service owns its ResilientCache and breaker state, so rebuilding it per
        lookup would silently discard the health it just learned."""
        settings.AI_SDK_INTEGRATIONS = {"weather": "tests.mocks.integrations.ExampleWeatherService"}

        first = (await get_all_integrations())["weather"]
        second = (await get_all_integrations())["weather"]

        assert first is second

    async def test_name_and_label_are_backfilled_from_the_settings_key(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"my-weather": "tests.mocks.integrations.UnnamedService"}

        service = (await get_all_integrations())["my-weather"]

        assert service.name == "my-weather"  # the key is authoritative
        assert service.label == "My-Weather"

    async def test_a_broken_entry_does_not_hide_the_others(self, settings, caplog):
        settings.AI_SDK_INTEGRATIONS = {
            "broken": "tests.unit.does.not.Exist",
            "weather": "tests.mocks.integrations.ExampleWeatherService",
        }

        integrations = await get_all_integrations()

        assert list(integrations) == ["weather"]
        assert "broken" in caplog.text

    async def test_a_broken_entry_is_only_reported_once(self, settings, caplog):
        """The first lookup may well be a chat request; a broken entry must not re-run
        a failing import on every subsequent one."""
        settings.AI_SDK_INTEGRATIONS = {"broken": "tests.unit.does.not.Exist"}

        await get_all_integrations()
        first_count = caplog.text.count("Could not load integration")
        await get_all_integrations()

        assert first_count == 1
        assert caplog.text.count("Could not load integration") == 1

    async def test_get_integrations_skips_unknown_names(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"weather": "tests.mocks.integrations.ExampleWeatherService"}

        assert list(await get_integrations(["weather", "nope"])) == ["weather"]

    async def test_explicitly_registered_service_wins_over_settings(self, settings):
        """The app-based escape hatch and the settings mapping share one registry;
        a deliberately constructed service takes precedence."""
        settings.AI_SDK_INTEGRATIONS = {"weather": "tests.mocks.integrations.ExampleWeatherService"}
        explicit = ExampleWeatherService()
        register(explicit)

        assert (await get_all_integrations())["weather"] is explicit

    def test_register_rejects_a_nameless_service(self):
        class Nameless(APIIntegration):
            tools = []

        with pytest.raises(ValueError, match="non-empty `name`"):
            register(Nameless())

    async def test_non_dict_settings_is_a_configuration_error(self, settings):
        settings.AI_SDK_INTEGRATIONS = ["weather"]

        with pytest.raises(ImproperlyConfigured, match="must be a dict"):
            await get_all_integrations()


class TestExtensibility:
    """A third integration backend (neither MCP nor APIIntegration) must be pluggable
    with no special-casing."""

    async def test_bare_api_integration_subclass_needs_no_init_override(self):
        class MinimalIntegration(APIIntegration):
            name = "minimal"
            tools = []

        integration = MinimalIntegration()  # must not raise — no required args
        assert await integration.get_tools() == []
        assert await integration.get_status() == IntegrationStatus.ACTIVE

    async def test_a_hand_rolled_service_satisfies_the_contract(self, settings):
        class CustomBackendService(IntegrationService):
            name = "custom"
            label = "Custom"

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                return ["custom-tool"]

            async def get_status(self, user=None, assistant=None):
                return IntegrationStatus.ACTIVE

        instance = CustomBackendService()
        settings.AI_SDK_INTEGRATIONS = {"custom": instance}

        resolved = (await get_all_integrations())["custom"]
        assert resolved is instance
        assert await resolved.get_tools() == ["custom-tool"]
        assert resolved.kind == "api"  # the contract's default, no MCP assumptions

    async def test_assistant_get_tools_threads_assistant_into_factory(self, settings):
        """A tool factory that runs its own LLM call (e.g. translation) needs the
        calling assistant's model — that's why `assistant` is in the contract."""
        from django_ai_sdk.assistant import Assistant

        received: dict = {}

        def factory(*, user=None, assistant=None, thread_id=""):
            received["assistant"] = assistant
            return []

        class ModelAwareIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "model-aware"
            tools = [factory]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["model-aware"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        settings.AI_SDK_INTEGRATIONS = {"model-aware": ModelAwareIntegration()}

        assistant = FakeAssistant()
        await assistant._get_integration_tools()

        assert received["assistant"] is assistant
        assert received["assistant"].model == "gpt-fake"


class TestIntegrationFailureIsolation:
    """A broken or slow integration must not affect another's tools, and must not
    serialize with it either — the guarantee _get_integration_tools makes via
    asyncio.gather plus a per-integration try/except."""

    async def test_one_failing_integration_does_not_drop_others_tools(self, settings):
        from django_ai_sdk.assistant import Assistant

        class BrokenIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "broken"
            tools = []

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                raise RuntimeError("upstream is down")

        class HealthyIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "healthy"
            tools = [lambda **kwargs: "healthy-tool"]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["broken", "healthy"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        settings.AI_SDK_INTEGRATIONS = {
            "broken": BrokenIntegration(),
            "healthy": HealthyIntegration(),
        }

        assert await FakeAssistant()._get_integration_tools() == ["healthy-tool"]

    async def test_integrations_are_awaited_concurrently_not_serially(self, settings):
        """If a slow integration and another were awaited one at a time, total
        wall-clock time would be additive. Assert it isn't."""
        from django_ai_sdk.assistant import Assistant

        delay = 0.2

        class SlowIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "slow"
            tools = []

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                await asyncio.sleep(delay)
                return ["slow-tool"]

        class OtherSlowIntegration(SlowIntegration):
            name = "other-slow"

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                await asyncio.sleep(delay)
                return ["other-slow-tool"]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["slow", "other-slow"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        settings.AI_SDK_INTEGRATIONS = {
            "slow": SlowIntegration(),
            "other-slow": OtherSlowIntegration(),
        }

        start = time.monotonic()
        tools = await FakeAssistant()._get_integration_tools()
        elapsed = time.monotonic() - start

        assert set(tools) == {"slow-tool", "other-slow-tool"}
        assert elapsed < delay * 2

    async def test_an_unpermitted_integration_contributes_no_tools(self, settings):
        """Permissions are enforced before tools reach the model, not after."""
        from django_ai_sdk.assistant import Assistant

        class ForbiddenIntegration(APIIntegration):
            name = "forbidden"
            tools = [lambda **kwargs: "secret-tool"]

            async def has_perms(self, user, operation=None, *, raise_on_deny=False):
                return False

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["forbidden"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        settings.AI_SDK_INTEGRATIONS = {"forbidden": ForbiddenIntegration()}

        assert await FakeAssistant()._get_integration_tools() == []


@pytest.mark.django_db
class TestOAuthTokenRefresh:
    """refresh_oauth_token() goes through Authlib's AsyncOAuth2Client. These cover the
    HTTP exchange plus the optimistic-concurrency guard against a concurrent refresh
    (another request, or an overlapping refresh_integrations run) landing first."""

    async def _make_token(self, user, server_name="notion", refresh_token="old-refresh"):
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        token_obj = MCPOAuthToken(user=user, server_name=server_name)
        token_obj.set_tokens(
            {"access_token": "old-access", "refresh_token": refresh_token, "expires_in": -10}
        )
        await token_obj.asave()
        return token_obj

    async def test_refresh_success_persists_new_tokens(self, monkeypatch):
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user)

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(
            monkeypatch,
            _mock_transport(
                {"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}
            ),
        )

        config = OAuthMCPIntegrationConfig(
            url="https://mcp.example.com", client_id="client-1", client_secret="secret-1"
        )
        result = await refresh_oauth_token(token_obj, config)

        assert result is not None
        assert result.get_access_token() == "new-access"
        assert result.get_refresh_token() == "new-refresh"

        stored = await MCPOAuthToken.objects.aget(pk=token_obj.pk)
        assert stored.get_access_token() == "new-access"

    async def test_refresh_returns_none_on_oauth_error(self, monkeypatch):
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user)

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(
            monkeypatch, _mock_transport({"error": "invalid_grant"}, status_code=400)
        )

        config = OAuthMCPIntegrationConfig(
            url="https://mcp.example.com", client_id="client-1", client_secret="secret-1"
        )
        result = await refresh_oauth_token(token_obj, config)

        assert result is None
        stored = await MCPOAuthToken.objects.aget(pk=token_obj.pk)
        assert stored.get_access_token() == "old-access"  # untouched by the failure

    async def test_refresh_without_a_refresh_token_gives_up(self, monkeypatch):
        """Nothing to exchange — this must fail fast rather than call the token
        endpoint with an empty grant."""
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user, refresh_token="")

        def explodes(request):
            raise AssertionError("must not call the token endpoint without a refresh token")

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(monkeypatch, httpx.MockTransport(explodes))

        config = OAuthMCPIntegrationConfig(url="https://mcp.example.com", client_id="c")
        assert await refresh_oauth_token(token_obj, config) is None

    async def test_refresh_loses_race_reloads_winner_instead_of_clobbering_it(self, monkeypatch):
        """By the time our refresh response comes back, a concurrent refresh has already
        rotated the row's refresh_token out from under us. Our conditional update must
        no-op, and the caller must get the winner's tokens — not silently overwrite
        them with our own."""
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user)

        winner = MCPOAuthToken(user=user, server_name=token_obj.server_name)
        winner.set_tokens(
            {"access_token": "winner-access", "refresh_token": "winner-refresh", "expires_in": 3600}
        )
        await MCPOAuthToken.objects.filter(pk=token_obj.pk).aupdate(
            access_token=winner.access_token,
            refresh_token=winner.refresh_token,
            expires_at=winner.expires_at,
        )

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(
            monkeypatch,
            _mock_transport(
                {
                    "access_token": "loser-access",
                    "refresh_token": "loser-refresh",
                    "expires_in": 3600,
                }
            ),
        )

        config = OAuthMCPIntegrationConfig(
            url="https://mcp.example.com", client_id="client-1", client_secret="secret-1"
        )
        # token_obj still holds the pre-race refresh_token in memory — exactly the stale
        # value a real caller would have if it read the row before the race.
        result = await refresh_oauth_token(token_obj, config)

        assert result is not None
        assert result.get_access_token() == "winner-access"


@pytest.mark.django_db
class TestExchangeToken:
    """The authorization_code exchange goes through Authlib."""

    async def test_exchange_returns_token_dict(self, monkeypatch):
        from django_ai_sdk.integrations.mcp.services import exchange_token

        _patch_oauth_transport(
            monkeypatch,
            _mock_transport(
                {"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}
            ),
        )

        token = await exchange_token(
            "https://auth.example.com/token",
            code="auth-code",
            redirect_uri="https://app.example.com/callback",
            verifier="verifier-value",
            client_id="client-1",
            client_secret="secret-1",
        )

        assert token["access_token"] == "new-access"
        assert token["refresh_token"] == "new-refresh"

    async def test_exchange_raises_on_oauth_error(self, monkeypatch):
        from django_ai_sdk.integrations.mcp.services import exchange_token

        _patch_oauth_transport(
            monkeypatch, _mock_transport({"error": "invalid_grant"}, status_code=400)
        )

        with pytest.raises(ValueError):
            await exchange_token(
                "https://auth.example.com/token",
                code="auth-code",
                redirect_uri="https://app.example.com/callback",
                verifier="verifier-value",
                client_id="client-1",
                client_secret="secret-1",
            )

    async def test_exchange_raises_when_access_token_missing(self, monkeypatch):
        from django_ai_sdk.integrations.mcp.services import exchange_token

        _patch_oauth_transport(monkeypatch, _mock_transport({"token_type": "Bearer"}))

        with pytest.raises(ValueError, match="access_token"):
            await exchange_token(
                "https://auth.example.com/token",
                code="auth-code",
                redirect_uri="https://app.example.com/callback",
                verifier="verifier-value",
                client_id="client-1",
                client_secret="secret-1",
            )


@pytest.mark.django_db
class TestOAuthRedirectFlow:
    """The browser-facing half of MCP OAuth: ``oauth_start`` hands the user off to the
    provider, ``oauth_callback`` brings them back and persists the token.

    This path had no coverage at all before, which matters more now that it is shared
    SDK infrastructure rather than one project's local code — every consumer inherits
    whatever is (or isn't) verified here.
    """

    @staticmethod
    def _request(user, path="/api/integrations/oauth/notion/start/", **params):
        from django.contrib.sessions.backends.db import SessionStore
        from django.test import RequestFactory

        request = RequestFactory().get(path, data=params)
        request.user = user
        request.session = SessionStore()
        return request

    @staticmethod
    def _oauth_integration(name="notion"):
        return MCPIntegration(
            name,
            OAuthMCPIntegrationConfig(
                url="https://mcp.example.com/mcp",
                client_id="client-1",
                client_secret="secret-1",
            ),
        )

    async def test_start_redirects_to_the_provider_and_stores_pkce_state(
        self, settings, monkeypatch
    ):
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views

        settings.AI_SDK_INTEGRATIONS = {"notion": self._oauth_integration()}
        _patch_discovery(monkeypatch)

        async def fake_register(*args, **kwargs):
            return "client-1", "secret-1"

        monkeypatch.setattr(
            "django_ai_sdk.integrations.mcp.services.get_or_register_client", fake_register
        )

        user = await UserFactory.acreate()
        request = self._request(user)
        response = await oauth_views.oauth_start(request, "notion")

        assert response.status_code == 302
        assert response["Location"].startswith("https://auth.example.com/authorize?")
        # PKCE state must be held server-side; the callback compares against it.
        assert request.session[loader_module._K_STATE.format("notion")]
        assert request.session[loader_module._K_VERIFIER.format("notion")]

    async def test_start_requires_authentication(self, settings):
        from django.contrib.auth.models import AnonymousUser

        from django_ai_sdk.integrations.mcp import oauth_views

        settings.AI_SDK_INTEGRATIONS = {"notion": self._oauth_integration()}

        response = await oauth_views.oauth_start(self._request(AnonymousUser()), "notion")

        assert response.status_code == 401

    async def test_start_rejects_an_unknown_server(self, settings):
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp import oauth_views

        settings.AI_SDK_INTEGRATIONS = {}
        user = await UserFactory.acreate()

        response = await oauth_views.oauth_start(self._request(user), "nope")

        assert response.status_code == 404

    async def test_callback_exchanges_the_code_and_persists_the_token(self, settings, monkeypatch):
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        settings.AI_SDK_INTEGRATIONS = {"notion": self._oauth_integration()}
        settings.AI_SDK_MCP_OAUTH_SUCCESS_URL = "/settings/integrations"
        _patch_oauth_transport(
            monkeypatch,
            _mock_transport(
                {
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "expires_in": 3600,
                }
            ),
        )

        user = await UserFactory.acreate()
        request = self._request(
            user,
            path="/api/integrations/oauth/notion/callback/",
            code="auth-code",
            state="the-state",
        )
        request.session[loader_module._K_STATE.format("notion")] = "the-state"
        request.session[loader_module._K_VERIFIER.format("notion")] = "the-verifier"
        request.session[loader_module._K_TOKEN_ENDPOINT.format("notion")] = (
            "https://auth.example.com/token"
        )

        response = await oauth_views.oauth_callback(request, "notion")

        assert response.status_code == 302
        assert response["Location"] == "/settings/integrations?connected=notion"

        stored = await MCPOAuthToken.objects.aget(user=user, server_name="notion")
        assert stored.get_access_token() == "fresh-access"
        assert stored.get_refresh_token() == "fresh-refresh"
        # One-shot PKCE material must not survive the exchange.
        assert loader_module._K_VERIFIER.format("notion") not in request.session

    async def test_callback_rejects_a_mismatched_state(self, settings, monkeypatch):
        """CSRF protection for the OAuth handshake: a code arriving with someone else's
        (or a forged) state must never be exchanged."""
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        settings.AI_SDK_INTEGRATIONS = {"notion": self._oauth_integration()}

        def explodes(request):
            raise AssertionError("must not reach the token endpoint on a state mismatch")

        _patch_oauth_transport(monkeypatch, httpx.MockTransport(explodes))

        user = await UserFactory.acreate()
        request = self._request(
            user,
            path="/api/integrations/oauth/notion/callback/",
            code="auth-code",
            state="attacker-state",
        )
        request.session[loader_module._K_STATE.format("notion")] = "the-real-state"
        request.session[loader_module._K_VERIFIER.format("notion")] = "the-verifier"

        response = await oauth_views.oauth_callback(request, "notion")

        assert response.status_code == 400
        assert not await MCPOAuthToken.objects.filter(user=user).aexists()

    async def test_callback_reports_a_provider_error_without_exchanging(
        self, settings, monkeypatch
    ):
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp import oauth_views

        settings.AI_SDK_INTEGRATIONS = {"notion": self._oauth_integration()}

        def explodes(request):
            raise AssertionError("must not reach the token endpoint after a provider error")

        _patch_oauth_transport(monkeypatch, httpx.MockTransport(explodes))

        user = await UserFactory.acreate()
        request = self._request(
            user,
            path="/api/integrations/oauth/notion/callback/",
            error="access_denied",
            error_description="user said no",
        )

        response = await oauth_views.oauth_callback(request, "notion")

        assert response.status_code == 400

    async def test_callback_refuses_an_off_site_success_redirect(self, settings, monkeypatch):
        """A misconfigured AI_SDK_MCP_OAUTH_SUCCESS_URL must not turn the callback into
        an open redirect — it falls back to "/"."""
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views

        settings.AI_SDK_INTEGRATIONS = {"notion": self._oauth_integration()}
        settings.AI_SDK_MCP_OAUTH_SUCCESS_URL = "https://evil.example.com/steal"
        _patch_oauth_transport(
            monkeypatch, _mock_transport({"access_token": "fresh-access", "expires_in": 3600})
        )

        user = await UserFactory.acreate()
        request = self._request(
            user,
            path="/api/integrations/oauth/notion/callback/",
            code="auth-code",
            state="the-state",
        )
        request.session[loader_module._K_STATE.format("notion")] = "the-state"
        request.session[loader_module._K_VERIFIER.format("notion")] = "the-verifier"
        request.session[loader_module._K_TOKEN_ENDPOINT.format("notion")] = (
            "https://auth.example.com/token"
        )

        response = await oauth_views.oauth_callback(request, "notion")

        assert response.status_code == 302
        assert response["Location"].startswith("/?connected=notion")


class TestIntegrationPermissions:
    async def test_no_user_gets_no_integration_tools_by_default(self, settings):
        """The INTEGRATIONS domain default requires an authenticated user, so a system
        or anonymous context contributes no integration tools at all. Documented because
        it's easy to mistake for a registry miss when writing a test."""
        from django_ai_sdk.assistant import Assistant

        class DefaultPermsIntegration(APIIntegration):
            name = "default-perms"
            tools = [lambda **kwargs: "should-not-appear"]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["default-perms"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        settings.AI_SDK_INTEGRATIONS = {"default-perms": DefaultPermsIntegration()}

        assert await FakeAssistant()._get_integration_tools(user=None) == []
