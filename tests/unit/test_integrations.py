"""Tests for the integrations layer.

Three things are under test, in order of how much they'd hurt if wrong:

1. ``ResilientCache`` — the latency and failure guarantees the whole layer rests on.
   A dead integration must cost a bounded wait once and ~nothing after, and must be
   visible as DEGRADED rather than silently contributing no tools.
2. The registry and the ``Integration`` contract — a broken integration is
   isolated, integrations load concurrently, and a third-party backend plugs in with
   no special-casing.
3. The MCP OAuth flow — connect, callback, code exchange, refresh, and the concurrent
   refresh race. This is shared infrastructure several consumers rely on, so it is
   covered here rather than only in whichever project happens to use it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
import pytest
from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.base import (
    Integration,
    IntegrationStatus,
    ResilientCache,
)
from django_ai_sdk.integrations.mcp.loader import DynamicMCPIntegration
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
from tests.mocks.integrations import ExampleWeatherService


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global and caches instances for their breaker state, so
    every test starts and ends with an empty one."""
    reset_registry()
    yield
    reset_registry()


def _mock_transport(
    token_response: dict, status_code: int = 200
) -> httpx.MockTransport:
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


async def _coroutine(value):
    """Wrap a value so it can stand in for an awaitable request attribute."""
    return value


def _patch_discovery(
    monkeypatch, token_endpoint: str = "https://auth.example.com/token"
) -> None:
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

    async def test_circuit_breaker_opens_after_repeated_failures_and_stops_retrying(
        self,
    ):
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
        so it refuses to guess: callers (e.g. DynamicMCPIntegration.get_status()) must force
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
        integration = DynamicMCPIntegration(
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
        integration = DynamicMCPIntegration(
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
        token = DynamicMCPIntegration(
            "linear", TokenMCPIntegrationConfig(url="https://x", token="t")
        )
        oauth = DynamicMCPIntegration(
            "notion", OAuthMCPIntegrationConfig(url="https://x")
        )

        assert token.kind == "token"
        assert oauth.kind == "oauth"

    async def test_mcp_integration_tool_names_reads_config_without_connecting(
        self, monkeypatch
    ):
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def explodes_if_called(*args, **kwargs):
            raise AssertionError("get_tool_names() must not trigger a live connect")

        monkeypatch.setattr(loader_module, "_connect", explodes_if_called)

        integration = DynamicMCPIntegration(
            "linear",
            TokenMCPIntegrationConfig(
                url="https://x", token="t", tools=["list_issues"]
            ),
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

        integration = DynamicMCPIntegration(
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

        integration = DynamicMCPIntegration(
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
                lambda **kw: _fake_tool(
                    "kwargs_all" if "assistant" in kw else "kwargs_bad"
                ),
            ]

        names = [
            t.name for t in await DummyIntegration().get_tools(user="u", thread_id="t")
        ]
        assert names == ["only_user", "user_thread", "kwargs_all"]


class TestMCPIntegrationGetStatus:
    async def test_wrong_static_token_reports_degraded_on_first_check(
        self, monkeypatch
    ):
        """A token integration whose credential is simply wrong must not show as
        connected just because get_status() was never exercised by a chat turn yet."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def fails(*args, **kwargs):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(loader_module, "_connect", fails)

        integration = DynamicMCPIntegration(
            "linear-wrong-token",
            TokenMCPIntegrationConfig(url="https://example.com/mcp", token="bad"),
        )

        assert await integration.get_status() == IntegrationStatus.DEGRADED

    async def test_correct_static_token_reports_active_on_first_check(
        self, monkeypatch
    ):
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def succeeds(*args, **kwargs):
            return ["tool"]

        monkeypatch.setattr(loader_module, "_connect", succeeds)

        integration = DynamicMCPIntegration(
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

        integration = DynamicMCPIntegration(
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
            auth="token",
            url="https://example.com/mcp",
            label="Linear",
            tools=[],
            token="",
        )
        integration = DynamicMCPIntegration(
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
    """Each integration app registers its service from ``ready()``; the registry
    itself is a plain dict fed only by ``register()``."""

    async def test_registered_service_is_returned(self):
        instance = ExampleWeatherService()
        register(instance)

        assert (await get_all_integrations())["weather"] is instance

    async def test_get_integrations_skips_unknown_names(self):
        register(ExampleWeatherService())

        assert list(await get_integrations(["weather", "nope"])) == ["weather"]

    def test_register_rejects_a_nameless_service(self):
        class Nameless(APIIntegration):
            tools = []

        with pytest.raises(ValueError, match="non-empty `name`"):
            register(Nameless())

    async def test_reset_registry_clears_everything(self):
        register(ExampleWeatherService())
        reset_registry()

        assert await get_all_integrations() == {}


class TestIntegrationAppConfig:
    """``ready()`` is the only path a service takes into the registry."""

    async def test_ready_registers_the_configured_integration(self):
        import tests as tests_module
        from django_ai_sdk.integrations.apps import IntegrationAppConfig

        class WeatherConfig(IntegrationAppConfig):
            name = "tests"
            integration = "tests.mocks.integrations.ExampleWeatherService"

        config = WeatherConfig("tests", tests_module)
        config.ready()

        service = (await get_all_integrations())["weather"]
        assert isinstance(service, ExampleWeatherService)

    def test_ready_warns_and_registers_nothing_without_an_integration_set(self, caplog):
        import tests as tests_module
        from django_ai_sdk.integrations.apps import IntegrationAppConfig

        class EmptyConfig(IntegrationAppConfig):
            name = "tests"

        config = EmptyConfig("tests", tests_module)
        config.ready()

        assert "no `integration` set" in caplog.text


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

    async def test_a_hand_rolled_service_satisfies_the_contract(self):
        class CustomBackendService(Integration):
            name = "custom"
            label = "Custom"

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                return ["custom-tool"]

            async def get_status(self, user=None, assistant=None):
                return IntegrationStatus.ACTIVE

        instance = CustomBackendService()
        register(instance)

        resolved = (await get_all_integrations())["custom"]
        assert resolved is instance
        assert await resolved.get_tools() == ["custom-tool"]
        assert resolved.kind == "api"  # the contract's default, no MCP assumptions

    async def test_assistant_get_tools_threads_assistant_into_factory(self):
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

        register(ModelAwareIntegration())

        assistant = FakeAssistant()
        await assistant._get_integration_tools()

        assert received["assistant"] is assistant
        assert received["assistant"].model == "gpt-fake"


class TestIntegrationFailureIsolation:
    """A broken or slow integration must not affect another's tools, and must not
    serialize with it either — the guarantee _get_integration_tools makes via
    asyncio.gather plus a per-integration try/except."""

    async def test_one_failing_integration_does_not_drop_others_tools(self):
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

        register(BrokenIntegration())
        register(HealthyIntegration())

        assert await FakeAssistant()._get_integration_tools() == ["healthy-tool"]

    async def test_integrations_are_awaited_concurrently_not_serially(self):
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

        register(SlowIntegration())
        register(OtherSlowIntegration())

        start = time.monotonic()
        tools = await FakeAssistant()._get_integration_tools()
        elapsed = time.monotonic() - start

        assert set(tools) == {"slow-tool", "other-slow-tool"}
        assert elapsed < delay * 2

    async def test_an_unpermitted_integration_contributes_no_tools(self):
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

        register(ForbiddenIntegration())

        assert await FakeAssistant()._get_integration_tools() == []


class TestIntegrationToolNamespacing:
    """Two MCP servers can define the same tool name (GitHub and Linear both have
    ``list_issues``) — nothing prevents it. Haystack requires unique names across
    everything handed to one agent, so without namespacing this would fail assistant
    construction outright as soon as both were enabled together."""

    @dataclass
    class FakeTool:
        name: str

    async def test_same_named_tools_from_two_integrations_do_not_collide(self):
        from django_ai_sdk.assistant import Assistant

        class FirstIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "first"
            tools = [
                lambda **kwargs: TestIntegrationToolNamespacing.FakeTool(
                    name="list_issues"
                )
            ]

        class SecondIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "second"
            tools = [
                lambda **kwargs: TestIntegrationToolNamespacing.FakeTool(
                    name="list_issues"
                )
            ]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["first", "second"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        register(FirstIntegration())
        register(SecondIntegration())

        tools = await FakeAssistant()._get_integration_tools()

        assert {t.name for t in tools} == {"first_list_issues", "second_list_issues"}


class TestIntegrationHint:
    """A tool's own description says what it does, not what this deployment's
    instance of it actually contains -- Integration.hint fills that gap by getting
    prepended to every tool's description at the same point tools get namespaced."""

    @dataclass
    class FakeToolWithDescription:
        name: str
        description: str

    async def test_hint_is_prepended_to_every_tool_description(self):
        from django_ai_sdk.assistant import Assistant

        class HintedIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "notion"
            hint = "Contains our company wiki and HR docs"
            tools = [
                lambda **kwargs: TestIntegrationHint.FakeToolWithDescription(
                    name="search", description="Search pages"
                )
            ]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["notion"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        register(HintedIntegration())

        tools = await FakeAssistant()._get_integration_tools()

        assert len(tools) == 1
        assert tools[0].description == (
            "Search pages\n\nContains our company wiki and HR docs"
        )

    async def test_no_hint_leaves_the_description_untouched(self):
        from django_ai_sdk.assistant import Assistant

        class UnhintedIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "plain"
            tools = [
                lambda **kwargs: TestIntegrationHint.FakeToolWithDescription(
                    name="search", description="Search pages"
                )
            ]

        class FakeAssistant(Assistant):
            name = "Fake"
            description = ""
            model = "gpt-fake"
            integrations = ["plain"]

            async def get_pipeline_adapter(self, thread_id=None, user=None):
                raise NotImplementedError

        register(UnhintedIntegration())

        tools = await FakeAssistant()._get_integration_tools()

        assert tools[0].description == "Search pages"


@pytest.mark.django_db
class TestOAuthTokenRefresh:
    """refresh_oauth_token() goes through Authlib's AsyncOAuth2Client. These cover the
    HTTP exchange plus the optimistic-concurrency guard against a concurrent refresh
    (another request, or an overlapping refresh_integrations run) landing first."""

    async def _make_token(
        self, user, server_name="notion", refresh_token="old-refresh"
    ):
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken

        token_obj = MCPOAuthToken(user=user, server_name=server_name)
        token_obj.set_tokens(
            {
                "access_token": "old-access",
                "refresh_token": refresh_token,
                "expires_in": -10,
            }
        )
        await token_obj.asave()
        return token_obj

    async def test_refresh_success_persists_new_tokens(self, monkeypatch):
        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user)

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(
            monkeypatch,
            _mock_transport(
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                }
            ),
        )

        config = OAuthMCPIntegrationConfig(
            url="https://mcp.example.com",
            client_id="client-1",
            client_secret="secret-1",
        )
        result = await refresh_oauth_token(token_obj, config)

        assert result is not None
        assert result.get_access_token() == "new-access"
        assert result.get_refresh_token() == "new-refresh"

        stored = await MCPOAuthToken.objects.aget(pk=token_obj.pk)
        assert stored.get_access_token() == "new-access"

    async def test_refresh_returns_none_on_oauth_error(self, monkeypatch):
        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user)

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(
            monkeypatch, _mock_transport({"error": "invalid_grant"}, status_code=400)
        )

        config = OAuthMCPIntegrationConfig(
            url="https://mcp.example.com",
            client_id="client-1",
            client_secret="secret-1",
        )
        result = await refresh_oauth_token(token_obj, config)

        assert result is None
        stored = await MCPOAuthToken.objects.aget(pk=token_obj.pk)
        assert stored.get_access_token() == "old-access"  # untouched by the failure

    async def test_refresh_without_a_refresh_token_gives_up(self, monkeypatch):
        """Nothing to exchange — this must fail fast rather than call the token
        endpoint with an empty grant."""
        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user, refresh_token="")

        def explodes(request):
            raise AssertionError(
                "must not call the token endpoint without a refresh token"
            )

        _patch_discovery(monkeypatch)
        _patch_oauth_transport(monkeypatch, httpx.MockTransport(explodes))

        config = OAuthMCPIntegrationConfig(url="https://mcp.example.com", client_id="c")
        assert await refresh_oauth_token(token_obj, config) is None

    async def test_refresh_loses_race_reloads_winner_instead_of_clobbering_it(
        self, monkeypatch
    ):
        """By the time our refresh response comes back, a concurrent refresh has already
        rotated the row's refresh_token out from under us. Our conditional update must
        no-op, and the caller must get the winner's tokens — not silently overwrite
        them with our own."""
        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        token_obj = await self._make_token(user)

        winner = MCPOAuthToken(user=user, server_name=token_obj.server_name)
        winner.set_tokens(
            {
                "access_token": "winner-access",
                "refresh_token": "winner-refresh",
                "expires_in": 3600,
            }
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
            url="https://mcp.example.com",
            client_id="client-1",
            client_secret="secret-1",
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
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                }
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
    """The browser-facing half of MCP OAuth: ``IntegrationService.connect()`` (via
    the generic ``POST /{name}/connect``) hands the user off to the provider,
    ``oauth_callback`` brings them back and persists the token. There is no dedicated
    "start" URL — see ``TestNoStartUrl`` below.

    This path had no coverage at all before, which matters more now that it is shared
    SDK infrastructure rather than one project's local code — every consumer inherits
    whatever is (or isn't) verified here.
    """

    @staticmethod
    def _request(user, path="/api/integrations/notion/connect", **params):
        from django.contrib.sessions.backends.db import SessionStore
        from django.test import RequestFactory

        request = RequestFactory().get(path, data=params)
        request.user = user
        # RequestFactory skips middleware, so `auser` — which AuthenticationMiddleware
        # attaches and every async view here awaits — is missing without this.
        request.auser = lambda: _coroutine(user)
        request.session = SessionStore()
        return request

    @staticmethod
    def _oauth_integration(name="notion"):
        return DynamicMCPIntegration(
            name,
            OAuthMCPIntegrationConfig(
                url="https://mcp.example.com/mcp",
                client_id="client-1",
                client_secret="secret-1",
            ),
        )

    async def test_connect_redirects_to_the_provider_and_stores_pkce_state(
        self, monkeypatch
    ):
        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.services import IntegrationService
        from tests.factories.db import UserFactory

        register(self._oauth_integration())
        _patch_discovery(monkeypatch)

        async def fake_register(*args, **kwargs):
            return "client-1", "secret-1"

        monkeypatch.setattr(
            "django_ai_sdk.integrations.mcp.services.get_or_register_client",
            fake_register,
        )

        user = await UserFactory.acreate()
        request = self._request(user)
        result = await IntegrationService.connect(
            "notion",
            user,
            request=request,
            redirect_uri="https://app.example.com/api/integrations/oauth/notion/callback/",
        )

        assert result["redirect_url"].startswith("https://auth.example.com/authorize?")
        # PKCE state must be held server-side; the callback compares against it.
        assert request.session[loader_module._K_STATE.format("notion")]
        assert request.session[loader_module._K_VERIFIER.format("notion")]

    async def test_connect_rejects_an_unknown_server(self):
        from django_ai_sdk.integrations.services import IntegrationService
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()

        result = await IntegrationService.connect(
            "nope",
            user,
            request=self._request(user),
            redirect_uri="https://app.example.com/cb",
        )

        assert result is None

    async def test_callback_exchanges_the_code_and_persists_the_token(
        self, settings, monkeypatch
    ):
        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
        from tests.factories.db import UserFactory

        register(self._oauth_integration())
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

    async def test_callback_clears_a_status_cached_before_the_token_existed(
        self, settings, monkeypatch
    ):
        """Connecting must take effect immediately, not after the cache TTL.

        The settings page checks status before the user connects, which caches
        "no tools" for them. Without an invalidation on callback that entry
        outlives the handshake by up to AI_SDK_INTEGRATION_CACHE_TTL (900s), so
        the assistant keeps treating a freshly connected server as unconfigured.
        """
        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views
        from tests.factories.db import UserFactory

        integration = self._oauth_integration()
        register(integration)
        settings.AI_SDK_MCP_OAUTH_SUCCESS_URL = "/settings/integrations"
        _patch_oauth_transport(
            monkeypatch,
            _mock_transport(
                {"access_token": "fresh-access", "refresh_token": "fresh-refresh"}
            ),
        )

        user = await UserFactory.acreate()
        key = integration._cache_key(user)

        fetches = 0

        async def fetch():
            nonlocal fetches
            fetches += 1
            return []

        # Stand in for the pre-connect status check, then prove it is cached.
        await integration._cache.get(key, fetch)
        await integration._cache.get(key, fetch)
        assert fetches == 1

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

        # The stale entry is gone, so the next read goes back to the server.
        await integration._cache.get(key, fetch)
        assert fetches == 2

    async def test_callback_rejects_a_mismatched_state(self, monkeypatch):
        """CSRF protection for the OAuth handshake: a code arriving with someone else's
        (or a forged) state must never be exchanged."""
        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
        from tests.factories.db import UserFactory

        register(self._oauth_integration())

        def explodes(request):
            raise AssertionError(
                "must not reach the token endpoint on a state mismatch"
            )

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
        self, monkeypatch
    ):
        from django_ai_sdk.integrations.mcp import oauth_views
        from tests.factories.db import UserFactory

        register(self._oauth_integration())

        def explodes(request):
            raise AssertionError(
                "must not reach the token endpoint after a provider error"
            )

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

    async def test_callback_refuses_an_off_site_success_redirect(
        self, settings, monkeypatch
    ):
        """A misconfigured AI_SDK_MCP_OAUTH_SUCCESS_URL must not turn the callback into
        an open redirect — it falls back to "/"."""
        from django_ai_sdk.integrations.mcp import loader as loader_module
        from django_ai_sdk.integrations.mcp import oauth_views
        from tests.factories.db import UserFactory

        register(self._oauth_integration())
        settings.AI_SDK_MCP_OAUTH_SUCCESS_URL = "https://evil.example.com/steal"
        _patch_oauth_transport(
            monkeypatch,
            _mock_transport({"access_token": "fresh-access", "expires_in": 3600}),
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


class TestNoStartUrl:
    """There is no ``oauth-start`` URL: the generic router's ``POST /{name}/connect``
    covers it. Only the callback must stay a fixed URL, since the identity provider is
    the one redirecting the browser there."""

    def test_oauth_start_no_longer_resolves(self):
        from django.urls import NoReverseMatch, reverse

        with pytest.raises(NoReverseMatch):
            reverse("integrations_mcp:oauth-start", kwargs={"server_name": "notion"})

    def test_oauth_callback_still_resolves(self):
        from django.urls import reverse

        assert reverse(
            "integrations_mcp:oauth-callback", kwargs={"server_name": "notion"}
        )


@pytest.mark.django_db
class TestIntegrationService:
    """The facade views.py delegates to — mirrors AssistantService's shape (resolve by
    name, permission-check, delegate to the instance)."""

    async def test_list_for_user_drops_unpermitted_rows(self):
        from django_ai_sdk.integrations.services import IntegrationService
        from tests.factories.db import UserFactory

        class ForbiddenIntegration(APIIntegration):
            name = "forbidden"
            tools = []

            async def has_perms(self, user, operation=None, *, raise_on_deny=False):
                return False

        class VisibleIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "visible"
            tools = []

        register(ForbiddenIntegration())
        register(VisibleIntegration())
        user = await UserFactory.acreate()

        rows = await IntegrationService.list_for_user(user)

        assert [r.name for r in rows] == ["visible"]

    async def test_list_for_user_isolates_a_broken_integration(self):
        """One integration's get_status() raising must not drop the others, and must
        report DEGRADED for itself rather than propagating."""
        from django_ai_sdk.integrations.base import IntegrationStatus
        from django_ai_sdk.integrations.services import IntegrationService
        from tests.factories.db import UserFactory

        class BrokenIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "broken"
            tools = []

            async def get_status(self, user=None, assistant=None):
                raise RuntimeError("upstream is down")

        class HealthyIntegration(APIIntegration):
            permissions = [AllowAll]
            name = "healthy"
            tools = []

        register(BrokenIntegration())
        register(HealthyIntegration())
        user = await UserFactory.acreate()

        rows = {r.name: r for r in await IntegrationService.list_for_user(user)}

        assert rows["broken"].status == IntegrationStatus.DEGRADED
        assert rows["healthy"].status == IntegrationStatus.ACTIVE

    async def test_connect_raises_permission_denied_without_manage_perm(self):
        from django_ai_sdk.integrations.services import IntegrationService
        from django_ai_sdk.permissions import PermissionDenied
        from tests.factories.db import UserFactory

        class OnlyUsable(APIIntegration):
            name = "only-usable"
            tools = []
            supports_connect = True

            async def has_perms(self, user, operation=None, *, raise_on_deny=False):
                from django_ai_sdk.permissions import Operation

                return operation == Operation.USE_INTEGRATION

        register(OnlyUsable())
        user = await UserFactory.acreate()

        with pytest.raises(PermissionDenied):
            await IntegrationService.connect(
                "only-usable",
                user,
                request=None,
                redirect_uri="https://app.example.com/cb",
            )

    async def test_connect_returns_none_for_an_unknown_integration(self):
        from django_ai_sdk.integrations.services import IntegrationService
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()

        result = await IntegrationService.connect(
            "nope", user, request=None, redirect_uri="https://app.example.com/cb"
        )

        assert result is None

    async def test_disconnect_and_reconnect_return_none_for_an_unknown_integration(
        self,
    ):
        from django_ai_sdk.integrations.services import IntegrationService
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()

        assert await IntegrationService.disconnect("nope", user) is None
        assert await IntegrationService.reconnect("nope", user) is None


class TestIntegrationPermissions:
    async def test_no_user_gets_no_integration_tools_by_default(self):
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

        register(DefaultPermsIntegration())

        assert await FakeAssistant()._get_integration_tools(user=None) == []
