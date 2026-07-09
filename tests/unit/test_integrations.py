"""Unit tests for the Integration abstraction: ResilientCache latency-safety
machinery, MCPIntegration cache-key isolation, and APIIntegration's
health_check-driven status reporting.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.base import Integration, IntegrationStatus, ResilientCache
from django_ai_sdk.integrations.mcp.loader import MCPIntegration
from django_ai_sdk.integrations.mcp.schemas import (
    OAuthMCPIntegrationConfig,
    StaticMCPIntegrationConfig,
    TokenMCPIntegrationConfig,
)
from django_ai_sdk.integrations.registry import _build

import httpx


def _mock_transport(token_response: dict, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=token_response)

    return httpx.MockTransport(handler)


def _patch_oauth_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    """Inject a fake token-endpoint transport into every AsyncOAuth2Client this
    module builds, whether called from loader.refresh_oauth_token directly or
    from services.exchange_token (which imports build_oauth_client locally, so
    patching the loader's copy covers both)."""
    from django_ai_sdk.integrations.mcp import loader as loader_module

    original = loader_module.build_oauth_client

    def wrapper(client_id, client_secret, **kwargs):
        kwargs.setdefault("transport", transport)
        return original(client_id, client_secret, **kwargs)

    monkeypatch.setattr(loader_module, "build_oauth_client", wrapper)


def _patch_discovery(monkeypatch, token_endpoint: str = "https://auth.example.com/token") -> None:
    import django_ai_sdk.integrations.mcp.discovery as discovery_module

    async def fake_discover(*args, **kwargs):
        return discovery_module.OAuthDiscovery(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint=token_endpoint,
        )

    monkeypatch.setattr(discovery_module, "discover", fake_discover)


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

        # Each failure is truthfully reported as DEGRADED while the breaker is still closed.
        for _ in range(3):
            await cache.get("k", always_fails)
            assert cache.status_for("k") == IntegrationStatus.DEGRADED

        # Breaker is now open — further gets short-circuit to empty without a live fetch.
        result = await cache.get("k", always_fails)
        assert result == []
        assert calls == 3  # the 4th get() didn't attempt another live fetch
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

    async def test_single_failure_immediately_reports_degraded_not_active(self):
        """A wrong/invalid token must never show as ACTIVE — status reflects the last
        real attempt, not an optimistic default, even before the breaker trips."""
        cache = ResilientCache(ttl=60, timeout=1)

        async def wrong_token():
            raise RuntimeError("401 Unauthorized")

        await cache.get("k", wrong_token)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

    def test_never_attempted_key_defaults_to_active(self):
        """Documents the contract: status_for() alone can't distinguish "never
        checked" from "healthy" — callers (e.g. MCPIntegration.get_status()) must
        force a real attempt via get() first if they want a truthful answer."""
        cache = ResilientCache(ttl=60, timeout=1)
        assert cache.status_for("never-checked") == IntegrationStatus.ACTIVE

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
        """invalidate() ("retry now") must drop a previously cached *successful*
        value, not just reset the breaker — otherwise reconnect keeps serving stale."""
        cache = ResilientCache(ttl=60, timeout=1)
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return [f"tools-{calls}"]

        assert await cache.get("k", fetch) == ["tools-1"]
        assert await cache.get("k", fetch) == ["tools-1"]  # served from cache, no re-fetch
        assert calls == 1

        await cache.invalidate("k")
        assert await cache.get("k", fetch) == ["tools-2"]  # cache cleared -> real re-fetch
        assert calls == 2


class TestResilientCacheRecovery:
    """The breaker recovers on its own via half-open probing after the cooldown —
    there is no terminal give-up state (that bespoke BROKEN behaviour was removed in
    favour of the standard circuit-breaker lifecycle)."""

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
        integration = MCPIntegration(
            "notion", OAuthMCPIntegrationConfig(url="https://example.com/mcp")
        )
        user_a = type("U", (), {"pk": 1})()
        user_b = type("U", (), {"pk": 2})()

        key_a = integration._cache_key(user_a)
        key_b = integration._cache_key(user_b)

        assert key_a != key_b
        assert key_a == ("notion", 1)
        assert integration._cache_key(None) is None  # no user — nothing to key on


class TestIntegrationDisplayMetadata:
    """`kind`/`get_tool_names()` must answer without any live I/O — callers like
    AssistantService.get_integration_status() rely on that to stay cheap."""

    async def test_mcp_integration_kind_reflects_config_type(self):
        static = MCPIntegration("linear", TokenMCPIntegrationConfig(url="https://x", token="t"))
        oauth = MCPIntegration("notion", OAuthMCPIntegrationConfig(url="https://x"))

        assert static.kind == "token"
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
        """Auto-discovery (no `tools` configured) has no static name list to read —
        must fall back to the real, cached discovered tools instead of reporting
        none at all (the bug: with no allow-list, tools were invisible to the
        assistant-info UI's Integrations grouping and leaked into the flat Tools
        list instead)."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        calls = 0

        async def fake_connect(*args, **kwargs):
            nonlocal calls
            calls += 1
            return [
                type("T", (), {"name": "list_issues"})(),
                type("T", (), {"name": "get_issue"})(),
            ]

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
            return [type("T", (), {"name": "list_issues"})()]

        monkeypatch.setattr(loader_module, "_connect", fake_connect)

        integration = MCPIntegration(
            "linear", TokenMCPIntegrationConfig(url="https://x", token="t")
        )

        await integration.get_status()
        assert await integration.get_tool_names() == ["list_issues"]
        assert calls == 1  # second call was a cache hit, not a second live fetch

    async def test_api_integration_kind_defaults_to_api(self):
        class DummyIntegration(APIIntegration):
            tools = []

        assert DummyIntegration().kind == "api"

    async def test_api_integration_tool_names_derives_from_get_tools(self):
        class DummyIntegration(APIIntegration):
            tools = [lambda **kwargs: type("T", (), {"name": "dummy_tool"})()]

        names = await DummyIntegration().get_tool_names()
        assert names == ["dummy_tool"]

    async def test_api_integration_factory_receives_only_declared_kwargs(self):
        """A tool factory may declare any subset of user/assistant/thread_id (or
        **kwargs) — it's called with only what it accepts, never a spurious TypeError."""

        def tool(name):
            return type("T", (), {"name": name})()

        class DummyIntegration(APIIntegration):
            tools = [
                lambda user: tool("only_user"),
                lambda user, thread_id: tool("user_thread"),
                lambda **kw: tool("kwargs_all" if "assistant" in kw else "kwargs_bad"),
            ]

        names = [t.name for t in await DummyIntegration().get_tools(user="u", thread_id="t")]
        assert names == ["only_user", "user_thread", "kwargs_all"]


class TestMCPIntegrationGetStatus:
    async def test_wrong_static_token_reports_degraded_on_first_check(self, monkeypatch):
        """Reproduces the reported bug: a static/token integration whose credential
        is simply wrong must not show as connected just because get_status() was
        never exercised via a real chat turn yet."""
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
        """reconnect() is an immediate retry-now: after the underlying problem (e.g. a
        wrong token) is fixed, it clears the cached failure and re-probes without
        waiting out the breaker cooldown."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        async def fails(*args, **kwargs):
            raise RuntimeError("dead")

        monkeypatch.setattr(loader_module, "_connect", fails)

        integration = MCPIntegration(
            "linear-broken", TokenMCPIntegrationConfig(url="https://example.com/mcp", token="bad")
        )

        assert await integration.get_status() == IntegrationStatus.DEGRADED

        async def succeeds(*args, **kwargs):
            return ["tool"]

        monkeypatch.setattr(loader_module, "_connect", succeeds)
        await integration.reconnect()
        assert await integration.get_status() == IntegrationStatus.ACTIVE


class TestAPIIntegrationGetStatus:
    """A hand-written API integration must report real health too — a down
    backend shows up as DEGRADED, not a hardcoded ACTIVE."""

    async def test_no_health_check_is_always_active(self):
        class NoHealthCheckIntegration(APIIntegration):
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
            name = "healthy"
            tools = []
            health_check = staticmethod(succeeds)

        assert await HealthyIntegration().get_status() == IntegrationStatus.ACTIVE

    async def test_reconnect_recovers_a_degraded_health_check(self):
        should_fail = True

        async def probe():
            if should_fail:
                raise RuntimeError("dead")

        class RecoverableIntegration(APIIntegration):
            name = "recoverable"
            tools = []
            health_check = staticmethod(probe)

        integration = RecoverableIntegration()
        assert await integration.get_status() == IntegrationStatus.DEGRADED

        should_fail = False
        await integration.reconnect()  # retry now, don't wait out the cooldown
        assert await integration.get_status() == IntegrationStatus.ACTIVE


class TestExtensibility:
    """A third integration backend (neither MCP nor the built-in APIIntegration
    convenience base) must be pluggable with zero special-casing: implement
    Integration directly, or subclass APIIntegration with no boilerplate."""

    async def test_bare_api_integration_subclass_needs_no_init_override(self):
        class MinimalIntegration(APIIntegration):
            tools = []

        integration = MinimalIntegration()  # must not raise — no required args
        assert await integration.get_tools() == []
        assert await integration.get_status() == IntegrationStatus.ACTIVE

    def test_registry_backfills_name_from_settings_key(self):
        class UnnamedIntegration(APIIntegration):
            tools = []

        integration = _build("my-integration-key", UnnamedIntegration())
        assert integration.name == "my-integration-key"

    def test_registry_rejects_unrecognized_config_with_clear_error(self):
        with pytest.raises(ImproperlyConfigured, match="not a recognized integration"):
            _build("bogus", {"this": "is not a valid config"})

    def test_registry_accepts_a_hand_rolled_integration_instance_directly(self):
        class CustomBackendIntegration(Integration):
            name = "custom"
            label = "Custom"

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                return ["custom-tool"]

            async def get_status(self, user=None, assistant=None):
                return IntegrationStatus.ACTIVE

        instance = CustomBackendIntegration()
        integration = _build("custom", instance)
        assert integration is instance

    async def test_assistant_get_tools_threads_assistant_into_factory(self, settings):
        """A tool factory that needs to run its own LLM call (e.g. translation)
        must be able to read the calling assistant's model — this is the whole
        point of threading `assistant` through the Integration contract."""
        from django_ai_sdk.assistant import Assistant
        from django_ai_sdk.integrations.registry import reset_registry_cache

        received: dict = {}

        def factory(*, user=None, assistant=None, thread_id=""):
            received["assistant"] = assistant
            return []

        class ModelAwareIntegration(APIIntegration):
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
        reset_registry_cache()
        try:
            assistant = FakeAssistant()
            await assistant._get_integration_tools()
        finally:
            reset_registry_cache()

        assert received["assistant"] is assistant
        assert received["assistant"].model == "gpt-fake"


class TestIntegrationFailureIsolation:
    """A broken or slow integration must not affect any other configured
    integration's tools, and must not serialize with it either — this is the
    guarantee _get_integration_tools makes via asyncio.gather + a per-integration
    try/except (see django_ai_sdk.assistant.Assistant._get_integration_tools)."""

    async def test_one_failing_integration_does_not_drop_others_tools(self, settings):
        from django_ai_sdk.assistant import Assistant
        from django_ai_sdk.integrations.registry import reset_registry_cache

        class BrokenIntegration(APIIntegration):
            name = "broken"
            tools = []

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                raise RuntimeError("upstream is down")

        class HealthyIntegration(APIIntegration):
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
        reset_registry_cache()
        try:
            tools = await FakeAssistant()._get_integration_tools()
        finally:
            reset_registry_cache()

        assert tools == ["healthy-tool"]

    async def test_integrations_are_awaited_concurrently_not_serially(self, settings):
        """If a slow integration and a fast one were awaited one at a time, total
        wall-clock time would be additive. Assert it isn't."""
        from django_ai_sdk.assistant import Assistant
        from django_ai_sdk.integrations.registry import reset_registry_cache

        DELAY = 0.2

        class SlowIntegration(APIIntegration):
            name = "slow"
            tools = []

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                await asyncio.sleep(DELAY)
                return ["slow-tool"]

        class OtherSlowIntegration(APIIntegration):
            name = "other-slow"
            tools = []

            async def get_tools(self, user=None, assistant=None, thread_id=""):
                await asyncio.sleep(DELAY)
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
        reset_registry_cache()
        try:
            start = time.monotonic()
            tools = await FakeAssistant()._get_integration_tools()
            elapsed = time.monotonic() - start
        finally:
            reset_registry_cache()

        assert set(tools) == {"slow-tool", "other-slow-tool"}
        assert elapsed < DELAY * 2


@pytest.mark.django_db
class TestOAuthTokenRefresh:
    """refresh_oauth_token() goes through Authlib's AsyncOAuth2Client now. These
    cover the actual HTTP exchange plus the optimistic-concurrency guard against a
    concurrent refresh (another request, or an overlapping refresh_mcp_tokens run)
    landing first."""

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
        from django_ai_sdk.integrations.mcp.schemas import OAuthMCPIntegrationConfig

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
        from django_ai_sdk.integrations.mcp.schemas import OAuthMCPIntegrationConfig

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
        assert stored.get_access_token() == "old-access"  # untouched by the failed attempt

    async def test_refresh_loses_race_reloads_winner_instead_of_clobbering_it(self, monkeypatch):
        """By the time our refresh response comes back, a concurrent refresh has
        already rotated the DB row's refresh_token out from under us. Our own
        conditional update must then no-op, and the caller must get the winner's
        tokens back — not silently overwrite them with our own stale result."""
        from tests.factories.db import UserFactory

        from django_ai_sdk.integrations.mcp.loader import refresh_oauth_token
        from django_ai_sdk.integrations.mcp.models import MCPOAuthToken
        from django_ai_sdk.integrations.mcp.schemas import OAuthMCPIntegrationConfig

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
        # token_obj still holds the pre-race refresh_token in memory — exactly the
        # stale value a real caller would have if it read the row before the race.
        result = await refresh_oauth_token(token_obj, config)

        assert result is not None
        assert result.get_access_token() == "winner-access"


@pytest.mark.django_db
class TestExchangeToken:
    """The authorization_code exchange also goes through Authlib now."""

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
