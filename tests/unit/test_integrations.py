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


class TestResilientCache:
    async def test_cache_hit_returns_immediately(self):
        cache = ResilientCache(
            ttl=60, timeout=5, cb_threshold=3, cb_cooldown=60, cb_max_cooldown=60
        )
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
        cache = ResilientCache(
            ttl=0.01, timeout=5, cb_threshold=3, cb_cooldown=60, cb_max_cooldown=60
        )
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.2)  # slow refresh — must not block the caller
            return [f"tool-{calls}"]

        first = await cache.get("k", fetch)
        assert first == ["tool-1"]

        await asyncio.sleep(0.02)  # let the entry go stale

        start = time.monotonic()
        second = await cache.get("k", fetch)
        elapsed = time.monotonic() - start

        assert second == ["tool-1"]  # stale value served immediately, not the new one
        assert elapsed < 0.05  # nowhere near the 0.2s refresh — never blocked on it

        await asyncio.sleep(0.3)  # let the background refresh finish
        third = await cache.get("k", fetch)
        assert third == ["tool-2"]  # now warmed by the background refresh

    async def test_cache_miss_bounded_by_timeout(self):
        cache = ResilientCache(
            ttl=60, timeout=0.05, cb_threshold=3, cb_cooldown=60, cb_max_cooldown=60
        )

        async def hangs_forever():
            await asyncio.sleep(10)
            return ["never"]

        start = time.monotonic()
        result = await cache.get("k", hangs_forever)
        elapsed = time.monotonic() - start

        assert result == []  # degrades to empty rather than hanging
        assert elapsed < 0.5  # bounded by `timeout`, not the 10s fetch

    async def test_circuit_breaker_opens_after_threshold_and_stops_retrying(self):
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=2, cb_cooldown=60, cb_max_cooldown=60
        )
        calls = 0

        async def always_fails():
            nonlocal calls
            calls += 1
            raise RuntimeError("dead server")

        await cache.get("k", always_fails)  # failure 1 — below cb_threshold, circuit still closed
        assert cache.status_for("k") == IntegrationStatus.DEGRADED  # but status is truthful

        await cache.get("k", always_fails)  # failure 2 — threshold reached, circuit opens
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

        result = await cache.get("k", always_fails)  # circuit open — must not call fetch
        assert result == []
        assert calls == 2  # third get() didn't attempt another live fetch

    async def test_single_failure_immediately_reports_degraded_not_active(self):
        """A wrong/invalid token must never show as ACTIVE just because the
        consecutive-failure count hasn't reached cb_threshold yet — that's the bug
        this cache is designed to avoid: status reflects the last real attempt, not
        an optimistic default."""
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=3, cb_cooldown=60, cb_max_cooldown=60
        )

        async def wrong_token():
            raise RuntimeError("401 Unauthorized")

        await cache.get("k", wrong_token)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

    def test_never_attempted_key_defaults_to_active(self):
        """Documents the contract: status_for() alone can't distinguish "never
        checked" from "healthy" — callers (e.g. MCPIntegration.get_status()) must
        force a real attempt via get() first if they want a truthful answer."""
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=3, cb_cooldown=60, cb_max_cooldown=60
        )
        assert cache.status_for("never-checked") == IntegrationStatus.ACTIVE

    async def test_circuit_breaker_resets_on_success(self):
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=2, cb_cooldown=60, cb_max_cooldown=60
        )

        async def fails():
            raise RuntimeError("dead")

        async def succeeds():
            return ["ok"]

        await cache.get("k", fails)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED

        cache.invalidate("k")  # simulate an explicit disconnect/reset
        result = await cache.get("k", succeeds)
        assert result == ["ok"]
        assert cache.status_for("k") == IntegrationStatus.ACTIVE


class TestResilientCacheBackoffAndBroken:
    """A persistently broken integration must stop being probed forever at a flat
    interval: cooldown should grow with repeated failure, and eventually give up
    entirely (BROKEN) rather than retry at the capped interval indefinitely."""

    async def test_cooldown_doubles_on_each_repeated_failure_up_to_the_cap(self):
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=1, cb_cooldown=10, cb_max_cooldown=45
        )

        async def fails():
            raise RuntimeError("dead")

        await cache.get("k", fails)  # opens at level 0 -> cooldown 10s
        assert cache._circuits["k"].open_until - time.monotonic() == pytest.approx(10, abs=1)

        cache._circuits["k"].open_until = 0  # simulate cooldown having elapsed
        await cache.get("k", fails)  # level 1 -> cooldown 20s
        assert cache._circuits["k"].open_until - time.monotonic() == pytest.approx(20, abs=1)

        cache._circuits["k"].open_until = 0
        await cache.get("k", fails)  # level 2 -> cooldown 40s, still under the 45s cap
        assert cache._circuits["k"].open_until - time.monotonic() == pytest.approx(40, abs=1)

        cache._circuits["k"].open_until = 0
        await cache.get("k", fails)  # level 3 -> would be 80s, capped at 45s
        assert cache._circuits["k"].open_until - time.monotonic() == pytest.approx(45, abs=1)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED  # still auto-retrying

    async def test_gives_up_and_reports_broken_after_one_full_cycle_at_the_cap(self):
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=1, cb_cooldown=10, cb_max_cooldown=10
        )

        async def fails():
            raise RuntimeError("dead")

        await cache.get("k", fails)  # first failure immediately hits the cap (cooldown == max)
        assert cache.status_for("k") == IntegrationStatus.DEGRADED  # one cap cycle still allowed

        cache._circuits["k"].open_until = 0  # let that one capped cooldown elapse
        result = await cache.get("k", fails)  # fails again while already at the cap -> give up
        assert result == []
        assert cache.status_for("k") == IntegrationStatus.BROKEN

    async def test_broken_key_is_never_probed_again_without_invalidate(self):
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=1, cb_cooldown=10, cb_max_cooldown=10
        )
        calls = 0

        async def fails():
            nonlocal calls
            calls += 1
            raise RuntimeError("dead")

        await cache.get("k", fails)
        cache._circuits["k"].open_until = 0
        await cache.get("k", fails)
        assert cache.status_for("k") == IntegrationStatus.BROKEN
        assert calls == 2

        cache._circuits["k"].open_until = 0  # even if "cooldown" elapses, BROKEN doesn't expire
        result = await cache.get("k", fails)
        assert result == []
        assert calls == 2  # no third attempt — broken keys are never auto-probed

    async def test_invalidate_recovers_a_broken_key(self):
        cache = ResilientCache(
            ttl=60, timeout=1, cb_threshold=1, cb_cooldown=10, cb_max_cooldown=10
        )

        async def fails():
            raise RuntimeError("dead")

        async def succeeds():
            return ["ok"]

        await cache.get("k", fails)
        cache._circuits["k"].open_until = 0
        await cache.get("k", fails)
        assert cache.status_for("k") == IntegrationStatus.BROKEN

        cache.invalidate("k")  # the manual "reconnect" action
        result = await cache.get("k", succeeds)
        assert result == ["ok"]
        assert cache.status_for("k") == IntegrationStatus.ACTIVE


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

    async def test_reconnect_recovers_a_broken_integration(self, settings, monkeypatch):
        """reconnect() is the manual way out of BROKEN — e.g. after fixing a wrong
        URL, something has to call this before auto-retry will ever try again."""
        from django_ai_sdk.integrations.mcp import loader as loader_module

        settings.AI_SDK_INTEGRATION_CB_THRESHOLD = 1
        settings.AI_SDK_INTEGRATION_CB_COOLDOWN = 10
        settings.AI_SDK_INTEGRATION_CB_MAX_COOLDOWN = 10

        async def fails(*args, **kwargs):
            raise RuntimeError("dead")

        monkeypatch.setattr(loader_module, "_connect", fails)

        integration = MCPIntegration(
            "linear-broken", TokenMCPIntegrationConfig(url="https://example.com/mcp", token="bad")
        )

        await integration.get_status()
        integration._cache._circuits["linear-broken"].open_until = (
            0  # let the capped cooldown elapse
        )
        assert await integration.get_status() == IntegrationStatus.BROKEN

        async def succeeds(*args, **kwargs):
            return ["tool"]

        monkeypatch.setattr(loader_module, "_connect", succeeds)
        await integration.reconnect()
        assert await integration.get_status() == IntegrationStatus.ACTIVE


class TestAPIIntegrationGetStatus:
    """A hand-written API integration must report real health too — a down
    backend shows up as DEGRADED/BROKEN, not a hardcoded ACTIVE."""

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

    async def test_reconnect_recovers_a_broken_health_check(self, settings):
        settings.AI_SDK_INTEGRATION_CB_THRESHOLD = 1
        settings.AI_SDK_INTEGRATION_CB_COOLDOWN = 10
        settings.AI_SDK_INTEGRATION_CB_MAX_COOLDOWN = 10

        should_fail = True

        async def probe():
            if should_fail:
                raise RuntimeError("dead")

        class RecoverableIntegration(APIIntegration):
            name = "recoverable"
            tools = []
            health_check = staticmethod(probe)

        integration = RecoverableIntegration()
        await integration.get_status()
        integration._cache._circuits["recoverable"].open_until = 0  # let the cooldown elapse
        assert await integration.get_status() == IntegrationStatus.BROKEN

        should_fail = False
        await integration.reconnect()
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
