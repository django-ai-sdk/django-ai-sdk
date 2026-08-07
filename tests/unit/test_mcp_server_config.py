"""DB-declared MCP servers: ``MCPServerConfig`` and how the registry merges them in
alongside code-declared (INSTALLED_APPS) integrations.

Three things matter here: a row builds the same ``DynamicMCPIntegration`` a
code-declared server would (so nothing downstream needs to know the difference), an
installed app always wins on a name collision, and a built integration is cached
across calls so its ResilientCache/circuit-breaker state survives -- otherwise every
request would look like the server's first-ever contact.
"""

from __future__ import annotations

import pytest
from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.mcp.loader import DynamicMCPIntegration
from django_ai_sdk.integrations.mcp.models import MCPServerConfig
from django_ai_sdk.integrations.registry import (
    get_all_integrations,
    get_integrations,
    invalidate_db_rows_cache,
    register,
    reset_registry,
)

from django.db import connection
from django.test.utils import CaptureQueriesContext

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class TestToIntegration:
    async def test_builds_a_dynamic_mcp_integration(self):
        row = await MCPServerConfig.objects.acreate(
            name="zendesk-1",
            label="Zendesk",
            hint="Support tickets and macros",
            url="https://zendesk.example.com/mcp",
            auth="static",
        )

        integration = row.to_integration()

        assert isinstance(integration, DynamicMCPIntegration)
        assert integration.name == "zendesk-1"
        assert integration.label == "Zendesk"
        assert integration.hint == "Support tickets and macros"

    async def test_reports_needs_setup_instead_of_raising_on_a_bad_row(self):
        row = await MCPServerConfig.objects.acreate(
            name="broken-1", url="https://x", auth="token", token=""
        )

        integration = row.to_integration()

        assert integration._needs_setup is not None


class TestRegistryMerge:
    async def test_enabled_db_row_is_returned(self):
        await MCPServerConfig.objects.acreate(
            name="zendesk-2", url="https://x", auth="static"
        )

        integrations = await get_all_integrations()

        assert "zendesk-2" in integrations
        assert isinstance(integrations["zendesk-2"], DynamicMCPIntegration)

    async def test_disabled_db_row_is_not_returned(self):
        await MCPServerConfig.objects.acreate(
            name="zendesk-3", url="https://x", auth="static", enabled=False
        )

        assert "zendesk-3" not in await get_all_integrations()

    async def test_get_integrations_resolves_db_declared_names_too(self):
        await MCPServerConfig.objects.acreate(
            name="zendesk-4", url="https://x", auth="static"
        )

        assert list(await get_integrations(["zendesk-4", "nope"])) == ["zendesk-4"]

    async def test_installed_app_wins_over_a_same_named_db_row(self):
        class CodeZendesk(APIIntegration):
            name = "zendesk-5"
            tools = []

        code_instance = CodeZendesk()
        register(code_instance)
        await MCPServerConfig.objects.acreate(
            name="zendesk-5", url="https://x", auth="static"
        )

        integrations = await get_all_integrations()

        assert integrations["zendesk-5"] is code_instance

    async def test_warns_once_when_an_enabled_row_is_shadowed(self, caplog):
        """An enabled row that an installed app already claims is never used --
        without a warning that's a silent no-op an operator has no way to explain."""

        class CodeShadow(APIIntegration):
            name = "zendesk-5b"
            tools = []

        register(CodeShadow())
        await MCPServerConfig.objects.acreate(
            name="zendesk-5b", url="https://x", auth="static", enabled=True
        )

        with caplog.at_level("WARNING"):
            await get_all_integrations()
            await get_all_integrations()  # second call must not warn again

        shadow_warnings = [r for r in caplog.records if "zendesk-5b" in r.message]
        assert len(shadow_warnings) == 1
        assert "already registers an integration" in shadow_warnings[0].message

    async def test_does_not_warn_when_the_shadowed_row_is_disabled(self, caplog):
        """A disabled row shadowed by an installed app is unremarkable -- it's off
        on purpose, not silently failing to take effect."""

        class CodeShadowOff(APIIntegration):
            name = "zendesk-5c"
            tools = []

        register(CodeShadowOff())
        await MCPServerConfig.objects.acreate(
            name="zendesk-5c", url="https://x", auth="static", enabled=False
        )

        with caplog.at_level("WARNING"):
            await get_all_integrations()

        assert not [r for r in caplog.records if "zendesk-5c" in r.message]

    async def test_built_integration_is_cached_across_calls(self):
        await MCPServerConfig.objects.acreate(
            name="zendesk-6", url="https://x", auth="static"
        )

        first = (await get_all_integrations())["zendesk-6"]
        second = (await get_all_integrations())["zendesk-6"]

        assert first is second

    async def test_cache_is_invalidated_when_the_row_changes(self):
        row = await MCPServerConfig.objects.acreate(
            name="zendesk-7", url="https://x", auth="static"
        )

        first = (await get_all_integrations())["zendesk-7"]
        row.label = "Renamed"
        await row.asave()
        second = (await get_all_integrations())["zendesk-7"]

        assert first is not second
        assert second.label == "Renamed"


class TestRowsListCache:
    """The enabled-rows list itself is cached (see registry._rows_cache), separate
    from the per-row built-integration cache above -- otherwise every chat request
    would run a SELECT just to check whether anything changed."""

    async def test_second_call_reuses_the_cached_rows_list(self):
        row = await MCPServerConfig.objects.acreate(
            name="zendesk-8", url="https://x", auth="static"
        )
        await get_all_integrations()  # primes the rows-list cache

        # .aupdate() is a queryset UPDATE, not save() -- it sends no post_save signal,
        # simulating a change this process's cache can't know about (another process,
        # a bulk update). If the second call actually re-queried, it would see this.
        await MCPServerConfig.objects.filter(pk=row.pk).aupdate(enabled=False)

        assert "zendesk-8" in await get_all_integrations()

        invalidate_db_rows_cache()

        assert "zendesk-8" not in await get_all_integrations()

    async def test_saving_a_row_invalidates_the_cache_immediately(self):
        """mcp.apps connects a post_save signal for exactly this -- a row enabled/
        disabled/added in this process must be visible on the very next call, not
        after AI_SDK_MCP_SERVER_LIST_CACHE_TTL elapses."""
        row = await MCPServerConfig.objects.acreate(
            name="zendesk-9", url="https://x", auth="static", enabled=False
        )
        assert "zendesk-9" not in await get_all_integrations()

        row.enabled = True
        await row.asave()

        assert "zendesk-9" in await get_all_integrations()

    async def test_deleting_a_row_invalidates_the_cache_immediately(self):
        row = await MCPServerConfig.objects.acreate(
            name="zendesk-10", url="https://x", auth="static"
        )
        assert "zendesk-10" in await get_all_integrations()

        await row.adelete()

        assert "zendesk-10" not in await get_all_integrations()


class TestUrlValidation:
    """save() rejects a url pointing at a loopback/private/link-local/metadata
    address, since this app's own process is what fetches it (see
    mcp.models._validate_public_url)."""

    async def test_rejects_a_loopback_url(self):
        with pytest.raises(ValueError):
            await MCPServerConfig.objects.acreate(
                name="ssrf-1", url="http://127.0.0.1:8000/mcp", auth="static"
            )

    async def test_rejects_localhost(self):
        with pytest.raises(ValueError):
            await MCPServerConfig.objects.acreate(
                name="ssrf-2", url="http://localhost/mcp", auth="static"
            )

    async def test_rejects_a_private_ip(self):
        with pytest.raises(ValueError):
            await MCPServerConfig.objects.acreate(
                name="ssrf-3", url="http://10.0.0.5/mcp", auth="static"
            )

    async def test_rejects_the_cloud_metadata_ip(self):
        with pytest.raises(ValueError):
            await MCPServerConfig.objects.acreate(
                name="ssrf-4",
                url="http://169.254.169.254/latest/meta-data/",
                auth="static",
            )

    async def test_allows_a_public_hostname(self):
        row = await MCPServerConfig.objects.acreate(
            name="ssrf-5", url="https://zendesk.example.com/mcp", auth="static"
        )
        assert row.pk is not None
