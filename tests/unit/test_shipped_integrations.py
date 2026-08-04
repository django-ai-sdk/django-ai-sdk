"""The integration apps the SDK ships, and the boot-safety guarantee they rely on.

These are the templates projects copy, so a broken one is both a bug and bad
documentation. Each app is exercised through the same path Django uses --
IntegrationAppConfig.ready() -> import_string(self.integration)() -- rather than by
importing the class directly, so a wrong dotted path in apps.py fails here.
"""

from __future__ import annotations

import pytest
from django.utils.module_loading import import_string
from django_ai_sdk.integrations.registry import get_all_integrations, reset_registry

SHIPPED = [
    ("django_ai_sdk.integrations.github.apps.GitHubConfig", "github", "token"),
    ("django_ai_sdk.integrations.linear.apps.LinearConfig", "linear", "token"),
    ("django_ai_sdk.integrations.notion.apps.NotionConfig", "notion", "oauth"),
    ("django_ai_sdk.integrations.weather.apps.WeatherConfig", "weather", None),
]


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def _run_ready(config_path):
    """Instantiate and ready() an AppConfig exactly the way Django does."""
    config_class = import_string(config_path)
    module = import_string(f"{config_class.name}.apps")
    config_class(config_class.name, module).ready()


@pytest.mark.parametrize(("config_path", "expected_name", "auth"), SHIPPED)
class TestShippedIntegrationApps:
    def test_appconfig_is_the_default_for_its_module(self, config_path, expected_name, auth):
        """IntegrationAppConfig sets default = False so importing it doesn't leave two
        candidates in the subclass's module; each shipped app must opt back in."""
        config_class = import_string(config_path)

        assert config_class.default is True
        assert config_class.name.endswith(f".{expected_name}")

    def test_app_label_is_namespaced(self, config_path, expected_name, auth):
        """An unqualified label would claim the global name "notion"/"github"/... and
        collide with a host project's own app of that name, which Django rejects
        outright at boot."""
        config_class = import_string(config_path)

        assert config_class.label == f"django_ai_sdk_{expected_name}"

    async def test_ready_registers_it_under_the_expected_name(
        self, config_path, expected_name, auth, settings
    ):
        settings.AI_SDK_INTEGRATIONS = {expected_name: {"TOKEN": "test-token"}}
        _run_ready(config_path)

        registered = await get_all_integrations()
        assert expected_name in registered
        assert registered[expected_name].name == expected_name

    def test_declares_its_auth_kind(self, config_path, expected_name, auth):
        if auth is None:
            pytest.skip("API-backed integration, no MCP auth kind")
        integration_class = import_string(import_string(config_path).integration)

        assert integration_class.auth == auth
        assert integration_class.url.startswith("https://")


class TestBootSafety:
    """A misconfigured integration must register as "needs setup", never raise out of
    ready(). An exception here is a failed deploy rather than one degraded feature."""

    async def test_a_missing_token_registers_instead_of_raising(self, settings):
        settings.AI_SDK_INTEGRATIONS = {}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        integration = (await get_all_integrations())["linear"]
        assert integration._needs_setup
        assert await integration.get_tools() == []

    async def test_a_malformed_settings_entry_does_not_break_ready(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"linear": "should-have-been-a-dict"}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        assert (await get_all_integrations())["linear"]._needs_setup


class TestPerDeploymentOverrides:
    """The class attributes are defaults, not decisions. Anything that genuinely varies
    between deployments must be reachable from settings without subclassing."""

    async def test_url_and_tools_come_from_settings_when_set(self, settings):
        settings.AI_SDK_INTEGRATIONS = {
            "linear": {
                "TOKEN": "lin",
                "URL": "https://mcp.internal.example/linear",
                "TOOLS": ["list_issues", "create_issue"],
            },
        }
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        integration = (await get_all_integrations())["linear"]
        assert not integration._needs_setup
        assert integration.config.url == "https://mcp.internal.example/linear"
        assert integration.config.tools == ["list_issues", "create_issue"]

    async def test_class_attributes_are_the_default(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"linear": {"TOKEN": "lin"}}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        integration = (await get_all_integrations())["linear"]
        assert integration.config.url == "https://mcp.linear.app/mcp"
        assert integration.config.tools == ["list_issues"]

    async def test_github_defaults_to_a_shared_token(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"github": {"TOKEN": "ghp_shared"}}
        _run_ready("django_ai_sdk.integrations.github.apps.GitHubConfig")

        integration = (await get_all_integrations())["github"]
        assert integration.kind == "token"
        assert integration.supports_connect is False
        assert integration.connect_kind is None

    async def test_github_switches_to_per_user_oauth_from_settings(self, settings):
        """Same server, same app, different credential mechanism -- a deployment
        choice, so it must not require subclassing a shipped integration."""
        settings.AI_SDK_INTEGRATIONS = {
            "github": {
                "AUTH": "oauth",
                "CLIENT_ID": "gh-client",
                "CLIENT_SECRET": "gh-secret",
            },
        }
        _run_ready("django_ai_sdk.integrations.github.apps.GitHubConfig")

        integration = (await get_all_integrations())["github"]
        assert not integration._needs_setup
        assert integration.kind == "oauth"
        # supports_connect/connect_kind are derived from the config, never declared,
        # so the Connect button appears without touching the integration class.
        assert integration.supports_connect is True
        assert integration.connect_kind == "oauth"

    async def test_an_unknown_auth_is_named_rather_than_silently_static(self, settings):
        """Falling through to "static" would look like a healthy server that simply
        never authenticates -- the worst possible failure for a typo."""
        settings.AI_SDK_INTEGRATIONS = {"github": {"AUTH": "0auth", "TOKEN": "ghp"}}
        _run_ready("django_ai_sdk.integrations.github.apps.GitHubConfig")

        integration = (await get_all_integrations())["github"]
        assert integration._needs_setup
        assert "0auth" in integration.detail
        assert await integration.get_tools() == []

    async def test_an_unrecognised_key_is_named(self, settings, caplog):
        """A silently-ignored typo reports "token must not be empty", which points at
        the wrong thing entirely."""
        settings.AI_SDK_INTEGRATIONS = {"linear": {"TOKENS": "lin"}}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        assert "TOKENS" in caplog.text
        # Still only a warning: an unknown key must not stop a deploy.
        assert "linear" in await get_all_integrations()


class TestMissingMCPApp:
    """Every MCP integration reuses the toolkit app's OAuth token models. Forgetting it
    used to defer the failure until someone called disconnect()."""

    async def test_reports_the_missing_app_instead_of_appearing_healthy(
        self, settings, monkeypatch
    ):
        from django.apps import apps as django_apps

        monkeypatch.setattr(
            django_apps, "is_installed", lambda name: name != "django_ai_sdk.integrations.mcp"
        )
        settings.AI_SDK_INTEGRATIONS = {"linear": {"TOKEN": "lin"}}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        integration = (await get_all_integrations())["linear"]
        assert integration._needs_setup
        assert "INSTALLED_APPS" in integration.detail
        assert await integration.get_tools() == []

    async def test_a_missing_secret_still_wins_the_report(self, settings, monkeypatch):
        """Two problems at once should name the credential, not the app: it's the one
        the deployer is actively working on."""
        from django.apps import apps as django_apps

        monkeypatch.setattr(
            django_apps, "is_installed", lambda name: name != "django_ai_sdk.integrations.mcp"
        )
        settings.AI_SDK_INTEGRATIONS = {}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        assert "token" in (await get_all_integrations())["linear"].detail


class TestOrphanedConfig:
    async def test_config_without_an_installed_app_is_named(self, settings, caplog):
        """The easiest mistake here, and silent before this warning: the integration
        just didn't exist, which reads like an SDK bug rather than a missing app."""
        settings.AI_SDK_INTEGRATIONS = {"zendesk": {"TOKEN": "zd"}}

        await get_all_integrations()

        assert "zendesk" in caplog.text
        assert "INSTALLED_APPS" in caplog.text

    async def test_warns_only_once_per_process(self, settings, caplog):
        """This sits on the read path, and the condition can't change without a
        restart."""
        settings.AI_SDK_INTEGRATIONS = {"zendesk": {"TOKEN": "zd"}}

        await get_all_integrations()
        await get_all_integrations()

        assert caplog.text.count("no installed app registers them") == 1

    async def test_silent_when_every_entry_has_its_app(self, settings, caplog):
        settings.AI_SDK_INTEGRATIONS = {"linear": {"TOKEN": "lin"}}
        _run_ready("django_ai_sdk.integrations.linear.apps.LinearConfig")

        await get_all_integrations()

        assert "no installed app registers them" not in caplog.text
