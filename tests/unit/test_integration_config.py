from __future__ import annotations

import pytest
from django_ai_sdk.integrations.api.base import APIIntegration
from django_ai_sdk.integrations.config import configured_names, get_integration_config


class ExampleIntegration(APIIntegration):
    name = "github"
    tools = []


class TestGetIntegrationConfig:
    def test_returns_the_slice_for_this_integration(self, settings):
        settings.AI_SDK_INTEGRATIONS = {
            "github": {"TOKEN": "ghp_secret"},
            "linear": {"TOKEN": "lin_secret"},
        }

        assert get_integration_config("github") == {"TOKEN": "ghp_secret"}

    def test_upper_cases_keys_so_either_style_resolves(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"linear": {"token": "lin", "Tools": ["x"]}}

        assert get_integration_config("linear") == {"TOKEN": "lin", "TOOLS": ["x"]}

    def test_empty_when_the_setting_is_missing(self, settings):
        del settings.AI_SDK_INTEGRATIONS

        assert get_integration_config("github") == {}

    def test_empty_when_this_integration_is_unconfigured(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"linear": {"TOKEN": "lin"}}

        assert get_integration_config("github") == {}

    @pytest.mark.parametrize("bad", [["github"], "github", 7])
    def test_a_malformed_setting_degrades_instead_of_raising(self, settings, caplog, bad):
        settings.AI_SDK_INTEGRATIONS = bad

        assert get_integration_config("github") == {}
        assert "AI_SDK_INTEGRATIONS" in caplog.text

    def test_a_malformed_entry_degrades_instead_of_raising(self, settings, caplog):
        settings.AI_SDK_INTEGRATIONS = {"github": "ghp_secret"}

        assert get_integration_config("github") == {}
        assert "github" in caplog.text


class TestConfiguredNames:
    def test_lists_configured_entries(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"github": {}, "linear": {"TOKEN": "x"}}

        assert configured_names() == {"github", "linear"}

    @pytest.mark.parametrize("bad", [None, ["github"], "github"])
    def test_empty_for_a_missing_or_malformed_setting(self, settings, bad):
        settings.AI_SDK_INTEGRATIONS = bad

        assert configured_names() == set()


class TestIntegrationSecret:
    def test_reads_the_configured_value(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"github": {"TOKEN": "ghp_secret"}}

        assert ExampleIntegration().secret("token") == "ghp_secret"

    def test_returns_the_default_when_unset(self, settings):
        settings.AI_SDK_INTEGRATIONS = {}
        integration = ExampleIntegration()

        assert integration.secret("token") == ""
        assert integration.secret("token", default="fallback") == "fallback"

    def test_an_empty_value_falls_back_to_the_default(self, settings):
        settings.AI_SDK_INTEGRATIONS = {"github": {"TOKEN": ""}}

        assert ExampleIntegration().secret("token", default="fallback") == "fallback"

    def test_a_non_string_value_is_refused_not_stringified(self, settings, caplog):
        settings.AI_SDK_INTEGRATIONS = {"github": {"TOKEN": ["ghp_a", "ghp_b"]}}

        assert ExampleIntegration().secret("token") == ""
        assert "TOKEN" in caplog.text

    def test_a_refused_value_is_never_logged(self, settings, caplog):
        settings.AI_SDK_INTEGRATIONS = {"github": {"TOKEN": ["ghp_supersecret"]}}

        ExampleIntegration().secret("token")

        assert "ghp_supersecret" not in caplog.text

    def test_ignores_the_environment(self, settings, monkeypatch):
        settings.AI_SDK_INTEGRATIONS = {}
        monkeypatch.setenv("AI_SDK_GITHUB_TOKEN", "not-used")
        monkeypatch.setenv("GITHUB_TOKEN", "ci-runner-token")

        assert ExampleIntegration().secret("token") == ""
