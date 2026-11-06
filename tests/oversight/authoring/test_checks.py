"""The authoring checks: its settings, and the floor every authored policy keeps."""

from __future__ import annotations

import pytest
from django.core import checks

from django_ai_sdk.oversight.authoring.checks import check_authoring, check_authoring_deploy
from tests.oversight.helpers import check_ids as ids


def test_a_fresh_install_has_nothing_to_say_until_deploy():
    assert check_authoring() == []
    assert ids(check_authoring_deploy()) == ["ai_sdk.oversight.authoring.W001"]


@pytest.mark.django_db  # Django 6.1's model checks ask the database what it supports.
def test_the_floor_warning_runs_under_check_deploy_only():
    plain = ids(checks.run_checks(include_deployment_checks=False))
    deploy = ids(checks.run_checks(include_deployment_checks=True))
    assert "ai_sdk.oversight.authoring.W001" in deploy
    assert "ai_sdk.oversight.authoring.W001" not in plain


def test_a_floor_set_is_quiet_on_deploy(settings):
    settings.AI_SDK_OVERSIGHT_AUTHORING = {"MUST_EXTEND": ["au-base"]}
    assert check_authoring_deploy() == []


def test_a_floor_naming_no_code_policy_is_an_error(settings):
    settings.AI_SDK_OVERSIGHT_AUTHORING = {"MUST_EXTEND": ["nope"]}
    assert ids(check_authoring()) == ["ai_sdk.oversight.authoring.E002"]


def test_unknown_keys_and_regex_lenses_are_errors(settings):
    settings.AI_SDK_OVERSIGHT_AUTHORING = {"MUST_EXTND": [], "INLINE": ["patterns"]}
    found = check_authoring()
    assert ids(found) == ["ai_sdk.oversight.authoring.E012"] * 2
    assert "patterns" in found[1].msg


def test_a_setting_that_is_not_a_dict_is_reported_not_raised(settings):
    settings.AI_SDK_OVERSIGHT_AUTHORING = ["security"]
    assert ids(check_authoring()) == ["ai_sdk.oversight.authoring.E012"]
    assert check_authoring_deploy() == []
