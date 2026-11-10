"""W001, the switches app's check: runtime agents nothing guards until someone switches them."""

from __future__ import annotations

from typing import Any

from django.core import checks
from django.test import override_settings

from django_ai_sdk.oversight.switches.checks import check_switches

BASES = {"AI_SDK_RUNTIME_AGENT_BASES": ["django_ai_sdk.agents.runtime.RuntimeAgent"]}


def ids(**settings: Any) -> set[str]:
    from django_ai_sdk.agents.config import get_runtime_agent_bases

    with override_settings(**settings):
        if hasattr(get_runtime_agent_bases, "cache_clear"):
            get_runtime_agent_bases.cache_clear()
        return {message.id for message in check_switches()}


def test_unguarded_runtime_agents_are_reported():
    assert ids(**BASES) == {"ai_sdk.oversight.switches.W001"}


def test_a_default_policy_covers_them():
    assert ids(**BASES, AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "sw-parent"}) == set()


def test_a_project_without_runtime_agents_hears_nothing():
    assert ids(AI_SDK_RUNTIME_AGENT_BASES=[]) == set()


def test_nor_does_one_with_oversight_off():
    assert ids(**BASES, AI_SDK_OVERSIGHT={"ENABLED": False}) == set()


def test_the_app_registers_it():
    assert check_switches in checks.registry.registry.get_checks()
