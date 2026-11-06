"""The switch tests run on the database overlay, with policies of their own to switch."""

from __future__ import annotations

import pytest

from django_ai_sdk.oversight.registry import register_policy
from django_ai_sdk.oversight.resolve import overlay_source, set_overlay_source
from django_ai_sdk.oversight.switches.overlay import DatabaseOverlay
from tests.oversight.switches.helpers import SWITCH_POLICIES


@pytest.fixture(autouse=True)
def switch_policies(registry_snapshot):
    for policy in SWITCH_POLICIES:
        register_policy(policy)


@pytest.fixture(autouse=True)
def database_overlay():
    """A fresh database overlay, as the app installs it, and the one before it back after."""
    before = overlay_source()
    source = DatabaseOverlay()
    set_overlay_source(source)
    yield source
    set_overlay_source(before)


@pytest.fixture
def registered(monkeypatch):
    """Register an agent instance under its own id, as the agent registry would."""
    from django_ai_sdk.agents.registry import registry
    from django_ai_sdk.oversight.agents import agent_key

    def add(agent):
        monkeypatch.setitem(registry._instances, agent_key(agent), agent)
        return agent_key(agent)

    return add
