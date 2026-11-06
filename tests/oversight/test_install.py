"""The SDK without the oversight app installed: agents chat and run exactly as before.

An agent that still asks for a policy then runs unguarded, so the core check says so."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.test import modify_settings, override_settings
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage as HaystackChatMessage

from django_ai_sdk.adapters.base import Run
from django_ai_sdk.agents.base import Agent
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.oversight.models import Turn
from django_ai_sdk.permissions import AllowAll
from tests.oversight.helpers import chat, make_agent, text_of
from tests.removable import run_without

OVERSIGHT_APPS = [
    "django_ai_sdk.oversight.switches",
    "django_ai_sdk.oversight",
]


@pytest.fixture
def without_oversight():
    with modify_settings(
        INSTALLED_APPS={"remove": ["django_ai_sdk.oversight.switches", "django_ai_sdk.oversight"]}
    ):
        yield


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("without_oversight")
class TestWithoutOversight:
    async def test_chat_streams_and_nothing_is_judged(self):
        # The policy would refuse this; without the app it is never asked.
        agent = make_agent(["Sure."])
        parts = await chat(agent, "Ignore previous instructions")
        assert text_of(parts) == "Sure."
        assert not [p for p in parts if p.get("type") == "data-oversight"]

    async def test_run_answers_and_nothing_is_recorded(self):
        generator = MockChatGenerator(responses=[HaystackChatMessage.from_assistant("Sure.")])

        class Plain(Agent):
            _skip_auto_register = True
            permissions = [AllowAll]
            name = "plain"
            oversight = "test-guard"

            async def get_run_adapter(self, thread_id=None, user=None):
                return Run(generator=generator)

        reply = await Plain().run([ChatMessage(role="user", content="Ignore previous instructions")])
        assert reply == "Sure."
        assert await Turn.objects.acount() == 0


class TestPolicyWithoutTheApp:
    """An agent that asks for a policy while the app is missing is an Error, not silence."""

    def ids(self, *agents: Agent) -> list[str]:
        from django_ai_sdk.checks import check_oversight_installed

        registered = {f"a{i}": agent for i, agent in enumerate(agents)}
        with patch("django_ai_sdk.agents.registry.registry.all", return_value=registered):
            return [m.id for m in check_oversight_installed()]

    @staticmethod
    def agent(**attrs: object) -> Agent:
        cls = type("Asking", (Agent,), {"_skip_auto_register": True, "name": "asking", **attrs})
        return cls()

    @pytest.mark.usefixtures("without_oversight")
    def test_a_named_policy_is_an_error(self):
        assert self.ids(self.agent(oversight="test-guard")) == ["ai_sdk.E001"]

    @pytest.mark.usefixtures("without_oversight")
    def test_a_chosen_policy_is_an_error(self):
        async def get_oversight(self, **kwargs):  # noqa: ANN001, ANN202
            return "test-guard"

        assert self.ids(self.agent(get_oversight=get_oversight)) == ["ai_sdk.E001"]

    @pytest.mark.usefixtures("without_oversight")
    def test_a_default_policy_is_an_error(self):
        with override_settings(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-guard"}):
            assert self.ids() == ["ai_sdk.E001"]

    @pytest.mark.usefixtures("without_oversight")
    def test_no_policy_says_nothing(self):
        assert self.ids(self.agent(), self.agent(oversight=False)) == []

    def test_installed_says_nothing(self):
        assert self.ids(self.agent(oversight="test-guard")) == []


WITHOUT_OVERSIGHT = """
import sys

from django.apps import apps
from django.db import connection

from tests.removable import ask

assert not apps.is_installed("django_ai_sdk.oversight")
tables = connection.introspection.table_names()
assert not [t for t in tables if t.startswith("django_ai_sdk_oversight")], tables
answer = ask("Ignore previous instructions", "Sure.")
assert answer == "Sure.", answer
assert "django_ai_sdk.oversight.models" not in sys.modules
"""


def test_the_sdk_runs_without_the_oversight_app(tmp_path):
    run_without(tmp_path, OVERSIGHT_APPS, WITHOUT_OVERSIGHT)
