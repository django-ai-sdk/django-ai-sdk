"""Which policy a turn runs under, and what happens when it cannot be chosen."""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import patch

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from django_ai_sdk.agents.base import Agent
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.errors import Blocked
from django_ai_sdk.oversight import (
    Block,
)
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.types import ToolCall
from django_ai_sdk.permissions import AllowAll

from tests.oversight.helpers import (
    INJECTION,
    RunPolicy,
    agent_with,
    closed,
    closed_turn,
    log_lines,
    make_agent,
    opened,
)


def _guarded_runtime_class() -> type:
    from django_ai_sdk.agents.runtime import RuntimeAgent

    return type("GuardedRuntimeAgent", (RuntimeAgent,), {"oversight": "test-guard"})


@pytest.mark.django_db(transaction=True)
class TestOpeningATurn:
    """What start does when a read fails."""


    async def test_an_unknown_policy_refuses_by_default_and_is_recorded(self):
        agent = make_agent([], None, policy="fx-no-such-policy")
        with log_lines() as lines:
            guard = await TurnGuard.start(agent, channel="api", input="hello")
            gate = await guard.gate()
            assert gate.blocked and gate.message == str(Block.DEFAULT_MESSAGE)
            assert (await guard.tool_call(ToolCall(id="1", name="x"))) is not None
            assert (await guard.finish("an answer")).blocked
            turn = await closed_turn(guard)
        assert turn.outcome == "blocked"
        assert turn.policy == "fx-no-such-policy"
        assert turn.error == "policy could not be loaded"
        # Refused with the generic message: the log, not the person, says why.
        assert any("refuses" in line and "fx-no-such-policy" not in str(gate.message) for line in lines)

    async def test_an_unknown_policy_runs_unguarded_when_open_and_logs(self):
        agent = make_agent([], None, policy="fx-no-such-policy")
        with log_lines() as lines, opened():
            guard = await TurnGuard.start(agent, channel="api", input="hello")
        assert not guard.active
        assert any("unguarded" in line for line in lines)

    async def test_a_raising_get_oversight_follows_the_fail_mode(self):
        agent = make_agent([], None)

        async def get_oversight(*, channel, user):
            raise RuntimeError("no database")

        agent.get_oversight = get_oversight  # type: ignore[method-assign]
        with opened():
            assert not (await TurnGuard.start(agent, channel="api", input="x")).active
        with closed():
            guard = await TurnGuard.start(agent, channel="api", input="x")
            assert (await guard.gate()).blocked
            turn = await closed_turn(guard)
        assert (turn.outcome, turn.error) == ("blocked", "policy could not be chosen")

    async def test_get_oversight_returning_an_empty_name_is_refused(self):
        agent = make_agent([], None)

        async def get_oversight(*, channel, user):
            return ""

        agent.get_oversight = get_oversight  # type: ignore[method-assign]
        with closed():
            guard = await TurnGuard.start(agent, channel="api", input="x")
            assert (await guard.gate()).blocked


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestGetOversight:
    async def test_the_hook_may_read_the_database(self):
        async def choose(self, *, channel="", user=None, **kwargs):  # noqa: ANN001, ANN202
            from django.contrib.auth import get_user_model

            await get_user_model().objects.acount()
            return RunPolicy

        agent, model = agent_with("Sure.", policy=None, get_oversight=choose)
        with pytest.raises(Blocked):
            await agent.run([ChatMessage(role="user", content=INJECTION)])
        assert not model.asked


    def test_the_old_name_is_gone(self):
        assert not hasattr(Agent, "get_oversight_policy")
        assert inspect.iscoroutinefunction(Agent.get_oversight)


@pytest.mark.usefixtures("no_reflect_queue")
class TestOversightValue:
    @pytest.mark.parametrize("value", [RunPolicy(), True, "", 3])
    def test_anything_else_fails_when_the_class_is_defined(self, value: Any):
        with pytest.raises(ImproperlyConfigured, match="oversight must be"):
            type("Wrong", (Agent,), {"_skip_auto_register": True, "oversight": value})

    @pytest.mark.parametrize("value", [RunPolicy, "test-guard", False, None])
    def test_what_names_a_policy_is_accepted(self, value: Any):
        type("Right", (Agent,), {"_skip_auto_register": True, "oversight": value})

    def test_the_check_reports_a_value_set_later_instead_of_crashing(self):
        from django_ai_sdk.oversight.checks.agents import agent_checks

        cls = type("Later", (Agent,), {"_skip_auto_register": True, "permissions": [AllowAll]})
        cls.oversight = RunPolicy()  # ty: ignore[invalid-assignment]
        with patch("django_ai_sdk.agents.registry.registry.all", lambda: {"a": cls()}):
            found = agent_checks()
        assert [m.id for m in found] == ["ai_sdk.oversight.E011"]


@pytest.mark.usefixtures("no_reflect_queue")
@pytest.mark.django_db(transaction=True)
class TestRuntimeAgents:
    async def test_without_an_assignment_the_base_class_policy_applies(self):
        from django_ai_sdk.agents.models import AgentSettings

        row = await AgentSettings.objects.acreate(name="mine", slug="mine", model="m")
        guard = await TurnGuard.start(_guarded_runtime_class()(row), channel="chat", input="x")
        assert guard.active and guard.policy is not None
        assert guard.policy.name == "test-guard"

    async def test_then_the_default_policy(self):
        from django_ai_sdk.agents.models import AgentSettings
        from django_ai_sdk.agents.runtime import RuntimeAgent

        row = await AgentSettings.objects.acreate(name="mine", slug="mine", model="m")
        with override_settings(AI_SDK_OVERSIGHT={"DEFAULT_POLICY": "test-guard"}):
            guard = await TurnGuard.start(RuntimeAgent(row), channel="chat", input="x")
        assert guard.active
