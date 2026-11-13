"""Turns no policy opens: an inert guard, and an empty policy name."""

from __future__ import annotations


import pytest
from django_ai_sdk.oversight.plugin.turn import open_turn

from tests.oversight.helpers import make_agent, opened


@pytest.mark.django_db(transaction=True)
class TestTurn:
    async def test_an_inert_guard_gates_to_the_question_itself(self):
        from django_ai_sdk.common import ChatMessage
        from django_ai_sdk.oversight.guard import TurnGuard

        agent = make_agent([], None, policy="")
        messages = [ChatMessage(role="user", content="hi"), ChatMessage(role="user", content="go")]
        inert = await TurnGuard.start(agent, channel="api", messages=messages)
        assert not inert.active and (await inert.gate()).input == "go"
        with opened():
            unknown = await TurnGuard.start(agent, channel="api", input="hi", oversight="fc-none")
        assert not unknown.active and (await unknown.gate()).input == "hi"

    async def test_an_empty_policy_name_opens_no_turn_at_any_layer(self):
        from django_ai_sdk.oversight.guard import TurnGuard

        agent = make_agent([], None)
        assert not (await TurnGuard.start(agent, channel="api", input="hi", oversight="")).active
        _, opened_own = await open_turn(agent, channel="api", input="hi", oversight="")
        assert not opened_own
