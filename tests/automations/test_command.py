"""`manage.py automations_tick`: one tick, a loop of ticks, or one automation now."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from django.core.management import CommandError, call_command


class _Stop(Exception):
    pass


def test_without_loop_it_ticks_once():
    with patch("django_ai_sdk.automations.runner.tick", AsyncMock(return_value=[])) as tick:
        call_command("automations_tick")
    assert tick.await_count == 1


def test_loop_keeps_ticking_with_the_interval_between():
    # The third sleep stops the loop, so two intervals ran between three ticks.
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise _Stop

    with (
        patch("django_ai_sdk.automations.runner.tick", AsyncMock(return_value=[])) as tick,
        patch("time.sleep", sleep),
        pytest.raises(_Stop),
    ):
        call_command("automations_tick", "--loop", "60")
    assert tick.await_count == 3
    assert sleeps == [60, 60, 60]


def test_a_name_runs_that_automation_now():
    with patch("django_ai_sdk.automations.runner.dispatch_now", AsyncMock(return_value=None)) as run:
        call_command("automations_tick", "digest")
    run.assert_awaited_once_with("digest")


def test_an_unknown_name_is_a_command_error():
    with (
        patch("django_ai_sdk.automations.runner.dispatch_now", AsyncMock(side_effect=ValueError("no"))),
        pytest.raises(CommandError),
    ):
        call_command("automations_tick", "nope")
