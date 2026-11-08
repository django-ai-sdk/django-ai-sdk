"""Switches read while they are written, and operators switching one rule at once.

SQLite serialises every writer, so these need PostgreSQL: set `DATABASE_URL`
(see tests/settings.py).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from asgiref.sync import async_to_sync
from django.test import override_settings

from django_ai_sdk.oversight.models import AccessRecord
from django_ai_sdk.oversight.policy import build
from django_ai_sdk.oversight.switches.models import Switch
from django_ai_sdk.oversight.switches.overlay import DatabaseOverlay
from django_ai_sdk.oversight.switches.schemas import RuleModeIn
from django_ai_sdk.oversight.switches.services import SwitchService
from django_ai_sdk.permissions import ConflictError
from tests.oversight.helpers import race
from tests.oversight.review.helpers import reader
from tests.oversight.switches.helpers import SWITCH

SWITCHES = 20
READERS = 3
OPERATORS = 6


pytestmark = pytest.mark.usefixtures("postgres_only")


def switch(user: Any, mode: str) -> Any:
    return async_to_sync(SwitchService.set_rule_mode)(
        "sw-parent", "block_x", RuleModeIn(mode=mode, reason="race"), user=user
    )


@pytest.mark.usefixtures("sent")
@pytest.mark.django_db(transaction=True)
class TestSwitchRacesOnPostgres:
    def test_a_turn_reading_while_an_operator_switches_sees_one_mode_or_the_other(self):
        user = async_to_sync(reader)(SWITCH)
        switch(user, "enforce")
        done = threading.Event()

        def write() -> int:
            try:
                for n in range(SWITCHES):
                    switch(user, "report" if n % 2 == 0 else "enforce")
                return SWITCHES
            finally:
                done.set()

        def read() -> list[str]:
            source = DatabaseOverlay()
            seen: list[str] = []
            while not done.is_set() or not seen:
                modes = async_to_sync(source.rule_modes)(["sw-parent"])["sw-parent"]
                # Always the one newest switch.
                assert list(modes) == ["block_x"], modes
                resolved = build("sw-parent", modes=modes)
                assert not resolved.rejected, resolved.rejected
                seen.append(str(resolved.rules["block_x"].mode))
            return seen

        with override_settings(AI_SDK_OVERSIGHT={"RUNTIME_CACHE_SECONDS": 0}):
            results = race(write, *[read for _ in range(READERS)])

        assert results[0] == SWITCHES
        for seen in results[1:]:
            assert set(seen) <= {"enforce", "report"}
        assert Switch.objects.order_by("-version").first().value == "enforce"
        assert AccessRecord.objects.filter(operation="switch").count() == SWITCHES + 1

    def test_operators_switching_one_rule_at_once_leave_one_switch_in_force(self):
        users = [async_to_sync(reader)(SWITCH) for _ in range(OPERATORS)]
        modes = ["enforce", "report", "off"] * (OPERATORS // 3)

        def attempt(user: Any, mode: str) -> str | None:
            try:
                switch(user, mode)
            except ConflictError:
                return None
            return mode

        results = race(*[lambda u=u, m=m: attempt(u, m) for u, m in zip(users, modes, strict=True)])

        won = [mode for mode in results if mode is not None]
        assert won, results
        # Every switch that went through is on the record, and none that did not.
        assert AccessRecord.objects.filter(operation="switch").count() == len(won)
        assert Switch.objects.count() == len(won)
