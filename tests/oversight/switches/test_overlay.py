"""What turns read of the switches: within RUNTIME_CACHE_SECONDS, and the last good read."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.db import OperationalError
from django.test import override_settings

from django_ai_sdk.oversight.resolve import aassigned_policy, aget_policy, invalidate
from django_ai_sdk.oversight.switches.models import Switch
from django_ai_sdk.oversight.types import Mode


async def append(**key: str) -> None:
    """Switch as another process would: in the database, no cache forgotten here."""
    value = key.pop("value")
    version = await Switch.objects.filter(**key).acount() + 1
    await Switch.objects.acreate(**key, value=value, version=version)


async def switch_rule(mode: str) -> None:
    await append(kind="rule_mode", policy="sw-parent", target="block_x", value=mode)


async def assign(agent_id: str, policy: str) -> None:
    await append(kind="agent_policy", target=agent_id, value=policy)


def age(cache: dict, key: str, seconds: float) -> None:
    """Make a cached resolution `seconds` older."""
    read_at, *rest = cache[key]
    cache[key] = (read_at - seconds, *rest)


async def mode() -> Mode:
    resolved = await aget_policy("sw-parent")
    assert resolved is not None
    return resolved.rules["block_x"].mode


@pytest.mark.django_db(transaction=True)
class TestCache:
    async def test_a_switch_from_another_process_applies_within_the_cache_time(self):
        from django_ai_sdk.oversight import resolve

        assert await mode() is Mode.REPORT
        await switch_rule("enforce")
        assert await mode() is Mode.REPORT
        age(resolve._cache, "sw-parent", 9)
        assert await mode() is Mode.REPORT
        age(resolve._cache, "sw-parent", 2)
        assert await mode() is Mode.ENFORCE

    async def test_agent_choices_follow_the_same_time(self, database_overlay):
        from django_ai_sdk.oversight import resolve

        assert await aassigned_policy("a-1") is None
        await assign("a-1", "sw-child")
        assert await aassigned_policy("a-1") is None
        resolve._agents_read_at -= 11
        assert await aassigned_policy("a-1") == "sw-child"
        await assign("a-1", "")
        invalidate()  # what the switching process does
        assert await aassigned_policy("a-1") == ""

    @override_settings(AI_SDK_OVERSIGHT={"RUNTIME_CACHE_SECONDS": 0})
    async def test_zero_reads_the_switches_on_every_turn(self):
        assert await mode() is Mode.REPORT
        await switch_rule("enforce")
        assert await mode() is Mode.ENFORCE


@pytest.fixture
def no_cache():
    with override_settings(AI_SDK_OVERSIGHT={"RUNTIME_CACHE_SECONDS": 0}):
        yield


@pytest.mark.usefixtures("no_cache")
@pytest.mark.django_db(transaction=True)
class TestUnreadable:
    async def test_a_failed_read_keeps_the_last_good_resolution(self):
        await switch_rule("enforce")
        assert await mode() is Mode.ENFORCE
        with patch.object(Switch.objects, "filter", side_effect=OperationalError("db down")):
            assert await mode() is Mode.ENFORCE

    async def test_a_failed_read_keeps_the_agents_choices(self):
        await assign("a-1", "sw-child")
        assert await aassigned_policy("a-1") == "sw-child"
        with patch.object(Switch.objects, "filter", side_effect=OperationalError("db down")):
            assert await aassigned_policy("a-1") == "sw-child"

    async def test_with_nothing_read_yet_code_decides_until_a_read_succeeds(self):
        await assign("a-1", "sw-child")
        with patch.object(Switch.objects, "filter", side_effect=OperationalError("db down")):
            assert await aassigned_policy("a-1") is None
        assert await aassigned_policy("a-1") == "sw-child"
