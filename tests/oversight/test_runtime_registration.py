"""The seams for policies registered at runtime: loaders, withdrawing a policy, and
building or dry-running a draft before it is registered."""

from __future__ import annotations

import pytest
from django_ai_sdk.oversight import Block, L, PatternLens, Policy, Rule
from django_ai_sdk.oversight.policy import build
from django_ai_sdk.oversight.registry import get_policy_class, register_policy, unregister_policy
from django_ai_sdk.oversight.resolve import _loaders, add_policy_loader, aget_policy, invalidate
from django_ai_sdk.oversight.testing import dry_run
from django_ai_sdk.oversight.types import Outcome, Stage


def draft(name: str = "draft") -> type[Policy]:
    return type(
        "Draft",
        (Policy,),
        {
            "name": name,
            "lenses": [PatternLens(name="refund", stages=[Stage.GATE], keywords=["refund"])],
            "rules": [
                Rule("no_refunds", when=L("refund").is_true, then=Block("No."), mode="enforce")
            ],
        },
    )


class TestDrafts:
    def test_build_takes_a_class_that_is_not_registered(self, registry_snapshot):
        resolved = build("draft", policy_class=draft())
        assert set(resolved.rules) == {"no_refunds"}
        assert get_policy_class("draft") is None

    @pytest.mark.django_db(transaction=True)
    async def test_dry_run_takes_a_built_draft(self, registry_snapshot):
        result = await dry_run(build("draft", policy_class=draft()), input="I want a refund")
        assert result.outcome is Outcome.BLOCKED
        assert result.stage(Stage.GATE).message == "No."


@pytest.mark.django_db(transaction=True)
class TestLoaders:
    async def test_a_loader_runs_before_a_lookup_and_may_register(self, registry_snapshot):
        calls = []

        async def loader():
            calls.append(1)
            if get_policy_class("loaded") is None:
                register_policy(draft("loaded"))

        add_policy_loader(loader)
        try:
            resolved = await aget_policy("loaded")
        finally:
            _loaders.remove(loader)
        assert resolved is not None and "no_refunds" in resolved.rules
        assert calls

    async def test_loaders_run_at_most_once_per_runtime_cache_seconds(self, registry_snapshot):
        calls = []

        async def loader():
            calls.append(1)

        add_policy_loader(loader)
        try:
            await aget_policy("missing")
            await aget_policy("missing")
            assert len(calls) == 1
            invalidate()
            await aget_policy("missing")
            assert len(calls) == 2
        finally:
            _loaders.remove(loader)

    async def test_a_failing_loader_keeps_what_is_registered(self, registry_snapshot):
        register_policy(draft("kept"))

        async def broken():
            raise RuntimeError("database gone")

        add_policy_loader(broken)
        try:
            assert await aget_policy("kept") is not None
        finally:
            _loaders.remove(broken)

    async def test_a_withdrawn_policy_is_unknown_at_once(self, registry_snapshot):
        register_policy(draft("gone"))
        assert await aget_policy("gone") is not None
        unregister_policy("gone")
        assert get_policy_class("gone") is None
        assert await aget_policy("gone") is None


async def test_lookups_during_a_load_wait_for_it(registry_snapshot):
    """A lookup that starts while the loaders run loads too, instead of finding nothing."""
    import asyncio

    from django_ai_sdk.oversight import Policy, resolve
    from django_ai_sdk.oversight.registry import register_policy

    class Late(Policy):
        name = "rt-late"

    async def slow_loader() -> None:
        await asyncio.sleep(0.01)
        register_policy(Late)

    resolve._loaders.append(slow_loader)
    resolve._loaded_at = float("-inf")
    try:
        found = await asyncio.gather(resolve.aget_policy("rt-late"), resolve.aget_policy("rt-late"))
    finally:
        resolve._loaders.remove(slow_loader)
    assert all(policy is not None for policy in found)
