"""Custody: core keeps content as it is, forgets it by deleting it, and chains nothing."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from django.test import override_settings

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.checks import check_oversight_deploy
from django_ai_sdk.oversight.content import load_content
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.models import LensAnswer, RuleMatch, Turn, TurnContent
from tests.oversight.helpers import (
    make_agent,
)

pytestmark = pytest.mark.usefixtures("no_reflect_queue")


class Recording:
    """Custody that remembers what it was handed, and can refuse."""

    def __init__(self, *, forgotten: bool = False, chain_fails: bool = False) -> None:
        self.forgotten = forgotten
        self.chain_fails = chain_fails
        self.chained_kinds: list[str] = []
        self.forgot: list[str] = []

    def actor(self, user_pk: str) -> str:
        return user_pk

    def store(self, scope: str, content: dict[str, Any]) -> dict[str, Any]:
        if self.forgotten:
            raise custody.Forgotten(scope)
        return {"wrapped": content}

    def load_many(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any] | None]:
        return [stored["wrapped"] for _scope, stored in items]

    def forget(self, scope: str) -> None:
        self.forgot.append(scope)

    def chained(self, items: list[tuple[str, Any]]) -> None:
        if self.chain_fails:
            raise RuntimeError("chain refused")
        self.chained_kinds.extend(kind for kind, _ in items)

    def pruning(self, turn_ids: Any, *, access_ids: Any = ()) -> None:
        return None

    def expiring(self, content_ids: Any) -> None:
        return None


@contextlib.contextmanager
def installed(keeper: custody.Custody):
    custody.install(keeper)
    try:
        yield keeper
    finally:
        custody.install(None)


async def close_a_turn(text: str = "What about my salary?", thread_id: str | None = None) -> str:
    guard = await TurnGuard.start(
        make_agent([], None), channel="api", input=text, thread_id=thread_id
    )
    await guard.gate()
    await guard.finish("It is fine.")
    return await guard.close()


def capture_ids() -> list[str]:
    # The test settings run tasks inline (W014): only what capture says counts here.
    return [m.id for m in check_oversight_deploy() if m.id != "ai_sdk.oversight.W014"]


@pytest.mark.django_db(transaction=True)
class TestPlainCustody:
    async def test_content_is_kept_as_it_is(self):
        assert custody.is_plain()
        turn_id = await close_a_turn()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        assert turn.content is not None
        assert turn.content.stored["input"] == "What about my salary?"
        assert turn.content.scope == f"turn:{turn_id}"

    async def test_forgetting_a_scope_deletes_its_content_and_keeps_the_turn(self):
        turn_id = await close_a_turn(thread_id="t-1")
        assert await TurnContent.objects.filter(scope="thread:t-1").aexists()
        from asgiref.sync import sync_to_async

        await sync_to_async(custody.get().forget)("thread:t-1")
        assert not await TurnContent.objects.filter(scope="thread:t-1").aexists()
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.content_id is None
        assert await RuleMatch.objects.filter(turn=turn).aexists()


@pytest.mark.django_db(transaction=True)
class TestAnotherCustody:
    async def test_content_goes_through_it_both_ways(self):
        with installed(Recording()) as keeper:
            turn_id = await close_a_turn()
        turn = await Turn.objects.select_related("content").aget(id=turn_id)
        assert set(turn.content.stored) == {"wrapped"}
        assert keeper.chained_kinds[0] == "turn"
        assert {"answer", "match"} <= set(keeper.chained_kinds)
        with installed(keeper):
            from asgiref.sync import sync_to_async

            opened = await sync_to_async(load_content)(turn)
        assert opened["input"] == "What about my salary?"

    async def test_a_forgotten_scope_keeps_the_turn_without_content(self):
        with installed(Recording(forgotten=True)):
            turn_id = await close_a_turn()
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.content_id is None
        assert await LensAnswer.objects.filter(turn=turn).aexists()
        assert not await TurnContent.objects.aexists()

    async def test_a_chain_that_refuses_writes_nothing(self):
        with installed(Recording(chain_fails=True)):
            await close_a_turn()
        assert not await Turn.objects.aexists()
        assert not await TurnContent.objects.aexists()

    async def test_a_deleted_thread_is_handed_to_custody(self):
        from django_ai_sdk.conversation.models import Thread

        thread = await Thread.objects.acreate()
        thread_id = str(thread.id)
        with installed(Recording()) as keeper:
            await close_a_turn(thread_id=thread_id)
            await thread.adelete()
        assert keeper.forgot == [f"thread:{thread_id}"]


class TestInstall:
    def test_a_custody_missing_a_method_is_refused(self):
        class Behind(Recording):
            load_many = None
            expiring = None

        with pytest.raises(TypeError, match="lacks load_many, expiring"):
            custody.install(Behind())
        assert custody.is_plain()

    def test_the_check_names_every_protocol_method(self):
        assert set(custody.METHODS) == {
            "actor",
            "store",
            "load_many",
            "forget",
            "chained",
            "pruning",
            "expiring",
        }
        assert set(custody.METHODS) <= set(vars(custody.PlainCustody))

    def test_actor_names_are_custodys_name_and_the_primary_key(self):
        class Renaming(Recording):
            def actor(self, user_pk: str) -> str:
                return f"u-{user_pk}" if user_pk else ""

        assert custody.actor_names("7") == {"7"}
        assert custody.actor_names("") == set()
        with installed(Renaming()):
            assert custody.actor_names("7") == {"7", "u-7"}
            assert custody.actor_names("") == set()


class TestDeployCheck:
    def test_full_capture_in_the_clear_warns(self):
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            assert capture_ids() == ["ai_sdk.oversight.W001"]
            with installed(Recording()):
                assert capture_ids() == []

    def test_masked_capture_says_nothing(self):
        assert capture_ids() == []
