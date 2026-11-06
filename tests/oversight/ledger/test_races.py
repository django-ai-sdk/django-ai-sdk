"""The chain under writers at once: turns, reads, forgetting and checkpoints, each on its own
connection. One gapless sequence, one tip, and a chain that verifies.

SQLite serialises every writer, so these need PostgreSQL: set `DATABASE_URL`
(see tests/settings.py).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from asgiref.sync import async_to_sync
from django.db import transaction
from django.test import override_settings

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.content import load_content
from django_ai_sdk.oversight.guard import TurnGuard
from django_ai_sdk.oversight.ledger import chain
from django_ai_sdk.oversight.ledger.checkpoints import checkpoint_matches_chain, write_checkpoint
from django_ai_sdk.oversight.ledger.models import Checkpoint, LedgerEntry, LedgerHead
from django_ai_sdk.oversight.models import AccessRecord, Turn, TurnContent
from django_ai_sdk.oversight.services import OversightService
from tests.oversight.helpers import make_agent, race
from tests.oversight.review.helpers import METRICS, reader

WRITERS = 6
TURNS = 4


pytestmark = pytest.mark.usefixtures("postgres_only")


@pytest.fixture(autouse=True)
def _quiet_close():
    """Patched once for every thread: patching inside a thread is not thread safe."""
    with (
        patch("django_ai_sdk.oversight.persist.enqueue_reflect"),
        patch("django_ai_sdk.oversight.persist.enqueue_notifications"),
        patch("django_ai_sdk.oversight.persist.enqueue_notifications"),
        override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}),
    ):
        yield


def turn(thread_id: str = "", text: str = "What about my salary?") -> str:
    async def run() -> str:
        guard = await TurnGuard.start(
            make_agent([], None, policy="test-guard"),
            channel="api",
            input=text,
            thread_id=thread_id or None,
        )
        await guard.gate()
        await guard.finish("Ask jan@example.com.")
        turn_id = await guard.close()
        assert turn_id is not None
        return turn_id

    return async_to_sync(run)()


def gapless() -> list[int]:
    seqs = list(LedgerEntry.objects.order_by("seq").values_list("seq", flat=True))
    assert seqs == list(range(1, len(seqs) + 1))
    head = LedgerHead.objects.get(id=1)
    assert head.seq == len(seqs)
    assert head.entry_hash == LedgerEntry.objects.get(seq=head.seq).entry_hash
    return seqs


@pytest.mark.django_db(transaction=True)
class TestChainRacesOnPostgres:
    def test_appenders_at_once_make_one_gapless_chain(self):
        def append() -> int:
            for n in range(10):
                with transaction.atomic():
                    record = AccessRecord.objects.create(
                        operation="list", reader_id="r", turn_ids=[], query={"n": n}
                    )
                    chain.append_many([("access", record)])
            return 10

        assert sum(race(*[append for _ in range(WRITERS)])) == 10 * WRITERS
        assert len(gapless()) == 10 * WRITERS
        result = chain.verify()
        assert result.ok, result.problems

    def test_turns_reads_and_checkpoints_at_once_verify(self):
        user = async_to_sync(reader)(METRICS)
        async_to_sync(OversightService.list_turns)(user=user)  # the permission cache warms up

        def turns(thread: str) -> list[str]:
            return [turn(thread) for _ in range(TURNS)]

        def reads() -> int:
            for _ in range(TURNS):
                async_to_sync(OversightService.list_turns)(user=user)
            return TURNS

        def checkpoints() -> int:
            return sum(1 for _ in range(TURNS) if write_checkpoint())

        works = [lambda n=n: turns(f"t-{n}") for n in range(WRITERS)]
        results = race(*works, reads, checkpoints)

        turn_ids = [t for ids in results[:WRITERS] for t in ids]
        assert Turn.objects.filter(id__in=turn_ids).count() == WRITERS * TURNS
        gapless()
        # Written open at the gate and again at close: two entries for every turn.
        assert LedgerEntry.objects.filter(kind="turn").count() == 2 * WRITERS * TURNS
        result = chain.verify()
        assert result.ok, result.problems
        # Every checkpoint written while others appended still matches the chain.
        assert all(checkpoint_matches_chain(c) for c in Checkpoint.objects.all())

    def test_forgetting_a_thread_while_it_takes_turns_leaves_nothing_readable(self):
        # The thread's key exists before the race, as it does once a thread has spoken.
        turn("t-forget")

        def talk() -> list[str]:
            return [turn("t-forget", "my salary again") for _ in range(TURNS)]

        def forget() -> bool:
            custody.get().forget("thread:t-forget")
            return True

        results = race(talk, talk, forget)

        later = Turn.objects.filter(id__in=results[0] + results[1]).select_related("content")
        assert later.count() == 2 * TURNS
        # Whether a turn wrote before or after the shred, nothing of the thread opens.
        assert all(load_content(t) is None for t in later)
        assert not any(
            "salary" in str(stored)
            for stored in TurnContent.objects.filter(scope="thread:t-forget").values_list(
                "stored", flat=True
            )
        )
        gapless()
        result = chain.verify()
        assert result.ok, result.problems
        assert LedgerEntry.objects.filter(kind="forget").count() == 1
