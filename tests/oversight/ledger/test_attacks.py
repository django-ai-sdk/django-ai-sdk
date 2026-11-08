"""What someone who can write the database tries, and what catches it."""

from __future__ import annotations

import base64
from io import StringIO

import pytest
from asgiref.sync import sync_to_async
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction

from django_ai_sdk.oversight.ledger import chain, checkpoints
from django_ai_sdk.oversight.ledger.models import Checkpoint, LedgerEntry, LedgerHead
from django_ai_sdk.oversight.ledger.services import verify_all
from django_ai_sdk.oversight.models import AccessRecord, LensAnswer, Turn, TurnContent
from tests.oversight.ledger.test_ledger import turn_in, verified


async def problems() -> set[str]:
    return {p["problem"] for p in (await verified()).problems}


def record() -> AccessRecord:
    with transaction.atomic():
        made = AccessRecord.objects.create(operation="list")
        chain.append_many([("access", made)])
    return made


@pytest.mark.django_db(transaction=True)
class TestTheContentLink:
    async def test_content_moved_to_another_turn_is_caught(self):
        thread = "t-swap"
        first, second = await turn_in(thread), await turn_in(thread)
        a = await Turn.objects.aget(id=first)
        b = await Turn.objects.aget(id=second)
        await Turn.objects.filter(id=first).aupdate(content_id=None)
        await Turn.objects.filter(id=second).aupdate(content_id=a.content_id)
        await Turn.objects.filter(id=first).aupdate(content_id=b.content_id)
        assert "record altered" in await problems()

    async def test_content_put_in_by_hand_is_caught(self):
        turn_id = await turn_in()
        fake = await TurnContent.objects.acreate(
            scope=f"turn:{turn_id}", capture="full", stored={"input": "FAKE", "output": "FAKE"}
        )
        await Turn.objects.filter(id=turn_id).aupdate(content_id=fake.pk)
        found = await problems()
        assert {"record not chained", "record altered"} <= found


@pytest.mark.django_db(transaction=True)
class TestRowsBehindItsBack:
    async def test_an_inserted_reviewer_answer_is_caught(self):
        turn_id = await turn_in()
        await LensAnswer.objects.acreate(
            turn_id=turn_id, lens="pii_out", stage="reflect", answerer="reviewer", value=False
        )
        assert "record not chained" in await problems()

    async def test_an_appended_turn_entry_is_caught(self):
        """Rewriting a closed turn by chaining it once more."""
        turn_id = await turn_in()
        await Turn.objects.filter(id=turn_id).aupdate(outcome="blocked")

        def append() -> None:
            with transaction.atomic():
                chain.append_many([(chain.TURN, Turn.objects.get(id=turn_id))])

        await sync_to_async(append)()
        assert "turn chained too often" in await problems()


@pytest.mark.django_db
class TestCheckpointsHoldHistory:
    def rehash(self) -> None:
        """Change the first entry and recompute every link after it, head included."""
        previous = "0" * 64
        for entry in LedgerEntry.objects.order_by("seq"):
            payload = "f" * 64 if entry.seq == 1 else entry.payload_hash
            digest = chain._entry_digest(entry.seq, entry.kind, entry.ref, payload, previous)
            LedgerEntry.objects.filter(seq=entry.seq).update(
                payload_hash=payload, prev_hash=previous, entry_hash=digest
            )
            previous = digest
        LedgerHead.objects.filter(id=1).update(entry_hash=previous)

    def test_no_checkpoint_is_written_over_a_rewritten_chain(self, settings):
        settings.AI_SDK_LEDGER = {"ORIGIN": "test/log"}
        record()
        checkpoints.write_checkpoint()
        record()
        self.rehash()
        with pytest.raises(checkpoints.ChainTruncated, match="no longer matches"):
            checkpoints.write_checkpoint()

    def test_an_older_checkpoint_shows_the_rewrite_under_a_newer_one(self, settings):
        settings.AI_SDK_LEDGER = {"ORIGIN": "test/log"}
        record()
        checkpoints.write_checkpoint()
        record()
        self.rehash()
        # A newer checkpoint, written as if over the rewritten chain.
        root = checkpoints.chain_roots([2])[2]
        Checkpoint.objects.create(tree_size=2, root_hash=root.hex(), note="x")
        out = verify_all()
        assert (out.ok, out.checkpoint_ok) == (False, False)
        assert [p["tree_size"] for p in out.problems if p["problem"] == "checkpoint mismatch"] == [1]

    def test_a_published_note_catches_deleted_checkpoint_rows(self, settings, tmp_path):
        settings.AI_SDK_LEDGER = {"ORIGIN": "test/log"}
        record()
        published = tmp_path / "checkpoint.txt"
        published.write_text(checkpoints.write_checkpoint().note)
        record()
        self.rehash()
        Checkpoint.objects.all().delete()
        with pytest.raises(CommandError):
            call_command("ledger_verify", "--checkpoint", str(published), stdout=StringIO())

    def test_with_a_verifier_the_newest_checkpoint_must_be_signed(self, settings):
        key = base64.b64encode(b"s" * 32).decode()
        settings.AI_SDK_LEDGER = {"ORIGIN": "test/log"}
        record()
        checkpoints.write_checkpoint()  # unsigned
        verifier = checkpoints.verifier_key("test/log", key)
        with pytest.raises(CommandError):
            call_command("ledger_verify", "--verifier", verifier, stdout=StringIO())


def test_prefix_roots_match_the_tree_hash_at_every_size():
    leaves = [bytes([n]) * 32 for n in range(1, 20)]
    roots = checkpoints.prefix_roots(leaves, range(20))
    assert all(roots[n] == checkpoints.merkle_root(leaves[:n]) for n in range(20))


OLD_SECRET = "o" * 32
NEW_SECRET = "n" * 32


@pytest.mark.django_db(transaction=True)
async def test_erasure_finds_turns_named_under_a_former_pseudonym_secret(settings):
    from django_ai_sdk.oversight.signals import forget_actor
    from tests.oversight.review.helpers import a_turn, reader

    settings.AI_SDK_LEDGER = {"PSEUDONYM_SECRET": OLD_SECRET}
    user = await reader()
    turn_id = await a_turn(user=user)
    settings.AI_SDK_LEDGER = {"PSEUDONYM_SECRET": [NEW_SECRET, OLD_SECRET]}

    assert await sync_to_async(forget_actor)(str(user.pk)) == 1
    assert (await Turn.objects.aget(id=turn_id)).actor_id == ""


@pytest.mark.django_db(transaction=True)
async def test_a_thread_deleted_before_its_first_turn_is_written_stays_forgotten():
    from django_ai_sdk.oversight.signals import forget_deleted_thread
    from django_ai_sdk.vault import is_shredded

    class Thread:
        pk = "t-early"

    await sync_to_async(forget_deleted_thread)(None, Thread())
    assert await sync_to_async(is_shredded)("thread:t-early")
