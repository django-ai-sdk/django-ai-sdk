"""Who asked, under the ledger: a pseudonym of at least 64 bits under its own secret, and a
reveal that needs its own permission and leaves a chained record with its reason."""

from __future__ import annotations

import pytest
from asgiref.sync import sync_to_async

from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.ledger import chain
from django_ai_sdk.oversight.ledger.models import LedgerEntry
from django_ai_sdk.oversight.ledger.privacy import find_user_by_pseudonym, is_pseudonym, pseudonym
from django_ai_sdk.oversight.ledger.services import LedgerService
from django_ai_sdk.oversight.models import AccessRecord, Turn
from django_ai_sdk.permissions import PermissionDenied
from tests.oversight.review.helpers import REVIEW, a_turn, reader

REVEAL = "django_ai_sdk_oversight_ledger.reveal_actor"
REASON = "Complaint 2026-118 names this conversation"
SECRET = "a-pseudonym-secret-of-at-least-32-characters"


class TestPseudonyms:
    def test_at_least_64_bits(self):
        name = pseudonym("42")
        assert name.startswith("u-") and len(name) == 2 + 16
        assert is_pseudonym(name)
        # 16 base32 characters: 80 bits.
        assert len(name[2:]) * 5 >= 64

    def test_stable_and_one_per_user(self):
        assert pseudonym("42") == pseudonym(42)
        assert len({pseudonym(str(pk)) for pk in range(2_000)}) == 2_000

    def test_nobody_has_none(self):
        assert pseudonym("") == pseudonym(None) == ""

    def test_its_own_secret_changes_every_pseudonym(self, settings):
        derived = pseudonym("42")
        settings.AI_SDK_OVERSIGHT_LEDGER = {"PSEUDONYM_SECRET": SECRET}
        own = pseudonym("42")
        assert own != derived
        settings.SECRET_KEY = "another-secret-key-entirely-for-this-test"
        assert pseudonym("42") == own


@pytest.mark.django_db(transaction=True)
class TestOnTheRecord:
    async def test_a_turn_names_who_asked_by_pseudonym(self):
        asker = await reader()
        turn_id = await a_turn(user=asker)
        turn = await Turn.objects.aget(id=turn_id)
        assert turn.actor_id == pseudonym(asker.pk)
        assert str(asker.pk) != turn.actor_id

    async def test_find_takes_a_pseudonym_back_to_its_user(self):
        asker = await reader()
        await reader()
        assert (await sync_to_async(find_user_by_pseudonym)(pseudonym(asker.pk))).pk == asker.pk
        assert await sync_to_async(find_user_by_pseudonym)(pseudonym("no-such-user")) is None
        assert await sync_to_async(find_user_by_pseudonym)("42") is None

    async def test_a_reviewer_still_cannot_answer_their_own_turn(self):
        from django_ai_sdk.oversight.schemas import ResolveIn
        from django_ai_sdk.oversight.services import OversightService

        from django_ai_sdk.oversight.models import ReviewItem

        asker = await reader(REVIEW)
        item = await ReviewItem.objects.aget(turn_id=await a_turn(user=asker))
        with pytest.raises(PermissionDenied):
            await OversightService.resolve_review(
                str(item.id), ResolveIn(lens="topic", value="other"), user=asker
            )

    async def test_a_turn_from_before_the_ledger_is_forgotten_when_its_user_goes(
        self, custody_under_test
    ):
        from django_ai_sdk.oversight import custody
        from django_ai_sdk.oversight.models import TurnContent

        asker = await reader()
        custody.install(custody.PlainCustody())
        turn_id = await a_turn(user=asker)
        custody.install(custody_under_test)
        before = await Turn.objects.aget(id=turn_id)
        assert before.actor_id == str(asker.pk)
        assert await TurnContent.objects.filter(scope=f"turn:{turn_id}").aexists()
        await asker.adelete()
        assert (await Turn.objects.aget(id=turn_id)).actor_id == ""
        assert not await TurnContent.objects.filter(scope=f"turn:{turn_id}").aexists()
        assert (await sync_to_async(chain.verify)()).ok

    async def test_a_reviewer_cannot_answer_their_own_turn_from_before_the_ledger(
        self, custody_under_test
    ):
        from django_ai_sdk.oversight import custody
        from django_ai_sdk.oversight.models import ReviewItem
        from django_ai_sdk.oversight.schemas import ResolveIn
        from django_ai_sdk.oversight.services import OversightService

        asker = await reader(REVIEW)
        custody.install(custody.PlainCustody())
        turn_id = await a_turn(user=asker)
        custody.install(custody_under_test)
        item = await ReviewItem.objects.aget(turn_id=turn_id)
        with pytest.raises(PermissionDenied):
            await OversightService.resolve_review(
                str(item.id), ResolveIn(lens="topic", value="other"), user=asker
            )

    async def test_a_deleted_user_leaves_no_pseudonym(self):
        asker = await reader()
        turn_id = await a_turn(user=asker)
        await asker.adelete()
        assert (await Turn.objects.aget(id=turn_id)).actor_id == ""


@pytest.mark.django_db(transaction=True)
class TestReveal:
    async def test_reveals_who_asked_and_records_why(self):
        asker = await reader()
        turn_id = await a_turn(user=asker)
        revealer = await reader(REVEAL)
        out = await LedgerService.reveal(turn_id, user=revealer, reason=REASON)
        assert out is not None
        assert (out.user_id, out.actor_id) == (str(asker.pk), pseudonym(asker.pk))
        assert str(asker.pk) in out.label
        record = await AccessRecord.objects.aget(operation="reveal")
        assert record.turn_ids == [turn_id]
        assert record.query == {"reason": REASON}
        assert record.reader_id == str(revealer.pk)
        last = await LedgerEntry.objects.order_by("-seq").afirst()
        assert (last.kind, last.ref) == ("access", str(record.pk))
        assert (await sync_to_async(chain.verify)()).ok

    async def test_needs_its_own_permission(self):
        from tests.oversight.review.helpers import CONTENT, METRICS

        turn_id = await a_turn(user=await reader())
        with pytest.raises(PermissionDenied):
            await LedgerService.reveal(
                turn_id, user=await reader(METRICS, CONTENT, REVIEW), reason=REASON
            )
        assert not await AccessRecord.objects.filter(operation="reveal").aexists()

    async def test_needs_a_reason(self):
        turn_id = await a_turn(user=await reader())
        with pytest.raises(UserError, match="reason"):
            await LedgerService.reveal(turn_id, user=await reader(REVEAL), reason="  because ")

    async def test_a_turn_out_of_reach_is_not_found(self):
        revealer = await reader(REVEAL)
        assert await LedgerService.reveal("nope", user=revealer, reason=REASON) is None
        missing = "00000000-0000-0000-0000-000000000000"
        assert await LedgerService.reveal(missing, user=revealer, reason=REASON) is None
        assert not await AccessRecord.objects.filter(operation="reveal").aexists()

    async def test_a_deleted_user_is_not_revealed_but_the_read_is_recorded(self):
        asker = await reader()
        turn_id = await a_turn(user=asker)
        pseudonym_before = (await Turn.objects.aget(id=turn_id)).actor_id
        await asker.adelete()
        out = await LedgerService.reveal(turn_id, user=await reader(REVEAL), reason=REASON)
        assert out is not None and out.user_id == "" and out.actor_id == ""
        assert pseudonym_before
        assert await AccessRecord.objects.filter(operation="reveal").aexists()
