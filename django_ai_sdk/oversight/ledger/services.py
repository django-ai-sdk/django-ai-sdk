"""Ledger service for the API: checkpoints, verify and reveal.

Every call writes an access record, which is chained. A reveal's record keeps its reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import sync_to_async

from django_ai_sdk.oversight.ledger.schemas import (
    MIN_REASON_CHARS,
    CheckpointOut,
    RevealOut,
    VerifyOut,
)
from django_ai_sdk.oversight.services.access import (
    OversightAccess,
    reader_label,
    require_reason,
)
from django_ai_sdk.oversight.services.paging import page_size
from django_ai_sdk.permissions import Operation

if TYPE_CHECKING:
    from django_ai_sdk.types import UserType

VIEW = Operation.VIEW_OVERSIGHT_LEDGER
CHECKPOINT = Operation.CHECKPOINT_OVERSIGHT
REVEAL = Operation.REVEAL_OVERSIGHT_ACTOR


# The access record's operation for each ledger call.
CHECKPOINTED, VERIFIED, REVEALED = "checkpoint", "verify", "reveal"


class LedgerService(OversightAccess):
    """Checkpoints, verify and reveal for the review API."""

    @classmethod
    async def checkpoints(cls, *, user: UserType, limit: int = 50) -> list[CheckpointOut]:
        """The newest checkpoints first."""
        from django_ai_sdk.oversight.ledger.models import Checkpoint

        await cls.has_perms(user, VIEW)
        rows = Checkpoint.objects.order_by("-tree_size")[: page_size(limit)]
        return [CheckpointOut.model_validate(row, from_attributes=True) async for row in rows]

    @classmethod
    async def write_checkpoint(cls, *, user: UserType) -> CheckpointOut | None:
        """Write a checkpoint. None when nothing was chained since the last one.

        Raises ChainTruncated when the chain ends before its head.
        """
        from django_ai_sdk.oversight.ledger.checkpoints import write_checkpoint

        await cls.has_perms(user, CHECKPOINT)
        checkpoint = await sync_to_async(write_checkpoint)()
        if checkpoint is None:
            return None
        await cls.record(user, CHECKPOINTED, query={"tree_size": checkpoint.tree_size})
        return CheckpointOut.model_validate(checkpoint, from_attributes=True)

    @classmethod
    async def verify(cls, *, user: UserType) -> VerifyOut:
        """Walk the whole chain and check the newest checkpoint against it."""

        await cls.has_perms(user, VIEW)
        result = await sync_to_async(verify_all)()
        await cls.record(user, VERIFIED, query={"state": result.state})
        return result

    @classmethod
    async def reveal(cls, turn_id: str, *, user: UserType, reason: str) -> RevealOut | None:
        """Reveal who asked `turn_id`. None when the turn is outside the reader's scope."""
        from django_ai_sdk.oversight.ledger.privacy import find_user_by_pseudonym
        from django_ai_sdk.oversight.models import Turn
        from django_ai_sdk.utils import as_uuid

        await cls.has_perms(user, REVEAL)
        reason = require_reason(
            reason,
            "Give a reason of at least a few words to reveal who asked.",
            min_chars=MIN_REASON_CHARS,
        )
        scoped = await cls.scoped(user, REVEAL, Turn.objects.all())
        pk = as_uuid(turn_id)
        turn = await scoped.filter(pk=pk).afirst() if pk else None
        if turn is None:
            return None
        # Record first, so a reveal that fails halfway is still logged.
        await cls.record(user, REVEALED, turn_ids=[turn.pk], query={"reason": reason})
        person = await sync_to_async(find_user_by_pseudonym)(turn.actor_id)
        if person is None:
            return RevealOut(turn_id=str(turn.pk), actor_id=turn.actor_id)
        return RevealOut(
            turn_id=str(turn.pk),
            actor_id=turn.actor_id,
            user_id=str(person.pk),
            label=reader_label(person),
        )


def verify_all() -> VerifyOut:
    """Verify the whole chain and every checkpoint against it. Synchronous.

    A rewrite of history leaves every checkpoint after the rewritten entry disagreeing,
    even when a newer checkpoint was signed over the rewrite.
    """
    from django_ai_sdk.oversight.ledger import chain
    from django_ai_sdk.oversight.ledger.checkpoints import chain_roots
    from django_ai_sdk.oversight.ledger.models import Checkpoint

    result = chain.verify()
    stored = list(Checkpoint.objects.order_by("tree_size").values_list("tree_size", "root_hash"))
    if stored:
        roots = chain_roots(size for size, _ in stored)
        wrong = [
            size
            for size, root in stored
            if (found := roots.get(size)) is None or found.hex() != root
        ]
        for size in wrong:
            result.fail(size, "checkpoint mismatch", tree_size=size)
        result.checkpoint_size = stored[-1][0]
        result.checkpoint_ok = not wrong
        result.ok = result.ok and not wrong
    return result


__all__ = ["LedgerService", "verify_all"]
