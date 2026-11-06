"""The ledger's tables. The hash chain, its head, and checkpoints.

The chain only holds hashes, never content. Entries point to rows by plain id, not
a foreign key, so deleting a row keeps its entry and verify can spot the gap.
"""

from __future__ import annotations

import uuid

from django.db import models

APP_LABEL = "django_ai_sdk_oversight_ledger"
# prev_hash of the first entry, and the head of an empty chain.
GENESIS = "0" * 64


class LedgerEntry(models.Model):
    """One chain entry. `seq` has no gaps and `entry_hash` includes the previous hash."""

    seq = models.BigIntegerField(primary_key=True)
    kind = models.CharField(max_length=32)
    # The row's id. For "prune" and "erase" entries, the seq run they cover.
    ref = models.CharField(max_length=64, db_index=True)
    payload_hash = models.CharField(max_length=64)
    prev_hash = models.CharField(max_length=64)
    entry_hash = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_ledger"
        ordering = ["seq"]
        indexes = [models.Index(fields=["kind", "ref"])]
        # Backs REVEAL_OVERSIGHT_ACTOR. Grant it per user or group.
        permissions = [("reveal_actor", "Can reveal who asked a turn, for a recorded reason")]

    def __str__(self) -> str:
        return f"#{self.seq} {self.kind} {self.entry_hash[:12]}"


class LedgerHead(models.Model):
    """The chain's tip. A single row (id 1, created by the migration).

    Appenders lock this row, which keeps them in order.
    """

    id = models.PositiveSmallIntegerField(primary_key=True, default=1)
    seq = models.BigIntegerField(default=0)
    entry_hash = models.CharField(max_length=64, default=GENESIS)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_ledger_head"


class Checkpoint(models.Model):
    """Merkle root of the first `tree_size` entries, with an optionally signed note."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    tree_size = models.BigIntegerField(unique=True)
    root_hash = models.CharField(max_length=64)
    # C2SP signed note with origin, size, root and signature lines.
    note = models.TextField()
    signed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_checkpoints"
        ordering = ["-tree_size"]


__all__ = ["APP_LABEL", "GENESIS", "Checkpoint", "LedgerEntry", "LedgerHead"]
