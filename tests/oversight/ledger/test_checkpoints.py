"""Checkpoints: an RFC 6962 root over the chain, written once per size, signed when a key
is set, and sent to the notifiers."""

from __future__ import annotations

import base64
import hashlib
import sys
from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction
from django.test import override_settings

from django_ai_sdk.oversight.ledger import chain, checkpoints
from django_ai_sdk.oversight.ledger.models import Checkpoint
from django_ai_sdk.oversight.models import AccessRecord
from django_ai_sdk.vault.crypto import new_key_b64

# RFC 6962 test inputs, as used by the certificate-transparency reference code.
LEAVES = [
    b"",
    b"\x00",
    b"\x10",
    b"\x20\x21",
    b"\x30\x31",
    b"\x40\x41\x42\x43",
    b"\x50\x51\x52\x53\x54\x55\x56\x57",
    b"\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f",
]
ROOTS = {
    1: "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    2: "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    3: "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    8: "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
}
KEY = base64.b64encode(b"s" * 32).decode()
LEDGER = {"ORIGIN": "test/log"}


@pytest.fixture
def ledger_origin(settings):
    settings.AI_SDK_OVERSIGHT_LEDGER = LEDGER


def record() -> AccessRecord:
    """One chained access record: one more entry."""
    with transaction.atomic():
        made = AccessRecord.objects.create(operation="list")
        chain.append_many([("access", made)])
    return made


@pytest.fixture
def no_cryptography():
    """The package not installed: every cryptography module fails to import."""
    hidden = {name: None for name in sys.modules if name.split(".")[0] == "cryptography"}
    with patch.dict(sys.modules, {**hidden, "cryptography": None}):
        yield


class TestMerkle:
    def test_rfc_6962_roots(self):
        for size, root in ROOTS.items():
            assert checkpoints.merkle_root(LEAVES[:size]).hex() == root

    def test_empty_tree(self):
        assert checkpoints.merkle_root([]) == hashlib.sha256(b"").digest()

    def test_signed_notes(self):
        key = new_key_b64()
        body = checkpoints.checkpoint_body("example.com/log", 8, bytes.fromhex(ROOTS[8]))
        note = checkpoints.sign_note(body, name="example.com/log", private_key_b64=key)
        verifier = checkpoints.verifier_key("example.com/log", key)
        assert checkpoints.verify_note(note, verifier)
        assert not checkpoints.verify_note(note.replace("\n8\n", "\n9\n"), verifier)
        other = checkpoints.verifier_key("example.com/log", new_key_b64())
        assert not checkpoints.verify_note(note, other)

    def test_a_key_that_is_not_base64_of_32_bytes_is_refused(self):
        from django.core.exceptions import ImproperlyConfigured

        with pytest.raises(ImproperlyConfigured, match="SIGNING_KEY'\\] must be base64"):
            checkpoints.private_key(base64.b64encode(b"x").decode())

    def test_a_key_without_cryptography_is_refused(self, no_cryptography):
        from django.core.exceptions import ImproperlyConfigured

        with pytest.raises(ImproperlyConfigured, match="cryptography"):
            checkpoints.private_key(KEY)


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("ledger_origin")
class TestWriteCheckpoint:
    def test_nothing_chained_writes_nothing(self):
        assert checkpoints.write_checkpoint() is None

    def test_one_checkpoint_per_size(self):
        for _ in range(5):
            record()
        checkpoint = checkpoints.write_checkpoint()
        assert checkpoint.tree_size == 5 and not checkpoint.signed
        assert checkpoint.note.startswith("test/log\n5\n")
        assert checkpoints.checkpoint_matches_chain(checkpoint)
        assert checkpoints.write_checkpoint() is None
        record()
        assert checkpoints.write_checkpoint().tree_size == 6

    def test_signed_with_the_configured_key(self):
        record()
        with override_settings(AI_SDK_OVERSIGHT_LEDGER={**LEDGER, "SIGNING_KEY": KEY}):
            checkpoint = checkpoints.write_checkpoint()
        assert checkpoint.signed
        assert checkpoints.signed_by(checkpoint, checkpoints.verifier_key("test/log", KEY))
        # The signature covers the note, not the columns beside it.
        checkpoint.root_hash = "0" * 64
        assert not checkpoints.signed_by(checkpoint, checkpoints.verifier_key("test/log", KEY))

    def test_two_writers_at_once_agree(self):
        record()
        real = checkpoints.merkle_root

        def racing(leaves):
            # The other writer inserts the same size between our check and our insert.
            root = real(leaves)
            Checkpoint.objects.create(tree_size=len(leaves), root_hash=root.hex(), note="first")
            return root

        with patch.object(checkpoints, "merkle_root", racing):
            checkpoint = checkpoints.write_checkpoint()
        assert checkpoint.note == "first"
        assert Checkpoint.objects.count() == 1

    def test_a_new_checkpoint_goes_to_the_notifiers_once(self):
        record()
        with patch("django_ai_sdk.oversight.persist.enqueue_notifications") as enqueue:
            checkpoint = checkpoints.write_checkpoint()
            assert checkpoints.write_checkpoint() is None
        [[[(event, names)]]] = [call.args for call in enqueue.call_args_list]
        assert event["type"] == "oversight.checkpoint"
        assert (event["tree_size"], event["root_hash"]) == (1, checkpoint.root_hash)
        assert event["id"] == checkpoints.checkpoint_event(checkpoint)["id"]
        assert names == []

    def test_a_rewritten_chain_no_longer_matches(self):
        from django_ai_sdk.oversight.ledger.models import LedgerEntry

        record()
        record()
        checkpoint = checkpoints.write_checkpoint()
        LedgerEntry.objects.filter(seq=1).update(entry_hash="f" * 64)
        assert not checkpoints.checkpoint_matches_chain(checkpoint)

    def test_a_chain_shorter_than_its_head_is_an_error_not_nothing_new(self):
        """Deleting the newest entries is tampering, so it must not pass quietly."""
        from django_ai_sdk.oversight.ledger.models import LedgerEntry

        record()
        record()
        LedgerEntry.objects.filter(seq=2).delete()
        with pytest.raises(checkpoints.ChainTruncated):
            checkpoints.write_checkpoint()
        with pytest.raises(CommandError, match="ends before its head"):
            call_command("ledger_checkpoint", stdout=StringIO())


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("ledger_origin")
class TestCheckpointCommand:
    def run(self) -> str:
        out = StringIO()
        call_command("ledger_checkpoint", stdout=out)
        return out.getvalue()

    def test_writes_a_checkpoint_once(self):
        record()
        assert self.run().startswith("Checkpoint at 1 entries")
        assert self.run().strip() == "Nothing chained since the last checkpoint."

    def test_an_unusable_key_fails_the_command(self):
        record()
        with (
            override_settings(AI_SDK_OVERSIGHT_LEDGER={"SIGNING_KEY": "nope"}),
            pytest.raises(CommandError, match="base64 of 32 bytes"),
        ):
            self.run()
