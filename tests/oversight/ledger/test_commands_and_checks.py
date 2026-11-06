"""The ledger's commands and system checks."""

from __future__ import annotations

import base64
import json
from datetime import timedelta
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction
from django.test import override_settings
from django.utils import timezone

from django_ai_sdk.oversight.ledger import chain, checkpoints
from django_ai_sdk.oversight.ledger.checks import (
    check_ledger,
    check_ledger_checkpoints,
    check_ledger_deploy,
)
from django_ai_sdk.oversight.ledger.models import Checkpoint, LedgerEntry
from django_ai_sdk.oversight.models import AccessRecord
from django_ai_sdk.vault.crypto import new_key_b64
from django_ai_sdk.vault.models import DataKey

KEY = base64.b64encode(b"s" * 32).decode()
SECRET = "a-pseudonym-secret-of-at-least-32-characters"
OTHER_MASTER = "b3RoZXItdmF1bHQtbWFzdGVyLWtleS1mb3ItdGVzdHM="


def record() -> AccessRecord:
    with transaction.atomic():
        made = AccessRecord.objects.create(operation="list")
        chain.append_many([("access", made)])
    return made


def run(command: str, *args: str) -> str:
    out = StringIO()
    call_command(command, *args, stdout=out, stderr=StringIO())
    return out.getvalue()


def ids(messages) -> set[str]:
    return {m.id for m in messages}


class TestChecks:
    def test_a_configured_ledger_passes(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {"SIGNING_KEY": KEY, "PSEUDONYM_SECRET": SECRET}
        assert check_ledger() == [] and check_ledger_deploy() == []

    def test_an_unusable_signing_key_is_an_error(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {"SIGNING_KEY": base64.b64encode(b"x").decode()}
        [error] = check_ledger()
        assert error.id == "ai_sdk.oversight.ledger.E001"
        assert "SIGNING_KEY'] must be base64 of 32 bytes" in error.msg

    def test_settings_that_are_not_settings_are_errors(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {"SIGNING_KEYS": KEY, "CHECKPOINT_DAYS": "2", "PSEUDONYM_SECRET": "x"}
        found = check_ledger()
        assert ids(found) == {"ai_sdk.oversight.ledger.E012"}
        text = " ".join(m.msg for m in found)
        assert "'SIGNING_KEYS'" in text and "'CHECKPOINT_DAYS'" in text and "32 characters" in text

    def test_no_signing_key_and_no_secret_warn_on_deploy(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {}
        assert ids(check_ledger_deploy()) == {"ai_sdk.oversight.ledger.W001", "ai_sdk.oversight.ledger.W002"}

    @pytest.mark.django_db  # Django 6.1's model checks ask the database what it supports.
    def test_they_are_deploy_checks(self):
        from django.core import checks

        deploy = ids(checks.run_checks(include_deployment_checks=True))
        plain = ids(checks.run_checks())
        assert {"ai_sdk.oversight.ledger.W001", "ai_sdk.oversight.ledger.W002"} <= deploy
        assert not {"ai_sdk.oversight.ledger.W001", "ai_sdk.oversight.ledger.W002"} & plain


@pytest.mark.django_db(transaction=True)
class TestCheckpointCheck:
    def test_entries_no_checkpoint_covers_for_days_warn(self):
        record()
        assert check_ledger_checkpoints(databases=["default"]) == []
        LedgerEntry.objects.update(created_at=timezone.now() - timedelta(days=3))
        assert ids(check_ledger_checkpoints(databases=["default"])) == {"ai_sdk.oversight.ledger.W003"}
        checkpoints.write_checkpoint()
        assert check_ledger_checkpoints(databases=["default"]) == []

    def test_a_fresh_chain_is_fine(self):
        assert check_ledger_checkpoints(databases=["default"]) == []


class TestKeysCommand:
    def test_prints_new_keys(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {}
        out = run("ledger_keys")
        assert "SIGNING_KEY" in out and "PSEUDONYM_SECRET" in out and "verifier key: " in out

    def test_prints_the_configured_keys_verifier(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {"SIGNING_KEY": KEY, "ORIGIN": "me/log"}
        assert run("ledger_keys").strip() == (
            f"verifier key: {checkpoints.verifier_key('me/log', KEY)}"
        )

    def test_refuses_an_unusable_key(self, settings):
        settings.AI_SDK_OVERSIGHT_LEDGER = {"SIGNING_KEY": "nope"}
        with pytest.raises(CommandError, match="base64 of 32 bytes"):
            run("ledger_keys")


@pytest.mark.django_db(transaction=True)
class TestForgetCommand:
    def test_forgets_a_thread_once_through_the_ledger(self):
        assert run("oversight_forget_thread", "t-9").strip() == "Forgot thread:t-9."
        assert run("oversight_forget_thread", "t-9").strip() == "Forgot thread:t-9."
        assert DataKey.objects.get(scope="thread:t-9").shredded_at is not None
        assert list(LedgerEntry.objects.values_list("kind", flat=True)) == ["forget"]
        assert chain.verify().ok

    def test_needs_a_thread(self):
        with pytest.raises(CommandError, match="thread_ids"):
            run("oversight_forget_thread")


@pytest.mark.django_db(transaction=True)
class TestVerifyCommand:
    ORIGIN = {"ORIGIN": "test/log"}

    def verifier(self) -> str:
        return checkpoints.verifier_key("test/log", KEY)

    def checkpoint(self) -> Checkpoint:
        with override_settings(AI_SDK_OVERSIGHT_LEDGER={**self.ORIGIN, "SIGNING_KEY": KEY}):
            return checkpoints.write_checkpoint()

    def test_an_intact_ledger_verifies(self):
        for _ in range(3):
            record()
        self.checkpoint()
        out = run("ledger_verify")
        assert out.startswith("3 entries checked, 0 pruned, 0 forgotten, 0 unreadable")
        assert "checkpoint at 3 matches" in out and "The ledger verifies." in out
        report = json.loads(run("ledger_verify", "--json", "--verifier", self.verifier()))
        assert report["ok"] and report["state"] == "ok"
        assert report["checkpoint_ok"] and report["bad_signatures"] == 0

    def test_tampering_fails_the_command(self):
        made = record()
        record()
        AccessRecord.objects.filter(pk=made.pk).update(operation="view")
        with pytest.raises(CommandError, match="does not verify"):
            run("ledger_verify")

    def test_a_rewritten_checkpointed_chain_fails(self):
        record()
        self.checkpoint()
        LedgerEntry.objects.filter(seq=1).update(entry_hash="f" * 64)
        with pytest.raises(CommandError, match="does not verify"):
            run("ledger_verify")

    def test_a_signature_by_another_key_fails(self):
        record()
        self.checkpoint()
        other = checkpoints.verifier_key("test/log", new_key_b64())
        with pytest.raises(CommandError):
            run("ledger_verify", "--verifier", other)
        with pytest.raises(CommandError, match="not a verifier key"):
            run("ledger_verify", "--verifier", "nonsense")

    @pytest.mark.usefixtures("no_reflect_queue")
    def test_unreadable_content_fails_unless_allowed(self, settings):
        from asgiref.sync import async_to_sync

        from tests.oversight.review.helpers import a_turn

        async_to_sync(a_turn)()
        settings.AI_SDK_VAULT = {"MASTER_KEYS": [OTHER_MASTER]}
        with pytest.raises(CommandError, match="master key not configured"):
            run("ledger_verify")
        assert "1 unreadable" in run("ledger_verify", "--allow-unreadable")
        report = json.loads(run("ledger_verify", "--json", "--allow-unreadable"))
        assert (report["ok"], report["state"], report["unreadable"]) == (True, "unreadable", 1)
