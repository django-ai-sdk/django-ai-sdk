"""The vault: content kept under a scope's own key, forgotten by shredding it, and master
keys that rotate without losing what they wrapped."""

from __future__ import annotations

import base64
import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import override_settings
from django.utils import timezone

from django_ai_sdk import vault
from django_ai_sdk.vault.checks import check_vault, check_vault_keys, check_vault_rotation
from django_ai_sdk.vault.keys import SettingsKeyProvider, key_id
from django_ai_sdk.vault.models import DataKey
from tests.oversight.helpers import check_ids as ids

OLD = base64.b64encode(b"o" * 32).decode()
NEW = base64.b64encode(b"n" * 32).decode()


def keys(*master: str):
    return override_settings(AI_SDK_VAULT={"MASTER_KEYS": list(master)})


@pytest.mark.django_db
class TestKeeping:
    def test_json_comes_back_and_the_envelope_holds_none_of_it(self):
        envelope = vault.encrypt_json("thread:1", {"input": "my salary", "n": 3})
        assert "salary" not in json.dumps(envelope)
        assert envelope["scope"] == "thread:1"
        assert vault.decrypt_json(envelope) == {"input": "my salary", "n": 3}

    def test_text_comes_back(self):
        token = vault.encrypt_text("user:7", "Jan de Vries")
        assert token.startswith("vault1:")
        assert "Jan" not in token
        assert vault.decrypt_text(token) == "Jan de Vries"

    def test_one_key_per_scope(self):
        vault.encrypt_text("thread:1", "a")
        vault.encrypt_text("thread:1", "b")
        vault.encrypt_text("thread:2", "c")
        assert DataKey.objects.count() == 2

    def test_a_key_is_wrapped_under_the_current_master_key_and_says_which(self):
        with keys(NEW, OLD):
            vault.encrypt_text("thread:1", "a")
        row = DataKey.objects.get(scope="thread:1")
        assert row.key_id == key_id(base64.b64decode(NEW))
        assert len(row.key_id) == 16

    def test_an_envelope_moved_to_another_scope_does_not_open(self):
        envelope = vault.encrypt_json("thread:1", {"a": 1})
        vault.encrypt_json("thread:2", {"b": 2})
        other = DataKey.objects.get(scope="thread:2")
        with pytest.raises(vault.KeyMismatch):
            vault.decrypt_json({**envelope, "scope": "thread:2", "key": str(other.id)})

    def test_altered_bytes_do_not_open(self):
        envelope = vault.encrypt_json("thread:1", {"a": 1})
        blob = bytearray(base64.b64decode(envelope["ct"]))
        blob[-1] ^= 1
        with pytest.raises(vault.KeyMismatch):
            vault.decrypt_json({**envelope, "ct": base64.b64encode(bytes(blob)).decode()})

    def test_what_is_not_an_envelope_does_not_open(self):
        with pytest.raises(vault.KeyMismatch):
            vault.decrypt_json({"input": "plain"})
        with pytest.raises(vault.KeyMismatch):
            vault.decrypt_text("plain")

    def test_the_key_rows_of_many_envelopes_are_read_at_once(self, django_assert_num_queries):
        envelopes = [vault.encrypt_json(f"thread:{i}", {"i": i}) for i in range(5)]
        with django_assert_num_queries(1):
            rows = vault.key_rows(envelopes)
            values = [vault.decrypt_json(e, rows=rows) for e in envelopes]
        assert values == [{"i": i} for i in range(5)]

    def test_a_digest_is_keyed_by_the_scope(self):
        vault.encrypt_text("thread:1", "a")
        vault.encrypt_text("thread:2", "a")
        first = vault.digest("thread:1", b"same")
        assert first == vault.digest("thread:1", b"same")
        assert first != vault.digest("thread:2", b"same")
        with pytest.raises(vault.KeyMismatch):
            vault.digest("thread:3", b"same")


@pytest.mark.django_db
class TestShredding:
    def test_a_shredded_scope_no_longer_opens_and_keeps_nothing_new(self):
        envelope = vault.encrypt_json("thread:1", {"a": 1})
        token = vault.encrypt_text("thread:1", "Jan")
        row, shredded = vault.shred("thread:1")
        assert shredded and row.wrapped is None and row.shredded_at is not None
        assert vault.decrypt_json(envelope) is None
        assert vault.decrypt_text(token) is None
        assert vault.is_shredded("thread:1")
        with pytest.raises(vault.ScopeShredded):
            vault.encrypt_json("thread:1", {"b": 2})
        with pytest.raises(vault.ScopeShredded):
            vault.digest("thread:1", b"x")

    def test_shredding_twice_says_so(self):
        vault.encrypt_text("thread:1", "a")
        assert vault.shred("thread:1")[1] is True
        assert vault.shred("thread:1")[1] is False

    def test_a_scope_shredded_before_its_first_use_keeps_nothing(self):
        assert vault.shred("thread:never")[1] is True
        with pytest.raises(vault.ScopeShredded):
            vault.encrypt_text("thread:never", "a")

    def test_other_scopes_still_open(self):
        kept = vault.encrypt_text("thread:2", "b")
        vault.encrypt_text("thread:1", "a")
        vault.shred("thread:1")
        assert vault.decrypt_text(kept) == "b"


@pytest.mark.django_db
class TestRotation:
    def test_a_previous_master_key_still_opens_what_it_wrapped(self):
        with keys(OLD):
            token = vault.encrypt_text("thread:1", "a")
        with keys(NEW, OLD):
            assert vault.decrypt_text(token) == "a"
            vault.encrypt_text("thread:2", "b")
        assert DataKey.objects.get(scope="thread:2").key_id == key_id(base64.b64decode(NEW))

    def test_rewrap_moves_every_key_to_the_current_master_key(self):
        with keys(OLD):
            tokens = [vault.encrypt_text(f"thread:{i}", str(i)) for i in range(3)]
            vault.encrypt_text("thread:gone", "x")
            vault.shred("thread:gone")
        with keys(NEW, OLD):
            out = StringIO()
            call_command("vault_rewrap", "--batch", "2", stdout=out)
            assert "rewrapped 3 data keys" in out.getvalue()
        current = key_id(base64.b64decode(NEW))
        assert set(
            DataKey.objects.filter(shredded_at__isnull=True).values_list("key_id", flat=True)
        ) == {current}
        assert DataKey.objects.filter(rewrapped_at__isnull=False).count() == 3
        with keys(NEW):
            assert [vault.decrypt_text(t) for t in tokens] == ["0", "1", "2"]

    def test_a_dropped_master_key_leaves_its_keys_unopened_and_fails_the_check(self):
        with keys(OLD):
            token = vault.encrypt_text("thread:1", "a")
        with keys(NEW):
            with pytest.raises(vault.KeyMismatch):
                vault.decrypt_text(token)
            assert ids(check_vault_keys(databases=["default"])) == ["ai_sdk.vault.E002"]
            with pytest.raises(CommandError, match="open under no configured master key"):
                call_command("vault_rewrap", stdout=StringIO())

    def test_a_previous_key_still_in_use_warns_on_deploy(self):
        with keys(OLD):
            vault.encrypt_text("thread:1", "a")
        with keys(NEW, OLD):
            assert ids(check_vault_rotation(databases=["default"])) == ["ai_sdk.vault.W001"]
            assert check_vault_keys(databases=["default"]) == []
            call_command("vault_rewrap", stdout=StringIO())
            assert check_vault_rotation(databases=["default"]) == []


class TestChecks:
    @pytest.mark.parametrize(
        "configured",
        [[], [""], ["not base64"], [base64.b64encode(b"short").decode()]],
        ids=["empty", "blank", "not-base64", "short"],
    )
    def test_no_usable_master_key_is_an_error(self, configured):
        with override_settings(AI_SDK_VAULT={"MASTER_KEYS": configured}):
            assert ids(check_vault()) == ["ai_sdk.vault.E001"]

    def test_a_key_provider_that_does_not_import_is_an_error(self):
        with override_settings(AI_SDK_VAULT={"MASTER_KEYS": [NEW], "KEY_PROVIDER": "no.such.X"}):
            assert ids(check_vault()) == ["ai_sdk.vault.E001"]

    def test_the_test_settings_pass(self):
        assert check_vault() == []

    def test_a_blank_previous_key_is_skipped(self):
        with keys(NEW, ""):
            assert SettingsKeyProvider().key_ids() == [key_id(base64.b64decode(NEW))]


class CountingProvider(SettingsKeyProvider):
    """A KeyProvider of its own, as a KMS would be: counts what it wraps and opens."""

    calls: list[str] = []  # noqa: RUF012

    def wrap(self, data_key: bytes, aad: bytes) -> tuple[str, bytes]:
        self.calls.append("wrap")
        return super().wrap(data_key, aad)

    def unwrap(self, key_id: str, wrapped: bytes, aad: bytes) -> bytes:
        self.calls.append("unwrap")
        return super().unwrap(key_id, wrapped, aad)


@pytest.mark.django_db
def test_a_key_provider_of_ones_own_wraps_and_opens_the_keys():
    CountingProvider.calls = []
    path = f"{__name__}.CountingProvider"
    with override_settings(AI_SDK_VAULT={"MASTER_KEYS": [NEW], "KEY_PROVIDER": path}):
        token = vault.encrypt_text("thread:1", "a")
        assert vault.decrypt_text(token) == "a"
    assert CountingProvider.calls == ["wrap", "unwrap"]


@pytest.mark.django_db
def test_the_rows_of_one_read_unwrap_each_key_once():
    CountingProvider.calls = []
    path = f"{__name__}.CountingProvider"
    with override_settings(AI_SDK_VAULT={"MASTER_KEYS": [NEW], "KEY_PROVIDER": path}):
        envelopes = [vault.encrypt_json("thread:1", {"n": n}) for n in range(5)]
        envelopes.append(vault.encrypt_json("thread:2", {"n": 5}))
        CountingProvider.calls = []
        rows = vault.key_rows(envelopes)
        assert [vault.decrypt_json(e, rows=rows) for e in envelopes] == [
            {"n": n} for n in range(6)
        ]
        assert vault.digest("thread:1", b"x", row=rows[envelopes[0]["key"]]) == vault.digest(
            "thread:1", b"x"
        )
    # Two keys for six envelopes and a digest; the digest without the row reads it again.
    assert CountingProvider.calls == ["unwrap", "unwrap", "unwrap"]


@pytest.mark.django_db
def test_a_cached_key_is_not_used_once_its_row_changes():
    envelope = vault.encrypt_json("thread:1", {"a": 1})
    rows = vault.key_rows([envelope])
    assert vault.decrypt_json(envelope, rows=rows) == {"a": 1}
    row = rows[envelope["key"]]
    row.wrapped = b"not the wrapped key"
    with pytest.raises(vault.KeyMismatch):
        vault.decrypt_json(envelope, rows=rows)
    row.shredded_at, row.wrapped = timezone.now(), None
    assert vault.decrypt_json(envelope, rows=rows) is None
    with pytest.raises(vault.ScopeShredded):
        vault.digest("thread:1", b"x", row=row)


@pytest.mark.django_db
def test_a_key_row_without_its_wrapped_key_is_altered_not_shredded():
    """Only `shred` clears `wrapped`, and it stamps `shredded_at` with it."""
    envelope = vault.encrypt_json("thread:1", {"a": 1})
    DataKey.objects.filter(scope="thread:1").update(wrapped=None)
    with pytest.raises(vault.KeyMismatch):
        vault.decrypt_json(envelope)
    with pytest.raises(vault.KeyMismatch):
        vault.encrypt_json("thread:1", {"a": 2})
    assert not vault.is_shredded("thread:1")


@pytest.mark.django_db
def test_a_digest_refuses_the_key_row_of_another_scope():
    vault.encrypt_json("thread:1", {"a": 1})
    other = vault.encrypt_json("thread:2", {"a": 1})
    rows = vault.key_rows([other])
    with pytest.raises(vault.KeyMismatch):
        vault.digest("thread:1", b"x", row=rows[other["key"]])


def test_vault_keys_prints_a_new_key_and_the_configured_ids():
    out = StringIO()
    with keys(NEW, OLD):
        call_command("vault_keys", stdout=out)
    lines = out.getvalue().splitlines()
    new = lines[0].split(": ")[1]
    assert len(base64.b64decode(new)) == 32
    assert lines[1] == f"current key id: {key_id(base64.b64decode(NEW))}"
    assert lines[2] == f"previous key id: {key_id(base64.b64decode(OLD))}"
