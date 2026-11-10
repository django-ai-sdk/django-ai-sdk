"""The ledger and the vault are opt-in: without them turns run on core's own custody, and
nothing imports them. The ledger goes on its own: the vault stays usable without it."""

from __future__ import annotations

from tests.removable import run_without

LEDGER = "django_ai_sdk.oversight.ledger"
VAULT = "django_ai_sdk.vault"

# A guarded turn on core's own custody, its input kept as captured.
TURN = """
import sys

from django.db import connection

from django_ai_sdk.oversight import Block, L, PatternLens, Policy, Rule, custody, register_policy
from django_ai_sdk.oversight.models import Turn, TurnContent
from django_ai_sdk.oversight.types import Stage
from tests.removable import ask


class Watch(Policy):
    name = "watch"
    capture = "full"
    lenses = [PatternLens(name="x", stages=[Stage.GATE], patterns=[r"xylophone"])]
    rules = [Rule("block_x", when=L("x").is_true, then=Block("No x."), mode="enforce")]


register_policy(Watch)

assert custody.is_plain()
answer = ask("Hello", "Fine.", policy="watch")
assert answer == "Fine.", answer
assert Turn.objects.count() == 1
assert TurnContent.objects.get().stored["input"] == "Hello"
tables = connection.introspection.table_names()
"""

WITHOUT_EITHER = (
    TURN
    + """
extra = [t for t in tables if "ledger" in t or "data_keys" in t or "checkpoints" in t]
assert not extra, extra
loaded = [m for m in sys.modules if m.startswith(("django_ai_sdk.oversight.ledger", "django_ai_sdk.vault"))]
assert not loaded, loaded
"""
)

WITHOUT_THE_LEDGER = (
    TURN
    + """
extra = [t for t in tables if "ledger" in t or "checkpoints" in t]
assert not extra, extra
assert "django_ai_sdk_data_keys" in tables
loaded = [m for m in sys.modules if m.startswith("django_ai_sdk.oversight.ledger")]
assert not loaded, loaded

from django_ai_sdk import vault

envelope = vault.encrypt_json("invoice:7", {"iban": "NL91"})
assert vault.decrypt_json(envelope) == {"iban": "NL91"}
vault.shred("invoice:7")
assert vault.decrypt_json(envelope) is None
assert custody.is_plain()
"""
)


def test_oversight_runs_without_the_ledger_and_the_vault(tmp_path):
    run_without(tmp_path, [LEDGER, VAULT], WITHOUT_EITHER)


def test_the_ledger_goes_on_its_own_and_the_vault_still_keeps(tmp_path):
    run_without(tmp_path, [LEDGER], WITHOUT_THE_LEDGER)
