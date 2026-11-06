"""The review layer is opt-in: with its modules gone, core still guards turns and queues reviews."""

from __future__ import annotations

from tests.removable import run_without

REVIEW = (
    "django_ai_sdk.oversight.services",
    "django_ai_sdk.oversight.schemas",
    "django_ai_sdk.oversight.permissions",
    "django_ai_sdk.oversight.retention",
    "django_ai_sdk.oversight.management.commands.oversight_prune",
    "django_ai_sdk.contrib.ninja.oversight",
)

# The review layer is modules inside the oversight app, not an app: hide them from imports.
HIDE_REVIEW = f"""
import importlib.abc
import sys

REVIEW = {REVIEW!r}


class Gone(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == m or name.startswith(m + ".") for m in REVIEW):
            raise ImportError(f"{{name}} is removed for this test")
        return None


sys.meta_path.insert(0, Gone())
"""

SCRIPT = """
from django_ai_sdk.oversight import L, PatternLens, Policy, Queue, Rule, register_policy
from django_ai_sdk.oversight.models import ReviewItem, Turn
from django_ai_sdk.oversight.types import Stage
from tests.removable import ask


class Watch(Policy):
    name = "watch"
    lenses = [PatternLens(name="pay", stages=[Stage.GATE], patterns=[r"salary"])]
    rules = [Rule("pay_to_hr", when=L("pay").is_true, then=Queue("hr"), mode="enforce")]


register_policy(Watch)

answer = ask("What about my salary?", "Ask HR.", policy="watch")
assert answer == "Ask HR.", answer
assert Turn.objects.count() == 1
assert ReviewItem.objects.filter(queue="hr").count() == 1
assert not [m for m in sys.modules if m.startswith(REVIEW)], sorted(sys.modules)
"""


def test_core_runs_without_the_review_layer(tmp_path):
    # The switches, authoring and ledger apps sit on the review layer: they go with it.
    apps = [
        "django_ai_sdk.oversight.switches",
        "django_ai_sdk.oversight.authoring",
        "django_ai_sdk.oversight.ledger",
    ]
    run_without(tmp_path, apps, SCRIPT, before_setup=HIDE_REVIEW)
