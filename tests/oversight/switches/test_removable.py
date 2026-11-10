"""The switches are opt-in: without the app, turns run on code alone and nothing imports it."""

from __future__ import annotations

from django_ai_sdk.oversight.resolve import overlay_source, set_overlay_source
from django_ai_sdk.oversight.switches.overlay import DatabaseOverlay

from tests.removable import run_without

SCRIPT = """
import sys

from django_ai_sdk.oversight import Block, L, PatternLens, Policy, Rule, register_policy
from django_ai_sdk.oversight.models import Turn
from django_ai_sdk.oversight.resolve import OverlaySource, overlay_source
from django_ai_sdk.oversight.types import Stage
from tests.removable import ask


class Watch(Policy):
    name = "watch"
    lenses = [PatternLens(name="x", stages=[Stage.GATE], patterns=[r"xylophone"])]
    rules = [Rule("block_x", when=L("x").is_true, then=Block("No x."), mode="enforce")]


register_policy(Watch)

assert type(overlay_source()) is OverlaySource
answer = ask("Hello", "Fine.", policy="watch")
assert answer == "Fine.", answer
assert Turn.objects.count() == 1
switches = [m for m in sys.modules if m.startswith("django_ai_sdk.oversight.switches")]
assert not switches, switches
"""


def test_the_app_installs_the_database_overlay_on_start():
    from django.apps import apps

    set_overlay_source(None)
    apps.get_app_config("django_ai_sdk_oversight_switches").ready()
    assert isinstance(overlay_source(), DatabaseOverlay)


def test_core_and_review_run_without_the_switches(tmp_path):
    run_without(tmp_path, ["django_ai_sdk.oversight.switches"], SCRIPT)
