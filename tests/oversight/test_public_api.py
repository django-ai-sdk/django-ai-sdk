"""The oversight surface beyond tests/test_public_api.py: the judges' exports and the
manual's settings table."""

from __future__ import annotations

import re
from pathlib import Path

import django_ai_sdk.oversight as oversight

MANUAL = Path(__file__).resolve().parents[2] / "docs/content/manual/oversight.md"


def test_judges_are_exported_from_the_package_and_load_on_first_use():
    assert {"TypeSafe", "LocalClassifier", "Judge", "Classifier", "certainty"} <= set(
        oversight.__all__
    )
    assert {"ask_judge", "build_judge", "load_local_classifier"} <= set(oversight.__all__)
    # Each judge loads on first use, so declaring a lens never imports a model client.
    assert "TypeSafe" not in vars(oversight)
    assert oversight.TypeSafe.__name__ == "TypeSafe"


def shown(value) -> str:
    """A default as the manual's settings table writes it."""
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, dict):
        return "{" + ", ".join(f'"{k}": {shown(v)}' for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(shown(v) for v in value) + "]"
    return str(value)


def test_the_settings_table_is_the_defaults():
    from django_ai_sdk.oversight.conf import DEFAULTS

    manual = MANUAL.read_text(encoding="utf-8")
    table = manual.split("## Settings\n", 1)[1].split("\nOutside the dict:", 1)[0]
    rows = re.findall(r"^\| `([A-Z_]+)` \| `([^|]*)` \|", table, re.MULTILINE)
    assert [key for key, _ in rows] == list(DEFAULTS)
    assert {key: default for key, default in rows} == {
        key: shown(value) for key, value in DEFAULTS.items()
    }


def test_adapter_plumbing_stays_in_its_modules():
    from django_ai_sdk.oversight import guard, history, stream
    from django_ai_sdk.oversight.judges import base

    plumbing = {
        guard: ["guarding", "closing_turn", "outlived_turn"],
        stream: ["StreamTurn", "CONTENT_FILTER"],
        history: ["history_for"],
        base: ["judge_label"],
    }
    for module, names in plumbing.items():
        for name in names:
            assert hasattr(module, name)
            assert name not in oversight.__all__
