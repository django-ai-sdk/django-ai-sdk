"""The oversight surface beyond tests/test_public_api.py: the judges' exports and the
manual's settings table."""

from __future__ import annotations

import re
from pathlib import Path

import django_ai_sdk.oversight as oversight

MANUAL = Path(__file__).resolve().parents[2] / "docs/content/manual/oversight.md"


def test_judges_are_exported_from_their_own_module():
    import django_ai_sdk.oversight.judges as judges

    assert "TypeSafe" not in oversight.__all__ and not hasattr(oversight, "TypeSafe")
    assert "TypeSafe" in judges.__all__
    assert not {"ask", "mask", "resolve"} & set(judges.__all__)
    # Each judge loads on first use, so declaring a lens never imports a model client.
    assert judges.TypeSafe.__name__ == "TypeSafe"


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
    from django_ai_sdk.oversight.conf import setting

    manual = MANUAL.read_text(encoding="utf-8")
    table = manual.split("## Settings\n", 1)[1].split("\nOutside the dict:", 1)[0]
    rows = re.findall(r"^\| `([A-Z_]+)` \| `([^|]*)` \|", table, re.MULTILINE)
    assert [key for key, _ in rows] == list(setting.defaults)
    assert {key: default for key, default in rows} == {
        key: shown(value) for key, value in setting.defaults.items()
    }
