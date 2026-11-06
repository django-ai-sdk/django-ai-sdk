"""The audit-grade oversight page matches the vault and ledger code: the settings, the
commands, the check ids and the verify problems (tests/test_public_api.py: exports)."""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import django_ai_sdk.vault as vault
import pytest
from django_ai_sdk.checks import SETTINGS_CHECK

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "docs/content/oversight-ledger.md"


def text() -> str:
    return PAGE.read_text(encoding="utf-8")


def section(heading: str) -> str:
    return text().split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]


def table(heading: str) -> list[list[str]]:
    rows = [
        line
        for line in section(heading).splitlines()
        if line.startswith("| ") and not line.startswith("| ---")
    ]
    return [[cell.strip() for cell in row.strip("|").split(" | ")] for row in rows[1:]]


@pytest.mark.parametrize(
    ("heading", "module"),
    [
        ("### Vault settings", "django_ai_sdk.vault.conf"),
        ("## Ledger settings", "django_ai_sdk.oversight.ledger.conf"),
    ],
)
def test_the_settings_tables_are_the_defaults(heading, module):
    defaults = importlib.import_module(module).setting.defaults
    rows = {key.strip("`"): default.strip("`") for key, default, _ in table(heading)}
    assert list(rows) == list(defaults)
    assert rows == {
        key: f'"{value}"' if isinstance(value, str) else repr(value)
        for key, value in defaults.items()
    }


def test_every_command_is_listed():
    listed = {re.match(r"`(\w+)", row[0]).group(1) for row in table("## Commands")}
    shipped = {
        path.stem
        for app in ("vault", "oversight/ledger")
        for path in (ROOT / "django_ai_sdk" / app / "management/commands").glob("*.py")
        if path.stem != "__init__"
    }
    assert listed == shipped


def test_the_checks_table_is_the_ids_the_code_raises():
    raised = set()
    for module in ("django_ai_sdk.vault.checks", "django_ai_sdk.oversight.ledger.checks"):
        source = inspect.getsource(importlib.import_module(module))
        prefix = re.search(r'^PREFIX = "([\w.]+)"', source, re.M).group(1)
        raised |= {f"{prefix}.{i}" for i in re.findall(r'id=f"\{PREFIX\}\.(\w+)"', source)}
        if ".errors(PREFIX)" in source:
            raised.add(f"{prefix}.{SETTINGS_CHECK}")
    listed = {row[0].strip("`") for row in table("## Checks")}
    assert listed == raised


def test_the_problems_table_names_every_problem_verify_reports():
    sources = inspect.getsource(importlib.import_module("django_ai_sdk.oversight.ledger.chain"))
    sources += inspect.getsource(
        importlib.import_module("django_ai_sdk.oversight.ledger.management.commands.ledger_verify")
    )
    reported = set(re.findall(r'fail\((?:[\w.]+, )?"([a-z ]+)"', sources))
    listed = {name for row in table("### Verify") for name in re.findall(r"`([a-z ]+)`", row[0])}
    assert listed == reported


@pytest.mark.django_db
def test_the_vault_example_runs_as_shown():
    block = section("## The vault").split("```python\n", 1)[1].split("```", 1)[0]
    *lines, last = [line for line in block.splitlines() if line.strip()]
    namespace: dict = {}
    for line in lines:
        code, _, shown = line.partition("  # ")
        if code.startswith(("from ", "envelope", "token", "vault.shred")):
            exec(code, namespace)  # noqa: S102 - the page's own example
        else:
            expected = shown.partition(":")[0] if shown.startswith("None") else shown
            assert repr(eval(code, namespace)) == expected.replace('"', "'")  # noqa: S307
    assert last.endswith("# raises vault.ScopeShredded")
    with pytest.raises(vault.ScopeShredded):
        exec(last.partition("  #")[0], namespace)  # noqa: S102
