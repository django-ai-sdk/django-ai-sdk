"""Automations are opt-in: without the app there are no tables, no command and no import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import sys

import django

django.setup()

from django.core.management import call_command, get_commands
from django.db import connection

call_command("check", fail_level="ERROR")
call_command("migrate", verbosity=0)
call_command("makemigrations", check=True, dry_run=True, verbosity=0)

assert "automations_tick" not in get_commands()
tables = connection.introspection.table_names()
assert not [t for t in tables if t.startswith("django_ai_sdk_automation")], tables
assert not [m for m in sys.modules if m.startswith("django_ai_sdk.automations")]
"""


def test_the_sdk_runs_without_the_app(tmp_path):
    # A fresh interpreter, so nothing this test run imported leaks in.
    (tmp_path / "without_automations.py").write_text(
        "from tests.settings import *  # noqa: F403\n\n"
        "INSTALLED_APPS = [a for a in INSTALLED_APPS if a != 'django_ai_sdk.automations']  # noqa: F405\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env={
            **os.environ,
            "DJANGO_SETTINGS_MODULE": "without_automations",
            "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)]),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
