"""Run a script in a fresh interpreter on the test settings without some apps.

A fresh interpreter, so nothing this test run imported or registered leaks in. Each opt-in
app keeps its own small test that calls `run_without`, so the app and its test come and go
together.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]

# Runs before every script: the project still checks, migrates and has no missing migration.
SETUP = """
import django

django.setup()

from django.core.management import call_command

call_command("check", fail_level="ERROR")
call_command("migrate", verbosity=0)
call_command("makemigrations", check=True, dry_run=True, verbosity=0)
"""


def run_without(
    tmp_path: Path,
    apps: Iterable[str],
    script: str,
    *,
    before_setup: str = "",
    settings: str = "",
    files: Mapping[str, str] | None = None,
) -> None:
    """Run `script` with `apps` left out of INSTALLED_APPS, and fail with its stderr unless
    it gets to the end. `settings` adds lines to the settings module, `files` adds modules
    next to it, and `before_setup` runs before Django is set up."""
    removed = sorted(apps)
    # A file, not :memory:: a run's queries can come from a thread with a connection of its own.
    database = str(tmp_path / "db.sqlite3")
    (tmp_path / "removed_settings.py").write_text(
        "from tests.settings import *  # noqa: F403\n\n"
        f"INSTALLED_APPS = [a for a in INSTALLED_APPS if a not in {removed!r}]  # noqa: F405\n"
        f"DATABASES = {{'default': {{'ENGINE': 'django.db.backends.sqlite3', 'NAME': {database!r}}}}}\n"
        f"{settings}\n"
    )
    for name, text in (files or {}).items():
        (tmp_path / name).write_text(text)
    env = {
        **os.environ,
        "DJANGO_SETTINGS_MODULE": "removed_settings",
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)]),
    }
    env.pop("DATABASE_URL", None)
    done = subprocess.run(
        [sys.executable, "-c", f'{before_setup}\n{SETUP}\n{script}\nprint("ok")\n'],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")


def ask(question: str, reply: str, *, policy: str | None = None) -> str:
    """For use inside a script: run an agent that answers `reply`, under `policy` if given."""
    import asyncio

    from haystack.components.generators.chat import MockChatGenerator
    from haystack.dataclasses import ChatMessage as HaystackChatMessage

    from django_ai_sdk import Agent
    from django_ai_sdk.adapters.base import Run
    from django_ai_sdk.permissions import AllowAll

    async def get_run_adapter(self, thread_id=None, user=None):  # noqa: ANN001, ANN202
        generator = MockChatGenerator(responses=[HaystackChatMessage.from_assistant(reply)])
        return Run(generator=generator)

    plain = type(
        "Plain",
        (Agent,),
        {
            "_skip_auto_register": True,
            "permissions": [AllowAll],
            "name": "plain",
            "oversight": policy,
            "title_generation": False,
            "get_run_adapter": get_run_adapter,
        },
    )
    return asyncio.run(plain().run([{"role": "user", "content": question}]))
