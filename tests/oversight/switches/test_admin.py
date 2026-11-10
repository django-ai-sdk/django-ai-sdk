"""The admin shows every switch version, and changes none of them.

The test settings do not install django.contrib.admin, so this runs in a process
of its own that does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = """
import django

django.setup()

from django.contrib import admin

admin.autodiscover()

from django_ai_sdk.oversight.switches.models import Switch

model_admin = admin.site._registry[Switch]
assert model_admin.check() == [], model_admin.check()
assert not model_admin.has_add_permission(None)
assert not model_admin.has_change_permission(None)
assert not model_admin.has_delete_permission(None)
assert "reason" in model_admin.search_fields
print("ok")
"""


def test_the_admin_shows_the_switches_and_changes_none(tmp_path):
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "with_admin_settings.py").write_text(
        "from tests.settings import *  # noqa: F403\n"
        "INSTALLED_APPS = ['django.contrib.admin', 'django.contrib.messages', *INSTALLED_APPS]"
        "  # noqa: F405\n"
    )
    env = {
        **os.environ,
        "DJANGO_SETTINGS_MODULE": "with_admin_settings",
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(root)]),
    }
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
