"""The workflows package imports without the app registry (tests/test_public_api.py: exports)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_package_imports_before_the_app_registry_is_ready():
    """Django imports an installed app's package while it populates the registry, so
    the package must not reach the models."""
    code = (
        "import django\n"
        "from django.conf import settings\n"
        "settings.configure(INSTALLED_APPS=[])\n"
        "import django_ai_sdk.workflows as w\n"
        "assert 'WorkflowService' not in vars(w)\n"
        "import sys\n"
        "assert 'django_ai_sdk.workflows.models' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert result.returncode == 0, result.stderr
