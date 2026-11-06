"""Authoring is opt-in: without the app, policies are code's alone and nothing imports it."""

from __future__ import annotations

from tests.removable import run_without

SCRIPT = """
import sys

from django.db import connection

from django_ai_sdk.oversight import resolve

assert not resolve._loaders, resolve._loaders
assert "django_ai_sdk_oversight_authored_policies" not in connection.introspection.table_names()
authoring = [m for m in sys.modules if m.startswith("django_ai_sdk.oversight.authoring")]
assert not authoring, authoring
"""


def test_the_app_adds_its_loader_on_start():
    from django_ai_sdk.oversight import resolve
    from django_ai_sdk.oversight.authoring.loader import loader

    assert loader.load in resolve._loaders


def test_everything_else_runs_without_the_app(tmp_path):
    run_without(tmp_path, ["django_ai_sdk.oversight.authoring"], SCRIPT)
