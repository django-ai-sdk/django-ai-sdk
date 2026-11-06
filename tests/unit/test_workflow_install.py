"""Workflows are an opt-in app: the SDK runs without it, and with it the app sets itself up."""

from __future__ import annotations

import pytest
from django.core.exceptions import ImproperlyConfigured
from tests.removable import run_without

# The automations app runs workflows on a schedule: it goes with them.
WORKFLOW_APPS = ["django_ai_sdk.workflows", "django_ai_sdk.automations"]

WITHOUT_WORKFLOWS = """
import sys

from django.apps import apps
from django.core import checks
from django.db import connection

from tests.removable import ask

assert not apps.is_installed("django_ai_sdk.workflows")
tables = connection.introspection.table_names()
assert not [t for t in tables if "workflow" in t], tables
names = [check.__name__ for check in checks.registry.registry.get_checks(include_deployment_checks=True)]
assert not [name for name in names if "workflow" in name], names

answer = ask("Hello", "Sure.")
assert answer == "Sure.", answer
loaded = sorted(name for name in sys.modules if name.startswith("django_ai_sdk.workflows"))
assert not loaded, loaded

# The DRF router leaves the workflow endpoints out.
import importlib.util

if importlib.util.find_spec("rest_framework"):
    from django_ai_sdk.contrib.drf.urls import router

    assert not [prefix for prefix, *_ in router.registry if prefix == "workflows"], router.registry
"""


def test_the_sdk_runs_without_the_workflows_app(tmp_path):
    run_without(
        tmp_path,
        WORKFLOW_APPS,
        WITHOUT_WORKFLOWS,
        # tests.urls mounts the workflow endpoints on purpose; this process has none.
        settings='ROOT_URLCONF = "no_urls"',
        files={"no_urls.py": "urlpatterns = []\n"},
    )


class TestTheAppConfig:
    def test_it_registers_its_own_checks(self):
        from django.core import checks
        from django_ai_sdk.workflows.checks import check_workflows, check_workflows_deploy

        assert check_workflows in checks.registry.registry.get_checks()
        assert check_workflows_deploy in checks.registry.registry.get_checks(
            include_deployment_checks=True
        )

    def test_it_must_come_after_the_core_app(self):
        from django.apps import apps

        config = apps.get_app_config("django_ai_sdk_workflows")
        core = apps.get_app_config("django_ai_sdk")
        reordered = [config, core]

        with (
            pytest.MonkeyPatch.context() as patch,
            pytest.raises(ImproperlyConfigured, match="after 'django_ai_sdk'"),
        ):
            patch.setattr(apps, "get_app_configs", lambda: reordered)
            config.ready()

    def test_the_core_app_holds_no_workflow_model(self):
        import django_ai_sdk.models as core

        assert not [name for name in core.__all__ if "Workflow" in name]


# What may import the workflows package from outside it: the opt-in HTTP contrib
# packages, which mount its routes, the default workflow permission, which reads its
# models lazily (a follow-up moves it into the app), and the opt-in automations app,
# which runs workflows on a schedule.
MAY_IMPORT_WORKFLOWS = (
    "django_ai_sdk.contrib",
    "django_ai_sdk.permissions",
    "django_ai_sdk.automations",
)


def test_the_rest_of_the_sdk_does_not_import_workflows():
    from tests.oversight.test_layers import (
        PACKAGE,
        imported_modules,
        is_under,
        module_name,
        sources,
    )

    leaning = {}
    for path in sources(PACKAGE):
        module = module_name(path)
        if is_under(module, "django_ai_sdk.workflows") or any(
            is_under(module, allowed) for allowed in MAY_IMPORT_WORKFLOWS
        ):
            continue
        found = sorted(t for t in imported_modules(path) if is_under(t, "django_ai_sdk.workflows"))
        if found:
            leaning[module] = found
    assert not leaning, f"imports of the opt-in workflows app: {leaning}"
