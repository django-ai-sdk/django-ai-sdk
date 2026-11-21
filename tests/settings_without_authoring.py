"""The test settings without the opt-in authoring app: policies are code's alone.

    uv run pytest tests/oversight --ds=tests.settings_without_authoring \
        --ignore=tests/oversight/authoring
"""

from tests.settings import *  # noqa: F403

INSTALLED_APPS = [app for app in INSTALLED_APPS if app != "django_ai_sdk.oversight.authoring"]  # noqa: F405
