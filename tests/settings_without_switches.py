"""The test settings without the opt-in switches app: core and review must run the same.

    uv run pytest tests/oversight --ds=tests.settings_without_switches \\
        --ignore=tests/oversight/switches
"""

from tests.settings import *  # noqa: F403

INSTALLED_APPS = [app for app in INSTALLED_APPS if app != "django_ai_sdk.oversight.switches"]  # noqa: F405
