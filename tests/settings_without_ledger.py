"""The test settings without the opt-in ledger and vault apps: core, review and switches must
run the same, on core's own custody.

    uv run pytest tests/oversight --ds=tests.settings_without_ledger \\
        --ignore=tests/oversight/ledger
"""

from tests.settings import *  # noqa: F403

INSTALLED_APPS = [  # noqa: F405
    app
    for app in INSTALLED_APPS  # noqa: F405
    if app not in {"django_ai_sdk.oversight.ledger", "django_ai_sdk.vault"}
]
