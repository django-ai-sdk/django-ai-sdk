"""Create or upgrade DBOS's tables. For a release step, before web processes queue runs.

The worker migrates them too when it launches, so this matters only when a web
process may queue a run before the first worker has started.
"""

from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create or upgrade DBOS's system tables (AI_SDK_DBOS['DATABASE_URL'], its SCHEMA)."

    def handle(self, *args: Any, **options: Any) -> None:
        from dbos import DBOS
        from django_ai_sdk.durable.conf import database_url, setting

        DBOS.migrate(database_url(), schema=setting("SCHEMA"))
        self.stdout.write(f"DBOS tables are up to date in schema {setting('SCHEMA')!r}.")
