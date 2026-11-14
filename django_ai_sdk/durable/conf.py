"""The AI_SDK_DBOS setting, read at call time. docs/content/manual/settings.md lists it."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.conf import setting_from

DEFAULTS: dict[str, Any] = {
    # Where DBOS keeps its tables, such as postgresql://user:pass@host/db. Empty uses
    # DATABASES['default'].
    "DATABASE_URL": "",
    # The Postgres schema DBOS keeps its tables in.
    "SCHEMA": "dbos",
    # Who owns the workflows, queues and schedules in the system tables.
    "APP_NAME": "django-ai-sdk",
    # Pinned, so a deploy recovers the runs the last one left. Change it only when step
    # code changes in a way an unfinished run cannot replay.
    "VERSION": "1",
    # False behind a pooler in transaction mode (PgBouncer), which drops LISTEN.
    "USE_LISTEN_NOTIFY": True,
    # Queues by name, with DBOS register_queue options. No cap on "workflows": a run
    # waiting days for an approval holds its slot.
    "QUEUES": {"workflows": {}, "default": {"worker_concurrency": 8}},
}


def setting(key: str) -> Any:
    return setting_from("AI_SDK_DBOS", DEFAULTS, key)


def database_url() -> str:
    """Where DBOS keeps its tables: AI_SDK_DBOS['DATABASE_URL'], or the default database.

    Postgres shares the project's database, in its own schema. SQLite gets a file
    next to the project's, since two writers in one file would lock each other.
    """
    if url := setting("DATABASE_URL"):
        return url
    db = settings.DATABASES["default"]
    engine = db.get("ENGINE", "")
    if "postgresql" in engine:
        auth = quote(db.get("USER") or "", safe="")
        if db.get("PASSWORD"):
            auth += ":" + quote(str(db["PASSWORD"]), safe="")
        host = db.get("HOST") or "localhost"
        port = f":{db['PORT']}" if db.get("PORT") else ""
        return f"postgresql://{auth}@{host}{port}/{quote(str(db['NAME']), safe='')}"
    if "sqlite" in engine:
        name = str(db.get("NAME") or "")
        if not name or name == ":memory:" or name.startswith("file:"):
            raise ImproperlyConfigured(
                "An in-memory SQLite database has no file for DBOS to sit next to. "
                "Set AI_SDK_DBOS['DATABASE_URL']."
            )
        return f"sqlite:///{name}.dbos.sqlite3"
    raise ImproperlyConfigured(
        f"DBOS runs on Postgres or SQLite, not {engine!r}. Set AI_SDK_DBOS['DATABASE_URL']."
    )


__all__ = ["DEFAULTS", "database_url", "setting"]
