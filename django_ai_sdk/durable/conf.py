"""The AI_SDK_DBOS setting, read at call time. docs/content/manual/settings.md lists it."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

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
    # Pinned so a deploy recovers the last one's runs. Change only when unfinished runs
    # cannot replay new step code: runs under the old version are then left alone.
    "VERSION": "1",
    # False behind a pooler in transaction mode (PgBouncer), which drops LISTEN.
    "USE_LISTEN_NOTIFY": True,
    # DBOS register_queue options by name. No cap on "workflows": a run waiting on an
    # approval holds its slot.
    "QUEUES": {"workflows": {}, "default": {"worker_concurrency": 8}},
}


def setting(key: str) -> Any:
    return setting_from("AI_SDK_DBOS", DEFAULTS, key)


# DATABASES OPTIONS that are Django's own, not libpq connection parameters.
DJANGO_ONLY_OPTIONS = frozenset(
    {"isolation_level", "assume_role", "server_side_binding", "pool", "cursor_factory"}
)


def database_url() -> str:
    """Where DBOS keeps its tables: AI_SDK_DBOS['DATABASE_URL'], or the default database.

    Postgres: same database and OPTIONS, own schema. SQLite: a file next to the project's.
    """
    if url := setting("DATABASE_URL"):
        return url
    db = settings.DATABASES["default"]
    engine = db.get("ENGINE", "")
    if "postgresql" in engine:
        return _postgres_url(db)
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


def _postgres_url(db: dict[str, Any]) -> str:
    auth = quote(db.get("USER") or "", safe="")
    if db.get("PASSWORD"):
        auth += ":" + quote(str(db["PASSWORD"]), safe="")
    params = {
        key: str(value)
        for key, value in (db.get("OPTIONS") or {}).items()
        if key not in DJANGO_ONLY_OPTIONS
        and isinstance(value, str | int)
        and not isinstance(value, bool)
    }
    host, port = str(db.get("HOST") or "localhost"), str(db.get("PORT") or "")
    if host.startswith("/"):
        # libpq takes a unix socket directory as ?host=, not in the authority.
        params.update(host=host, **({"port": port} if port else {}))
        host, port = "", ""
    if port:
        host = f"{host}:{port}"
    query = f"?{urlencode(params)}" if params else ""
    return f"postgresql://{auth}@{host}/{quote(str(db['NAME']), safe='')}{query}"


__all__ = ["DEFAULTS", "database_url", "setting"]
