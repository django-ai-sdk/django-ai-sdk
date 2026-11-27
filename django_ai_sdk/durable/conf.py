from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.conf import setting_from

DEFAULTS: dict[str, Any] = {
    "DATABASE_URL": "",
    "SCHEMA": "dbos",
    "APP_NAME": "django-ai-sdk",
    "VERSION": "1",
    "USE_LISTEN_NOTIFY": True,
    "QUEUES": {"workflows": {}, "default": {"worker_concurrency": 8}},
}


def setting(key: str) -> Any:
    return setting_from("AI_SDK_DBOS", DEFAULTS, key)


# DATABASES OPTIONS that are Django's own, not libpq connection parameters.
DJANGO_ONLY_OPTIONS = frozenset(
    {"isolation_level", "assume_role", "server_side_binding", "pool", "cursor_factory"}
)


def database_url() -> str:
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
        params.update(host=host, **({"port": port} if port else {}))
        host, port = "", ""
    if port:
        host = f"{host}:{port}"
    query = f"?{urlencode(params)}" if params else ""
    return f"postgresql://{auth}@{host}/{quote(str(db['NAME']), safe='')}{query}"


__all__ = ["DEFAULTS", "database_url", "setting"]
