"""The DBOS database URL follows DATABASES, connection options included."""

from __future__ import annotations

from typing import Any

import pytest
from django.test import override_settings

from django_ai_sdk.durable.conf import database_url


def _url(**db: Any) -> str:
    default = {"ENGINE": "django.db.backends.postgresql", "NAME": "app", "USER": "u", "PASSWORD": "p@ss"}
    with override_settings(DATABASES={"default": {**default, **db}}, AI_SDK_DBOS={}):
        return database_url()


def test_a_managed_postgres_keeps_its_ssl_options() -> None:
    url = _url(HOST="db.example.com", PORT=5432, OPTIONS={"sslmode": "require", "connect_timeout": 5})
    assert url == "postgresql://u:p%40ss@db.example.com:5432/app?sslmode=require&connect_timeout=5"


def test_django_only_options_stay_out_of_the_url() -> None:
    url = _url(HOST="db", OPTIONS={"isolation_level": 1, "server_side_binding": True, "pool": {"min_size": 2}})
    assert url == "postgresql://u:p%40ss@db/app"


def test_a_socket_directory_goes_in_the_query() -> None:
    url = _url(HOST="/var/run/postgresql", PORT="5433")
    assert url == "postgresql://u:p%40ss@/app?host=%2Fvar%2Frun%2Fpostgresql&port=5433"


def test_no_host_means_localhost() -> None:
    assert _url() == "postgresql://u:p%40ss@localhost/app"


@pytest.mark.parametrize("url", ["postgresql://dbos@elsewhere/dbos?sslmode=verify-full"])
def test_an_explicit_url_wins(url: str) -> None:
    with override_settings(AI_SDK_DBOS={"DATABASE_URL": url}):
        assert database_url() == url


def test_the_durable_app_says_what_to_install_without_dbos(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from django.apps import apps
    from django.core.exceptions import ImproperlyConfigured

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "dbos" else real(name, *a))
    with pytest.raises(ImproperlyConfigured, match=r"django-ai-sdk\[dbos\]"):
        apps.get_app_config("django_ai_sdk_durable").ready()
