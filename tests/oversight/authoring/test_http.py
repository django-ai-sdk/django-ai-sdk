"""The authoring routes: who may call them, what they answer, and that each checks a permission."""

from __future__ import annotations

import pytest
from django.test import AsyncClient

from tests.oversight.authoring.helpers import AUTHOR, VIEW, refunds
from tests.oversight.helpers import browser

BASE = "/api/oversight/authoring"


pytestmark = pytest.mark.urls("tests.oversight.review.urls")


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestRoutes:
    async def test_save_read_try_and_withdraw(self):
        client = await browser(AUTHOR, VIEW)
        saved = await client.put(
            f"{BASE}/policies/refunds/", refunds(), content_type="application/json"
        )
        assert saved.status_code == 200, saved.content
        assert saved.json()["loaded"] is True

        listed = await client.get(f"{BASE}/policies/")
        assert [p["name"] for p in listed.json()] == ["refunds"]
        detail = await client.get(f"{BASE}/policies/refunds/")
        assert detail.json()["versions"][0]["version"] == 1

        tried = await client.post(
            f"{BASE}/policies/refunds/try/",
            {"input": "I want a refund"},
            content_type="application/json",
        )
        assert tried.status_code == 200, tried.content
        assert tried.json()["outcome"] == "blocked"

        gone = await client.post(
            f"{BASE}/policies/refunds/withdraw/",
            {"reason": "replaced"},
            content_type="application/json",
        )
        assert gone.status_code == 200 and gone.json()["withdrawn"] is True

    async def test_the_catalog_names_aliases_but_never_their_config(self, settings):
        settings.AI_SDK_OVERSIGHT = {
            **getattr(settings, "AI_SDK_OVERSIGHT", {}),
            "JUDGES": {"judge": {"class": "LLMJudge", "api_key_env": "SECRET_ENV"}},
        }
        client = await browser(VIEW)
        catalog = (await client.get(f"{BASE}/catalog/")).json()
        assert catalog["aliases"] == ["judge"]
        assert "SECRET_ENV" not in str(catalog)
        assert catalog["lens_kinds"] == ["check", "judge", "pattern"]
        assert catalog["answers"] == ["choice", "entities", "flag", "score"]
        assert catalog["inline"]["pii"] == ["kinds"]
        assert "patterns" not in catalog["inline"]
        assert {"name": "au-base", "label": "au-base", "origin": "code"} in catalog["policies"]

    async def test_a_definition_that_does_not_check_is_a_400(self):
        client = await browser(AUTHOR, VIEW)
        bad = await client.put(
            f"{BASE}/policies/au-base/", refunds(), content_type="application/json"
        )
        assert bad.status_code == 400
        assert "declared in code" in bad.json()["message"]

    @pytest.mark.parametrize(
        ("method", "url", "body"),
        [
            ("get", "/catalog/", None),
            ("get", "/policies/", None),
            ("put", "/policies/refunds/", refunds()),
            ("post", "/policies/refunds/try/", {"input": "hi"}),
            ("post", "/policies/refunds/withdraw/", {"reason": "x"}),
        ],
    )
    async def test_every_route_checks_a_permission(self, method, url, body):
        client = await browser()
        call = getattr(client, method)
        kwargs = {"content_type": "application/json"} if body is not None else {}
        response = await call(f"{BASE}{url}", *([body] if body is not None else []), **kwargs)
        assert response.status_code == 403, (url, response.content)

    async def test_a_stranger_is_turned_away(self):
        response = await AsyncClient().get(f"{BASE}/policies/")
        assert response.status_code == 401
