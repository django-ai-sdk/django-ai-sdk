"""The review routes over HTTP: the service's answers, with 403, 404 and 409 where they belong."""

from __future__ import annotations

import pytest
from asgiref.sync import sync_to_async
from django.test import AsyncClient

from django_ai_sdk.oversight.models import ReviewItem
from tests.oversight.helpers import client_for
from tests.oversight.review.helpers import CONTENT, METRICS, REVIEW, a_turn, reader

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.urls("tests.oversight.review.urls"),
]


async def test_a_member_is_refused():
    await a_turn()
    client = await client_for(await reader(staff=True))
    for url in ("/api/oversight/turns/", "/api/oversight/reviews/", "/api/oversight/metrics/"):
        response = await client.get(url)
        assert response.status_code == 403, url


async def test_signed_out_is_refused():
    response = await AsyncClient().get("/api/oversight/turns/")
    assert response.status_code == 401


async def test_signed_out_is_refused_under_an_api_without_auth():
    for url in ("/bare/oversight/turns/", "/bare/oversight/metrics/"):
        assert (await AsyncClient().get(url)).status_code == 401, url


async def test_the_router_signs_in_by_session_and_checks_csrf():
    """Its own auth, `django_auth`: whatever the API around it uses, a POST needs the token."""
    await a_turn()
    user = await reader(REVIEW)
    item = await ReviewItem.objects.aget()
    client = AsyncClient(enforce_csrf_checks=True)
    await sync_to_async(client.force_login)(user)
    url = f"/api/oversight/reviews/{item.id}/"
    refused = await client.post(url, {"dismiss": True}, content_type="application/json")
    assert refused.status_code == 403
    await item.arefresh_from_db()
    assert item.status == ReviewItem.Status.OPEN
    assert (await client.get("/api/oversight/reviews/")).status_code == 200


def test_the_router_takes_another_auth_when_given_one():
    from ninja.security import django_auth, django_auth_superuser

    from django_ai_sdk.contrib.ninja import get_oversight_router

    assert get_oversight_router().auth is django_auth
    assert get_oversight_router(auth=django_auth_superuser).auth is django_auth_superuser


async def test_a_reviewer_of_their_own_turn_is_refused():
    asker = await reader(REVIEW)
    item = await ReviewItem.objects.aget(turn_id=await a_turn(user=asker))
    client = await client_for(asker)
    response = await client.post(
        f"/api/oversight/reviews/{item.id}/",
        {"agree": False},
        content_type="application/json",
    )
    assert response.status_code == 403


async def test_the_feed_and_one_turn():
    turn_id = await a_turn()
    client = await client_for(await reader(METRICS, CONTENT))
    page = (await client.get("/api/oversight/turns/", {"label": "topic", "value": "hr"})).json()
    assert [row["id"] for row in page["items"]] == [turn_id]
    assert page["next_cursor"] == ""
    detail = (await client.get(f"/api/oversight/turns/{turn_id}/")).json()
    assert detail["content"]["input"] == "What about my salary?"
    assert (await client.get("/api/oversight/turns/nope/")).status_code == 404
    assert (await client.get("/api/oversight/turns/", {"cursor": "bad"})).status_code == 400


async def test_an_unknown_filter_value_is_refused_not_an_empty_page():
    client = await client_for(await reader(METRICS, REVIEW))
    for url, params in (
        ("/api/oversight/turns/", {"status": "closd"}),
        ("/api/oversight/turns/", {"outcome": "blokced"}),
        ("/api/oversight/turns/", {"review": "maybe"}),
        ("/api/oversight/reviews/", {"status": "opn"}),
    ):
        assert (await client.get(url, params)).status_code == 422, params
    assert (await client.get("/api/oversight/turns/", {"status": "closed"})).status_code == 200


async def test_resolving_twice_is_a_conflict():
    await a_turn()
    client = await client_for(await reader(REVIEW))
    [item] = (await client.get("/api/oversight/reviews/")).json()["items"]
    url = f"/api/oversight/reviews/{item['id']}/"
    body = {"agree": False, "note": "Not HR."}
    first = await client.post(url, body, content_type="application/json")
    assert first.status_code == 200, first.content
    assert first.json()["status"] == "done"
    again = await client.post(url, body, content_type="application/json")
    assert again.status_code == 409
    assert (await client.get("/api/oversight/queues/")).json() == {}


async def test_a_close_without_a_verdict_is_a_400():
    await a_turn()
    client = await client_for(await reader(REVIEW))
    [item] = (await client.get("/api/oversight/reviews/")).json()["items"]
    bad = await client.post(
        f"/api/oversight/reviews/{item['id']}/",
        {"note": "Hmm."},
        content_type="application/json",
    )
    assert bad.status_code == 400


async def test_metrics():
    await a_turn()
    client = await client_for(await reader(METRICS))
    metrics = (await client.get("/api/oversight/metrics/", {"days": 7})).json()
    assert metrics["turns"] == 1
    assert metrics["queues"] == {"hr": 1}


def test_the_router_drops_what_it_is_told_to():
    from django_ai_sdk.contrib.ninja import get_oversight_router

    router = get_oversight_router(exclude={"get_oversight_metrics"})
    paths = set(router.path_operations)
    assert "/oversight/metrics/" not in paths
    assert "/oversight/turns/" in paths
