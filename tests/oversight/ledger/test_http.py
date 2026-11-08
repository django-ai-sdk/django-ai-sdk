"""The ledger routes over HTTP: each its own permission, a reveal only with a reason."""

from __future__ import annotations

import pytest
from django.test import AsyncClient

from django_ai_sdk.oversight.ledger.privacy import pseudonym
from django_ai_sdk.oversight.models import AccessRecord
from tests.oversight.helpers import client_for
from tests.oversight.review.helpers import CONTENT, METRICS, REVIEW, a_turn, reader

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.urls("tests.oversight.review.urls"),
]

VIEW = "django_ai_sdk_oversight_ledger.view_checkpoint"
WRITE = "django_ai_sdk_oversight_ledger.add_checkpoint"
REVEAL = "django_ai_sdk_oversight_ledger.reveal_actor"
REASON = "Complaint 2026-118 names this conversation"


async def test_signed_out_is_refused():
    assert (await AsyncClient().get("/api/oversight/ledger/checkpoints/")).status_code == 401


async def test_the_review_permissions_reach_none_of_it():
    turn_id = await a_turn(user=await reader())
    client = await client_for(await reader(METRICS, CONTENT, REVIEW, staff=True))
    assert (await client.get("/api/oversight/ledger/checkpoints/")).status_code == 403
    assert (await client.post("/api/oversight/ledger/checkpoints/")).status_code == 403
    assert (await client.post("/api/oversight/ledger/verify/")).status_code == 403
    reveal = await client.post(
        f"/api/oversight/ledger/turns/{turn_id}/reveal/", {"reason": REASON}, content_type="application/json"
    )
    assert reveal.status_code == 403


async def test_write_list_and_verify_checkpoints():
    await a_turn()
    client = await client_for(await reader(VIEW, WRITE))
    made = await client.post("/api/oversight/ledger/checkpoints/")
    assert made.status_code == 201
    assert made.json()["tree_size"] > 0
    # The checkpoint's own access record is chained after it: one more entry to cover.
    again = await client.post("/api/oversight/ledger/checkpoints/")
    assert again.status_code == 201
    assert again.json()["tree_size"] == made.json()["tree_size"] + 1
    listed = (await client.get("/api/oversight/ledger/checkpoints/")).json()
    assert [c["tree_size"] for c in listed] == [again.json()["tree_size"], made.json()["tree_size"]]
    verified = (await client.post("/api/oversight/ledger/verify/")).json()
    assert verified["ok"] and verified["state"] == "ok" and verified["checkpoint_ok"]
    assert await AccessRecord.objects.filter(operation="verify").aexists()


async def test_reveal():
    asker = await reader()
    turn_id = await a_turn(user=asker)
    client = await client_for(await reader(REVEAL))
    url = f"/api/oversight/ledger/turns/{turn_id}/reveal/"
    short = await client.post(url, {"reason": "why"}, content_type="application/json")
    assert short.status_code == 400
    found = await client.post(url, {"reason": REASON}, content_type="application/json")
    assert found.status_code == 200
    assert found.json()["user_id"] == str(asker.pk)
    assert found.json()["actor_id"] == pseudonym(asker.pk)
    missing = await client.post(
        "/api/oversight/ledger/turns/nope/reveal/", {"reason": REASON}, content_type="application/json"
    )
    assert missing.status_code == 404


async def test_a_truncated_chain_is_a_conflict_not_a_server_error():
    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    await a_turn()
    last = await LedgerEntry.objects.order_by("-seq").afirst()
    await last.adelete()
    client = await client_for(await reader(VIEW, WRITE))
    response = await client.post("/api/oversight/ledger/checkpoints/")
    assert response.status_code == 409
    assert "ends before its head" in response.json()["message"]
