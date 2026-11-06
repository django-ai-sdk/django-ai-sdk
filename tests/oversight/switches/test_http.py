"""The switch routes: who may call them, what they answer, and that each checks a permission."""

from __future__ import annotations

import pytest
from django.test import AsyncClient
from django_ai_sdk.oversight.models import AccessRecord

from tests.oversight.helpers import browser, make_agent
from tests.oversight.switches.helpers import SWITCH, VIEW

pytestmark = pytest.mark.urls("tests.oversight.review.urls")


async def post(client: AsyncClient, url: str, body: dict):
    return await client.post(f"/api/oversight/{url}", body, content_type="application/json")


@pytest.mark.usefixtures("no_reflect_queue", "sent")
@pytest.mark.django_db(transaction=True)
class TestRoutes:
    async def test_a_rule_from_report_to_enforce_and_back(self):
        root = await browser(SWITCH, VIEW)
        on = await post(root, "switches/rules/sw-parent/block_x/", {"mode": "enforce"})
        assert on.status_code == 200, on.content
        assert on.json()["effective"] == "enforce"
        listed = await root.get("/api/oversight/switches/")
        assert [s["target"] for s in listed.json()] == ["block_x"]
        back = await post(
            root, "switches/rules/sw-parent/block_x/", {"mode": None, "reason": "too eager"}
        )
        assert back.json()["effective"] == "report"
        assert await AccessRecord.objects.filter(operation="switch").acount() == 2

    async def test_what_is_refused_and_why(self, registered):
        root = await browser(SWITCH)
        url = "switches/rules/sw-parent/block_x/"
        assert (await post(root, url, {"mode": "enforced"})).status_code == 422
        assert (await post(root, "switches/rules/nope/x/", {"mode": "off"})).status_code == 404
        no_rule = await post(root, "switches/rules/sw-parent/nope/", {"mode": "off"})
        assert no_rule.status_code == 404
        assert (await post(root, "switches/agents/missing/", {"policy": ""})).status_code == 404
        agent_id = registered(make_agent([], None, policy="sw-parent"))
        no_reason = await post(root, f"switches/agents/{agent_id}/", {"policy": ""})
        assert no_reason.status_code == 400
        assert "Say why" in no_reason.json()["message"]
        floor = await post(
            root, "switches/rules/sw-floored/always_block_x/", {"mode": "off", "reason": "r"}
        )
        assert floor.status_code == 400
        ok = await post(root, f"switches/agents/{agent_id}/", {"policy": "", "reason": "retired"})
        assert ok.status_code == 200, ok.content
        assert ok.json()["effective"] == ""

    async def test_every_route_checks_a_permission(self):
        from django_ai_sdk.contrib.ninja.oversight_switches import routes

        nobody = await browser()
        bodies = {"POST": {"mode": "enforce", "policy": "sw-parent"}}
        for method, route, _func, _kwargs in routes._ops:
            url = "/api" + route.replace("{policy}", "sw-parent").replace("{rule}", "block_x")
            url = url.replace("{agent_id}", "a-1")
            if method == "GET":
                response = await nobody.get(url)
            else:
                response = await nobody.post(url, bodies["POST"], content_type="application/json")
            assert response.status_code == 403, (method, route, response.content)

    async def test_anonymous_readers_are_turned_away(self):
        response = await AsyncClient().get("/api/oversight/switches/")
        assert response.status_code == 401
