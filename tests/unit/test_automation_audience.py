"""Audience resolution: who a scheduled run acts as.

Getting this wrong is quiet: an automation resolves to nobody and runs zero times.
"""

from __future__ import annotations

import pytest

from django_ai_sdk.automations import Audience


class TestAppPrincipal:
    async def test_resolves_to_a_single_userless_run(self):
        assert await Audience.APP.resolve(None) == [None]

    def test_describes_itself(self):
        assert Audience.APP.describe() == "app"


@pytest.mark.django_db(transaction=True)
class TestSubscribedUsers:
    async def test_no_subscribers_resolves_to_nobody(self):
        from tests.factories.db import UserFactory

        await UserFactory.acreate()
        assert await Audience.SUBSCRIBED.resolve(_automation("example")) == []

    async def test_resolves_only_users_with_an_enabled_row(self):
        from django_ai_sdk.automations.models import AutomationSubscription
        from tests.factories.db import UserFactory

        subscribed = await UserFactory.acreate(email="subscribed@example.com")
        await UserFactory.acreate(email="never-subscribed@example.com")
        opted_out = await UserFactory.acreate(email="opted-out@example.com")

        await AutomationSubscription.objects.acreate(name="example", user=subscribed, enabled=True)
        await AutomationSubscription.objects.acreate(name="example", user=opted_out, enabled=False)

        resolved = await Audience.SUBSCRIBED.resolve(_automation("example"))
        assert [u.get_username() for u in resolved] == ["subscribed@example.com"]

    async def test_only_matches_this_automations_name(self):
        from django_ai_sdk.automations.models import AutomationSubscription
        from tests.factories.db import UserFactory

        user = await UserFactory.acreate()
        await AutomationSubscription.objects.acreate(
            name="other-automation", user=user, enabled=True
        )

        assert await Audience.SUBSCRIBED.resolve(_automation("example")) == []

    def test_describes_itself(self):
        assert Audience.SUBSCRIBED.describe() == "subscribed"


def _automation(name: str):
    class _Stub:
        pass

    stub = _Stub()
    stub.name = name
    return stub
