"""Reviewers racing each other and reflect, each on its own connection.

SQLite serialises every writer, so these need PostgreSQL: set `DATABASE_URL`
(see tests/settings.py).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from asgiref.sync import async_to_sync
from django_ai_sdk.oversight.models import LensAnswer, ReviewItem, RuleMatch, Turn
from django_ai_sdk.oversight.reflect import areflect
from django_ai_sdk.oversight.schemas import ResolveIn
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.permissions import ConflictError

from tests.oversight.helpers import race
from tests.oversight.review.helpers import REVIEW, a_turn, reader
from tests.oversight.test_durability import close, start

ROUNDS = 3


pytestmark = pytest.mark.usefixtures("postgres_only")


def reflected_later() -> str:
    """A closed turn whose reflect lens `later` will answer yes: rule `tell_later` notifies."""

    async def make() -> str:
        guard = await start("test-durable-reflect", "hello")
        await guard.gate()
        await guard.finish("a bad answer")
        return str((await close(guard)).id)

    return async_to_sync(make)()


@pytest.mark.django_db(transaction=True)
class TestReviewRacesOnPostgres:
    def test_of_many_reviewers_resolving_one_item_one_wins(self):
        for _ in range(ROUNDS):
            turn_id = async_to_sync(a_turn)()
            item = ReviewItem.objects.get(turn_id=turn_id)
            reviewers = [async_to_sync(reader)(REVIEW) for _ in range(6)]

            def resolve(user: Any) -> Any:
                try:
                    return async_to_sync(OversightService.resolve_review)(
                        str(item.id), ResolveIn(lens="topic", value="other"), user=user
                    )
                except ConflictError:
                    return None

            results = race(*[lambda user=user: resolve(user) for user in reviewers])

            [won] = [r for r in results if r is not None]
            item.refresh_from_db()
            assert item.status == ReviewItem.Status.DONE
            assert item.resolution["by"] == won.resolution["by"]
            # Only the winner's answer was written: the losers wrote nothing.
            assert list(
                LensAnswer.objects.filter(turn_id=turn_id)
                .exclude(reviewer="")
                .values_list("reviewer", flat=True)
            ) == [won.resolution["by"]]

    def test_reflect_and_a_reviewer_matching_one_rule_at_once_match_it_once(self):
        reviewer = async_to_sync(reader)(REVIEW)
        for _ in range(ROUNDS):
            turn_id = reflected_later()
            item = ReviewItem.objects.create(turn_id=turn_id, queue="later", lens="later")

            def review() -> Any:
                return async_to_sync(OversightService.resolve_review)(
                    str(item.id), ResolveIn(value=True), user=reviewer
                )

            with patch("django_ai_sdk.oversight.persist.enqueue_notifications") as enqueue:
                race(lambda: async_to_sync(areflect)(turn_id), review)

            # Both answers are kept; the rule matched once, by whichever wrote first.
            answers = LensAnswer.objects.filter(turn_id=turn_id, lens="later")
            assert sorted(answers.values_list("reviewer", flat=True)) == ["", str(reviewer.pk)]
            assert RuleMatch.objects.filter(turn_id=turn_id, rule="tell_later").count() == 1
            assert Turn.objects.get(id=turn_id).reflect_state == "done"
            # And it notified once: the one that matched it sends.
            events = [event for call in enqueue.call_args_list for event, _ in call.args[0]]
            assert [event["rule"] for event in events] == ["tell_later"]
