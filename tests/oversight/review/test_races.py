"""Reviewers racing each other, each on its own connection.

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
                        str(item.id), ResolveIn(agree=True), user=user
                    )
                except ConflictError:
                    return None

            results = race(*[lambda user=user: resolve(user) for user in reviewers])

            [won] = [r for r in results if r is not None]
            item.refresh_from_db()
            assert item.status == ReviewItem.Status.DONE
            assert item.resolved_by_id == won.resolved_by
