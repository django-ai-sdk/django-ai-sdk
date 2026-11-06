"""The ledger's tests run on the ledger's custody, as a project with the app installed does."""

from __future__ import annotations

import pytest

from django_ai_sdk.oversight import custody
from django_ai_sdk.oversight.ledger.custody import LedgerCustody


@pytest.fixture(autouse=True)
def custody_under_test():
    before = custody.get()
    keeper = LedgerCustody()
    custody.install(keeper)
    yield keeper
    custody.install(before)
