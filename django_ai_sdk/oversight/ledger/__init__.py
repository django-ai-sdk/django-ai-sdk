"""The oversight ledger: hash-chains the record and keeps its content in the vault.

An opt-in app on top of oversight and the vault; docs/content/oversight-ledger.md covers
install, checkpoints, verify and forgetting.
"""

from __future__ import annotations

from django_ai_sdk.oversight.ledger.services import LedgerService

__all__ = ["LedgerService"]
