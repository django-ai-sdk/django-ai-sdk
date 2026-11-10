"""Ledger routes: checkpoints, chain verification and revealing who asked a turn.

`get_oversight_router()` adds them when the ledger app is installed; the oversight
manual's Review API lists the permission each operation takes.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest

from django_ai_sdk.contrib.ninja.routing import ApiRouter, Limit, found
from django_ai_sdk.oversight.ledger.schemas import CheckpointOut, RevealIn, RevealOut, VerifyOut
from django_ai_sdk.oversight.ledger.services import LedgerService

routes = ApiRouter()


@routes.get("/oversight/ledger/checkpoints/", response=list[CheckpointOut])
async def list_ledger_checkpoints(request: HttpRequest, limit: Limit = 50) -> Any:
    return await LedgerService.checkpoints(user=request.user, limit=limit)


# 204 when nothing was chained since the last checkpoint.
@routes.post("/oversight/ledger/checkpoints/", response={201: CheckpointOut, 204: None})
async def write_ledger_checkpoint(request: HttpRequest) -> Any:
    checkpoint = await LedgerService.write_checkpoint(user=request.user)
    return (201, checkpoint) if checkpoint is not None else (204, None)


@routes.post("/oversight/ledger/verify/", response=VerifyOut)
async def verify_ledger(request: HttpRequest) -> Any:
    return await LedgerService.verify(user=request.user)


@routes.post("/oversight/ledger/turns/{turn_id}/reveal/", response=RevealOut)
async def reveal_oversight_actor(request: HttpRequest, turn_id: str, payload: RevealIn) -> Any:
    return found(
        await LedgerService.reveal(turn_id, user=request.user, reason=payload.reason), "Turn"
    )
