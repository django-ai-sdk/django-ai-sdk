"""Service-layer facade for integrations — the ``AssistantService`` counterpart.

``Integration`` (``integrations/base.py``) is the per-item business logic, one
instance per configured integration — the role ``Assistant`` plays for assistants.
``IntegrationService`` resolves by name, permission-checks, and delegates to the
instance — the role ``AssistantService`` plays for assistants. ``integrations/views.py``
and ``AssistantService`` are its two callers today; either could be replaced by a
management command or another host app without duplicating this logic.

No ``PermissionsMixin`` here (contrast ``AssistantService``): every operation below
already has a concrete ``Integration`` instance to delegate ``has_perms`` to,
so there is no domain-level check that needs to run without one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from django_ai_sdk.integrations.base import IntegrationNotConnectable, IntegrationStatus
from django_ai_sdk.integrations.registry import get_all_integrations, get_integrations
from django_ai_sdk.integrations.schemas import IntegrationOut
from django_ai_sdk.permissions import Operation, PermissionDenied

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser
    from django.http import HttpRequest

    from django_ai_sdk.integrations.base import Integration

logger = logging.getLogger(__name__)


async def _safe_status_and_tools(
    name: str,
    svc: Integration,
    user: AbstractBaseUser | AnonymousUser | None,
) -> tuple[IntegrationStatus, list[str]]:
    """One integration's (status, tool_names), isolated from the rest of a fan-out.

    Shared by ``IntegrationService.list_for_user`` and
    ``AssistantService.get_integration_status`` — a slow/broken integration (a dead
    server, a DB hiccup fetching an OAuth token) degrades to DEGRADED/[] instead of
    failing whichever list it's part of.
    """
    try:
        status = await svc.get_status(user)
        tool_names = await svc.get_tool_names(user)
    except Exception:
        logger.exception("Failed to get status for integration %r", name)
        return IntegrationStatus.DEGRADED, []
    return status, tool_names


class IntegrationService:
    """Resolve, permission-check, and act on integrations by name."""

    @classmethod
    async def get(cls, name: str) -> Integration | None:
        """Return the named integration, or None if it isn't configured."""
        return (await get_integrations([name])).get(name)

    @classmethod
    async def list_for_user(
        cls, user: AbstractBaseUser | AnonymousUser | None
    ) -> list[IntegrationOut]:
        """Every integration ``user`` may use, with real (not guessed) status.

        Concurrent, not sequential — one cold/dead integration must not make the whole
        settings page pay N x its timeout (each ``get_status()``/``get_tool_names()``
        is individually bounded by its own ``ResilientCache``).
        """
        integrations = await get_all_integrations()

        async def _row(name: str, svc: Integration) -> IntegrationOut | None:
            if not await svc.has_perms(user, Operation.USE_INTEGRATION):
                return None
            status, _tool_names = await _safe_status_and_tools(name, svc, user)
            return IntegrationOut(
                name=name,
                label=svc.label,
                kind=svc.kind,
                status=status,
                supports_connect=svc.supports_connect,
                supports_test=svc.supports_test,
                connect_kind=svc.connect_kind,
                detail=svc.detail,
                connected=status == IntegrationStatus.ACTIVE,
            )

        rows = await asyncio.gather(*(_row(name, svc) for name, svc in integrations.items()))
        return [row for row in rows if row is not None]

    @classmethod
    async def connect(
        cls,
        name: str,
        user: AbstractBaseUser | AnonymousUser | None,
        *,
        request: HttpRequest,
        redirect_uri: str,
    ) -> dict[str, Any] | None:
        """Begin connecting ``name`` for ``user``.

        Returns None for an unknown integration, raises ``PermissionDenied`` when the
        user isn't allowed to manage it, and lets ``IntegrationNotConnectable``
        propagate for a capability mismatch (e.g. connect() on a non-OAuth server) —
        callers (the router) translate each into the matching HTTP status.
        """
        svc = await cls.get(name)
        if svc is None:
            return None
        if not await svc.has_perms(user, Operation.MANAGE_INTEGRATION):
            raise PermissionDenied(f"Not permitted to manage {name!r}")
        return await svc.connect(user, request=request, redirect_uri=redirect_uri)

    @classmethod
    async def disconnect(
        cls, name: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> bool | None:
        """Drop ``user``'s stored connection for ``name``. None if unknown, raises if forbidden."""
        svc = await cls.get(name)
        if svc is None:
            return None
        if not await svc.has_perms(user, Operation.MANAGE_INTEGRATION):
            raise PermissionDenied(f"Not permitted to manage {name!r}")
        return await svc.disconnect(user)

    @classmethod
    async def reconnect(
        cls, name: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> IntegrationStatus | None:
        """Force a fresh connection attempt for ``name`` and return the real outcome."""
        svc = await cls.get(name)
        if svc is None:
            return None
        if not await svc.has_perms(user, Operation.USE_INTEGRATION):
            raise PermissionDenied(f"Not permitted to use {name!r}")
        return await svc.test(user)


__all__ = ["IntegrationService", "IntegrationNotConnectable"]
