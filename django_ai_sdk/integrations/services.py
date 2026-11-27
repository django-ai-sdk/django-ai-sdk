from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

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
    integration: Integration,
    user: AbstractBaseUser | AnonymousUser | None,
) -> tuple[IntegrationStatus, list[str]]:
    try:
        status = await integration.get_status(user)
        tool_names = await integration.get_tool_names(user)
    except Exception:
        logger.exception("Failed to get status for integration %r", name)
        return IntegrationStatus.DEGRADED, []
    return status, tool_names


class IntegrationService:
    @classmethod
    async def get(cls, name: str) -> Integration | None:
        """Return the named integration, or None if it isn't configured."""
        return (await get_integrations([name])).get(name)

    @classmethod
    async def list_for_user(
        cls, user: AbstractBaseUser | AnonymousUser | None
    ) -> list[IntegrationOut]:
        """Every integration user may use, with live status, fetched concurrently."""
        integrations = await get_all_integrations()

        async def _row(name: str, integration: Integration) -> IntegrationOut | None:
            if not await integration.has_perms(user, Operation.USE_INTEGRATION):
                return None
            status, _tool_names = await _safe_status_and_tools(name, integration, user)
            return IntegrationOut(
                name=name,
                label=integration.label,
                hint=integration.hint,
                kind=integration.kind,
                status=status,
                supports_connect=integration.supports_connect,
                supports_test=integration.supports_test,
                connect_kind=integration.connect_kind,
                detail=integration.detail,
                connected=status == IntegrationStatus.ACTIVE,
            )

        rows = await asyncio.gather(
            *(_row(name, integration) for name, integration in integrations.items())
        )
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
        """None if unknown; raises PermissionDenied or IntegrationNotConnectable."""
        integration = await cls.get(name)
        if integration is None:
            return None
        if not await integration.has_perms(user, Operation.MANAGE_INTEGRATION):
            raise PermissionDenied(f"Not permitted to manage {name!r}")
        return await integration.connect(user, request=request, redirect_uri=redirect_uri)

    @classmethod
    async def disconnect(
        cls, name: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> bool | None:
        """Drop user's stored connection for name. None if unknown, raises if forbidden."""
        integration = await cls.get(name)
        if integration is None:
            return None
        if not await integration.has_perms(user, Operation.MANAGE_INTEGRATION):
            raise PermissionDenied(f"Not permitted to manage {name!r}")
        return await integration.disconnect(user)

    @classmethod
    async def reconnect(
        cls, name: str, user: AbstractBaseUser | AnonymousUser | None
    ) -> IntegrationStatus | None:
        """Force a fresh connection attempt for name and return the real outcome."""
        integration = await cls.get(name)
        if integration is None:
            return None
        if not await integration.has_perms(user, Operation.USE_INTEGRATION):
            raise PermissionDenied(f"Not permitted to use {name!r}")
        return await integration.test(user)


list_integrations_for_user = async_to_sync(IntegrationService.list_for_user)
connect_integration = async_to_sync(IntegrationService.connect)
disconnect_integration = async_to_sync(IntegrationService.disconnect)
reconnect_integration = async_to_sync(IntegrationService.reconnect)

__all__ = ["IntegrationNotConnectable", "IntegrationService"]
