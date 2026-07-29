"""OAuth 2.1 + PKCE callback view for MCP integrations.

A browser-redirect endpoint (not JSON): the identity provider redirects the browser
here after the user approves the connection, so it must live at a fixed URL — unlike
the *start* leg (building the authorization URL, PKCE, session state), which is plain
business logic on ``Integration.connect()`` reached via the generic
``POST /api/integrations/{name}/connect`` (see ``integrations/views.py``); the client
navigates to that response's ``redirect_url`` itself, no dedicated start endpoint
needed. This view validates the IdP response, exchanges the code, and stores the
token. It lives in the MCP toolkit and is wired under the integrations namespace by
the host project's URLconf. Config is resolved from the integrations registry — there
is no ``AI_SDK_INTEGRATIONS`` setting.
"""

from __future__ import annotations

import hmac
import logging
from http import HTTPStatus

import httpx
from asgiref.sync import sync_to_async
from django.conf import settings
from django.http import HttpRequest, HttpResponseRedirect, JsonResponse
from django.utils.http import url_has_allowed_host_and_scheme

from django_ai_sdk.integrations.mcp import services as mcp_service
from django_ai_sdk.integrations.mcp.discovery import discover
from django_ai_sdk.integrations.mcp.loader import (
    _K_STATE,
    _K_TOKEN_ENDPOINT,
    _K_VERIFIER,
    DynamicMCPIntegration,
    resolve_client_credentials,
)
from django_ai_sdk.integrations.mcp.schemas import OAuthMCPIntegrationConfig

logger = logging.getLogger(__name__)


class OAuthCallbackError(Exception):
    """Raised during OAuth callback validation with an HTTP error response."""

    def __init__(self, message: str, http_status: int = 400) -> None:
        self.message = message
        self.http_status = http_status


async def _get_oauth_config(server_name: str) -> OAuthMCPIntegrationConfig | None:
    """Resolve a registered OAuth MCP integration's config, or None."""
    from django_ai_sdk.integrations.registry import get_integrations

    integration = (await get_integrations([server_name])).get(server_name)
    if not isinstance(integration, DynamicMCPIntegration) or not isinstance(
        integration.config, OAuthMCPIntegrationConfig
    ):
        return None
    return integration.config


async def _validate_callback_params(
    request: HttpRequest, server_name: str
) -> tuple[str, str, str | None, OAuthMCPIntegrationConfig]:
    """Validate the OAuth callback and extract (code, verifier, token_endpoint, config)."""
    if error := request.GET.get("error"):
        desc = request.GET.get("error_description", "")
        logger.error("OAuth error for %r: %s — %s", server_name, error, desc)
        raise OAuthCallbackError(f"{error}: {desc}")

    code = request.GET.get("code")
    state = request.GET.get("state")
    if not code or not state:
        raise OAuthCallbackError("Missing code or state")

    stored_state = request.session.get(_K_STATE.format(server_name)) or ""
    if not hmac.compare_digest(state, stored_state):
        raise OAuthCallbackError("State mismatch")

    verifier = request.session.get(_K_VERIFIER.format(server_name))
    if not verifier:
        raise OAuthCallbackError("Missing code verifier")

    token_endpoint = request.session.get(_K_TOKEN_ENDPOINT.format(server_name))

    config = await _get_oauth_config(server_name)
    if config is None:
        raise OAuthCallbackError("Server not found or not OAuth type", http_status=404)

    return code, verifier, token_endpoint, config


async def _resolve_token_endpoint(
    config: OAuthMCPIntegrationConfig, session_endpoint: str | None
) -> str:
    """Resolve the token endpoint from session, config, or discovery."""
    if session_endpoint:
        return session_endpoint
    if config.token_endpoint:
        return config.token_endpoint
    try:
        discovery = await discover(config.oauth_discovery_url or config.url)
    except (httpx.HTTPError, ValueError) as e:
        raise OAuthCallbackError(f"Cannot determine token endpoint: {e}", http_status=500) from e
    return discovery.token_endpoint


async def oauth_callback(
    request: HttpRequest, server_name: str
) -> HttpResponseRedirect | JsonResponse:
    """Handle the OAuth 2.1 callback: validate, exchange the code, store the token."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Not authenticated"}, status=HTTPStatus.UNAUTHORIZED)

    try:
        code, verifier, raw_endpoint, config = await _validate_callback_params(request, server_name)
        client_id, client_secret = await resolve_client_credentials(server_name, config)

        for key_template in (_K_STATE, _K_VERIFIER, _K_TOKEN_ENDPOINT):
            request.session.pop(key_template.format(server_name), None)
        await sync_to_async(request.session.save)()

        token_endpoint = await _resolve_token_endpoint(config, raw_endpoint)
        redirect_uri = request.build_absolute_uri(request.path)

        try:
            token_response = await mcp_service.exchange_token(
                token_endpoint=token_endpoint,
                code=code,
                redirect_uri=redirect_uri,
                verifier=verifier,
                client_id=client_id,
                client_secret=client_secret,
            )
            await mcp_service.store_token(
                user=request.user, server_name=server_name, token_response=token_response
            )
        except (httpx.HTTPError, ValueError) as e:
            logger.exception("Token exchange/store failed for %r", server_name)
            return JsonResponse({"error": f"Token exchange failed: {e}"}, status=500)

        success_url = getattr(settings, "AI_SDK_MCP_OAUTH_SUCCESS_URL", "/")
        if not url_has_allowed_host_and_scheme(success_url, allowed_hosts={request.get_host()}):
            success_url = "/"
        sep = "&" if "?" in success_url else "?"
        return HttpResponseRedirect(f"{success_url}{sep}connected={server_name}")

    except OAuthCallbackError as e:
        return JsonResponse({"error": e.message}, status=e.http_status)
    except Exception:
        logger.exception("Unexpected error in OAuth callback for %r", server_name)
        return JsonResponse({"error": "Unexpected error"}, status=500)
