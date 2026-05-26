"""OAuth 2.1 + PKCE flow for MCP servers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from urllib.parse import urlencode

import httpx
from django.conf import settings
from django.http import HttpRequest, HttpResponseRedirect, JsonResponse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_http_methods

from django_ai_sdk.mcp.discovery import discover
from django_ai_sdk.mcp.models import MCPOAuthClient, MCPOAuthToken

logger = logging.getLogger(__name__)

# Session key templates — server_name is interpolated at runtime
_K_STATE = "mcp_oauth_state_{}"
_K_VERIFIER = "mcp_oauth_verifier_{}"
_K_TOKEN_ENDPOINT = "mcp_oauth_token_endpoint_{}"

_SESSION_KEYS = (
    _K_STATE,
    _K_VERIFIER,
    _K_TOKEN_ENDPOINT,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _mcp_servers() -> dict:
    return getattr(settings, "AI_SDK_MCP_SERVERS", {})


@require_http_methods(["GET"])
async def oauth_start(
    request: HttpRequest, server_name: str
) -> HttpResponseRedirect | JsonResponse:
    """Initiate OAuth 2.1 + PKCE flow for an MCP server."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    server = _mcp_servers().get(server_name)
    if server is None or server.type != "oauth":
        return JsonResponse({"error": "Server not found or not OAuth type"}, status=404)

    # Derive the callback URL from our own path so this works regardless of where
    # the SDK URLs are mounted (e.g. /api/mcp/ vs /mcp/).
    callback_path = request.path.removesuffix("/start/") + "/callback/"
    redirect_uri = request.build_absolute_uri(callback_path)

    try:
        if server.authorization_endpoint and server.token_endpoint:
            from django_ai_sdk.mcp.discovery import OAuthDiscovery

            discovery = OAuthDiscovery(
                authorization_endpoint=server.authorization_endpoint,
                token_endpoint=server.token_endpoint,
            )
            logger.info(
                "Using configured OAuth endpoints for %r: auth=%s token=%s",
                server_name,
                discovery.authorization_endpoint,
                discovery.token_endpoint,
            )
        else:
            discovery_url = server.oauth_discovery_url or server.url
            logger.debug("Discovering OAuth endpoints for %r from: %s", server_name, discovery_url)
            discovery = await discover(discovery_url)
            logger.debug(
                "Discovered OAuth endpoints for %r: auth_endpoint=%s token_endpoint=%s",
                server_name,
                discovery.authorization_endpoint,
                discovery.token_endpoint,
            )
    except (httpx.HTTPError, ValueError) as e:
        logger.exception("OAuth start failed for %r", server_name)
        return JsonResponse({"error": f"OAuth start failed: {e}"}, status=500)

    client_id = server.client_id
    client_secret = server.client_secret

    # Dynamic client registration (RFC 7591) if registration_endpoint is available
    if discovery.registration_endpoint and not client_id:
        try:
            # Check if we already have a registered client
            try:
                oauth_client = await MCPOAuthClient.objects.aget(server_name=server_name)
                client_id = oauth_client.client_id
                client_secret = oauth_client.get_client_secret()
                logger.info(
                    "Using previously registered client for %r: client_id=%s",
                    server_name,
                    client_id,
                )
            except MCPOAuthClient.DoesNotExist:
                # Perform dynamic registration
                client_name = getattr(settings, "AI_SDK_MCP_CLIENT_NAME", "MCP OAuth Client")
                registration_data = {
                    "client_name": client_name,
                    "redirect_uris": [redirect_uri],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                }
                async with httpx.AsyncClient(timeout=10) as client:
                    reg_response = await client.post(
                        discovery.registration_endpoint,
                        json=registration_data,
                    )
                reg_response.raise_for_status()
                reg_data = reg_response.json()
                client_id = reg_data.get("client_id")
                client_secret = reg_data.get("client_secret", "")
                if not client_id:
                    return JsonResponse(
                        {"error": "No client_id in registration response"}, status=500
                    )
                logger.info(
                    "Dynamically registered client for %r: client_id=%s", server_name, client_id
                )
                # Store in database
                oauth_client = MCPOAuthClient(server_name=server_name, redirect_uri=redirect_uri)
                oauth_client.set_credentials(client_id, client_secret)
                await oauth_client.asave()
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.exception("Dynamic client registration failed for %r", server_name)
            return JsonResponse({"error": f"Client registration failed: {e}"}, status=500)

    if not client_id:
        return JsonResponse(
            {"error": f"No client credentials available for {server_name!r}"}, status=500
        )

    logger.debug(
        "OAuth start for %r: client_id_prefix=%s, server.url=%s, redirect_uri=%s",
        server_name,
        client_id[:20] if client_id else "EMPTY",
        server.url,
        redirect_uri,
    )

    verifier = secrets.token_urlsafe(96)
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    state = secrets.token_urlsafe(32)

    request.session[_K_STATE.format(server_name)] = state
    request.session[_K_VERIFIER.format(server_name)] = verifier
    request.session[_K_TOKEN_ENDPOINT.format(server_name)] = discovery.token_endpoint
    await request.session.asave()

    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if server.scope:
        auth_params["scope"] = server.scope
    auth_url = discovery.authorization_endpoint + "?" + urlencode(auth_params)
    logger.info("Redirecting to auth: client_id=%s", client_id)
    return HttpResponseRedirect(auth_url)


async def oauth_callback(
    request: HttpRequest, server_name: str
) -> HttpResponseRedirect | JsonResponse:
    """Handle OAuth 2.1 callback and exchange the code for a token."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    server = _mcp_servers().get(server_name)
    if server is None or server.type != "oauth":
        return JsonResponse({"error": "Server not found or not OAuth type"}, status=404)

    if error := request.GET.get("error"):
        desc = request.GET.get("error_description", "")
        logger.error("OAuth error for %r: %s — %s", server_name, error, desc)
        return JsonResponse({"error": f"{error}: {desc}"}, status=400)

    code = request.GET.get("code")
    state = request.GET.get("state")
    if not code or not state:
        return JsonResponse({"error": "Missing code or state"}, status=400)

    stored_state = request.session.get(_K_STATE.format(server_name)) or ""
    if not hmac.compare_digest(state, stored_state):
        return JsonResponse({"error": "State mismatch"}, status=400)

    verifier = request.session.get(_K_VERIFIER.format(server_name))
    if not verifier:
        return JsonResponse({"error": "Missing code verifier"}, status=400)

    token_endpoint = request.session.get(_K_TOKEN_ENDPOINT.format(server_name))

    # Retrieve client credentials: database (dynamic) or config (static)
    client_id = server.client_id
    client_secret = server.client_secret
    try:
        oauth_client = await MCPOAuthClient.objects.aget(server_name=server_name)
        client_id = oauth_client.client_id
        client_secret = oauth_client.get_client_secret()
    except MCPOAuthClient.DoesNotExist:
        pass

    for key_template in _SESSION_KEYS:
        request.session.pop(key_template.format(server_name), None)
    await request.session.asave()

    if not token_endpoint:
        if server.token_endpoint:
            token_endpoint = server.token_endpoint
        else:
            try:
                discovery_url = server.oauth_discovery_url or server.url
                discovery = await discover(discovery_url)
                token_endpoint = discovery.token_endpoint
            except (httpx.HTTPError, ValueError) as e:
                return JsonResponse({"error": f"Cannot determine token endpoint: {e}"}, status=500)

    redirect_uri = request.build_absolute_uri(request.path)
    token_data: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
        "client_id": client_id,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if client_secret:
                token_data.pop("client_id")
                response = await client.post(
                    token_endpoint,
                    data=token_data,
                    auth=(client_id, client_secret),
                )
            else:
                response = await client.post(token_endpoint, data=token_data)

        logger.info("Token exchange %s for %r", response.status_code, server_name)
        response.raise_for_status()

    except httpx.HTTPError as e:
        logger.error("Token exchange failed for %r: %s", server_name, e)
        return JsonResponse({"error": f"Token exchange failed: {e}"}, status=500)

    try:
        token_response = response.json()
    except ValueError as e:
        logger.error("Invalid JSON in token response for %r: %s", server_name, e)
        return JsonResponse({"error": "Invalid token response"}, status=500)

    if "access_token" not in token_response:
        return JsonResponse({"error": "No access token in response"}, status=500)

    token_obj, _ = await MCPOAuthToken.objects.aget_or_create(
        user=request.user,
        server_name=server_name,
    )
    token_obj.set_tokens(token_response)
    await token_obj.asave()
    logger.info("Stored token for %r user=%s", server_name, request.user)

    success_url = getattr(settings, "AI_SDK_MCP_OAUTH_SUCCESS_URL", "/")
    if not url_has_allowed_host_and_scheme(success_url, allowed_hosts={request.get_host()}):
        success_url = "/"
    sep = "&" if "?" in success_url else "?"
    return HttpResponseRedirect(f"{success_url}{sep}connected={server_name}")
