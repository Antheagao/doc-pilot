"""API-key authentication for the MCP server, over HTTP and stdio.

A key (`dpk_<prefix>_<secret>`, app/auth/api_keys.py) names one organization and
stands in for a person. Over HTTP, `ApiKeyVerifier` implements the SDK's
`TokenVerifier` protocol and the endpoint (app/mcp_server/http.py) puts the
result on the ASGI scope as an `AuthenticatedUser`, so the SDK's own
`get_access_token()` sees it and a tool call reads its principal from
`ctx.request.scope`. Over stdio there is no request: the process holds one key
(DOCPILOT_API_KEY) and checks it again before every tool call, so a key revoked
while a client is connected stops working at its next call.

The `AccessToken` never holds the secret: its `token` is the key's prefix, which
is public (it is what a key list shows), because the SDK keeps and may log the
object.
"""

import uuid
from typing import TYPE_CHECKING

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from pydantic import SecretStr

from app.auth import api_keys
from app.auth.principal import Principal
from app.auth.tokens import API_KEY_SCHEME
from app.models import ApiKey
from app.tenancy import SessionFactory

if TYPE_CHECKING:
    from mcp.server import ServerRequestContext

SCOPE = "mcp:tools"
CLIENT_ID_PREFIX = "api-key:"


class InvalidApiKey(Exception):
    """The key behind a call is missing, unknown, or revoked."""


def access_token_for(key: ApiKey) -> AccessToken:
    return AccessToken(
        token=f"{API_KEY_SCHEME}_{key.prefix}",
        client_id=f"{CLIENT_ID_PREFIX}{key.id}",
        scopes=[SCOPE],
        subject=str(key.organization_id),
        claims={"organization_id": str(key.organization_id), "api_key_id": str(key.id)},
    )


class ApiKeyVerifier:
    """Looks a bearer token up as an active API key (`TokenVerifier`)."""

    def __init__(self, identity_sessions: SessionFactory) -> None:
        self._identity_sessions = identity_sessions

    async def verify_token(self, token: str) -> AccessToken | None:
        async with self._identity_sessions() as db:
            key = await api_keys.authenticate_api_key(db, token)
        return None if key is None else access_token_for(key)


async def principal_from_request_context(ctx: "ServerRequestContext") -> Principal:
    """Who an HTTP tool call is, from the user the endpoint put on the request's
    scope. Raises InvalidApiKey when there is none, which the endpoint's check
    makes impossible, so a call that gets here without one is refused rather
    than guessed at."""
    user = ctx.request.scope.get("user") if ctx.request is not None else None
    if not isinstance(user, AuthenticatedUser):
        raise InvalidApiKey("the request carries no authenticated API key")
    claims = user.access_token.claims or {}
    try:
        return Principal(
            organization_id=uuid.UUID(claims["organization_id"]),
            api_key_id=uuid.UUID(claims["api_key_id"]),
        )
    except (KeyError, ValueError):
        raise InvalidApiKey("the request's API key has no organization") from None


class StdioPrincipals:
    """The principal of every call over stdio: the process's one key, looked up
    again each time."""

    def __init__(self, identity_sessions: SessionFactory, key: SecretStr) -> None:
        self._identity_sessions = identity_sessions
        self._key = key

    async def authenticate(self) -> Principal | None:
        async with self._identity_sessions() as db:
            key = await api_keys.authenticate_api_key(db, self._key.get_secret_value())
        return None if key is None else Principal(organization_id=key.organization_id, api_key_id=key.id)

    async def __call__(self, ctx: "ServerRequestContext") -> Principal:
        principal = await self.authenticate()
        if principal is None:
            raise InvalidApiKey("the API key is no longer valid")
        return principal
