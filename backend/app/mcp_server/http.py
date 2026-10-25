"""The MCP server over Streamable HTTP, mounted on the API at /mcp.

The same server object as the stdio entry point (app/mcp_server/server.py),
served stateless with plain JSON responses: no session ids, no SSE streams, so
a request needs nothing from an earlier one and the API process holds no
per-client state. Both protocol eras the SDK serves work on this one route.

A request is checked in this order, cheapest and most hostile first:

1. The route is up. The SDK's session manager can be entered only once, so
   `run()` builds a fresh one each time the API's lifespan starts, and
   between runs the route answers 503.
2. Host and Origin (DNS rebinding). The API is for local use, and a web page
   open in the user's browser can make requests to 127.0.0.1 too, so only
   loopback Host values are served, and any Origin header is refused (the
   frontend is not an MCP client). This runs before the rate limiter, so a
   request that is not served spends none of a client's allowance.
3. The method. Only POST is served: a stateless server has no session to end
   and no server-to-client stream to open, and the SDK's handshake-era GET
   would hold a connection open indefinitely, so anything else is a 405.
4. The API key: `Authorization: Bearer dpk_...`, checked by `ApiKeyVerifier`
   (app/mcp_server/auth.py), refused with the API's own 401 body and
   `WWW-Authenticate: Bearer realm="doc-pilot"`. Every request is checked, so a
   key revoked mid-session stops working at its next call. The result goes on
   the ASGI scope as the SDK's `AuthenticatedUser`.
5. The rate limit (every JSON-RPC message is a POST): the "mcp" bucket of
   app/ratelimit.py keyed by API key, refused with the API's own 429 body and
   Retry-After. After the key check, so a request with no valid key spends
   nobody's allowance.
6. The SDK, which refuses a body over MAX_REQUEST_BODY_BYTES with a 413 before
   parsing it.

The Host and Origin checks stay in front of the key: they keep a browser tab or
another machine from reaching an endpoint a key opens, which is still
loopback-only and has no TLS. The key picks the organization the tools read.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException
from mcp.server import Server
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from starlette.authentication import AuthCredentials
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from app.config import Settings
from app.mcp_server.auth import ApiKeyVerifier
from app.ratelimit import enforce_rate_limit

# Only these Host values (any port) are served, and no Origin is: a browser
# sends Origin on a cross-origin request, an MCP client does not.
LOOPBACK_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
    allowed_origins=[],
)
# A tool call is a few hundred bytes; this leaves room for a long query.
MAX_REQUEST_BODY_BYTES = 65_536
RATE_LIMIT_BUCKET = "mcp"


class McpHttpEndpoint:
    """An ASGI app for the /mcp route; `run()` is its lifespan."""

    def __init__(self, server: Server, settings: Settings, verifier: ApiKeyVerifier) -> None:
        self._server = server
        self._settings = settings
        self._verifier = verifier
        self._security = TransportSecurityMiddleware(LOOPBACK_SECURITY)
        self._manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        """Serve requests while the block is open. A fresh session manager
        each time, because the SDK's cannot be entered twice."""
        manager = StreamableHTTPSessionManager(
            app=self._server,
            json_response=True,
            stateless=True,
            security_settings=LOOPBACK_SECURITY,
            max_request_body_size=MAX_REQUEST_BODY_BYTES,
        )
        async with manager.run():
            self._manager = manager
            try:
                yield
            finally:
                self._manager = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        manager = self._manager
        if manager is None:
            response = JSONResponse({"detail": "the MCP endpoint is not running"}, status_code=503)
            await response(scope, receive, send)
            return
        request = Request(scope, receive)
        rejection = await self._security.validate_request(request)
        if rejection is not None:
            await rejection(scope, receive, send)
            return
        if request.method != "POST":
            response = JSONResponse(
                {"detail": "the MCP endpoint takes POST only"}, status_code=405, headers={"Allow": "POST"}
            )
            await response(scope, receive, send)
            return
        scheme, _, bearer = request.headers.get("authorization", "").partition(" ")
        access_token = await self._verifier.verify_token(bearer) if scheme.lower() == "bearer" else None
        if access_token is None:
            response = JSONResponse(
                {"detail": "not authenticated"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="doc-pilot"'},
            )
            await response(scope, receive, send)
            return
        scope["user"] = AuthenticatedUser(access_token)
        scope["auth"] = AuthCredentials(access_token.scopes)
        try:
            enforce_rate_limit(
                request,
                RATE_LIMIT_BUCKET,
                self._settings.mcp_rate_limit_per_minute,
                key=access_token.client_id,
                per="API key",
            )
        except HTTPException as refusal:
            response = JSONResponse(
                {"detail": refusal.detail}, status_code=refusal.status_code, headers=refusal.headers
            )
            await response(scope, receive, send)
            return
        # The middleware makes the user the SDK's own get_access_token() sees.
        await AuthContextMiddleware(manager.handle_request)(scope, receive, send)
