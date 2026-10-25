import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.auth import passwords
from app.auth.principal import current_principal
from app.config import get_settings
from app.db import async_session_maker, engine, name_process, verify_tenant_isolation
from app.extraction import check_model_provider
from app.mcp_server.auth import ApiKeyVerifier, principal_from_request_context
from app.mcp_server.http import McpHttpEndpoint
from app.mcp_server.server import build_server
from app.retrieval.embeddings import get_embedder, warm_up_embedder
from app.routers import (
    api_keys,
    ask,
    auth,
    chat,
    documents,
    monitoring,
    review,
    search,
    stats,
)
from app.routers.documents import MAX_UPLOAD_SIZE
from app.telemetry import configure_tracing, instrument_api, otlp_enabled
from app.tenancy import tenant_session


class MaxBodySizeMiddleware:
    """Rejects oversized request bodies before they can be fully read.

    Starlette's multipart parser spools file parts to a temp file as it
    reads them — ahead of any application code (including
    UploadFile.read()) getting a chance to check the size — so a check
    inside the route handler alone is too late to stop an oversized
    upload from being written to disk. This ASGI middleware wraps
    `receive` so the body is aborted mid-stream as soon as the cumulative
    byte count crosses max_body_size, plus a cheap upfront check of the
    Content-Length header when the client provides one.
    """

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self.max_body_size
        ):
            response = JSONResponse({"detail": "File too large"}, status_code=413)
            await response(scope, receive, send)
            return

        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_size:
                    raise HTTPException(status_code=413, detail="File too large")
            return message

        await self.app(scope, limited_receive, send)


# Served as-is: the compose healthcheck's answer and the API docs describe no
# one's data.
CACHEABLE_PATHS = frozenset({"/healthz", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"})


class SecurityHeadersMiddleware:
    """Adds X-Content-Type-Options: nosniff to every response, and
    Cache-Control: private, no-store to every response that does not set its
    own (the health check and the API docs excepted).

    The API serves user-uploaded bytes back out via /documents/{id}/file
    with a stored Content-Type; nosniff tells browsers to honor that type
    rather than content-sniffing their own, which (combined with the
    magic-byte check at upload time) closes the stored-payload-served-
    as-image pattern from both ends. Every response is for one signed-in
    caller, so none may be stored by a shared cache or heuristically reused:
    FileResponse sends a Last-Modified, which lets a cache decide for itself.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cacheable = scope["path"] in CACHEABLE_PATHS

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-content-type-options", b"nosniff"))
                if not cacheable and not any(name.lower() == b"cache-control" for name, _ in headers):
                    headers.append((b"cache-control", b"private, no-store"))
            await send(message)

        await self.app(scope, receive, send_with_headers)


# The MCP server over Streamable HTTP at /mcp (app/mcp_server/). It shares this
# process's embedder (get_embedder() is lazy and cached), rate limiter and
# tracing; its session manager runs inside the lifespan below. It authenticates
# an API key itself (a bearer key, not the cookie the routers use).
mcp_endpoint = McpHttpEndpoint(
    build_server(
        tenant_session_factory=partial(tenant_session, async_session_maker),
        resolve_principal=principal_from_request_context,
        embedder=get_embedder(),
        settings=get_settings(),
    ),
    get_settings(),
    ApiKeyVerifier(async_session_maker),
)


# uvicorn's own logger, because the API configures no logging of its own and an
# INFO line from this module would never print under uvicorn's defaults.
startup_logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # First, so even the isolation check's connection is attributable.
    name_process("doc-pilot-api")
    # Before anything serves: a role that skips row-level security must stop
    # the API here, not turn the database layer off silently.
    await verify_tenant_isolation(engine)
    # Who answers model calls, so a log shows at a glance whether this API can
    # spend (the load stack's must say scripted).
    startup_logger.info(get_settings().model_provider_summary())
    check_model_provider(get_settings())
    # Build the dummy hash an unknown-email sign-in verifies against, so the
    # first one does not pay for two hashes and answer visibly slower.
    await passwords.warm_up()
    # Warm the embedding model in the background: /search embeds every
    # query, and the first one shouldn't pay the model load. Not awaited,
    # so the API is serving (and /healthz passing) while it loads.
    warm_up = asyncio.create_task(run_in_threadpool(warm_up_embedder))
    async with mcp_endpoint.run():
        yield
    warm_up.cancel()


# FastAPI's built-in OpenTelemetry (0.142+) is off: doc-pilot configures
# tracing itself (app/telemetry.py, opt-in via OTEL_EXPORTER_OTLP_ENDPOINT)
# and instruments the app with the OpenTelemetry ASGI middleware. Left on,
# FastAPI would start its own root span for every request whenever any
# tracer provider is set, and could add a second set of OTLP exporters
# from the same environment variables.
app = FastAPI(
    title="doc-pilot",
    lifespan=lifespan,
    telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's 422, except that a refused sign-in never echoes the request:
    each error carries the `input` that failed, which for a short password is the
    password and for a missing email is the whole body with it."""
    if request.url.path != "/auth/login":
        return await request_validation_exception_handler(request, exc)
    errors = [
        {key: value for key, value in error.items() if key not in ("input", "ctx")}
        for error in exc.errors()
    ]
    return JSONResponse({"detail": errors}, status_code=422)


# Credentialed CORS: the frontend's origin may send the session cookie and
# read the answer. The origins are explicit (a "*" is refused at load,
# config.py) and so are the methods and headers: the frontend sends GET, POST,
# and DELETE with a JSON content type, and Retry-After is exposed so a page can
# say when a 429 lifts. CORS only lets a page read a response; it is the Origin
# check on cookie-authenticated writes (auth/principal.py) that refuses a forged
# one. /mcp is not for pages: it answers any Origin with 403 itself
# (mcp_server/http.py), whatever these settings say.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["Retry-After"],
)

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(MaxBodySizeMiddleware, max_body_size=MAX_UPLOAD_SIZE)

# Every router but auth needs a signed-in caller; /auth/login and /auth/logout
# need none, and the two other /auth routes ask for it themselves.
authenticated = [Depends(current_principal)]
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"], dependencies=authenticated)
app.include_router(documents.router, prefix="/documents", tags=["documents"], dependencies=authenticated)
app.include_router(chat.router, prefix="/documents", tags=["chat"], dependencies=authenticated)
app.include_router(review.router, prefix="/review", tags=["review"], dependencies=authenticated)
app.include_router(stats.router, prefix="/stats", tags=["stats"], dependencies=authenticated)
app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"], dependencies=authenticated)
app.include_router(search.router, prefix="/search", tags=["search"], dependencies=authenticated)
app.include_router(ask.router, prefix="/ask", tags=["ask"], dependencies=authenticated)
# Plain Routes, not a Mount: a mount answers 307 to every request, which MCP
# clients do not follow on a POST. "/mcp/" is routed too, because Starlette
# would otherwise redirect it the same way, and a pasted URL often ends in a
# slash. Not APIRoutes, so neither is in /openapi.json.
app.router.routes.append(Route("/mcp", endpoint=mcp_endpoint))
app.router.routes.append(Route("/mcp/", endpoint=mcp_endpoint))

# Tracing is opt-in via OTEL_EXPORTER_OTLP_ENDPOINT (see app/telemetry.py);
# when it's off this is a no-op and no instrumentation is installed.
if configure_tracing("doc-pilot-api") and otlp_enabled():
    instrument_api(app, engine)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
