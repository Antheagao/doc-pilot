import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.db import engine
from app.retrieval.embeddings import warm_up_embedder
from app.routers import ask, chat, documents, monitoring, review, search, stats
from app.routers.documents import MAX_UPLOAD_SIZE
from app.telemetry import configure_tracing, instrument_api


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


class SecurityHeadersMiddleware:
    """Adds X-Content-Type-Options: nosniff to every response.

    The API serves user-uploaded bytes back out via /documents/{id}/file
    with a stored Content-Type; nosniff tells browsers to honor that type
    rather than content-sniffing their own, which (combined with the
    magic-byte check at upload time) closes the stored-payload-served-
    as-image pattern from both ends.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-content-type-options", b"nosniff"))
            await send(message)

        await self.app(scope, receive, send_with_headers)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Warm the embedding model in the background: /search embeds every
    # query, and the first one shouldn't pay the model load. Not awaited,
    # so the API is serving (and /healthz passing) while it loads.
    warm_up = asyncio.create_task(run_in_threadpool(warm_up_embedder))
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

# No cookies or HTTP auth are used anywhere, so credentialed CORS is
# deliberately NOT enabled -- allow_credentials would only widen the
# blast radius of a misconfigured origin for zero functional gain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(MaxBodySizeMiddleware, max_body_size=MAX_UPLOAD_SIZE)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/documents", tags=["chat"])
app.include_router(review.router, prefix="/review", tags=["review"])
app.include_router(stats.router, prefix="/stats", tags=["stats"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])

# Tracing is opt-in via OTEL_EXPORTER_OTLP_ENDPOINT (see app/telemetry.py);
# when it's off this is a no-op and no instrumentation is installed.
if configure_tracing("doc-pilot-api"):
    instrument_api(app, engine)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
