from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.routers import documents, review
from app.routers.documents import MAX_UPLOAD_SIZE


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


app = FastAPI(title="doc-pilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MaxBodySizeMiddleware, max_body_size=MAX_UPLOAD_SIZE)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(review.router, prefix="/review", tags=["review"])


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
