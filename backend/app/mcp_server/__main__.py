"""Serve the doc-pilot MCP server over stdio: `python -m app.mcp_server`.

For a local MCP client that starts its server as a subprocess (Claude Code,
Claude Desktop). It reads the database named by DATABASE_URL (backend/.env),
so run it with the backend virtualenv and, outside Docker, the host port the
compose database publishes.

stdout is the protocol channel, so nothing here prints: logs go to stderr, and
stdio_server points fd 1 at stderr while it serves, so a stray write from a
library lands there too instead of corrupting a frame.
"""

import asyncio
import logging
import sys
from functools import partial

import anyio
from mcp.server.stdio import stdio_server
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db import async_session_maker, engine, verify_tenant_isolation
from app.mcp_server.server import build_server, default_principal
from app.retrieval.embeddings import get_embedder, warm_up_embedder
from app.telemetry import (
    configure_tracing,
    instrument_engine,
    otlp_enabled,
    shutdown_tracing,
)
from app.tenancy import tenant_session

logger = logging.getLogger(__name__)


async def serve() -> None:
    # A host backend/.env that still names the database owner would otherwise
    # read every organization's documents; fail before the client connects.
    await verify_tenant_isolation(engine)
    if configure_tracing("doc-pilot-mcp") and otlp_enabled():
        instrument_engine(engine)
        logger.info("tracing enabled (OTLP export)")
    server = build_server(
        tenant_session_factory=partial(tenant_session, async_session_maker),
        resolve_principal=default_principal,
        embedder=get_embedder(),
        settings=get_settings(),
    )
    warm_up: asyncio.Task | None = None
    try:
        async with stdio_server() as (read_stream, write_stream):
            # Load the embedding model in the background, as the API does: the
            # client is connected and listing tools while it loads. Not awaited.
            # Started inside stdio_server, after it points fd 1 at stderr, so a
            # library that writes to stdout while loading cannot corrupt a frame.
            warm_up = asyncio.create_task(run_in_threadpool(warm_up_embedder))
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        if warm_up is not None:
            warm_up.cancel()
        await engine.dispose()
        shutdown_tracing()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    anyio.run(serve)
