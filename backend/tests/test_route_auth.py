"""Every route needs a sign-in unless it is listed here as public.

The test walks `app.routes`, so a route added later without authentication
fails CI by default: it must either answer 401 with no credentials or be named
in PUBLIC with its reason. Cookie sign-in itself is tests/test_auth.py; this
file is the net under every other route, including the two kinds of response
that are not plain JSON (a stream and a file)."""

import re
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.routing import iter_route_contexts
from httpx import AsyncClient

from app.auth import sessions
from app.main import app
from app.models import DEFAULT_ORGANIZATION_ID, Document, User
from app.routers.ask import get_anthropic_client

# No credential needed, and why.
PUBLIC = {
    "/healthz": "the compose healthcheck and load balancers",
    "/auth/login": "how a credential is obtained",
    "/auth/logout": "idempotent: with no session it only clears the cookie; the Origin rule still applies",
    "/openapi.json": "the schema describes routes, never data",
    "/docs": "the Swagger UI page",
    "/docs/oauth2-redirect": "the Swagger UI page",
    "/redoc": "the ReDoc page",
}
# Not yet behind a credential: /mcp has its own Host, Origin, and rate-limit
# checks and gets bearer API keys in the API key task, not cookie sign-in.
AWAITING_API_KEYS = {"/mcp", "/mcp/"}

PATH_PARAMETER = re.compile(r"\{[^}]+\}")
COOKIE = "docpilot_session"
ORIGIN = {"Origin": "http://localhost:3000"}


def _routes() -> list[tuple[set[str], str]]:
    """(methods, path template) of everything the app serves, included routers
    and plain Starlette routes alike (`app.routes` holds the routers lazily)."""
    return [
        (route.methods or {"GET"}, route.path)
        for route in iter_route_contexts(app.router.routes)
        if route.path is not None
    ]


def _walk() -> list[tuple[str, str]]:
    """Every (method, path) the app serves, path parameters filled with a
    random UUID (the routes' ids are all UUIDs)."""
    return [
        (method, PATH_PARAMETER.sub(lambda _: str(uuid.uuid4()), path))
        for methods, path in _routes()
        for method in sorted(methods - {"HEAD", "OPTIONS"})
    ]


def _template_paths() -> set[str]:
    return {path for _, path in _routes()}


def _guarded() -> list[tuple[str, str]]:
    exempt = set(PUBLIC) | AWAITING_API_KEYS  # none has a path parameter
    return [(method, path) for method, path in _walk() if path not in exempt]


def test_the_walk_finds_the_routes() -> None:
    paths = {path for _, path in _walk()}
    assert len(_guarded()) >= 20
    assert any(path.startswith("/documents/") and path.endswith("/file") for path in paths)
    assert "/ask/stream" in paths and "/auth/me" in paths


def test_every_listed_exception_is_a_real_route() -> None:
    """A route renamed or removed must leave the lists too, or they would
    quietly exempt whatever takes its place."""
    assert (set(PUBLIC) | AWAITING_API_KEYS) <= _template_paths()


@pytest.mark.parametrize(("method", "path"), _guarded(), ids=lambda value: str(value)[:48])
async def test_a_route_with_no_credentials_answers_401(
    anon_client: AsyncClient, method: str, path: str
) -> None:
    response = await anon_client.request(method, path, headers=ORIGIN)

    assert response.status_code == 401, f"{method} {path} is reachable without signing in"
    assert response.json() == {"detail": "not authenticated"}


@pytest.mark.parametrize(("method", "path"), _guarded(), ids=lambda value: str(value)[:48])
async def test_a_bearer_token_alone_is_a_401_until_api_keys_exist(
    anon_client: AsyncClient, method: str, path: str
) -> None:
    response = await anon_client.request(
        method, path, headers={"Authorization": "Bearer dpk_aaaaaaaaaaaa_" + "a" * 43}
    )

    assert response.status_code == 401


async def test_the_public_routes_answer_without_a_credential(anon_client: AsyncClient) -> None:
    assert (await anon_client.get("/healthz")).status_code == 200
    assert (await anon_client.get("/openapi.json")).status_code == 200
    assert (await anon_client.get("/docs")).status_code == 200
    # Reached, not turned away: the Origin rule is what refuses these.
    assert (await anon_client.post("/auth/login", json={})).status_code == 403
    assert (await anon_client.post("/auth/logout")).status_code == 403


async def _sign_in(anon_client: AsyncClient, db_session, owner: User) -> None:
    token = await sessions.create_session(
        db_session,
        user_id=owner.id,
        organization_id=DEFAULT_ORGANIZATION_ID,
        ttl=timedelta(hours=1),
        now=datetime.now(UTC),
    )
    anon_client.cookies.set(COOKIE, token, domain="test.local")


async def test_the_cookie_authenticates_a_file_download_the_way_an_img_or_iframe_sends_it(
    anon_client: AsyncClient, db_session, owner: User, tmp_path
) -> None:
    stored = tmp_path / "receipt.png"
    stored.write_bytes(b"\x89PNG not really")
    document = Document(filename="receipt.png", mime_type="image/png", storage_path=str(stored))
    db_session.add(document)
    await db_session.flush()
    url = f"/documents/{document.id}/file"
    assert (await anon_client.get(url)).status_code == 401
    await _sign_in(anon_client, db_session, owner)

    # A subresource request: the cookie and nothing else (no Origin, no headers).
    response = await anon_client.get(url)

    assert response.status_code == 200
    assert response.content == b"\x89PNG not really"
    assert response.headers["content-type"] == "image/png"


async def test_the_cookie_authenticates_the_streams(
    anon_client: AsyncClient, db_session, owner: User
) -> None:
    """Both stream routes answer 401 without a cookie and get past sign-in with
    one: with no model client, /ask/stream answers 503 and the chat stream 404
    for a document that does not exist, both of which come after the
    principal."""
    # backend/.env holds a real key that Settings() reads: never let a test
    # reach the model.
    app.dependency_overrides[get_anthropic_client] = lambda: None
    document_id = uuid.uuid4()
    streams = [
        ("/ask/stream", {"question": "q"}),
        (f"/documents/{document_id}/chat/stream", {"question": "q"}),
    ]
    for path, body in streams:
        assert (await anon_client.post(path, json=body, headers=ORIGIN)).status_code == 401
    await _sign_in(anon_client, db_session, owner)

    asked = await anon_client.post(streams[0][0], json=streams[0][1], headers=ORIGIN)
    chatted = await anon_client.post(streams[1][0], json=streams[1][1], headers=ORIGIN)
    assert asked.status_code == 503
    assert chatted.status_code == 404
    # The Origin rule applies to the signed-in stream like any write.
    assert (await anon_client.post("/ask/stream", json={"question": "q"})).status_code == 403


async def test_responses_are_private_and_not_stored_unless_the_route_says_otherwise(
    anon_client: AsyncClient, db_session, owner: User, tmp_path
) -> None:
    stored = tmp_path / "receipt.png"
    stored.write_bytes(b"\x89PNG not really")
    document = Document(filename="receipt.png", mime_type="image/png", storage_path=str(stored))
    db_session.add(document)
    await db_session.flush()
    assert (await anon_client.get("/documents")).headers["cache-control"] == "private, no-store"
    await _sign_in(anon_client, db_session, owner)

    file = await anon_client.get(f"/documents/{document.id}/file")
    listing = await anon_client.get("/documents")
    me = await anon_client.get("/auth/me")

    assert file.status_code == listing.status_code == 200
    assert file.headers["cache-control"] == listing.headers["cache-control"] == "private, no-store"
    # A route that sets its own is left alone.
    assert me.headers["cache-control"] == "no-store"
    # The health check and the docs describe no one's data.
    for path in ("/healthz", "/openapi.json", "/docs"):
        assert "cache-control" not in (await anon_client.get(path)).headers, path
