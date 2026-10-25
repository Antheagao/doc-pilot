"""API keys over HTTP (app/routers/api_keys.py, app/auth/principal.py): the
management routes, bearer access to the REST API, and the one place a secret
may appear. The MCP side of keys is in tests/test_mcp.py."""

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app import ratelimit
from app.auth import api_keys, sessions
from app.auth.principal import Principal, current_principal
from app.auth.tokens import digest, parse_api_key
from app.config import Settings, get_settings
from app.main import app
from app.models import DEFAULT_ORGANIZATION_ID, ApiKey, Organization, User
from app.ratelimit import SlidingWindowLimiter

ORIGIN = {"Origin": "http://localhost:3000"}
CANNOT_MANAGE = {"detail": "an API key cannot manage API keys"}


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _key(
    db: AsyncSession, organization_id: uuid.UUID = DEFAULT_ORGANIZATION_ID, name: str = "test"
) -> tuple[ApiKey, str]:
    return await api_keys.create_api_key(db, organization_id=organization_id, name=name)


async def _session_cookie(db: AsyncSession, user: User) -> str:
    return await sessions.create_session(
        db, user_id=user.id, organization_id=DEFAULT_ORGANIZATION_ID, ttl=timedelta(hours=1)
    )


def _as_member(owner: User) -> None:
    app.dependency_overrides[current_principal] = lambda: Principal(
        DEFAULT_ORGANIZATION_ID, owner.id, "member"
    )


# --- creating and listing ----------------------------------------------------


async def test_creating_a_key_returns_the_secret_once_in_a_201(
    client: AsyncClient, db_session: AsyncSession, owner: User
) -> None:
    response = await client.post("/api-keys", json={"name": "  laptop  "})

    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"api_key", "secret"}
    secret = body["secret"]
    assert parse_api_key(secret) == body["api_key"]["prefix"]
    assert body["api_key"]["name"] == "laptop"
    assert body["api_key"]["last_used_at"] is None and body["api_key"]["revoked_at"] is None
    assert set(body["api_key"]) == {"id", "name", "prefix", "created_at", "last_used_at", "revoked_at"}
    row = (await db_session.execute(select(ApiKey).where(ApiKey.id == uuid.UUID(body["api_key"]["id"])))).scalar_one()
    assert row.organization_id == DEFAULT_ORGANIZATION_ID and row.created_by_user_id == owner.id
    # Only the digest is stored.
    assert row.token_digest == digest(secret) and secret.encode() not in row.token_digest


async def test_the_secret_appears_in_the_201_and_nowhere_else(
    client: AsyncClient, caplog: pytest.LogCaptureFixture, spans
) -> None:
    with caplog.at_level(logging.DEBUG):
        created = (await client.post("/api-keys", json={"name": "laptop"})).json()
        secret = created["secret"]
        listed = await client.get("/api-keys")
        await client.delete(f"/api-keys/{created['api_key']['id']}")
        again = await client.get("/api-keys")
        # A refused request, with the key in play.
        await client.post("/api-keys", json={"name": ""})

    for response in (listed, again):
        assert secret not in response.text
    assert secret not in caplog.text
    assert secret not in "".join(str(record.args) for record in caplog.records)
    for span in spans.get_finished_spans():
        assert secret not in str(dict(span.attributes))
        assert all(secret not in str(event.attributes) for event in span.events)
    schema = (await client.get("/openapi.json")).text
    assert "dpk_" not in schema and "example" not in json.dumps(
        app.openapi()["components"]["schemas"]["ApiKeyCreated"]
    )


async def test_the_list_is_wrapped_newest_first_with_revoked_keys_marked(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    now = datetime.now(UTC)
    old, _ = await _key(db_session, name="old")
    new, _ = await _key(db_session, name="new")
    same, _ = await _key(db_session, name="same-time")
    await db_session.execute(update(ApiKey).where(ApiKey.id == old.id).values(created_at=now - timedelta(hours=2)))
    await db_session.execute(
        update(ApiKey).where(ApiKey.id.in_([new.id, same.id])).values(created_at=now)
    )
    await api_keys.revoke_api_key(db_session, organization_id=DEFAULT_ORGANIZATION_ID, key_id=old.id)

    body = (await client.get("/api-keys")).json()

    assert list(body) == ["api_keys"]
    listed = [item for item in body["api_keys"] if item["id"] in {str(k.id) for k in (old, new, same)}]
    tied = sorted([new.id, same.id])
    assert [item["id"] for item in listed] == [str(tied[0]), str(tied[1]), str(old.id)]
    assert listed[-1]["revoked_at"] is not None and listed[0]["revoked_at"] is None
    assert all("secret" not in item and "token_digest" not in item for item in listed)


async def test_the_list_is_this_organizations_only(
    client: AsyncClient, db_session: AsyncSession, second_organization: Organization
) -> None:
    mine, _ = await _key(db_session)
    theirs, _ = await _key(db_session, second_organization.id)

    ids = {item["id"] for item in (await client.get("/api-keys")).json()["api_keys"]}

    assert str(mine.id) in ids and str(theirs.id) not in ids


@pytest.mark.parametrize("name", ["", "x" * 101, "bad\x00name", "   "])
async def test_a_bad_name_is_a_422(client: AsyncClient, name: str) -> None:
    response = await client.post("/api-keys", json={"name": name})

    assert response.status_code == 422
    assert "secret" not in response.text


async def test_the_cap_of_25_active_keys_is_a_409_until_one_is_revoked(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    active = await db_session.scalar(
        select(func.count()).select_from(ApiKey).where(
            ApiKey.organization_id == DEFAULT_ORGANIZATION_ID, ApiKey.revoked_at.is_(None)
        )
    )
    made = []
    for index in range(api_keys.MAX_ACTIVE_KEYS - active):
        key, _ = await _key(db_session, name=f"k{index}")
        made.append(key)

    full = await client.post("/api-keys", json={"name": "one too many"})
    await api_keys.revoke_api_key(db_session, organization_id=DEFAULT_ORGANIZATION_ID, key_id=made[0].id)
    after = await client.post("/api-keys", json={"name": "fits now"})

    assert full.status_code == 409 and "at most 25" in full.json()["detail"]
    assert after.status_code == 201


# --- revoking ----------------------------------------------------------------


async def test_revoking_ends_the_key_and_then_it_is_a_404(
    anon_client: AsyncClient, db_session: AsyncSession, owner: User
) -> None:
    key, token = await _key(db_session)
    key_id = key.id
    signed_in = {**ORIGIN, "Cookie": f"docpilot_session={await _session_cookie(db_session, owner)}"}
    assert (await anon_client.get("/documents", headers=_bearer(token))).status_code == 200

    revoked = await anon_client.delete(f"/api-keys/{key_id}", headers=signed_in)
    used = await anon_client.get("/documents", headers=_bearer(token))
    again = await anon_client.delete(f"/api-keys/{key_id}", headers=signed_in)

    assert revoked.status_code == 204 and revoked.content == b""
    assert used.status_code == 401
    assert again.status_code == 404
    assert (await anon_client.delete(f"/api-keys/{uuid.uuid4()}", headers=signed_in)).status_code == 404


async def test_another_organizations_key_is_a_404_and_stays_active(
    client: AsyncClient, db_session: AsyncSession, second_organization: Organization
) -> None:
    theirs, token = await _key(db_session, second_organization.id)

    response = await client.delete(f"/api-keys/{theirs.id}")

    assert response.status_code == 404
    assert (await api_keys.find_active_api_key(db_session, token)) is not None


# --- who may manage keys --------------------------------------------------------


async def test_a_member_may_list_but_not_create_or_revoke(
    client: AsyncClient, db_session: AsyncSession, owner: User
) -> None:
    key, _ = await _key(db_session)
    key_id = key.id
    _as_member(owner)

    listed = await client.get("/api-keys")
    created = await client.post("/api-keys", json={"name": "nope"})
    revoked = await client.delete(f"/api-keys/{key_id}")

    assert listed.status_code == 200
    assert created.status_code == revoked.status_code == 403
    assert created.json() == {"detail": "only an owner can manage API keys"}
    assert (
        await db_session.scalar(select(ApiKey.revoked_at).where(ApiKey.id == key_id))
    ) is None  # the refused revoke changed nothing


async def test_a_key_cannot_manage_keys_or_sessions(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    key, token = await _key(db_session)
    key_id = key.id
    headers = {**_bearer(token), **ORIGIN}
    organization = {"organization_id": str(DEFAULT_ORGANIZATION_ID)}

    responses = [
        await anon_client.get("/api-keys", headers=headers),
        await anon_client.post("/api-keys", json={"name": "more"}, headers=headers),
        await anon_client.delete(f"/api-keys/{key_id}", headers=headers),
    ]

    for response in responses:
        assert response.status_code == 403
        assert response.json() == CANNOT_MANAGE
    me = await anon_client.get("/auth/me", headers=headers)
    switch = await anon_client.post("/auth/organization", json=organization, headers=headers)
    assert me.status_code == switch.status_code == 403
    assert me.json() == {"detail": "an API key cannot manage sessions"}
    # And the key is untouched by all of that.
    assert (await anon_client.get("/documents", headers=_bearer(token))).status_code == 200


# --- bearer access to the REST API ------------------------------------------------


async def test_a_key_calls_the_tenant_routes_with_no_cookie_and_no_origin(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    _, token = await _key(db_session)
    headers = _bearer(token)

    assert (await anon_client.get("/documents", headers=headers)).status_code == 200
    assert (await anon_client.get("/stats", headers=headers)).status_code == 200
    assert (await anon_client.get("/review/queue/count", headers=headers)).status_code == 200
    # A write: no Origin, and past the Origin rule (the field does not exist).
    resolve = await anon_client.post(
        f"/review/fields/{uuid.uuid4()}/resolve", json={"action": "approve"}, headers=headers
    )
    assert resolve.status_code == 404


async def test_a_cookie_write_still_needs_an_origin_but_a_key_write_does_not(
    anon_client: AsyncClient, db_session: AsyncSession, owner: User
) -> None:
    _, token = await _key(db_session)
    cookie = await _session_cookie(db_session, owner)
    path = f"/review/fields/{uuid.uuid4()}/resolve"

    with_cookie = await anon_client.post(
        path, json={"action": "approve"}, headers={"Cookie": f"docpilot_session={cookie}"}
    )
    with_key = await anon_client.post(path, json={"action": "approve"}, headers=_bearer(token))

    assert with_cookie.status_code == 403
    assert with_key.status_code == 404


@pytest.mark.parametrize(
    "value",
    ["Bearer", "Bearer ", "Bearer dpk_short", "Bearer " + "a" * 5000, "Bearer dpk_aaaaaaaaaaaa_" + "!" * 43],
    ids=["no token", "empty", "short", "huge", "bad alphabet"],
)
async def test_a_malformed_key_is_a_401_with_no_database_query(
    anon_client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    queries = []
    real_execute = db_session.execute

    async def counting(*args, **kwargs):
        queries.append(args[0])
        return await real_execute(*args, **kwargs)

    monkeypatch.setattr(db_session, "execute", counting)

    response = await anon_client.get("/documents", headers={"Authorization": value})

    assert response.status_code == 401
    assert queries == []


async def test_a_bad_key_is_a_401_even_with_a_good_cookie(
    anon_client: AsyncClient, db_session: AsyncSession, owner: User
) -> None:
    cookie = await _session_cookie(db_session, owner)
    revoked, token = await _key(db_session)
    await api_keys.revoke_api_key(db_session, organization_id=DEFAULT_ORGANIZATION_ID, key_id=revoked.id)

    response = await anon_client.get(
        "/documents", headers={**_bearer(token), "Cookie": f"docpilot_session={cookie}"}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "not authenticated"}


# --- last_used_at ----------------------------------------------------------------------


async def test_last_used_at_is_written_once_in_a_burst_and_again_a_minute_later(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    key, token = await _key(db_session)
    key_id = key.id
    assert key.last_used_at is None

    for _ in range(5):
        assert (await anon_client.get("/documents", headers=_bearer(token))).status_code == 200
    db_session.expire_all()
    first = (await db_session.execute(select(ApiKey.last_used_at).where(ApiKey.id == key_id))).scalar_one()
    for _ in range(3):
        await anon_client.get("/documents", headers=_bearer(token))
    db_session.expire_all()
    second = (await db_session.execute(select(ApiKey.last_used_at).where(ApiKey.id == key_id))).scalar_one()

    assert first is not None and second == first
    later = first + api_keys.LAST_USED_INTERVAL + timedelta(seconds=1)
    await api_keys.authenticate_api_key(db_session, token, now=later)
    db_session.expire_all()
    third = (await db_session.execute(select(ApiKey.last_used_at).where(ApiKey.id == key_id))).scalar_one()
    assert third == later


async def test_a_use_within_the_interval_costs_no_write(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, token = await _key(db_session)
    now = datetime.now(UTC)
    await api_keys.authenticate_api_key(db_session, token, now=now)
    statements: list[str] = []
    real_execute = db_session.execute

    async def spy(statement, *args, **kwargs):
        statements.append(type(statement).__name__)
        return await real_execute(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", spy)
    await api_keys.authenticate_api_key(db_session, token, now=now + timedelta(seconds=30))

    assert "Update" not in statements


async def test_a_failed_last_used_write_still_authenticates_the_key(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    key, token = await _key(db_session)
    key_id, organization_id = key.id, key.organization_id
    real_execute = db_session.execute

    async def failing_update(statement, *args, **kwargs):
        if type(statement).__name__ == "Update":
            raise RuntimeError("lock timeout")
        return await real_execute(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", failing_update)
    with caplog.at_level("WARNING", logger="app.auth.api_keys"):
        authenticated = await api_keys.authenticate_api_key(db_session, token)

    # The caller reads the key after the rollback without touching the database.
    assert authenticated is not None
    assert (authenticated.id, authenticated.organization_id) == (key_id, organization_id)
    assert "could not record last_used_at" in caplog.text
    assert token not in caplog.text


# --- rate limits ---------------------------------------------------------------------------


async def test_the_ask_limit_counts_each_key_apart(
    anon_client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(ratelimit, "_limiter", SlidingWindowLimiter())
    app.dependency_overrides[get_settings] = lambda: Settings(
        upload_dir=str(tmp_path), daily_budget_usd=0, ask_rate_limit_per_minute=1, anthropic_api_key=None
    )
    _, first = await _key(db_session)
    _, second = await _key(db_session)
    body = {"question": "q"}

    codes = [
        (await anon_client.post("/ask", json=body, headers=_bearer(token))).status_code
        for token in (first, first, second)
    ]

    assert codes == [503, 429, 503]
