"""Sign-in over HTTP (app/routers/auth.py, app/auth/principal.py): the cookie,
the checks in front of the password, logout, the Origin rule for writes, and the
organization switch. Every test runs the real `current_principal` through
`anon_client`, on the test's transaction, with the cheapest argon2 parameters."""

import logging
import threading
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from argon2 import PasswordHasher, profiles
from httpx import AsyncClient
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app import main as main_module
from app import ratelimit
from app.auth import accounts, passwords, sessions
from app.auth.tokens import digest
from app.config import Settings, get_settings
from app.main import app
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    AuthSession,
    Membership,
    Organization,
    User,
)
from app.ratelimit import SlidingWindowLimiter

EMAIL = "ada@example.com"
PASSWORD = "correct horse battery"
ORIGIN = {"Origin": "http://localhost:3000"}
COOKIE_NAME = "docpilot_session"


class CountingHasher:
    """A CHEAPEST-profile hasher that counts its calls."""

    def __init__(self) -> None:
        self._inner = PasswordHasher.from_parameters(profiles.CHEAPEST)
        self._lock = threading.Lock()
        self.hashes = 0
        self.verifies = 0

    def hash(self, password: bytes) -> str:
        with self._lock:
            self.hashes += 1
        return self._inner.hash(password)

    def verify(self, password_hash: str, password: bytes) -> bool:
        with self._lock:
            self.verifies += 1
        return self._inner.verify(password_hash, password)

    def check_needs_rehash(self, password_hash: str) -> bool:
        return self._inner.check_needs_rehash(password_hash)


@pytest.fixture(autouse=True)
def hasher(monkeypatch: pytest.MonkeyPatch) -> CountingHasher:
    counting = CountingHasher()
    monkeypatch.setattr(passwords, "_hasher", counting)
    monkeypatch.setattr(passwords, "_dummy_hash", None)
    return counting


@pytest.fixture
def fresh_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ratelimit, "_limiter", SlidingWindowLimiter())


def _settings(tmp_path, **overrides) -> None:
    """Replace the test settings. The login limits are off unless a test
    sets them."""
    values = {
        "upload_dir": str(tmp_path),
        "daily_budget_usd": 0,
        "ask_rate_limit_per_minute": 0,
        "upload_rate_limit_per_minute": 0,
        "login_rate_limit_per_minute": 0,
        "login_account_rate_limit_per_minute": 0,
        "anthropic_api_key": None,
    } | overrides
    app.dependency_overrides[get_settings] = lambda: Settings(**values)


async def _account(
    db: AsyncSession,
    email: str = EMAIL,
    *,
    password: str = PASSWORD,
    role: str = "owner",
    organization_id: uuid.UUID = DEFAULT_ORGANIZATION_ID,
    since: datetime | None = None,
) -> User:
    user = await accounts.create_user(db, email=email, password=password)
    membership = await accounts.add_membership(
        db, organization_id=organization_id, user_id=user.id, role=role
    )
    if since is not None:
        membership.created_at = since
        await db.flush()
    return user


async def _login(
    client: AsyncClient, email: str = EMAIL, password: str = PASSWORD, headers: dict | None = ORIGIN
):
    return await client.post(
        "/auth/login", json={"email": email, "password": password}, headers=headers or {}
    )


def _cookie_value(response) -> str:
    """The session token a login response sets, for a test to compare or send."""
    header = response.headers["set-cookie"]
    return header.split(";")[0].partition("=")[2]


async def _session_count(db: AsyncSession, user_id: uuid.UUID | None = None) -> int:
    stmt = select(func.count()).select_from(AuthSession)
    if user_id is not None:
        stmt = stmt.where(AuthSession.user_id == user_id)
    return (await db.execute(stmt)).scalar_one()


# --- login -------------------------------------------------------------------


async def test_login_sets_the_session_cookie_with_exact_flags_and_max_age(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    user = await _account(db_session)

    response = await _login(anon_client)

    assert response.status_code == 200
    name, _, rest = response.headers["set-cookie"].partition("=")
    token, _, flags = rest.partition("; ")
    assert name == COOKIE_NAME
    assert len(token) == 43  # token_urlsafe(32)
    assert sorted(flags.split("; ")) == sorted(["HttpOnly", "Max-Age=43200", "Path=/", "SameSite=lax"])
    assert response.headers["cache-control"] == "no-store"
    row = (
        await db_session.execute(select(AuthSession).where(AuthSession.token_digest == digest(token)))
    ).scalar_one()
    assert row.user_id == user.id and row.organization_id == DEFAULT_ORGANIZATION_ID
    lifetime = row.expires_at - datetime.now(UTC)
    assert timedelta(hours=11, minutes=59) < lifetime <= timedelta(hours=12)
    # The body says who signed in and never carries the token or the hash.
    assert response.json()["email"] == EMAIL
    assert response.json()["organization"]["role"] == "owner"
    assert token not in response.text and "argon2" not in response.text


async def test_the_cookie_name_ttl_and_secure_flag_come_from_settings(
    anon_client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    _settings(tmp_path, session_cookie_name="other_session", session_cookie_secure=True, session_ttl_hours=2)
    await _account(db_session)

    header = (await _login(anon_client)).headers["set-cookie"]

    assert header.startswith("other_session=")
    assert sorted(header.partition("; ")[2].split("; ")) == sorted(
        ["HttpOnly", "Max-Age=7200", "Path=/", "SameSite=lax", "Secure"]
    )


async def test_the_cookie_signs_the_browser_in_and_a_read_does_not_write(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    await _account(db_session)
    token = _cookie_value(await _login(anon_client))
    before = (await db_session.execute(select(AuthSession.expires_at))).scalar_one()

    me = await anon_client.get("/auth/me")
    documents = await anon_client.get("/documents")

    assert me.status_code == 200 and me.json()["email"] == EMAIL
    assert me.headers["cache-control"] == "no-store"
    assert documents.status_code == 200
    db_session.expire_all()
    assert (await db_session.execute(select(AuthSession.expires_at))).scalar_one() == before
    assert digest(token) == (await db_session.execute(select(AuthSession.token_digest))).scalar_one()


async def test_a_wrong_password_and_an_unknown_email_get_the_same_401_and_both_verify(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher
) -> None:
    await _account(db_session)
    await passwords.warm_up()
    verifies = hasher.verifies

    wrong = await _login(anon_client, password="not the password")
    assert hasher.verifies == verifies + 1
    unknown = await _login(anon_client, email="nobody@example.com")
    assert hasher.verifies == verifies + 2

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "email or password is incorrect"}
    assert "set-cookie" not in wrong.headers and "set-cookie" not in unknown.headers
    assert await _session_count(db_session) == 0


async def test_a_disabled_user_gets_the_same_401_and_a_verify(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher
) -> None:
    user = await _account(db_session)
    user.disabled_at = datetime.now(UTC)
    await db_session.flush()
    await passwords.warm_up()
    verifies = hasher.verifies

    response = await _login(anon_client)

    assert response.status_code == 401
    assert response.json() == {"detail": "email or password is incorrect"}
    assert hasher.verifies == verifies + 1
    assert await _session_count(db_session) == 0


async def test_the_warm_up_builds_the_dummy_hash_once_so_an_unknown_email_costs_one_hash_less(
    anon_client: AsyncClient, hasher: CountingHasher
) -> None:
    await passwords.warm_up()
    await passwords.warm_up()
    assert hasher.hashes == 1

    await _login(anon_client, email="nobody@example.com")

    assert (hasher.hashes, hasher.verifies) == (1, 1)


async def test_the_apps_lifespan_warms_up_the_password_hasher(
    monkeypatch: pytest.MonkeyPatch, hasher: CountingHasher
) -> None:
    monkeypatch.setattr(main_module, "warm_up_embedder", lambda: None)

    async with app.router.lifespan_context(app):
        assert hasher.hashes == 1


@pytest.mark.parametrize(
    "password", ["x" * 129, "short", ""], ids=["129 characters", "11 or fewer", "empty"]
)
async def test_a_password_outside_12_to_128_characters_is_a_422_and_the_hasher_never_runs(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher, password: str
) -> None:
    await _account(db_session)
    calls = (hasher.hashes, hasher.verifies)

    response = await _login(anon_client, password=password)

    assert response.status_code == 422
    assert (hasher.hashes, hasher.verifies) == calls
    # The refusal never repeats what was typed.
    assert "x" * 20 not in response.text


async def test_a_422_for_a_missing_field_does_not_echo_the_body(
    anon_client: AsyncClient, hasher: CountingHasher
) -> None:
    secret = "a password that must not come back"

    missing = await anon_client.post("/auth/login", json={"password": secret}, headers=ORIGIN)
    wrong_type = await anon_client.post("/auth/login", json={"email": EMAIL, "password": 12}, headers=ORIGIN)

    assert missing.status_code == wrong_type.status_code == 422
    assert secret not in missing.text
    assert '"input"' not in missing.text and '"input"' not in wrong_type.text
    assert missing.json()["detail"][0]["loc"] == ["body", "email"]
    assert (hasher.hashes, hasher.verifies) == (0, 0)


async def test_an_email_over_254_characters_is_a_422(anon_client: AsyncClient, hasher: CountingHasher) -> None:
    response = await _login(anon_client, email="a" * 250 + "@b.co")

    assert response.status_code == 422
    assert (hasher.hashes, hasher.verifies) == (0, 0)


async def test_the_password_length_counts_normalized_characters(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    # 12 characters once the decomposed "e" plus accent is composed, 13 as typed.
    typed = "café au lait"
    await _account(db_session, password=typed)

    assert (await _login(anon_client, password="café au lait")).status_code == 200


async def test_login_rehashes_a_password_stored_under_older_parameters(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher
) -> None:
    older = PasswordHasher(time_cost=2, memory_cost=8, parallelism=1).hash(PASSWORD.encode())
    user = await _account(db_session)
    user_id = user.id
    user.password_hash = older
    await db_session.flush()

    assert (await _login(anon_client)).status_code == 200

    db_session.expire_all()
    stored = (await db_session.execute(select(User.password_hash).where(User.id == user_id))).scalar_one()
    assert stored != older and not passwords.needs_rehash(stored)
    assert await passwords.verify_password(PASSWORD, stored)


async def test_login_leaves_a_current_hash_alone(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher
) -> None:
    user = await _account(db_session)
    user_id, stored = user.id, user.password_hash
    hashes = hasher.hashes

    await _login(anon_client)

    assert hasher.hashes == hashes
    db_session.expire_all()
    assert (await db_session.execute(select(User.password_hash).where(User.id == user_id))).scalar_one() == stored


async def test_login_sweeps_expired_sessions(anon_client: AsyncClient, db_session: AsyncSession) -> None:
    other = await _account(db_session, "bob@example.com")
    await sessions.create_session(
        db_session,
        user_id=other.id,
        organization_id=DEFAULT_ORGANIZATION_ID,
        ttl=timedelta(hours=1),
        now=datetime.now(UTC) - timedelta(days=2),
    )
    ada = await _account(db_session)

    await _login(anon_client)

    assert await _session_count(db_session, other.id) == 0
    assert await _session_count(db_session, ada.id) == 1


async def test_a_user_with_no_organization_gets_the_generic_401_and_a_log_line_after_the_verify(
    anon_client: AsyncClient,
    db_session: AsyncSession,
    hasher: CountingHasher,
    caplog: pytest.LogCaptureFixture,
) -> None:
    user = await accounts.create_user(db_session, email=EMAIL, password=PASSWORD)
    await passwords.warm_up()
    verifies = hasher.verifies

    with caplog.at_level(logging.WARNING, logger="app.routers.auth"):
        response = await _login(anon_client)

    assert response.status_code == 401
    assert response.json() == {"detail": "email or password is incorrect"}
    assert "set-cookie" not in response.headers
    assert hasher.verifies == verifies + 1
    assert await _session_count(db_session) == 0
    assert f"login refused: user {user.id} has no membership" in caplog.text
    assert PASSWORD not in caplog.text


async def test_login_picks_the_earliest_membership(
    anon_client: AsyncClient, db_session: AsyncSession, second_organization: Organization
) -> None:
    now = datetime.now(UTC)
    user = await _account(db_session, since=now)
    await accounts.add_membership(
        db_session, organization_id=second_organization.id, user_id=user.id, role="member"
    )
    await db_session.execute(
        update(Membership)
        .where(Membership.organization_id == second_organization.id)
        .values(created_at=now - timedelta(days=1))
    )

    body = (await _login(anon_client)).json()

    assert body["organization"]["id"] == str(second_organization.id)
    assert {item["id"] for item in body["organizations"]} == {
        str(DEFAULT_ORGANIZATION_ID),
        str(second_organization.id),
    }


# --- the limits come before any hashing ---------------------------------------


async def test_the_per_address_limit_answers_429_with_retry_after_before_hashing(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher, fresh_limiter, tmp_path
) -> None:
    _settings(tmp_path, login_rate_limit_per_minute=2)
    await _account(db_session)
    await passwords.warm_up()

    first = await _login(anon_client, password="not the password")
    second = await _login(anon_client, email="nobody@example.com")
    calls = (hasher.hashes, hasher.verifies)
    limited = await _login(anon_client)

    assert (first.status_code, second.status_code, limited.status_code) == (401, 401, 429)
    assert limited.headers["retry-after"] == "60"
    assert (hasher.hashes, hasher.verifies) == calls


async def test_the_per_account_limit_is_shared_by_every_spelling_of_one_email(
    anon_client: AsyncClient, db_session: AsyncSession, hasher: CountingHasher, fresh_limiter, tmp_path
) -> None:
    _settings(tmp_path, login_account_rate_limit_per_minute=2)
    await _account(db_session)
    await _account(db_session, "bob@example.com")
    await passwords.warm_up()
    wrong = "not the password"

    # Case differences, and the full-width letters NFKC folds to ASCII.
    first = await _login(anon_client, "Ada@Example.com", wrong)
    second = await _login(anon_client, "ADA@EXAMPLE.COM", wrong)
    calls = (hasher.hashes, hasher.verifies)
    limited = await _login(anon_client, "ａda@example.com", PASSWORD)
    assert (hasher.hashes, hasher.verifies) == calls
    other_account = await _login(anon_client, "bob@example.com", wrong)

    assert (first.status_code, second.status_code) == (401, 401)
    assert limited.status_code == 429 and limited.headers["retry-after"] == "60"
    assert "per account" in limited.json()["detail"]
    assert other_account.status_code == 401  # its own bucket


async def test_an_unknown_email_is_limited_too_so_the_limit_does_not_say_which_exist(
    anon_client: AsyncClient, hasher: CountingHasher, fresh_limiter, tmp_path
) -> None:
    _settings(tmp_path, login_account_rate_limit_per_minute=1)
    await passwords.warm_up()

    first = await _login(anon_client, "nobody@example.com")
    verifies = hasher.verifies
    second = await _login(anon_client, "NOBODY@example.com")
    elsewhere = await _login(anon_client, "someone@example.com")

    assert (first.status_code, second.status_code, elsewhere.status_code) == (401, 429, 401)
    assert hasher.verifies == verifies + 1  # only the third verified


# Spellings Python's casefold() and the database's lower() disagree on.
FOLD_PAIRS = [
    # (typed, respelled, whether the database treats them as one account)
    ("ali@example.com", "al\u0130@example.com", True),  # i and the dotted capital I
    ("stra\u00dfe@example.com", "STRA\u1e9eE@example.com", True),  # sharp s and capital sharp S
    ("strasse@example.com", "STRA\u1e9eE@example.com", False),
]


@pytest.mark.parametrize(("typed", "respelled", "same_account"), FOLD_PAIRS)
async def test_the_limit_answers_alike_whether_or_not_the_account_exists_for_every_spelling(
    anon_client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    typed: str,
    respelled: str,
    same_account: bool,
) -> None:
    """Five wrong tries on one spelling and a sixth on another must give the same
    answers when the account exists as when it does not, or the limiter tells
    anyone which emails have accounts."""
    _settings(tmp_path, login_account_rate_limit_per_minute=5)
    await passwords.warm_up()
    wrong = "not the password"

    async def run() -> list[int]:
        monkeypatch.setattr(ratelimit, "_limiter", SlidingWindowLimiter())
        tries = [typed] * 5 + [respelled]
        return [(await _login(anon_client, email, wrong)).status_code for email in tries]

    without_account = await run()
    await _account(db_session, typed)
    with_account = await run()

    assert with_account == without_account
    # A respelling the database reads as the same email shares the bucket.
    assert with_account == [401] * 5 + [429 if same_account else 401]


async def test_a_refused_origin_spends_none_of_the_login_allowance(
    anon_client: AsyncClient, db_session: AsyncSession, fresh_limiter, tmp_path
) -> None:
    _settings(tmp_path, login_rate_limit_per_minute=1)
    await _account(db_session)

    refused = await _login(anon_client, headers={"Origin": "https://evil.example"})
    served = await _login(anon_client)

    assert (refused.status_code, served.status_code) == (403, 200)


# --- logout ------------------------------------------------------------------


async def test_logout_deletes_the_session_clears_the_cookie_and_is_idempotent(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    user = await _account(db_session)
    token = _cookie_value(await _login(anon_client))
    assert await _session_count(db_session, user.id) == 1

    first = await anon_client.post("/auth/logout", headers=ORIGIN)
    again = await anon_client.post("/auth/logout", headers={**ORIGIN, "Cookie": f"{COOKIE_NAME}={token}"})
    anon_client.cookies.clear()
    bare = await anon_client.post("/auth/logout", headers=ORIGIN)

    assert (first.status_code, again.status_code, bare.status_code) == (204, 204, 204)
    assert first.content == b""
    cleared = first.headers["set-cookie"]
    assert cleared.startswith((f'{COOKIE_NAME}="";', f"{COOKIE_NAME}=;"))
    assert "Max-Age=0" in cleared and "Path=/" in cleared and "HttpOnly" in cleared
    assert first.headers["cache-control"] == "no-store"
    assert await _session_count(db_session, user.id) == 0
    stale = await anon_client.get("/auth/me", headers={"Cookie": f"{COOKIE_NAME}={token}"})
    assert stale.status_code == 401


async def test_logout_with_a_cookie_that_is_not_a_session_is_a_204(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    user = await _account(db_session)
    await _login(anon_client)

    response = await anon_client.post(
        "/auth/logout", headers={**ORIGIN, "Cookie": f"{COOKIE_NAME}=not-a-session"}
    )

    assert response.status_code == 204
    assert await _session_count(db_session, user.id) == 1  # someone else's session is untouched


# --- the Origin rule for cookie-authenticated writes ---------------------------


FOREIGN = [{}, {"Origin": "https://evil.example"}, {"Origin": "null"}, {"Origin": "http://localhost:3000.evil.example"}]
FOREIGN_IDS = ["no Origin", "foreign", "null", "lookalike"]


@pytest.mark.parametrize("headers", FOREIGN, ids=FOREIGN_IDS)
async def test_login_logout_and_a_write_with_a_valid_cookie_need_an_allowed_origin(
    anon_client: AsyncClient, db_session: AsyncSession, headers: dict
) -> None:
    user = await _account(db_session)
    await _login(anon_client)
    sessions_before = await _session_count(db_session, user.id)
    field_id = uuid.uuid4()

    login = await anon_client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD}, headers=headers)
    logout = await anon_client.post("/auth/logout", headers=headers)
    resolve = await anon_client.post(
        f"/review/fields/{field_id}/resolve", json={"action": "approve"}, headers=headers
    )
    switch = await anon_client.post(
        "/auth/organization", json={"organization_id": str(DEFAULT_ORGANIZATION_ID)}, headers=headers
    )

    assert [r.status_code for r in (login, logout, resolve, switch)] == [403] * 4
    assert login.json() == {"detail": "origin not allowed"}
    # The refused logout did not end the session, and the refused login made none.
    assert await _session_count(db_session, user.id) == sessions_before
    # The same write with the frontend's Origin gets past the check (the field does not exist).
    allowed = await anon_client.post(
        f"/review/fields/{field_id}/resolve", json={"action": "approve"}, headers=ORIGIN
    )
    assert allowed.status_code == 404


async def test_a_safe_request_needs_no_origin(anon_client: AsyncClient, db_session: AsyncSession) -> None:
    await _account(db_session)
    await _login(anon_client)

    assert (await anon_client.get("/documents")).status_code == 200
    assert (await anon_client.get("/documents", headers={"Origin": "https://evil.example"})).status_code == 200


async def test_a_write_with_no_credential_is_a_401_not_a_403(anon_client: AsyncClient) -> None:
    response = await anon_client.post("/ask", json={"question": "q"}, headers={"Origin": "https://evil.example"})

    assert response.status_code == 401


# --- resolving the principal ----------------------------------------------------


async def test_no_cookie_and_an_unknown_cookie_are_the_same_401(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    none = await anon_client.get("/documents")
    unknown = await anon_client.get("/documents", headers={"Cookie": f"{COOKIE_NAME}=nope"})

    assert none.status_code == unknown.status_code == 401
    assert none.json() == unknown.json() == {"detail": "not authenticated"}
    assert none.headers["www-authenticate"] == 'Bearer realm="doc-pilot"'


async def test_a_bearer_header_is_a_401_even_with_a_valid_cookie(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    await _account(db_session)
    await _login(anon_client)

    for value in ("Bearer dpk_aaaaaaaaaaaa_" + "a" * 43, "bearer x", "Bearer"):
        response = await anon_client.get("/documents", headers={"Authorization": value})
        assert response.status_code == 401, value
    # Another scheme is not a bearer token and leaves the cookie to decide.
    assert (await anon_client.get("/documents", headers={"Authorization": "Basic eDp5"})).status_code == 200


async def test_an_expired_session_is_a_401_and_its_row_is_deleted(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    user = await _account(db_session)
    await _login(anon_client)
    await db_session.execute(
        update(AuthSession).where(AuthSession.user_id == user.id).values(
            expires_at=datetime.now(UTC) - timedelta(seconds=1)
        )
    )

    response = await anon_client.get("/documents")

    assert response.status_code == 401
    assert await _session_count(db_session, user.id) == 0


async def test_a_disabled_user_loses_the_session_they_had(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    user = await _account(db_session)
    await _login(anon_client)
    assert (await anon_client.get("/documents")).status_code == 200

    user.disabled_at = datetime.now(UTC)
    await db_session.flush()

    assert (await anon_client.get("/documents")).status_code == 401


async def test_removing_a_membership_ends_its_sessions(
    anon_client: AsyncClient, db_session: AsyncSession
) -> None:
    user = await _account(db_session)
    await _login(anon_client)

    await db_session.execute(delete(Membership).where(Membership.user_id == user.id))

    assert await _session_count(db_session, user.id) == 0
    assert (await anon_client.get("/documents")).status_code == 401


async def test_set_password_ends_every_session(anon_client: AsyncClient, db_session: AsyncSession) -> None:
    user = await _account(db_session)
    await _login(anon_client)
    await _login(anon_client)
    assert await _session_count(db_session, user.id) == 2

    ended = await accounts.set_password(db_session, user, "a brand new passphrase")

    assert ended == 2 and await _session_count(db_session, user.id) == 0
    assert (await anon_client.get("/auth/me")).status_code == 401
    assert (await _login(anon_client)).status_code == 401
    assert (await _login(anon_client, password="a brand new passphrase")).status_code == 200


# --- the organization switch -----------------------------------------------------


async def test_switching_organization_rotates_the_token_and_keeps_the_expiry(
    anon_client: AsyncClient, db_session: AsyncSession, second_organization: Organization
) -> None:
    now = datetime.now(UTC)
    user = await _account(db_session, since=now - timedelta(days=1))
    await accounts.add_membership(
        db_session, organization_id=second_organization.id, user_id=user.id, role="member"
    )
    old_token = _cookie_value(await _login(anon_client))
    old_expiry = (await db_session.execute(select(AuthSession.expires_at))).scalar_one()

    switched = await anon_client.post(
        "/auth/organization", json={"organization_id": str(second_organization.id)}, headers=ORIGIN
    )

    assert switched.status_code == 200
    assert switched.json()["organization"] == {
        "id": str(second_organization.id),
        "name": second_organization.name,
        "slug": second_organization.slug,
        "role": "member",
    }
    assert switched.headers["cache-control"] == "no-store"
    new_token = _cookie_value(switched)
    assert new_token != old_token
    rows = (await db_session.execute(select(AuthSession))).scalars().all()
    assert [(row.token_digest, row.organization_id) for row in rows] == [
        (digest(new_token), second_organization.id)
    ]
    assert abs((rows[0].expires_at - old_expiry).total_seconds()) < 1  # a switch does not extend a sign-in
    # The new cookie works, the old one does not.
    assert (await anon_client.get("/auth/me")).json()["organization"]["id"] == str(second_organization.id)
    stale = await anon_client.get("/auth/me", headers={"Cookie": f"{COOKIE_NAME}={old_token}"})
    assert stale.status_code == 401


async def test_a_token_can_be_switched_with_only_once(
    anon_client: AsyncClient, db_session: AsyncSession, second_organization: Organization
) -> None:
    """The old token's delete is the guard against two switches racing on it:
    the second one finds no row and mints no session."""
    user = await _account(db_session)
    await accounts.add_membership(
        db_session, organization_id=second_organization.id, user_id=user.id, role="member"
    )
    old_token = _cookie_value(await _login(anon_client))
    body = {"organization_id": str(second_organization.id)}

    first = await anon_client.post("/auth/organization", json=body, headers=ORIGIN)
    second = await anon_client.post(
        "/auth/organization", json=body, headers={**ORIGIN, "Cookie": f"{COOKIE_NAME}={old_token}"}
    )

    assert first.status_code == 200
    assert second.status_code == 401
    assert await _session_count(db_session, user.id) == 1


async def test_switching_to_an_organization_the_user_is_not_in_is_a_404(
    anon_client: AsyncClient, db_session: AsyncSession, second_organization: Organization
) -> None:
    await _account(db_session)
    token = _cookie_value(await _login(anon_client))

    foreign = await anon_client.post(
        "/auth/organization", json={"organization_id": str(second_organization.id)}, headers=ORIGIN
    )
    missing = await anon_client.post(
        "/auth/organization", json={"organization_id": str(uuid.uuid4())}, headers=ORIGIN
    )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    # The session is as it was.
    row = (await db_session.execute(select(AuthSession))).scalar_one()
    assert row.token_digest == digest(token) and row.organization_id == DEFAULT_ORGANIZATION_ID


async def test_me_and_the_switch_need_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.get("/auth/me")).status_code == 401
    switch = await anon_client.post(
        "/auth/organization", json={"organization_id": str(DEFAULT_ORGANIZATION_ID)}, headers=ORIGIN
    )
    assert switch.status_code == 401


# --- CORS ------------------------------------------------------------------------


async def test_a_preflight_from_the_frontend_allows_credentials(anon_client: AsyncClient) -> None:
    preflight = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }

    allowed = await anon_client.options("/auth/login", headers={"Origin": "http://localhost:3000", **preflight})
    other = await anon_client.options("/auth/login", headers={"Origin": "https://evil.example", **preflight})

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "access-control-allow-origin" not in other.headers


async def test_the_allowed_methods_and_headers_are_the_frontends_only(anon_client: AsyncClient) -> None:
    async def preflight(method: str, headers: str):
        return await anon_client.options(
            "/documents",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": headers,
            },
        )

    assert (await preflight("GET", "content-type")).status_code == 200
    assert (await preflight("DELETE", "content-type")).status_code == 200
    assert (await preflight("PUT", "content-type")).status_code == 400
    assert (await preflight("POST", "x-custom")).status_code == 400


async def test_a_response_to_the_frontend_carries_credentials_and_exposes_retry_after(
    anon_client: AsyncClient,
) -> None:
    response = await anon_client.get("/documents", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 401
    assert response.headers["access-control-allow-credentials"] == "true"
    assert response.headers["access-control-expose-headers"] == "Retry-After"


# --- nothing secret is logged -----------------------------------------------------


async def test_no_password_token_or_cookie_value_reaches_the_logs(
    anon_client: AsyncClient, db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    await _account(db_session)
    wrong = "a wrong password 9f3c"
    junk_cookie = "a-junk-cookie-value-7d21"

    with caplog.at_level(logging.DEBUG):
        good = await _login(anon_client)
        token = _cookie_value(good)
        await _login(anon_client, password=wrong)
        await _login(anon_client, password="x" * 140)
        await anon_client.get("/auth/me")
        await anon_client.get("/documents", headers={"Cookie": f"{COOKIE_NAME}={junk_cookie}"})
        await anon_client.post("/auth/logout", headers=ORIGIN)

    assert caplog.records  # the capture was live
    for secret in (PASSWORD, wrong, "x" * 140, token, junk_cookie):
        assert secret not in caplog.text
        assert secret not in "".join(str(record.args) for record in caplog.records)
