"""The identity schema and the credential primitives (app/auth/, scripts/accounts.py).

Nothing here is wired into a route yet. Password hashing runs on argon2's
CHEAPEST profile (a fixture swaps the module's hasher), except the one test that
pins the production parameters. The database tests run in db_session's
rolled-back transaction; the one subprocess test commits real rows to the test
database and deletes them again, so no assertion here counts rows globally.
"""

import asyncio
import importlib.util
import io
import os
import string
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from argon2 import PasswordHasher, profiles
from argon2.low_level import Type
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import delete, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import accounts, api_keys, passwords, sessions, tokens
from app.config import BACKEND_DIR
from app.db import Base, engine
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    IDENTITY_TABLES,
    MEMBERSHIP_ROLES,
    ApiKey,
    AuthSession,
    Membership,
    Organization,
    User,
)

PASSWORD = "correct horse battery"
# The hasher as imported, before the fixture below swaps it.
PRODUCTION_HASHER = passwords._hasher

BASE32 = "abcdefghijklmnopqrstuvwxyz234567"
URLSAFE = string.ascii_letters + string.digits + "-_"


class CountingHasher:
    """A CHEAPEST-profile hasher that counts its calls and the most that ran at
    once."""

    def __init__(self, pause: float = 0.0) -> None:
        self._inner = PasswordHasher.from_parameters(profiles.CHEAPEST)
        self._pause = pause
        self._lock = threading.Lock()
        self.hashes = 0
        self.verifies = 0
        self._running = 0
        self.most_running = 0

    def _enter(self) -> None:
        with self._lock:
            self._running += 1
            self.most_running = max(self.most_running, self._running)
        time.sleep(self._pause)

    def _leave(self) -> None:
        with self._lock:
            self._running -= 1

    def hash(self, password: str) -> str:
        self._enter()
        try:
            with self._lock:
                self.hashes += 1
            return self._inner.hash(password)
        finally:
            self._leave()

    def verify(self, password_hash: str, password: str) -> bool:
        self._enter()
        try:
            with self._lock:
                self.verifies += 1
            return self._inner.verify(password_hash, password)
        finally:
            self._leave()

    def check_needs_rehash(self, password_hash: str) -> bool:
        return self._inner.check_needs_rehash(password_hash)


@pytest.fixture(autouse=True)
def cheap_hasher(monkeypatch: pytest.MonkeyPatch) -> CountingHasher:
    """The cheapest argon2 parameters, and a fresh dummy hash: it would
    otherwise be built once, under whichever parameters ran first."""
    hasher = CountingHasher()
    monkeypatch.setattr(passwords, "_hasher", hasher)
    monkeypatch.setattr(passwords, "_dummy_hash", None)
    return hasher


# --- passwords ---------------------------------------------------------------


def test_the_production_hasher_is_pinned_to_the_rfc_9106_low_memory_profile() -> None:
    # The numbers are spelled out so that a library upgrade that moved the
    # profile fails here instead of silently changing what is stored.
    assert PRODUCTION_HASHER.type is Type.ID
    assert PRODUCTION_HASHER.time_cost == 3
    assert PRODUCTION_HASHER.memory_cost == 65536  # 64 MiB
    assert PRODUCTION_HASHER.parallelism == 4
    assert PRODUCTION_HASHER.hash_len == 32
    assert PRODUCTION_HASHER.salt_len == 16
    assert PRODUCTION_HASHER._parameters == profiles.RFC_9106_LOW_MEMORY


async def test_a_production_hash_carries_the_pinned_parameters_and_verifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(passwords, "_hasher", PRODUCTION_HASHER)

    password_hash = await passwords.hash_password(PASSWORD)

    assert password_hash.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
    assert PASSWORD not in password_hash
    assert await passwords.verify_password(PASSWORD, password_hash) is True
    assert await passwords.verify_password(PASSWORD + "x", password_hash) is False
    assert passwords.needs_rehash(password_hash) is False


async def test_a_hash_made_under_other_parameters_needs_a_rehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cheap_hash = await passwords.hash_password(PASSWORD)
    assert passwords.needs_rehash(cheap_hash) is False

    monkeypatch.setattr(passwords, "_hasher", PRODUCTION_HASHER)
    assert passwords.needs_rehash(cheap_hash) is True


async def test_verify_accepts_the_password_and_refuses_anything_else() -> None:
    password_hash = await passwords.hash_password(PASSWORD)

    assert await passwords.verify_password(PASSWORD, password_hash) is True
    assert await passwords.verify_password("Correct horse battery", password_hash) is False
    # The same password hashes to a different string each time (random salt).
    assert await passwords.hash_password(PASSWORD) != password_hash


@pytest.mark.parametrize("stored", ["", "not a hash", "$argon2id$garbage", "$2b$12$abcdefghijklmnopqrstuv"])
async def test_a_stored_value_that_is_not_an_argon2_hash_never_matches(stored: str) -> None:
    assert await passwords.verify_password(PASSWORD, stored) is False


@pytest.mark.parametrize("length", [0, 1, passwords.PASSWORD_MIN_CHARS - 1, passwords.PASSWORD_MAX_CHARS + 1])
async def test_hashing_refuses_a_password_outside_the_length_limits(length: int) -> None:
    with pytest.raises(ValueError, match="12 to 128"):
        await passwords.hash_password("x" * length)


@pytest.mark.parametrize("length", [passwords.PASSWORD_MIN_CHARS, passwords.PASSWORD_MAX_CHARS])
async def test_hashing_accepts_the_length_limits_themselves(length: int) -> None:
    password = "x" * length
    assert await passwords.verify_password(password, await passwords.hash_password(password)) is True


async def test_a_password_over_the_cap_is_refused_without_being_hashed(
    cheap_hasher: CountingHasher,
) -> None:
    password_hash = await passwords.hash_password(PASSWORD)
    verifies = cheap_hasher.verifies

    assert await passwords.verify_password("x" * (passwords.PASSWORD_MAX_CHARS + 1), password_hash) is False
    assert cheap_hasher.verifies == verifies


async def test_at_most_two_hashes_run_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = CountingHasher(pause=0.05)
    monkeypatch.setattr(passwords, "_hasher", slow)

    await asyncio.gather(*(passwords.hash_password(f"{PASSWORD}{n}") for n in range(8)))

    assert slow.hashes == 8
    assert slow.most_running == passwords.HASH_CONCURRENCY == 2


async def test_the_dummy_hash_is_built_once_and_each_dummy_verify_costs_a_verification(
    cheap_hasher: CountingHasher,
) -> None:
    assert cheap_hasher.hashes == 0

    await asyncio.gather(*(passwords.verify_dummy_password(PASSWORD) for _ in range(5)))
    await passwords.verify_dummy_password("anything at all")

    assert cheap_hasher.hashes == 1
    assert cheap_hasher.verifies == 6


# --- tokens ------------------------------------------------------------------


def test_a_session_token_is_256_random_bits_in_the_urlsafe_alphabet() -> None:
    token = tokens.new_session_token()

    assert len(token) == 43
    assert set(token) <= set(URLSAFE)
    assert tokens.new_session_token() != token


def test_a_digest_is_the_32_byte_sha256_of_the_token() -> None:
    assert tokens.digest("abc").hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert len(tokens.digest(tokens.new_session_token())) == 32


def test_digests_are_compared_for_equality() -> None:
    assert tokens.digests_match(tokens.digest("a"), tokens.digest("a")) is True
    assert tokens.digests_match(tokens.digest("a"), tokens.digest("b")) is False


def test_a_minted_api_key_has_exactly_the_documented_format() -> None:
    minted = tokens.mint_api_key()

    scheme, prefix, secret = minted.token.split("_", 2)
    assert scheme == "dpk"
    assert len(prefix) == 12 and set(prefix) <= set(BASE32)
    assert len(secret) == 43 and set(secret) <= set(URLSAFE)
    assert minted.prefix == prefix
    assert minted.token_digest == tokens.digest(minted.token)
    assert tokens.parse_api_key(minted.token) == prefix
    assert tokens.mint_api_key().token != minted.token


def _has_api_key_shape(value: str) -> bool:
    """The documented shape, spelled out without the module's regular
    expression: dpk_ + 12 base32 characters + _ + 43 URL-safe characters."""
    return (
        len(value) == 4 + 12 + 1 + 43
        and value[:4] == "dpk_"
        and value[16] == "_"
        and set(value[4:16]) <= set(BASE32)
        and set(value[17:]) <= set(URLSAFE)
    )


valid_api_keys = st.builds(
    lambda prefix, secret: f"dpk_{prefix}_{secret}",
    st.text(BASE32, min_size=12, max_size=12),
    st.text(URLSAFE, min_size=43, max_size=43),
)


# The characters a loosened parser would let through: line ends and spaces (`$`
# before a trailing newline, `\s`), the separators, uppercase (re.I), non-ASCII
# digits and letters (`\d`, `\w`), the ASCII ones a base32 prefix
# excludes, lone surrogates, and anything else.
hard_characters = st.one_of(
    st.sampled_from("\n\r\t _-"),
    st.sampled_from(string.ascii_uppercase + "0189"),  # in a lowercase base32 prefix
    st.characters(categories=["Lu"]),
    st.characters(categories=["Nd"]),
    st.characters(categories=["Ll", "Lo"]),
    st.characters(categories=["Cs"]),
    st.characters(),
)


@st.composite
def near_misses(draw) -> str:
    """A valid key with one character replaced, inserted, or dropped, or one
    added at either end: the inputs a loose parser would let through. The
    position is aimed at the whole key, the 12-character prefix, or the
    43-character secret, so each class of fault is tried in each part."""
    key = draw(valid_api_keys)
    first, last = draw(st.sampled_from([(0, len(key) - 1), (4, 15), (17, len(key) - 1)]))
    i = draw(st.integers(first, last))
    c = draw(hard_characters)
    return draw(
        st.sampled_from([key[:i] + c + key[i + 1 :], key[:i] + c + key[i:], key[:i] + key[i + 1 :], key + c, c + key])
    )


def _assert_parses_exactly_the_documented_shape(value: str) -> None:
    assert tokens.parse_api_key(value) == (value[4:16] if _has_api_key_shape(value) else None)


@settings(max_examples=400)
@given(st.one_of(st.text(), valid_api_keys, near_misses(), near_misses()))
def test_parse_api_key_accepts_exactly_the_documented_shape_and_never_raises(value: str) -> None:
    _assert_parses_exactly_the_documented_shape(value)


# The same property, aimed so that a loosened pattern cannot get through by the
# luck of the draw: junk at either end (a `$` that allows a trailing newline, an
# unanchored search), one wrong character in the prefix (re.I, `\d` for the
# digits 2 to 7, `\w`), and one in the secret.
@given(
    valid_api_keys,
    st.sampled_from(["\n", "\r\n", "\r", " ", "\t", "\x00", "_", "-", "a", "A", "0", "dpk_"]),
    st.booleans(),
)
def test_junk_at_either_end_of_a_valid_key_is_refused(key: str, junk: str, at_start: bool) -> None:
    _assert_parses_exactly_the_documented_shape(junk + key if at_start else key + junk)


@given(valid_api_keys, st.integers(4, 15), st.sampled_from(string.ascii_uppercase + "0189_-\u0663\u00e9\u212a"))
def test_one_wrong_character_in_the_prefix_is_refused(key: str, i: int, c: str) -> None:
    _assert_parses_exactly_the_documented_shape(key[:i] + c + key[i + 1 :])


@given(valid_api_keys, st.integers(17, 59), st.sampled_from(" \n.+/=!\u0663\u00e9\u212a\uff21\ud800"))
def test_one_wrong_character_in_the_secret_is_refused(key: str, i: int, c: str) -> None:
    _assert_parses_exactly_the_documented_shape(key[:i] + c + key[i + 1 :])


@given(valid_api_keys)
def test_every_well_formed_key_parses_to_its_prefix(value: str) -> None:
    assert tokens.parse_api_key(value) == value[4:16]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "dpk_",
        f"dpk_{'a' * 12}_{'A' * 43}\n",  # a trailing newline
        f" dpk_{'a' * 12}_{'A' * 43}",
        f"dpk_{'A' * 12}_{'A' * 43}",  # an uppercase prefix
        f"dpk_{'a' * 11}1_{'A' * 43}",  # 1 is not base32
        f"dpk_{'a' * 12}_{'A' * 42}",
        f"dpk_{'a' * 12}_{'A' * 44}",
        f"dpk_{'a' * 12}_{'A' * 42}٣",  # an Arabic-Indic digit
        f"DPK_{'a' * 12}_{'A' * 43}",
        f"dpk-{'a' * 12}_{'A' * 43}",
        f"Bearer dpk_{'a' * 12}_{'A' * 43}",
    ],
)
def test_parse_api_key_refuses_near_misses(value: str) -> None:
    assert tokens.parse_api_key(value) is None


# --- fixtures and helpers for the database tests ---------------------------


async def _organization(session: AsyncSession, slug: str = "acme") -> Organization:
    return await accounts.create_organization(session, name=slug.title(), slug=slug)


async def _user(session: AsyncSession, email: str = "ada@example.com") -> User:
    return await accounts.create_user(session, email=email, password=PASSWORD)


async def _member(
    session: AsyncSession, organization: Organization, user: User, role: str = "member"
) -> Membership:
    return await accounts.add_membership(
        session, organization_id=organization.id, user_id=user.id, role=role
    )


async def _refuses(session: AsyncSession, row: object, constraint: str) -> None:
    """Flush `row` and expect the database to refuse it, naming `constraint`.
    The savepoint keeps the test's transaction usable afterwards."""
    with pytest.raises(IntegrityError, match=constraint):
        async with session.begin_nested():
            session.add(row)
            await session.flush()


async def _accepts(session: AsyncSession, row: object) -> None:
    async with session.begin_nested():
        session.add(row)
        await session.flush()


def _session_row(membership: Membership, digest: bytes) -> AuthSession:
    return AuthSession(
        token_digest=digest,
        organization_id=membership.organization_id,
        user_id=membership.user_id,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


# --- schema ------------------------------------------------------------------


async def test_the_default_organization_row_matches_the_model_constant(db_session: AsyncSession) -> None:
    organization = await db_session.get(Organization, DEFAULT_ORGANIZATION_ID)

    assert organization is not None
    assert (organization.name, organization.slug) == ("Default", "default")


async def test_the_models_and_the_migration_describe_the_same_identity_tables() -> None:
    def differences(connection):
        context = MigrationContext.configure(connection)
        found = []
        for diff in compare_metadata(context, Base.metadata):
            # A diff is an operation tuple, or a list of them for one table.
            for operation in diff if isinstance(diff, list) else [diff]:
                subject = operation[1]
                table = getattr(subject, "table", subject)
                if getattr(table, "name", None) in IDENTITY_TABLES:
                    found.append(operation)
        return found

    async with engine.connect() as connection:
        assert await connection.run_sync(differences) == []


async def test_every_foreign_key_of_the_identity_tables_has_an_index_that_leads_with_it() -> None:
    def unindexed(connection):
        inspector = inspect(connection)
        missing = []
        for table in sorted(IDENTITY_TABLES):
            covering = [inspector.get_pk_constraint(table)["constrained_columns"]]
            covering += [index["column_names"] for index in inspector.get_indexes(table)]
            covering += [constraint["column_names"] for constraint in inspector.get_unique_constraints(table)]
            for foreign_key in inspector.get_foreign_keys(table):
                columns = foreign_key["constrained_columns"]
                if not any(candidate[: len(columns)] == columns for candidate in covering):
                    missing.append((table, columns))
        return missing

    async with engine.connect() as connection:
        assert await connection.run_sync(unindexed) == []


def test_every_table_is_either_identity_or_content() -> None:
    content = {
        "documents",
        "jobs",
        "extractions",
        "extracted_fields",
        "reconciliations",
        "document_pages",
        "document_chunks",
        "ask_runs",
    }
    assert set(Base.metadata.tables) == content | IDENTITY_TABLES
    assert not content & IDENTITY_TABLES


@pytest.mark.parametrize("slug", ["ab", "a-b", "0a", "a" + "-" * 3 + "b", "x" * 63, "9" * 2])
async def test_a_valid_slug_is_accepted(db_session: AsyncSession, slug: str) -> None:
    await _accepts(db_session, Organization(name="Acme", slug=slug))


@pytest.mark.parametrize(
    "slug", ["", "a", "A1", "Ab", "-ab", "ab_c", "a b", "ab.c", "x" * 64, "ab\n", "café"]
)
async def test_a_bad_slug_is_refused(db_session: AsyncSession, slug: str) -> None:
    await _refuses(db_session, Organization(name="Acme", slug=slug), "ck_organizations_slug")


@pytest.mark.parametrize("name", ["", "x" * 101])
async def test_an_organization_name_outside_1_to_100_characters_is_refused(
    db_session: AsyncSession, name: str
) -> None:
    await _refuses(db_session, Organization(name=name, slug="acme"), "ck_organizations_name")


@pytest.mark.parametrize("name", ["x", "x" * 100])
async def test_an_organization_name_of_1_to_100_characters_is_accepted(
    db_session: AsyncSession, name: str
) -> None:
    await _accepts(db_session, Organization(name=name, slug="acme"))


async def test_a_slug_is_unique(db_session: AsyncSession) -> None:
    await _organization(db_session, "acme")
    await _refuses(db_session, Organization(name="Other", slug="acme"), "organizations_slug_key")


@pytest.mark.parametrize("email", ["", "a" * 253 + "@b"])
async def test_an_email_outside_1_to_254_characters_is_refused(db_session: AsyncSession, email: str) -> None:
    await _refuses(db_session, User(email=email, password_hash="x"), "ck_users_email")


async def test_an_email_of_254_characters_is_accepted(db_session: AsyncSession) -> None:
    await _accepts(db_session, User(email="a" * 252 + "@b", password_hash="x"))


async def test_an_email_is_unique_without_regard_to_case(db_session: AsyncSession) -> None:
    await _user(db_session, "Ada@Example.com")

    await _refuses(db_session, User(email="ada@example.com", password_hash="x"), "users_email_key")
    await _refuses(db_session, User(email="ADA@EXAMPLE.COM", password_hash="x"), "users_email_key")


async def test_a_user_is_found_by_email_in_any_case(db_session: AsyncSession) -> None:
    user = await _user(db_session, "Ada@Example.com")

    assert (await accounts.get_user(db_session, "ada@example.COM")).id == user.id
    assert (await accounts.get_user(db_session, "  ada@example.com ")).id == user.id
    # citext keeps the case it was stored with.
    assert user.email == "Ada@Example.com"


async def test_a_role_must_be_owner_or_member(db_session: AsyncSession) -> None:
    organization = await _organization(db_session)
    user = await _user(db_session)

    for role in ("admin", "Owner", "", "OWNER"):
        await _refuses(
            db_session,
            Membership(organization_id=organization.id, user_id=user.id, role=role),
            "ck_memberships_role",
        )
    for role in MEMBERSHIP_ROLES:
        other = await _user(db_session, f"{role}@example.com")
        await _accepts(db_session, Membership(organization_id=organization.id, user_id=other.id, role=role))


async def test_a_membership_needs_a_real_user_and_organization(db_session: AsyncSession) -> None:
    user = await _user(db_session)
    organization = await _organization(db_session)

    await _refuses(
        db_session,
        Membership(organization_id=uuid.uuid4(), user_id=user.id, role="member"),
        "memberships_organization_id_fkey",
    )
    await _refuses(
        db_session,
        Membership(organization_id=organization.id, user_id=uuid.uuid4(), role="member"),
        "memberships_user_id_fkey",
    )


@pytest.mark.parametrize("length", [0, 1, 31, 33, 64])
async def test_a_session_digest_that_is_not_32_bytes_is_refused(
    db_session: AsyncSession, length: int
) -> None:
    membership = await _member(db_session, await _organization(db_session), await _user(db_session))

    await _refuses(db_session, _session_row(membership, b"x" * length), "ck_auth_sessions_token_digest")


async def test_a_session_digest_of_32_bytes_is_accepted_once(db_session: AsyncSession) -> None:
    membership = await _member(db_session, await _organization(db_session), await _user(db_session))

    await _accepts(db_session, _session_row(membership, b"x" * 32))
    await _refuses(db_session, _session_row(membership, b"x" * 32), "auth_sessions_token_digest_key")


async def test_a_session_must_belong_to_a_membership(db_session: AsyncSession) -> None:
    organization = await _organization(db_session)
    user = await _user(db_session)  # a user, but not a member of the organization
    outsider = Membership(organization_id=organization.id, user_id=user.id, role="member")

    await _refuses(db_session, _session_row(outsider, b"x" * 32), "auth_sessions_organization_id_user_id_fkey")


@pytest.mark.parametrize("length", [0, 31, 33])
async def test_an_api_key_digest_that_is_not_32_bytes_is_refused(db_session: AsyncSession, length: int) -> None:
    organization = await _organization(db_session)

    await _refuses(
        db_session,
        ApiKey(organization_id=organization.id, name="k", prefix="abcdefghijkl", token_digest=b"x" * length),
        "ck_api_keys_token_digest",
    )


@pytest.mark.parametrize(
    "prefix",
    ["", "abcdefghijk", "abcdefghijklm", "ABCDEFGHIJKL", "abcdefghijk1", "abcdefghijk8", "abcdefghijk\n", "abcdefghijk-"],
)
async def test_an_api_key_prefix_that_is_not_12_base32_characters_is_refused(
    db_session: AsyncSession, prefix: str
) -> None:
    organization = await _organization(db_session)

    await _refuses(
        db_session,
        ApiKey(organization_id=organization.id, name="k", prefix=prefix, token_digest=b"x" * 32),
        "ck_api_keys_prefix",
    )


@pytest.mark.parametrize("name", ["", "x" * 101])
async def test_an_api_key_name_outside_1_to_100_characters_is_refused(db_session: AsyncSession, name: str) -> None:
    organization = await _organization(db_session)

    await _refuses(
        db_session,
        ApiKey(organization_id=organization.id, name=name, prefix="abcdefghijkl", token_digest=b"x" * 32),
        "ck_api_keys_name",
    )


async def test_an_api_key_prefix_is_unique(db_session: AsyncSession) -> None:
    organization = await _organization(db_session)
    await _accepts(
        db_session,
        ApiKey(organization_id=organization.id, name="a", prefix="abcdefghijk2", token_digest=b"x" * 32),
    )

    await _refuses(
        db_session,
        ApiKey(organization_id=organization.id, name="b", prefix="abcdefghijk2", token_digest=b"y" * 32),
        "api_keys_prefix_key",
    )


async def test_removing_a_membership_ends_its_sessions_and_only_those(db_session: AsyncSession) -> None:
    ada = await _user(db_session)
    acme = await _organization(db_session, "acme")
    other = await _organization(db_session, "other")
    kept = await _member(db_session, other, ada)
    removed = await _member(db_session, acme, ada)
    now = datetime.now(UTC)
    for membership in (kept, removed):
        await sessions.create_session(
            db_session,
            user_id=ada.id,
            organization_id=membership.organization_id,
            ttl=timedelta(hours=1),
            now=now,
        )

    await db_session.delete(removed)
    await db_session.flush()

    remaining = (await db_session.execute(select(AuthSession.organization_id).where(AuthSession.user_id == ada.id))).scalars().all()
    assert remaining == [other.id]


async def test_removing_a_user_removes_their_memberships_and_sessions_and_keeps_their_keys(
    db_session: AsyncSession,
) -> None:
    ada = await _user(db_session)
    acme = await _organization(db_session)
    await _member(db_session, acme, ada)
    await sessions.create_session(db_session, user_id=ada.id, organization_id=acme.id, ttl=timedelta(hours=1))
    key, _ = await api_keys.create_api_key(
        db_session, organization_id=acme.id, name="ci", created_by_user_id=ada.id
    )

    await db_session.delete(ada)
    await db_session.flush()
    await db_session.refresh(key)

    assert (await db_session.execute(select(Membership).where(Membership.user_id == ada.id))).first() is None
    assert (await db_session.execute(select(AuthSession).where(AuthSession.user_id == ada.id))).first() is None
    assert key.created_by_user_id is None
    assert key.revoked_at is None


async def test_removing_an_organization_removes_its_memberships_sessions_and_keys(
    db_session: AsyncSession,
) -> None:
    ada = await _user(db_session)
    acme = await _organization(db_session)
    await _member(db_session, acme, ada)
    await sessions.create_session(db_session, user_id=ada.id, organization_id=acme.id, ttl=timedelta(hours=1))
    key, _ = await api_keys.create_api_key(db_session, organization_id=acme.id, name="ci")

    await db_session.delete(acme)
    await db_session.flush()

    assert (await db_session.execute(select(Membership).where(Membership.organization_id == acme.id))).first() is None
    assert (await db_session.execute(select(AuthSession).where(AuthSession.organization_id == acme.id))).first() is None
    assert (await db_session.execute(select(ApiKey).where(ApiKey.id == key.id))).first() is None
    assert await db_session.get(User, ada.id) is not None


# --- accounts ----------------------------------------------------------------


async def test_creating_a_user_stores_a_hash_and_never_the_password(db_session: AsyncSession) -> None:
    user = await _user(db_session)

    assert user.password_hash.startswith("$argon2id$")
    assert PASSWORD not in user.password_hash
    assert await passwords.verify_password(PASSWORD, user.password_hash) is True


async def test_a_second_account_for_the_same_email_in_another_case_is_refused(db_session: AsyncSession) -> None:
    await _user(db_session, "ada@example.com")

    with pytest.raises(accounts.AccountError, match="already has an account"):
        await _user(db_session, "ADA@example.com")


@pytest.mark.parametrize("password", ["", "short", "x" * 11, "x" * 129])
async def test_creating_a_user_refuses_a_password_outside_the_limits(
    db_session: AsyncSession, password: str
) -> None:
    with pytest.raises(ValueError, match="12 to 128"):
        await accounts.create_user(db_session, email="ada@example.com", password=password)


@pytest.mark.parametrize("email", ["", "   ", "a" * 253 + "@b"])
async def test_creating_a_user_refuses_an_email_outside_the_limits(db_session: AsyncSession, email: str) -> None:
    with pytest.raises(accounts.AccountError, match="1 to 254"):
        await accounts.create_user(db_session, email=email, password=PASSWORD)


@pytest.mark.parametrize("slug", ["A", "a", "-ab", "has space", "x" * 64])
async def test_creating_an_organization_refuses_a_bad_slug(db_session: AsyncSession, slug: str) -> None:
    with pytest.raises(accounts.AccountError, match="slug"):
        await accounts.create_organization(db_session, name="Acme", slug=slug)


async def test_creating_an_organization_refuses_a_taken_slug_and_a_bad_name(db_session: AsyncSession) -> None:
    await _organization(db_session, "acme")

    with pytest.raises(accounts.AccountError, match="already taken"):
        await accounts.create_organization(db_session, name="Again", slug="acme")
    for name in ("", "x" * 101):
        with pytest.raises(accounts.AccountError, match="1 to 100"):
            await accounts.create_organization(db_session, name=name, slug="fresh")


async def test_adding_a_membership_refuses_an_unknown_role_and_a_repeat(db_session: AsyncSession) -> None:
    organization = await _organization(db_session)
    user = await _user(db_session)

    with pytest.raises(accounts.AccountError, match="owner, member"):
        await _member(db_session, organization, user, role="admin")
    await _member(db_session, organization, user, role="owner")
    with pytest.raises(accounts.AccountError, match="already a member"):
        await _member(db_session, organization, user)


async def test_unknown_organizations_and_users_are_reported(db_session: AsyncSession) -> None:
    with pytest.raises(accounts.AccountError, match="no organization"):
        await accounts.get_organization(db_session, "nowhere")
    with pytest.raises(accounts.AccountError, match="no user"):
        await accounts.get_user(db_session, "nobody@example.com")


async def test_set_password_replaces_the_hash_and_ends_every_session_of_that_user_only(
    db_session: AsyncSession,
) -> None:
    ada = await _user(db_session, "ada@example.com")
    bob = await _user(db_session, "bob@example.com")
    acme = await _organization(db_session, "acme")
    other = await _organization(db_session, "other")
    for user in (ada, bob):
        await _member(db_session, acme, user)
    await _member(db_session, other, ada)
    ada_tokens = [
        await sessions.create_session(db_session, user_id=ada.id, organization_id=org.id, ttl=timedelta(hours=1))
        for org in (acme, other)
    ]
    bob_token = await sessions.create_session(
        db_session, user_id=bob.id, organization_id=acme.id, ttl=timedelta(hours=1)
    )

    ended = await accounts.set_password(db_session, ada, "a different passphrase")

    assert ended == 2
    assert await passwords.verify_password("a different passphrase", ada.password_hash) is True
    assert await passwords.verify_password(PASSWORD, ada.password_hash) is False
    assert [await sessions.lookup_session(db_session, token) for token in ada_tokens] == [None, None]
    assert await sessions.lookup_session(db_session, bob_token) is not None


async def test_set_password_refuses_a_short_password_and_keeps_the_sessions(db_session: AsyncSession) -> None:
    ada = await _user(db_session)
    acme = await _organization(db_session)
    await _member(db_session, acme, ada)
    token = await sessions.create_session(db_session, user_id=ada.id, organization_id=acme.id, ttl=timedelta(hours=1))
    before = ada.password_hash

    with pytest.raises(ValueError, match="12 to 128"):
        await accounts.set_password(db_session, ada, "short")

    assert ada.password_hash == before
    assert await sessions.lookup_session(db_session, token) is not None


# --- sessions ----------------------------------------------------------------


async def _signed_in(session: AsyncSession) -> tuple[User, Organization]:
    user = await _user(session)
    organization = await _organization(session)
    await _member(session, organization, user)
    return user, organization


async def test_a_session_stores_the_digest_of_its_token_and_not_the_token(db_session: AsyncSession) -> None:
    user, organization = await _signed_in(db_session)
    now = datetime.now(UTC)

    token = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=12), now=now
    )

    row = await sessions.lookup_session(db_session, token, now)
    assert row is not None
    assert (row.user_id, row.organization_id) == (user.id, organization.id)
    assert row.token_digest == tokens.digest(token)
    assert token.encode() not in row.token_digest
    assert row.expires_at == now + timedelta(hours=12)
    assert await sessions.lookup_session(db_session, token + "x", now) is None


async def test_a_session_is_found_until_its_expiry_and_not_at_or_after_it(db_session: AsyncSession) -> None:
    user, organization = await _signed_in(db_session)
    now = datetime.now(UTC)
    token = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1), now=now
    )

    assert await sessions.lookup_session(db_session, token, now + timedelta(minutes=59)) is not None
    assert await sessions.lookup_session(db_session, token, now + timedelta(hours=1)) is None
    assert await sessions.lookup_session(db_session, token, now + timedelta(days=1)) is None


async def test_a_session_for_a_user_who_is_not_a_member_is_refused(db_session: AsyncSession) -> None:
    user, _ = await _signed_in(db_session)
    stranger = await _organization(db_session, "stranger")

    with pytest.raises(IntegrityError, match="auth_sessions_organization_id_user_id_fkey"):
        async with db_session.begin_nested():
            await sessions.create_session(
                db_session, user_id=user.id, organization_id=stranger.id, ttl=timedelta(hours=1)
            )


async def test_logout_deletes_only_that_session(db_session: AsyncSession) -> None:
    user, organization = await _signed_in(db_session)
    first, second = [
        await sessions.create_session(
            db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1)
        )
        for _ in range(2)
    ]

    assert await sessions.delete_session(db_session, first) is True
    assert await sessions.delete_session(db_session, first) is False
    assert await sessions.lookup_session(db_session, first) is None
    assert await sessions.lookup_session(db_session, second) is not None


async def test_sweeping_removes_expired_sessions_and_keeps_live_ones(db_session: AsyncSession) -> None:
    user, organization = await _signed_in(db_session)
    now = datetime.now(UTC)
    expired = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1), now=now - timedelta(hours=2)
    )
    expiring_now = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1), now=now - timedelta(hours=1)
    )
    live = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1), now=now
    )

    removed = await sessions.delete_expired_sessions(db_session, now)

    assert removed >= 2
    remaining = (
        await db_session.execute(select(AuthSession.token_digest).where(AuthSession.user_id == user.id))
    ).scalars().all()
    assert remaining == [tokens.digest(live)]
    assert tokens.digest(expired) not in remaining and tokens.digest(expiring_now) not in remaining


# --- API keys ----------------------------------------------------------------


async def test_an_api_key_is_stored_as_its_prefix_and_digest_and_found_by_its_token(
    db_session: AsyncSession,
) -> None:
    organization = await _organization(db_session)

    key, token = await api_keys.create_api_key(db_session, organization_id=organization.id, name="laptop")

    assert tokens.parse_api_key(token) == key.prefix
    assert key.token_digest == tokens.digest(token)
    assert token.encode() not in key.token_digest
    found = await api_keys.find_active_api_key(db_session, token)
    assert found is not None and found.id == key.id
    assert (key.last_used_at, key.revoked_at, key.created_by_user_id) == (None, None, None)


async def test_a_token_with_the_right_prefix_and_the_wrong_secret_is_not_a_key(
    db_session: AsyncSession,
) -> None:
    organization = await _organization(db_session)
    key, token = await api_keys.create_api_key(db_session, organization_id=organization.id, name="laptop")
    forged = f"dpk_{key.prefix}_{tokens.new_session_token()}"

    assert tokens.parse_api_key(forged) == key.prefix
    assert await api_keys.find_active_api_key(db_session, forged) is None
    assert await api_keys.find_active_api_key(db_session, token) is not None


class _NoQuerySession:
    async def execute(self, *args, **kwargs):
        raise AssertionError("a malformed key must not reach the database")


@pytest.mark.parametrize("value", ["", "garbage", "dpk_short_x", "Bearer " + "a" * 60, "\x00"])
async def test_a_malformed_key_is_refused_without_a_query(value: str) -> None:
    assert await api_keys.find_active_api_key(_NoQuerySession(), value) is None


async def test_a_revoked_key_stops_resolving_and_keeps_its_row(db_session: AsyncSession) -> None:
    organization = await _organization(db_session)
    key, token = await api_keys.create_api_key(db_session, organization_id=organization.id, name="laptop")
    now = datetime.now(UTC)

    assert await api_keys.revoke_api_key(db_session, organization_id=organization.id, key_id=key.id, now=now)
    assert not await api_keys.revoke_api_key(db_session, organization_id=organization.id, key_id=key.id)

    assert await api_keys.find_active_api_key(db_session, token) is None
    await db_session.refresh(key)
    assert key.revoked_at == now


async def test_a_key_is_revoked_only_through_its_own_organization(db_session: AsyncSession) -> None:
    acme = await _organization(db_session, "acme")
    other = await _organization(db_session, "other")
    key, token = await api_keys.create_api_key(db_session, organization_id=acme.id, name="laptop")

    assert not await api_keys.revoke_api_key(db_session, organization_id=other.id, key_id=key.id)
    assert await api_keys.find_active_api_key(db_session, token) is not None


# --- scripts/accounts.py -----------------------------------------------------


def _load_cli():
    spec = importlib.util.spec_from_file_location("accounts_cli", BACKEND_DIR / "scripts" / "accounts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return _load_cli()


async def _run(cli, session: AsyncSession, monkeypatch: pytest.MonkeyPatch, *argv: str, stdin: str = "") -> int:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    return await cli.run(session, cli.build_parser().parse_args(list(argv)))


async def test_the_cli_builds_an_organization_a_user_a_membership_and_a_key(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = partial(_run, cli, db_session, monkeypatch)

    assert await run("create-organization", "--name", "Acme Inc", "--slug", "acme-cli") == 0
    assert await run("create-user", "--email", "Ada@Example.com", "--password-stdin", stdin=PASSWORD + "\n") == 0
    assert await run("add-member", "--organization", "acme-cli", "--email", "ada@example.com", "--role", "owner") == 0
    capsys.readouterr()
    assert await run("create-api-key", "--organization", "acme-cli", "--name", "laptop") == 0

    out, err = capsys.readouterr()
    token = out.strip()
    assert out == token + "\n"  # the key alone on stdout, nothing else
    assert tokens.parse_api_key(token) is not None
    assert token not in err and tokens.digest(token).hex() not in out + err
    assert "created API key" in err and "acme-cli" in err

    user = await accounts.get_user(db_session, "ada@example.com")
    organization = await accounts.get_organization(db_session, "acme-cli")
    membership = await db_session.get(Membership, (organization.id, user.id))
    assert membership is not None and membership.role == "owner"
    assert await passwords.verify_password(PASSWORD, user.password_hash) is True
    key = await api_keys.find_active_api_key(db_session, token)
    assert key is not None and key.organization_id == organization.id and key.name == "laptop"


async def test_add_member_defaults_to_the_member_role(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await _organization(db_session, "acme-cli")
    user = await _user(db_session)

    await _run(cli, db_session, monkeypatch, "add-member", "--organization", "acme-cli", "--email", user.email)

    assert (await db_session.get(Membership, (organization.id, user.id))).role == "member"


async def test_the_cli_never_prints_a_password(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await _run(cli, db_session, monkeypatch, "create-user", "--email", "ada@example.com", "--password-stdin", stdin=PASSWORD)
    await _run(cli, db_session, monkeypatch, "set-password", "--email", "ada@example.com", "--password-stdin", stdin="new passphrase here")

    out, err = capsys.readouterr()
    assert PASSWORD not in out + err and "new passphrase here" not in out + err


async def test_set_password_through_the_cli_ends_the_users_sessions(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    user, organization = await _signed_in(db_session)
    token = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1)
    )

    await _run(
        cli, db_session, monkeypatch, "set-password", "--email", user.email, "--password-stdin", stdin="brand new passphrase\n"
    )

    assert "ended 1 session(s)" in capsys.readouterr().out
    assert await sessions.lookup_session(db_session, token) is None
    await db_session.refresh(user)
    assert await passwords.verify_password("brand new passphrase", user.password_hash) is True
    assert await passwords.verify_password(PASSWORD, user.password_hash) is False


async def test_prune_sessions_through_the_cli_removes_only_expired_rows(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    user, organization = await _signed_in(db_session)
    now = datetime.now(UTC)
    expired = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1), now=now - timedelta(days=1)
    )
    live = await sessions.create_session(
        db_session, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1), now=now
    )

    assert await _run(cli, db_session, monkeypatch, "prune-sessions") == 0

    assert "expired session(s)" in capsys.readouterr().out
    assert await sessions.lookup_session(db_session, live) is not None
    remaining = (await db_session.execute(select(AuthSession.token_digest))).scalars().all()
    assert tokens.digest(expired) not in remaining


async def test_the_cli_refuses_a_password_that_is_too_short_or_missing_and_commits_nothing(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="12 to 128"):
        await _run(cli, db_session, monkeypatch, "create-user", "--email", "ada@example.com", "--password-stdin", stdin="short\n")
    with pytest.raises(ValueError, match="first line of stdin"):
        await _run(cli, db_session, monkeypatch, "create-user", "--email", "ada@example.com", "--password-stdin", stdin="")

    assert (await db_session.execute(select(User).where(User.email == "ada@example.com"))).first() is None


async def test_the_cli_prompts_twice_without_echo_and_refuses_a_mismatch(
    cli, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter([PASSWORD, PASSWORD + "!"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))
    with pytest.raises(ValueError, match="do not match"):
        await _run(cli, db_session, monkeypatch, "create-user", "--email", "ada@example.com")

    answers = iter([PASSWORD, PASSWORD])
    await _run(cli, db_session, monkeypatch, "create-user", "--email", "ada@example.com")
    assert (await accounts.get_user(db_session, "ada@example.com")).email == "ada@example.com"


@pytest.mark.parametrize("flag", ["--password", "--pass", "--password=hunter2hunter2"])
def test_a_password_is_never_accepted_as_an_argument(cli, flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["create-user", "--email", "a@example.com", flag, "hunter2hunter2"])

    assert excinfo.value.code == 2


async def test_the_cli_command_line_end_to_end() -> None:
    """The real entry point against the test database, as a subprocess. It
    commits, so the rows are removed again."""
    suffix = uuid.uuid4().hex[:8]
    slug, email = f"cli-{suffix}", f"cli-{suffix}@example.com"
    env = {**os.environ, "ANTHROPIC_API_KEY": ""}

    def run(*argv: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/accounts.py", *argv],
            cwd=BACKEND_DIR,
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        assert run("create-organization", "--name", "CLI test", "--slug", slug).returncode == 0
        assert run("create-user", "--email", email, "--password-stdin", stdin=PASSWORD + "\n").returncode == 0
        assert run("add-member", "--organization", slug, "--email", email, "--role", "owner").returncode == 0
        made = run("create-api-key", "--organization", slug, "--name", "ci")
        assert made.returncode == 0, made.stderr
        assert made.stdout == made.stdout.strip() + "\n" and tokens.parse_api_key(made.stdout.strip())
        assert made.stdout.strip() not in made.stderr

        refused = run("set-password", "--email", email, "--password-stdin", stdin="short\n")
        assert refused.returncode == 1
        assert refused.stderr.startswith("error: a password is 12 to 128 characters long")
        assert "Traceback" not in refused.stderr and "short" not in refused.stdout

        unknown = run("add-member", "--organization", "no-such-org", "--email", email)
        assert unknown.returncode == 1 and "no organization" in unknown.stderr

        assert run("prune-sessions").returncode == 0
    finally:
        async with engine.begin() as connection:
            await connection.execute(delete(Organization).where(Organization.slug == slug))
            await connection.execute(delete(User).where(User.email == email))


# --- canonical forms ---------------------------------------------------------

NFC_EMAIL = unicodedata.normalize("NFC", "café@example.com")
NFD_EMAIL = unicodedata.normalize("NFD", NFC_EMAIL)
LONE_SURROGATE = "correct horse \ud800 battery"


def test_the_two_spellings_of_the_test_email_really_differ() -> None:
    assert NFC_EMAIL != NFD_EMAIL and len(NFD_EMAIL) == len(NFC_EMAIL) + 1


async def test_the_precomposed_and_decomposed_spellings_of_an_email_are_one_account(
    db_session: AsyncSession,
) -> None:
    user = await _user(db_session, NFC_EMAIL)

    assert user.email == NFC_EMAIL
    with pytest.raises(accounts.AccountError, match="already has an account"):
        await _user(db_session, NFD_EMAIL)
    assert (await accounts.get_user(db_session, NFD_EMAIL)).id == user.id
    assert (await accounts.get_user(db_session, NFC_EMAIL)).id == user.id


async def test_a_decomposed_email_is_stored_in_its_canonical_form(db_session: AsyncSession) -> None:
    user = await _user(db_session, NFD_EMAIL)

    assert user.email == NFC_EMAIL
    assert (await accounts.get_user(db_session, NFC_EMAIL)).id == user.id


async def test_compatibility_forms_of_an_email_are_one_account(db_session: AsyncSession) -> None:
    user = await _user(db_session, "ada@example.com")

    # Fullwidth letters and an ideographic space fold to plain ones under NFKC.
    assert (await accounts.get_user(db_session, "ａｄａ@example.com　")).id == user.id


async def test_uppercase_and_lowercase_emails_are_one_account(db_session: AsyncSession) -> None:
    user = await _user(db_session, "Ada.Lovelace@Example.COM")

    assert (await accounts.get_user(db_session, "ada.lovelace@example.com")).id == user.id
    assert (await accounts.get_user(db_session, "ADA.LOVELACE@EXAMPLE.COM")).id == user.id
    with pytest.raises(accounts.AccountError, match="already has an account"):
        await _user(db_session, "ada.lovelace@example.com")


async def test_a_decomposed_password_verifies_against_a_precomposed_one_and_back() -> None:
    precomposed = unicodedata.normalize("NFC", "passé word café über")
    decomposed = unicodedata.normalize("NFD", precomposed)
    assert precomposed != decomposed

    from_precomposed = await passwords.hash_password(precomposed)
    from_decomposed = await passwords.hash_password(decomposed)

    assert await passwords.verify_password(decomposed, from_precomposed) is True
    assert await passwords.verify_password(precomposed, from_decomposed) is True
    assert await passwords.verify_password(precomposed + "x", from_precomposed) is False
    # Compatibility forms too: a fullwidth letter is the plain one.
    assert await passwords.verify_password(PASSWORD.replace("c", "ｃ"), await passwords.hash_password(PASSWORD)) is True


async def test_a_password_set_in_one_form_signs_in_with_the_other(db_session: AsyncSession) -> None:
    user = await accounts.create_user(
        db_session, email="ada@example.com", password=unicodedata.normalize("NFC", "café au lait 1")
    )

    assert await passwords.verify_password(unicodedata.normalize("NFD", "café au lait 1"), user.password_hash)


async def test_password_length_counts_normalized_characters() -> None:
    six_accented_letters = unicodedata.normalize("NFD", "é" * 6)  # 12 code points, 6 characters
    assert len(six_accented_letters) == 12

    with pytest.raises(ValueError, match="12 to 128"):
        await passwords.hash_password(six_accented_letters)
    assert await passwords.hash_password(unicodedata.normalize("NFD", "é" * 12))


@pytest.mark.parametrize(
    "email",
    [
        "a",
        "@b",
        "a@",
        "a@@b",
        "a@b@c",
        "a b@c",
        "a@b c",
        "a\tb@c",
        "a\nb@c",
        "a\x00b@c",
        "a\x1fb@c",
        "a\x7fb@c",
        "a\u200bb@c",
        "a b@c",
        "a\ud800b@c",
    ],
)
async def test_an_email_without_exactly_one_at_or_with_a_space_or_control_character_is_refused(
    db_session: AsyncSession, email: str
) -> None:
    with pytest.raises(accounts.AccountError, match="exactly one"):
        await accounts.create_user(db_session, email=email, password=PASSWORD)
    with pytest.raises(accounts.AccountError, match="exactly one"):
        await accounts.get_user(db_session, email)


@pytest.mark.parametrize(
    "email",
    ["a", "@b", "a@", "a@@b", "a@b@c", "a b@c", "a@b c", "a\tb@c", "a\nb@c", "a@b\n", "\na@b", "a\x01b@c", "a\x7fb@c"],
)
async def test_the_database_refuses_an_email_of_the_wrong_shape(db_session: AsyncSession, email: str) -> None:
    await _refuses(db_session, User(email=email, password_hash="x"), "ck_users_email_shape")


@pytest.mark.parametrize("email", ["a@b", "a.b+c@d.example", NFC_EMAIL, "a@b-c.d"])
async def test_the_database_accepts_an_email_of_the_right_shape(db_session: AsyncSession, email: str) -> None:
    await _accepts(db_session, User(email=email, password_hash="x"))


# --- text that cannot be encoded ---------------------------------------------


async def test_a_password_with_a_lone_surrogate_never_matches_and_never_raises() -> None:
    password_hash = await passwords.hash_password(PASSWORD)

    assert await passwords.verify_password(LONE_SURROGATE, password_hash) is False
    assert await passwords.verify_dummy_password(LONE_SURROGATE) is None


async def test_hashing_a_password_with_a_lone_surrogate_is_a_clear_value_error(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(ValueError, match="cannot be encoded as UTF-8"):
        await passwords.hash_password(LONE_SURROGATE)
    with pytest.raises(ValueError, match="cannot be encoded as UTF-8"):
        await accounts.create_user(db_session, email="ada@example.com", password=LONE_SURROGATE)
    user = await _user(db_session)
    with pytest.raises(ValueError, match="cannot be encoded as UTF-8"):
        await accounts.set_password(db_session, user, LONE_SURROGATE)


def test_a_digest_of_any_string_is_32_bytes_and_distinct_strings_stay_distinct() -> None:
    assert len(tokens.digest(LONE_SURROGATE)) == 32
    assert len({tokens.digest(token) for token in ("\ud800", "\ud801", "?", "�", "")}) == 5


@given(st.text(st.characters(categories=["Cs"]) | st.characters()))
def test_a_digest_never_raises(token: str) -> None:
    assert len(tokens.digest(token)) == 32


async def test_a_cookie_or_key_with_a_lone_surrogate_finds_nothing(db_session: AsyncSession) -> None:
    assert await sessions.lookup_session(db_session, "\ud800" * 43) is None
    assert await sessions.delete_session(db_session, "\ud800" * 43) is False
    assert await api_keys.find_active_api_key(db_session, f"dpk_{'a' * 12}_{'\ud800' * 43}") is None


# --- names -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", "x" * 101, "a\x00b", "a\nb"])
async def test_an_api_key_name_must_be_1_to_100_characters_without_control_characters(
    db_session: AsyncSession, name: str
) -> None:
    organization = await _organization(db_session)

    with pytest.raises(accounts.AccountError, match="1 to 100"):
        await api_keys.create_api_key(db_session, organization_id=organization.id, name=name)


async def test_names_are_trimmed(db_session: AsyncSession) -> None:
    organization = await accounts.create_organization(db_session, name="  Acme Inc \n", slug="acme")
    key, _ = await api_keys.create_api_key(db_session, organization_id=organization.id, name="  laptop  ")

    assert (organization.name, key.name) == ("Acme Inc", "laptop")
    assert (await api_keys.create_api_key(db_session, organization_id=organization.id, name="x" * 100))[0]


@pytest.mark.parametrize("name", ["   ", "\n", "a\x00b"])
async def test_an_organization_name_that_is_blank_or_has_a_control_character_is_refused(
    db_session: AsyncSession, name: str
) -> None:
    with pytest.raises(accounts.AccountError, match="1 to 100"):
        await accounts.create_organization(db_session, name=name, slug="acme")


async def test_a_slug_that_cannot_exist_is_not_looked_up(db_session: AsyncSession) -> None:
    with pytest.raises(accounts.AccountError, match="no organization"):
        await accounts.get_organization(db_session, "a\x00b")


# --- CLI error output --------------------------------------------------------


async def test_the_cli_names_a_refused_constraint_and_nothing_else(
    cli, db_session: AsyncSession
) -> None:
    with pytest.raises(IntegrityError) as excinfo:
        async with db_session.begin_nested():
            db_session.add(User(email="a", password_hash="$argon2id$SECRET-HASH"))
            await db_session.flush()

    message = cli.refusal_message(excinfo.value)

    assert message == "the database refused the change (ck_users_email_shape)"
    assert "SECRET-HASH" not in message and "INSERT" not in message


async def test_the_cli_reports_a_refused_constraint_as_one_stderr_line_with_exit_1(
    cli, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Refused(Exception):
        constraint_name = "organizations_slug_key"

    async def refuse(session, **kwargs):
        raise IntegrityError(
            "INSERT INTO organizations (name, slug) VALUES ($1, $2)", {"name": "SECRET-NAME"}, Refused("duplicate key")
        )

    monkeypatch.setattr(cli.accounts, "create_organization", refuse)
    monkeypatch.setattr(cli, "engine", SimpleNamespace(dispose=AsyncMock()))

    code = await cli.main(["create-organization", "--name", "Acme", "--slug", "acme-cli"])

    out, err = capsys.readouterr()
    assert code == 1 and out == ""
    assert err == "error: the database refused the change (organizations_slug_key)\n"


async def test_a_password_typed_as_an_argument_is_not_echoed_in_the_error(
    cli, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (
        ["create-user", "--email", "a@example.com", "--password", "hunter2hunter2"],
        ["set-password", "--email", "a@example.com", "--password=hunter2hunter2"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(argv)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        assert "hunter2hunter2" not in err
        assert "unrecognized arguments (not shown)" in err and "--password-stdin" in err


async def test_other_usage_errors_still_say_what_is_wrong(cli, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["create-user"])

    assert "the following arguments are required: --email" in capsys.readouterr().err
