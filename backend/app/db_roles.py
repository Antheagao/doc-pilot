"""The application's login role, created outside Alembic.

Row-level security (the tenant_isolation policies, migration 73a7d8691932) binds
only a role that is not a superuser, has no BYPASSRLS, and owns no table. The
migration creates the NOLOGIN group role `docpilot_app` and gives it every
grant; the application connects through a login role that is a member of it.
A login role and its password do not belong in a migration, so this module
makes that role, from the user and password of DATABASE_URL, over a connection
as the database owner (MIGRATION_DATABASE_URL). The compose `migrate` service
and CI run it after `alembic upgrade head` (scripts/provision_db_role.py), and
the test session runs it for `docpilot_test_runtime` (tests/conftest.py).

The password is never put into a statement by Python: it travels as a bind
parameter into a transaction-local setting, and a DO block builds the CREATE or
ALTER ROLE with format('%I', ...) and format('%L', ...) on the server, so the
text that is sent, logged, or put in an exception holds no password. What is
sent is not even the password but its SCRAM-SHA-256 verifier, computed here the
way libpq does (RFC 5802 and 7677, PostgreSQL's documented
`SCRAM-SHA-256$<iterations>:<salt>$<StoredKey>:<ServerKey>` form), so a server
that logs every statement (log_statement = all) still never records it. Roles are
cluster-wide and the dev, demo, and test databases share a cluster, so this
refuses what could damage another database's login: it never touches the owner
role or a superuser, and it alters an existing role only when it is already a
member of docpilot_app (one this module made).
"""

import base64
import hashlib
import hmac
import secrets
import stringprep
import unicodedata

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

GROUP_ROLE = "docpilot_app"

# PostgreSQL's own default (scram_iterations): what a server-side hash would use.
_SCRAM_ITERATIONS = 4096

# Postgres's default port, so "localhost" and "localhost:5432" compare equal.
_DEFAULT_PORT = 5432


class ProvisioningError(RuntimeError):
    """The role cannot be provisioned safely; the message says what to change.
    It never contains a password."""


_PROVISION_SQL = """
DO $$
DECLARE
    login_role text := current_setting('docpilot.provision_role');
    verifier text := current_setting('docpilot.provision_verifier');
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'docpilot_app') THEN
        RAISE EXCEPTION 'role docpilot_app does not exist: run alembic upgrade head first';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = login_role) THEN
        EXECUTE format(
            'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB '
            'INHERIT IN ROLE docpilot_app',
            login_role, verifier);
    ELSIF EXISTS (
        SELECT 1
        FROM pg_auth_members m
        JOIN pg_roles member ON member.oid = m.member
        JOIN pg_roles grp ON grp.oid = m.roleid
        WHERE grp.rolname = 'docpilot_app' AND member.rolname = login_role AND NOT member.rolsuper
    ) THEN
        EXECUTE format(
            'ALTER ROLE %I WITH LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB INHERIT',
            login_role, verifier);
    ELSE
        RAISE EXCEPTION 'role % exists and is not a docpilot_app login role: refusing to alter it '
            '(if it is one of ours, GRANT docpilot_app TO it as the owner and run this again)', login_role;
    END IF;
END
$$
"""


def _saslprep(password: str) -> str:
    """The password as PostgreSQL's SASLprep (RFC 4013) treats it, the way
    libpq and asyncpg do before they hash it: an ASCII password is used as it
    is, and so is any password the profile cannot be applied to (prohibited
    characters, mixed-direction text), since the server falls back to the
    original there too."""
    if password.isascii():
        return password
    mapped = "".join(
        " " if stringprep.in_table_c12(c) else c for c in password if not stringprep.in_table_b1(c)
    )
    if not mapped:
        return password
    normalized = unicodedata.normalize("NFKC", mapped)
    if not normalized:
        return password
    prohibited = (
        stringprep.in_table_a1,
        stringprep.in_table_c12,
        stringprep.in_table_c21_c22,
        stringprep.in_table_c3,
        stringprep.in_table_c4,
        stringprep.in_table_c5,
        stringprep.in_table_c6,
        stringprep.in_table_c7,
        stringprep.in_table_c8,
        stringprep.in_table_c9,
    )
    if any(check(c) for c in normalized for check in prohibited):
        return password
    if any(stringprep.in_table_d1(c) for c in normalized):
        if not (stringprep.in_table_d1(normalized[0]) and stringprep.in_table_d1(normalized[-1])):
            return password
        if any(stringprep.in_table_d2(c) for c in normalized):
            return password
    return normalized


def scram_verifier(password: str, *, salt: bytes | None = None, iterations: int = _SCRAM_ITERATIONS) -> str:
    """The SCRAM-SHA-256 secret PostgreSQL stores for `password`, in the form
    ALTER ROLE accepts instead of a plaintext password. A fresh random salt
    unless one is given (a test passes one to compare against known values)."""
    salt = secrets.token_bytes(16) if salt is None else salt
    salted = hashlib.pbkdf2_hmac("sha256", _saslprep(password).encode("utf-8"), salt, iterations)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    b64 = base64.b64encode
    return (
        f"SCRAM-SHA-256${iterations}:{b64(salt).decode()}"
        f"${b64(stored_key).decode()}:{b64(server_key).decode()}"
    )


def check_urls(migration_url: str | URL, runtime_url: str | URL) -> URL:
    """Validate the pair of URLs and return the runtime one, parsed. Raises
    ProvisioningError when the runtime URL names no user or password, names
    the owner (provisioning would demote it), or names a different server or
    database than the owner URL (the role would be made on one cluster for an
    application that connects to another)."""
    owner = make_url(migration_url)
    runtime = make_url(runtime_url)
    if not runtime.username or not runtime.password:
        raise ProvisioningError("DATABASE_URL must name the runtime role's user and password")
    if runtime.username == owner.username:
        raise ProvisioningError(
            "DATABASE_URL names the same user as MIGRATION_DATABASE_URL (the database owner): "
            "the application must connect as its own role, which this script creates"
        )
    if runtime.username == GROUP_ROLE:
        raise ProvisioningError(f"DATABASE_URL cannot name {GROUP_ROLE}: it is the NOLOGIN group role")

    def where(url: URL) -> tuple[str | None, int, str | None]:
        return (url.host, url.port or _DEFAULT_PORT, url.database)

    if where(owner) != where(runtime):
        raise ProvisioningError(
            "DATABASE_URL and MIGRATION_DATABASE_URL name different servers or databases; "
            "they must differ only in the user and password"
        )
    return runtime


async def ensure_runtime_role(engine: AsyncEngine, role: str, password: str) -> None:
    """Create the login role `role`, or reset the password and attributes of
    the one this function made, as a member of docpilot_app. `engine` connects
    as a role that may create roles (the owner); roles are cluster-wide, so any
    database of the cluster will do, but docpilot_app must exist.

    Raises ProvisioningError when docpilot_app is missing or `role` exists and
    is not one of ours."""
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "SELECT set_config('docpilot.provision_role', :role, true), "
                    "set_config('docpilot.provision_verifier', :verifier, true)"
                ),
                {"role": role, "verifier": scram_verifier(password)},
            )
            await connection.execute(text(_PROVISION_SQL))
    except DBAPIError as exc:
        # SQLAlchemy's own message carries the parameters of the failed
        # statement, which here include the password; `from None` also keeps
        # the original out of the traceback.
        raise ProvisioningError(str(exc.orig)) from None


async def provision_runtime_role(migration_url: str | URL, runtime_url: str | URL) -> str:
    """Provision the login role DATABASE_URL names, over MIGRATION_DATABASE_URL,
    and return its name. See check_urls and ensure_runtime_role for the
    refusals."""
    runtime = check_urls(migration_url, runtime_url)
    engine = create_async_engine(migration_url, poolclass=NullPool, hide_parameters=True)
    try:
        await ensure_runtime_role(engine, runtime.username, runtime.password)
    finally:
        await engine.dispose()
    return runtime.username
