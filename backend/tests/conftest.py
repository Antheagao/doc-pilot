import asyncio
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
import pytest
import pytest_asyncio
from fastapi import Depends
from httpx import ASGITransport, AsyncClient
from hypothesis import settings as hypothesis_settings
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import BACKEND_DIR, Settings

# The suite runs against its own database on the same server as the dev one,
# so the compose worker polling `docpilot` can never claim the jobs the worker
# tests commit for real. This must happen before `app.db` is imported (it
# builds its engine from get_settings() at import), hence the module-level
# code ahead of the app imports below.
TEST_DATABASE_NAME = "docpilot_test"

# The suite must be unable to spend. backend/.env holds the real Anthropic key,
# and Settings() reads it from that file, so a test that reaches a model route
# without a fake would bill real calls. A variable in the environment beats the
# env file in pydantic-settings, so an empty one makes the key absent for every
# Settings() the suite builds (explicit Settings(anthropic_api_key=...) still
# wins). The one opt-in is RUN_LIVE_SMOKE=1 (tests/test_live_smoke.py), which
# needs the key from backend/.env. Before any app import: get_settings() caches.
if os.environ.get("RUN_LIVE_SMOKE") != "1":
    os.environ["ANTHROPIC_API_KEY"] = ""

# The suite connects as a non-owner login role, so row-level security binds
# every test the way it binds the running app. Roles are cluster-wide and the
# dev and demo databases share the cluster, so this is the only role the suite
# ever creates or alters: never the dev runtime role (docpilot_runtime) and
# never the owner. The password is dev-grade, like the owner's.
TEST_RUNTIME_ROLE = "docpilot_test_runtime"
TEST_RUNTIME_PASSWORD = "docpilot_test_runtime"


def _owner_database_url() -> URL:
    """The test database as its owner connects: the role that migrates it and
    provisions the runtime role. MIGRATION_DATABASE_URL names the owner when
    DATABASE_URL has become the dev runtime role."""
    settings = Settings()
    dev_url = make_url(settings.migration_database_url or settings.database_url)
    explicit = os.environ.get("TEST_DATABASE_URL")
    if not explicit:
        # Also fine when DATABASE_URL already names docpilot_test.
        url = dev_url.set(database=TEST_DATABASE_NAME)
    else:
        url = make_url(explicit)
        if (url.host, url.port, url.database) == (dev_url.host, dev_url.port, dev_url.database):
            raise pytest.UsageError(
                f"TEST_DATABASE_URL and DATABASE_URL both name {dev_url.database!r}; the tests "
                f"commit rows and must not share the dev database (unset TEST_DATABASE_URL to "
                f"use {TEST_DATABASE_NAME!r})"
            )
    if url.username == TEST_RUNTIME_ROLE:
        raise pytest.UsageError(
            f"the owner URL names {TEST_RUNTIME_ROLE!r}, the suite's own non-owner role: "
            "set MIGRATION_DATABASE_URL (or DATABASE_URL) to the database owner"
        )
    return url


OWNER_DATABASE_URL = _owner_database_url()
TEST_DATABASE_URL = OWNER_DATABASE_URL.set(username=TEST_RUNTIME_ROLE, password=TEST_RUNTIME_PASSWORD)
# The app (app.db) reads the runtime role; the alembic subprocess reads the owner.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL.render_as_string(hide_password=False)
os.environ["MIGRATION_DATABASE_URL"] = OWNER_DATABASE_URL.render_as_string(hide_password=False)

from tenant_rows import populate

from app.auth.principal import (
    Principal,
    current_principal,
    get_identity_session_factory,
)
from app.config import get_settings
from app.db import (
    TenantIsolationError,
    engine,
    get_session,
    get_tenant_session_factory,
    verify_tenant_isolation,
)
from app.db_roles import ProvisioningError, ensure_runtime_role
from app.main import app
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    MEMBERSHIP_ROLE_OWNER,
    Membership,
    Organization,
    User,
)
from app.routers.ask import get_anthropic_client
from app.tenancy import bind_tenant

if engine.url.database != TEST_DATABASE_URL.database:
    raise pytest.UsageError(
        f"the app engine is bound to {engine.url.database!r}, not the test database "
        f"{TEST_DATABASE_URL.database!r}: app.db was imported before tests/conftest.py set DATABASE_URL"
    )
if engine.url.username != TEST_RUNTIME_ROLE:
    raise pytest.UsageError(
        f"the app engine connects as {engine.url.username!r}, not {TEST_RUNTIME_ROLE!r}: "
        "app.db was imported before tests/conftest.py set DATABASE_URL"
    )


# Property tests (tests/test_reconcile_properties.py) draw the same examples on
# every run, so a failure reproduces on CI and a green run is repeatable; no
# deadline, because a slow shared runner is not a bug. A failing property stops
# at its first minimal failure (report_multiple_bugs=False) instead of shrinking
# every distinct failure it finds, which takes minutes on a wide strategy.
hypothesis_settings.register_profile(
    "doc-pilot", derandomize=True, deadline=None, report_multiple_bugs=False
)
hypothesis_settings.load_profile("doc-pilot")

# Serializes parallel pytest runs against one server: only one creates and
# migrates the database at a time.
SETUP_LOCK_ID = 7_243_001


async def _prepare_test_database(url: URL) -> None:
    """Create and migrate the test database (as its owner, `url`) and provision
    the runtime role the suite connects as."""
    # Connect to the server's maintenance database: a database cannot be
    # created from inside a transaction, or from itself.
    maintenance = url.set(drivername="postgresql", database="postgres")
    connection = await asyncpg.connect(maintenance.render_as_string(hide_password=False))
    try:
        # Held on this connection until the migration and the role are done, so
        # a second run on an empty server waits instead of racing `alembic
        # upgrade` or the role's CREATE and ALTER.
        await connection.execute("SELECT pg_advisory_lock($1)", SETUP_LOCK_ID)
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", url.database)
        if not exists:
            quoted = '"' + url.database.replace('"', '""') + '"'
            await connection.execute(f"CREATE DATABASE {quoted}")
        _migrate_to_head()
        await _provision_runtime_role(url)
        await connection.execute("SELECT pg_advisory_unlock($1)", SETUP_LOCK_ID)
    finally:
        await connection.close()


async def _provision_runtime_role(owner_url: URL) -> None:
    """Make the suite's login role (the only one it ever touches), then check,
    through a connection of its own, that it is the kind row-level security
    binds."""
    owner = create_async_engine(owner_url, poolclass=NullPool, hide_parameters=True)
    try:
        await ensure_runtime_role(owner, TEST_RUNTIME_ROLE, TEST_RUNTIME_PASSWORD)
    except ProvisioningError as exc:
        raise pytest.UsageError(f"cannot provision {TEST_RUNTIME_ROLE!r} for the suite: {exc}") from None
    finally:
        await owner.dispose()
    runtime = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    try:
        await verify_tenant_isolation(runtime)
    except TenantIsolationError as exc:
        raise pytest.UsageError(f"the suite's role would skip row-level security: {exc}") from None
    finally:
        await runtime.dispose()


def _migrate_to_head() -> None:
    # A subprocess runs alembic: its env.py reconfigures logging and starts
    # its own event loop, neither of which belongs inside the test process.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env=os.environ,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise pytest.UsageError(
            f"`alembic upgrade head` failed on the test database (exit {result.returncode}):\n"
            f"{result.stderr}\n{result.stdout}"
        )


def pytest_configure(config: pytest.Config) -> None:
    """Create the test database if it is missing, bring it to the migration
    head (which also installs the pgvector extension), and provision the
    non-owner role the suite runs as, so a fresh checkout or a fresh CI service
    needs no setup step."""
    asyncio.run(_prepare_test_database(OWNER_DATABASE_URL))


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _row_level_security_binds_the_suite() -> None:
    """The app's own engine, the one every test uses, connects as a role that
    row-level security binds. The suite must never run with the second tenancy
    layer silently off."""
    await verify_tenant_isolation(engine)


@pytest.fixture
def owner_database_url() -> URL:
    """The test database as its owner connects (a superuser, which row-level
    security never binds). For the tests of a layer on its own: the application
    layer without the database one, the policies' bypass rules, the downgrade."""
    return OWNER_DATABASE_URL


@pytest_asyncio.fixture
async def owner_engine(owner_database_url: URL) -> AsyncGenerator[AsyncEngine]:
    """An engine connecting as the database owner, with no pool so nothing it
    does outlives the test. Never use it for what the application does."""
    owner = create_async_engine(owner_database_url, poolclass=NullPool)
    try:
        yield owner
    finally:
        await owner.dispose()


@asynccontextmanager
async def _rolled_back_session(bind: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    async with bind.connect() as connection:
        trans = await connection.begin()
        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        await bind_tenant(session, DEFAULT_ORGANIZATION_ID)
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    """A session bound to a single connection/transaction that is always
    rolled back at the end of the test, so tests leave no rows in the test
    database. join_transaction_mode="create_savepoint"
    lets app code call session.commit() (as the upload endpoint does)
    without prematurely ending the outer transaction.

    It connects as the suite's runtime role, so row-level security binds it as
    it binds the app. The session is bound to the default organization, as
    every request and every worker handler is (app/tenancy.py), so what a test
    adds belongs to it and what it queries is scoped to it. A test that needs
    another organization uses `second_organization`, or binds one itself.

    One harness limit: a setting made inside the test's savepoint and released
    (a commit) stays for the rest of the test's transaction, where in the app
    the end of the transaction ends it. `docpilot.queue_scope` is the one to
    mind; tests/test_rls.py shows how to test it on a connection of its own.
    """
    async with _rolled_back_session(engine) as session:
        yield session


@pytest_asyncio.fixture
async def owner_session(owner_engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    """`db_session` on the owner's connection, which row-level security never
    binds: for the tests that show the application layer holds on its own."""
    async with _rolled_back_session(owner_engine) as session:
        yield session


@pytest_asyncio.fixture
async def second_organization(db_session: AsyncSession) -> Organization:
    """A second organization, created in the test's transaction. The session
    stays bound to the default organization; bind this one to act as it."""
    organization = Organization(name="Second", slug=f"second-{uuid.uuid4().hex[:8]}")
    db_session.add(organization)
    await db_session.flush()
    return organization


@pytest_asyncio.fixture
async def two_tenants(
    db_session: AsyncSession, second_organization: Organization
) -> tuple[dict[type, uuid.UUID], dict[type, uuid.UUID]]:
    """A row of every content model in the default organization and in the
    second one. The session ends up bound to the default organization with an
    empty identity map, so what a test reads comes from the database."""
    await bind_tenant(db_session, second_organization.id)
    theirs = await populate(db_session, "theirs")
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)
    ours = await populate(db_session, "ours")
    await bind_tenant(db_session, second_organization.id)
    await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)
    return ours, theirs


@pytest_asyncio.fixture
async def owner(db_session: AsyncSession) -> User:
    """A user who owns the default organization, created in the test's
    transaction. The password hash is a placeholder that matches nothing, so no
    test pays for a real hash it does not need; tests/test_auth.py makes real
    accounts."""
    user = User(email=f"owner-{uuid.uuid4().hex[:12]}@test.example", password_hash="!")
    db_session.add(user)
    await db_session.flush()
    db_session.add(
        Membership(
            organization_id=DEFAULT_ORGANIZATION_ID, user_id=user.id, role=MEMBERSHIP_ROLE_OWNER
        )
    )
    await db_session.flush()
    return user


class ModelCallAttempted(AssertionError):
    """A test reached the model without a fake."""


class _NoModelClient:
    """Stands in for the Anthropic client in the `client` fixtures: any use fails
    the test. A test that needs a model overrides `get_anthropic_client` with its
    scripted fake, which replaces this."""

    def __getattr__(self, name: str):
        raise ModelCallAttempted(
            f"a test used the model client ({name!r}) without overriding get_anthropic_client "
            "with a fake"
        )


def _guarded_anthropic_client(settings: Settings = Depends(get_settings)):
    """Like the real dependency, None (the route's 503) while the settings hold
    no key, and otherwise a client that refuses to be used."""
    return _NoModelClient() if settings.anthropic_api_key else None


@asynccontextmanager
async def _api_client(
    db_session: AsyncSession, tmp_path, principal: Principal | None
) -> AsyncGenerator[AsyncClient]:
    """The real app over ASGI with the database, settings, and identity
    sessions handed to the test's transaction. `principal` is who every request
    is made by; None leaves `current_principal` real, so the request must carry
    a session cookie."""

    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    def override_get_settings() -> Settings:
        # The spend cap and rate limits are off: spend committed to the test
        # database earlier in the day must not turn an upload into a 429,
        # and one in-process limiter serves the whole session. tests/test_budget.py and
        # tests/test_ratelimit.py turn them on explicitly.
        return Settings(
            upload_dir=str(tmp_path),
            daily_budget_usd=0,
            # One limiter serves the whole test session, from one client.
            ask_rate_limit_per_minute=0,
            upload_rate_limit_per_minute=0,
            login_rate_limit_per_minute=0,
            login_account_rate_limit_per_minute=0,
        )

    @asynccontextmanager
    async def shared_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_tenant_session_factory] = lambda: shared_session
    app.dependency_overrides[get_identity_session_factory] = lambda: shared_session
    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_anthropic_client] = _guarded_anthropic_client
    if principal is not None:
        app.dependency_overrides[current_principal] = lambda: principal
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession, owner: User, tmp_path) -> AsyncGenerator[AsyncClient]:
    """A client signed in as `owner`: `current_principal` is overridden, so a
    test of a route's own behavior needs no cookie. Tests of sign-in itself use
    `anon_client`."""
    principal = Principal(DEFAULT_ORGANIZATION_ID, owner.id, MEMBERSHIP_ROLE_OWNER)
    async with _api_client(db_session, tmp_path, principal) as ac:
        yield ac


@pytest_asyncio.fixture
async def anon_client(db_session: AsyncSession, tmp_path) -> AsyncGenerator[AsyncClient]:
    """A client with no override of `current_principal`: it reaches the real
    sign-in, cookie, and Origin checks (on the test's transaction)."""
    async with _api_client(db_session, tmp_path, None) as ac:
        yield ac


_EXPORTER = InMemorySpanExporter()


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """The spans the test emits, captured in memory. A process can install its
    global TracerProvider only once, so the first test to ask installs an SDK
    provider with the in-memory exporter (the app itself never installs one
    unless OTEL_EXPORTER_OTLP_ENDPOINT is set); the captured spans are cleared
    before each test."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    if not getattr(provider, "_docpilot_test_exporter", False):
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        provider._docpilot_test_exporter = True  # type: ignore[attr-defined]
    _EXPORTER.clear()
    return _EXPORTER
