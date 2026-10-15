import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from hypothesis import settings as hypothesis_settings
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BACKEND_DIR, Settings

# The suite runs against its own database on the same server as the dev one,
# so the compose worker polling `docpilot` can never claim the jobs the worker
# tests commit for real. This must happen before `app.db` is imported (it
# builds its engine from get_settings() at import), hence the module-level
# code ahead of the app imports below.
TEST_DATABASE_NAME = "docpilot_test"


def _test_database_url() -> URL:
    dev_url = make_url(Settings().database_url)
    explicit = os.environ.get("TEST_DATABASE_URL")
    if not explicit:
        # Also fine when DATABASE_URL already names docpilot_test.
        return dev_url.set(database=TEST_DATABASE_NAME)
    url = make_url(explicit)
    if (url.host, url.port, url.database) == (dev_url.host, dev_url.port, dev_url.database):
        raise pytest.UsageError(
            f"TEST_DATABASE_URL and DATABASE_URL both name {dev_url.database!r}; the tests "
            f"commit rows and must not share the dev database (unset TEST_DATABASE_URL to "
            f"use {TEST_DATABASE_NAME!r})"
        )
    return url


TEST_DATABASE_URL = _test_database_url()
os.environ["DATABASE_URL"] = TEST_DATABASE_URL.render_as_string(hide_password=False)

from app.config import get_settings
from app.db import engine, get_session, get_session_factory
from app.main import app

if engine.url.database != TEST_DATABASE_URL.database:
    raise pytest.UsageError(
        f"the app engine is bound to {engine.url.database!r}, not the test database "
        f"{TEST_DATABASE_URL.database!r}: app.db was imported before tests/conftest.py set DATABASE_URL"
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
    # Connect to the server's maintenance database: a database cannot be
    # created from inside a transaction, or from itself.
    maintenance = url.set(drivername="postgresql", database="postgres")
    connection = await asyncpg.connect(maintenance.render_as_string(hide_password=False))
    try:
        # Held on this connection until the migration finishes, so a second
        # run on an empty server waits instead of racing `alembic upgrade`.
        await connection.execute("SELECT pg_advisory_lock($1)", SETUP_LOCK_ID)
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", url.database)
        if not exists:
            quoted = '"' + url.database.replace('"', '""') + '"'
            await connection.execute(f"CREATE DATABASE {quoted}")
        _migrate_to_head()
        await connection.execute("SELECT pg_advisory_unlock($1)", SETUP_LOCK_ID)
    finally:
        await connection.close()


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
    """Create the test database if it is missing and bring it to the
    migration head (which also installs the pgvector extension), so a fresh
    checkout or a fresh CI service needs no setup step."""
    asyncio.run(_prepare_test_database(TEST_DATABASE_URL))


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    """A session bound to a single connection/transaction that is always
    rolled back at the end of the test, so tests leave no rows in the test
    database. join_transaction_mode="create_savepoint"
    lets app code call session.commit() (as the upload endpoint does)
    without prematurely ending the outer transaction.
    """
    async with engine.connect() as connection:
        trans = await connection.begin()
        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession, tmp_path) -> AsyncGenerator[AsyncClient]:
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
        )

    @asynccontextmanager
    async def shared_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_session_factory] = lambda: shared_session
    app.dependency_overrides[get_settings] = override_get_settings
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
