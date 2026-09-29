from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import engine, get_session
from app.main import app


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    """A session bound to a single connection/transaction that is always
    rolled back at the end of the test, so tests don't leak rows into the
    real Postgres database on 5434. join_transaction_mode="create_savepoint"
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
        # The spend cap is off: tests share the dev database, and real
        # spend recorded there today must not turn an upload into a 429.
        # tests/test_budget.py turns it on explicitly.
        return Settings(upload_dir=str(tmp_path), daily_budget_usd=0)

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_settings] = override_get_settings
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
