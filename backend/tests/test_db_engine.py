"""The process's database engine (app/db.py): its pool is sized by the DB_POOL_*
settings, and its connections say which process they belong to in
pg_stat_activity. The defaults are SQLAlchemy's own, so setting nothing changes
nothing."""

from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app import db as db_module
from app import main as main_module
from app import worker as worker_module
from app.config import Settings
from app.db import build_engine, engine, name_process
from app.main import app
from app.mcp_server import __main__ as mcp_main


class _Stopped(Exception):
    """Raised by a stub to end worker.main() right after its first connection."""


def _settings(**overrides) -> Settings:
    """Settings as the suite's environment has them, but from no env file."""
    return Settings(_env_file=None, **overrides)


@pytest.fixture
async def engines() -> AsyncGenerator[list[AsyncEngine]]:
    """Engines a test builds, disposed afterwards so no connection outlives it."""
    built: list[AsyncEngine] = []
    yield built
    for built_engine in built:
        await built_engine.dispose()


async def _application_name(connection_engine: AsyncEngine) -> str:
    async with connection_engine.connect() as connection:
        return (
            await connection.execute(
                text("SELECT application_name FROM pg_stat_activity WHERE pid = pg_backend_pid()")
            )
        ).scalar_one()


def _pool_numbers(pooled: AsyncEngine) -> tuple[int, int, float]:
    pool = pooled.pool
    return pool.size(), pool._max_overflow, pool._timeout


# --- the pool ----------------------------------------------------------------


def test_the_pool_defaults_are_the_ones_sqlalchemy_applies_when_given_none() -> None:
    sqlalchemy_default = create_async_engine("postgresql+asyncpg://user:pw@localhost/none")
    settings = _settings()

    assert _pool_numbers(sqlalchemy_default) == (5, 10, 30.0)
    assert (settings.db_pool_size, settings.db_max_overflow, settings.db_pool_timeout_seconds) == (5, 10, 30.0)
    assert _pool_numbers(build_engine(settings, {})) == _pool_numbers(sqlalchemy_default)
    assert _pool_numbers(engine) == _pool_numbers(sqlalchemy_default)


def test_the_pool_settings_reach_the_engine() -> None:
    sized = build_engine(
        _settings(db_pool_size=2, db_max_overflow=3, db_pool_timeout_seconds=7.5), {}
    )

    assert _pool_numbers(sized) == (2, 3, 7.5)


def test_the_pool_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_POOL_SIZE", "20")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "0")
    monkeypatch.setenv("DB_POOL_TIMEOUT_SECONDS", "2")

    settings = _settings()

    assert (settings.db_pool_size, settings.db_max_overflow, settings.db_pool_timeout_seconds) == (20, 0, 2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"db_pool_size": 0},  # SQLAlchemy reads 0 as "no limit"
        {"db_max_overflow": -1},  # and -1 as "unlimited overflow"
        {"db_pool_timeout_seconds": 0},
        {"db_pool_timeout_seconds": float("inf")},
    ],
)
def test_a_pool_setting_that_would_remove_a_limit_is_refused(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        _settings(**overrides)


# --- application_name --------------------------------------------------------


async def test_an_engines_connections_carry_its_application_name(engines: list[AsyncEngine]) -> None:
    named = build_engine(_settings(), {"application_name": "doc-pilot-test-engine"})
    engines.append(named)

    assert await _application_name(named) == "doc-pilot-test-engine"
    async with named.connect() as connection:
        visible = (
            await connection.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE application_name = 'doc-pilot-test-engine'")
            )
        ).scalar_one()
    assert visible >= 1


async def test_naming_the_process_names_the_connections_opened_afterwards(
    monkeypatch: pytest.MonkeyPatch, engines: list[AsyncEngine]
) -> None:
    monkeypatch.setitem(db_module._server_settings, "application_name", "doc-pilot")
    process_engine = build_engine(_settings(), db_module._server_settings)
    engines.append(process_engine)
    assert await _application_name(process_engine) == "doc-pilot"
    await process_engine.dispose()

    name_process("doc-pilot-worker")

    assert await _application_name(process_engine) == "doc-pilot-worker"


async def test_the_app_engine_sends_an_application_name() -> None:
    assert "application_name" in db_module._server_settings
    assert await _application_name(engine)


async def test_each_entry_point_names_its_process_before_its_first_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[str] = []

    async def isolation_check(checked_engine) -> None:
        calls.append("first connection")

    for module in (main_module, worker_module, mcp_main):
        monkeypatch.setattr(module, "name_process", lambda name: calls.append(name))
        monkeypatch.setattr(module, "verify_tenant_isolation", isolation_check)

    monkeypatch.setattr(main_module, "warm_up_embedder", lambda: None)
    async with app.router.lifespan_context(app):
        pass
    assert calls[:2] == ["doc-pilot-api", "first connection"]

    calls.clear()

    def stop_after_the_check(settings) -> None:
        raise _Stopped

    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: Settings(_env_file=None, worker_heartbeat_file=str(tmp_path / "heartbeat")),
    )
    monkeypatch.setattr(worker_module, "check_model_provider", stop_after_the_check)
    with pytest.raises(_Stopped):
        await worker_module.main()
    assert calls == ["doc-pilot-worker", "first connection"]

    calls.clear()

    async def nothing() -> None:
        return None

    monkeypatch.setattr(mcp_main, "engine", SimpleNamespace(dispose=nothing))
    monkeypatch.setattr(mcp_main, "get_settings", lambda: SimpleNamespace(mcp_stdio_api_key=None))
    assert await mcp_main.serve() == 2  # no DOCPILOT_API_KEY: refused after the check
    assert calls == ["doc-pilot-mcp", "first connection"]

