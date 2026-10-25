"""MODEL_PROVIDER=scripted end to end: an upload, the worker's extract and index
jobs, /ask and /ask/stream, and the judge job, with no model and no key.

Everything is real except the model: the real routes, claim, handlers,
reconciliation, chunking, search, agent loop and judge parser, on rows that are
committed (the worker opens sessions of its own, so nothing here can borrow the
test's rolled-back transaction). The settings come from the environment, the way
the load stack's containers get them. `anthropic.AsyncAnthropic.__init__` is
patched to raise, so the test fails if any code path builds a real client, and
the checks at the end read the stored rows: every one names `scripted`, and
every cost is 0.

Two organizations commit rows and are removed in a `finally`, as the isolation
suite's worker test does; the default organization is untouched.
"""

import json
import uuid
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from scripted_helpers import png_with_text_chunk
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.auth.principal import Principal, current_principal
from app.config import SCRIPTED_MODEL, get_settings
from app.db import async_session_maker
from app.evals.dataset import load_cases
from app.evals.online import process_judge_job
from app.main import app
from app.models import (
    IDENTITY_TABLES,
    JOB_KIND_EXTRACT,
    JOB_KIND_INDEX,
    JOB_KIND_JUDGE,
    AskRun,
    Base,
    Job,
    Organization,
)
from app.retrieval import index_job as index_job_module
from app.retrieval.embeddings import HashingEmbedder, get_embedder
from app.routers import ask as ask_module
from app.routers.ask import get_anthropic_client
from app.tenancy import tenant_session
from app.worker import run_once

EMBEDDER = HashingEmbedder()
CASES = {case.doc_id: case for case in load_cases()}
# The content tables, children first, the way app/db.py derives them for the
# row-level security check.
TABLES = tuple(
    table.name for table in reversed(Base.metadata.sorted_tables) if table.name not in IDENTITY_TABLES
)
# Each tenant uploads one corpus picture, made byte-unique by a metadata chunk.
UPLOADS = {
    "a": ("001-clean-coffee-receipt", "Cascade Coffee Roasters"),
    "b": ("018-dense-office-outfitters", "Northgate Office Outfitters"),
}


@pytest.fixture
def scripted_environment(tmp_path: Path) -> Iterator[None]:
    """The scripted provider as the load stack sets it: environment variables,
    read by the one cached get_settings(), which is rebuilt on the way in and on
    the way out so no other test sees this one's settings."""
    with pytest.MonkeyPatch.context() as env:
        for key, value in {
            "MODEL_PROVIDER": SCRIPTED_MODEL,
            "EXTRACTION_MODEL": SCRIPTED_MODEL,
            "TRANSCRIPTION_MODEL": SCRIPTED_MODEL,
            "AGENT_MODEL": SCRIPTED_MODEL,
            "JUDGE_MODEL": SCRIPTED_MODEL,
            "ANTHROPIC_API_KEY": "",
            "EMBEDDING_BACKEND": "hashing",
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "DAILY_BUDGET_USD": "0",
            "ASK_RATE_LIMIT_PER_MINUTE": "0",
            "UPLOAD_RATE_LIMIT_PER_MINUTE": "0",
            "ASK_JUDGE_SAMPLE_RATE": "1",
        }.items():
            env.setenv(key, value)
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture
def no_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a real Anthropic client was built")

    monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", refuse)
    monkeypatch.setattr(anthropic.Anthropic, "__init__", refuse)
    # The API caches its client for the process; this test builds it afresh.
    monkeypatch.setattr(ask_module, "_client", None)
    monkeypatch.setattr(index_job_module, "get_embedder", lambda: EMBEDDER)


@dataclass
class Tenant:
    tag: str
    organization_id: uuid.UUID
    doc_id: str
    vendor: str
    document_id: uuid.UUID | None = None


async def _remove_rows(owner_engine: AsyncEngine, organization_ids: list[uuid.UUID]) -> None:
    """Delete what the test committed, as the owner (row-level security never
    binds it), children first."""
    async with owner_engine.begin() as connection:
        for table in TABLES:
            await connection.execute(
                text(f"DELETE FROM {table} WHERE organization_id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": organization_ids},
            )
        await connection.execute(
            text("DELETE FROM organizations WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": organization_ids},
        )


@pytest_asyncio.fixture
async def tenants(owner_engine: AsyncEngine) -> AsyncGenerator[dict[str, Tenant]]:
    built: dict[str, Tenant] = {}
    try:
        for tag, (doc_id, vendor) in UPLOADS.items():
            async with async_session_maker() as session:
                organization = Organization(
                    name=f"Scripted {tag}", slug=f"scripted-{tag}-{uuid.uuid4().hex[:8]}"
                )
                session.add(organization)
                await session.commit()
                built[tag] = Tenant(tag, organization.id, doc_id, vendor)
        yield built
    finally:
        await _remove_rows(owner_engine, [tenant.organization_id for tenant in built.values()])


@pytest_asyncio.fixture
async def api(
    scripted_environment, no_real_client, tenants: dict[str, Tenant]
) -> AsyncGenerator[tuple[AsyncClient, dict[str, str]]]:
    """The real app over ASGI as one tenant at a time. Only the sign-in and the
    embedder are overridden: sessions, settings and the model client are the
    real dependencies."""
    caller = {"tag": "a"}
    app.dependency_overrides[current_principal] = lambda: Principal(
        tenants[caller["tag"]].organization_id
    )
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    assert get_anthropic_client not in app.dependency_overrides  # the real one, no key
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, caller
    finally:
        app.dependency_overrides.clear()


def _sse_events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def test_the_whole_pipeline_runs_on_the_scripted_model_and_bills_nothing(
    api, tenants: dict[str, Tenant], owner_engine: AsyncEngine
) -> None:
    client, caller = api
    settings = get_settings()
    assert settings.model_provider == "scripted" and not settings.anthropic_api_key

    # Upload through the real route, one document per tenant.
    for tag, tenant in tenants.items():
        caller["tag"] = tag
        raw = png_with_text_chunk(CASES[tenant.doc_id].image_path.read_bytes(), tag.encode())
        response = await client.post(
            "/documents", files={"file": (f"{tag}.png", raw, "image/png")}
        )
        assert response.status_code == 201, response.text
        tenant.document_id = uuid.UUID(response.json()["id"])

    # The worker's real claim and handlers: extract, then the index job it queued.
    for tenant in tenants.values():
        assert await run_once(document_id=tenant.document_id) is True
        assert await run_once(document_id=tenant.document_id) is True
        assert await run_once(document_id=tenant.document_id) is False

    async with owner_engine.connect() as connection:

        async def rows(sql: str, **parameters: Any) -> list[tuple]:
            return [tuple(row) for row in await connection.execute(text(sql), parameters)]

        for tenant in tenants.values():
            assert await rows(
                "SELECT kind, state FROM jobs WHERE document_id = :d ORDER BY kind",
                d=tenant.document_id,
            ) == [(JOB_KIND_EXTRACT, "done"), (JOB_KIND_INDEX, "done")]
            assert await rows(
                "SELECT status FROM documents WHERE id = :d", d=tenant.document_id
            ) == [("extracted",)]
            # The committed Sonnet run's answer, replayed through the real handler.
            assert await rows(
                "SELECT e.model, e.prompt_version, e.cost_usd::float8, e.input_tokens > 0, "
                "f.value->>'value' FROM extractions e JOIN extracted_fields f "
                "ON f.extraction_id = e.id WHERE e.document_id = :d AND f.field_name = 'vendor'",
                d=tenant.document_id,
            ) == [(SCRIPTED_MODEL, "extract_v1", 0.0, True, tenant.vendor)]
            assert await rows(
                "SELECT count(*) FROM reconciliations r JOIN extractions e "
                "ON e.id = r.extraction_id WHERE e.document_id = :d",
                d=tenant.document_id,
            ) == [(1,)]
            (page,) = await rows(
                "SELECT model, cost_usd::float8, source, text FROM document_pages "
                "WHERE document_id = :d",
                d=tenant.document_id,
            )
            assert page[:3] == (SCRIPTED_MODEL, 0.0, "transcription")
            assert tenant.vendor in page[3] and "\nRef " in page[3]
            assert (
                await rows(
                    "SELECT count(*) > 0 FROM document_chunks WHERE document_id = :d",
                    d=tenant.document_id,
                )
                == [(True,)]
            )

    # /ask and /ask/stream through the real get_anthropic_client dependency.
    runs: dict[str, list[dict[str, Any]]] = {}
    for tag, tenant in tenants.items():
        caller["tag"] = tag
        question = f"What was the total on my receipt from {tenant.vendor}?"
        asked = await client.post("/ask", json={"question": question})
        assert asked.status_code == 200, asked.text
        streamed = await client.post("/ask/stream", json={"question": question})
        assert streamed.status_code == 200
        events = _sse_events(streamed.text)
        assert [e["type"] for e in events if e["type"] in ("tool_call", "answer", "error")] == [
            "tool_call",
            "tool_call",
            "answer",
        ]
        assert [e["step"] for e in events if e["type"] == "model_call"] == [1, 2, 3]
        runs[tag] = [asked.json(), events[-1]["run"]]

        for run in runs[tag]:
            assert (run["status"], run["model"], run["cost_usd"]) == ("answered", SCRIPTED_MODEL, 0.0)
            assert run["steps"] == 3 and run["input_tokens"] > 0
            assert [call["name"] for call in run["tool_calls"]] == ["query_extractions", "get_page"]
            # Cited from this tenant's own rows only, resolved by the real code.
            assert {c["document_id"] for c in run["citations"]} == {str(tenant.document_id)}
            assert [c["kind"] for c in run["citations"]] == ["record", "page"]
            assert run["citations"][0]["cited_text"].startswith("total:")

    # The judge job each answer queued (ASK_JUDGE_SAMPLE_RATE=1), run by the
    # real handler in a session per job, as the worker gives it.
    for tenant in tenants.values():
        async with tenant_session(async_session_maker, tenant.organization_id) as session:
            judge_job_ids = list(
                (await session.execute(select(Job.id).where(Job.kind == JOB_KIND_JUDGE))).scalars()
            )
        assert len(judge_job_ids) == 2
        for job_id in judge_job_ids:
            async with tenant_session(async_session_maker, tenant.organization_id) as session:
                await process_judge_job(session, await session.get(Job, job_id))
                await session.commit()
        async with tenant_session(async_session_maker, tenant.organization_id) as session:
            graded = (await session.execute(select(AskRun))).scalars().all()
            assert [
                (r.model, r.judge_model, float(r.judge_cost_usd), r.judge_grounded, r.judge_error)
                for r in graded
            ] == [(SCRIPTED_MODEL, SCRIPTED_MODEL, 0.0, True, None)] * 2

    # Every stored row names the scripted model, and every cost is 0.
    organization_ids = [tenant.organization_id for tenant in tenants.values()]
    async with owner_engine.connect() as connection:
        models = {
            (table, column): set(
                (
                    await connection.execute(
                        text(
                            f"SELECT DISTINCT {column} FROM {table} "
                            "WHERE organization_id = ANY(CAST(:ids AS uuid[]))"
                        ),
                        {"ids": organization_ids},
                    )
                ).scalars()
            )
            for table, column in (
                ("extractions", "model"),
                ("document_pages", "model"),
                ("ask_runs", "model"),
                ("ask_runs", "judge_model"),
            )
        }
        spend = (
            await connection.execute(
                text(
                    "SELECT (SELECT coalesce(sum(cost_usd), 0) FROM extractions "
                    "WHERE organization_id = ANY(CAST(:ids AS uuid[]))) "
                    "+ (SELECT coalesce(sum(cost_usd), 0) FROM document_pages "
                    "WHERE organization_id = ANY(CAST(:ids AS uuid[]))) "
                    "+ (SELECT coalesce(sum(cost_usd + coalesce(judge_cost_usd, 0)), 0) "
                    "FROM ask_runs WHERE organization_id = ANY(CAST(:ids AS uuid[])))"
                ),
                {"ids": organization_ids},
            )
        ).scalar_one()
    assert models == {key: {SCRIPTED_MODEL} for key in models}
    assert spend == 0


async def test_without_a_key_the_scripted_ask_dependency_still_answers(
    scripted_environment, no_real_client
) -> None:
    """The unscripted API answers 503 with no key (tests/test_no_spend.py); the
    scripted one has a client without one."""
    client = get_anthropic_client(get_settings())

    assert client is not None
    assert type(client).__name__ == "ScriptedModelClient"
