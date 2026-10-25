"""Tenant B cannot reach tenant A's data: the whole product, asked from outside.

This module is the CI gate named "Tenant isolation". The layers each have
their own tests (tests/test_tenancy.py for the application filter,
tests/test_rls.py for the policies, tests/test_route_auth.py for sign-in); this
file is the end-to-end claim above them. Two organizations hold the same kinds
of data, B signs in (by session cookie, or with an API key) and asks for A's
ids by every route, the agent, MCP, and the worker, and nothing of A's comes
back or changes.

Every test that acts as B runs twice, once per kind of credential. A control
for each claim shows the call works for its own tenant, so a 404 is the
isolation and not a broken id.

The HTTP, agent, and MCP tests run on the suite's rolled-back transaction as
the non-owner runtime role, so both layers are on: the ORM filter and
row-level security. Each layer alone is proved in the two modules above.
Only the worker test commits (its claim opens sessions of its own); it removes
every row it made in a `finally`.
"""

import copy
import json
import re
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
import pytest_asyncio
from fastapi import Depends
from fastapi.routing import iter_route_contexts
from httpx import AsyncClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from test_agent import (
    EMBEDDER,
    _client,
    _message,
    _sse_events,
    _text,
    _tool_use,
    _usage,
)
from test_extraction import REALISTIC_TOOL_INPUT, TINY_PNG
from test_mcp import BASE_URL, _bearer, _http

from app import extraction as extraction_module
from app.agent.tools import ToolContext, execute_tool
from app.auth import api_keys, sessions
from app.auth.principal import Principal, current_principal
from app.config import Settings, get_settings
from app.db import async_session_maker, get_session, get_tenant_session_factory
from app.evals import online as online_module
from app.extraction_fields import SCHEMA_V1
from app.main import app
from app.models import (
    DEFAULT_ORGANIZATION_ID,
    JOB_KIND_EXTRACT,
    JOB_KIND_JUDGE,
    MEMBERSHIP_ROLE_OWNER,
    ApiKey,
    AskRun,
    Document,
    DocumentChunk,
    DocumentPage,
    ExtractedField,
    Extraction,
    Job,
    Membership,
    Organization,
    Reconciliation,
    User,
)
from app.retrieval import index_job as index_job_module
from app.retrieval.embeddings import get_embedder
from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages
from app.routers.ask import get_anthropic_client
from app.tenancy import bind_tenant, tenant_session
from app.worker import dispatch_job, run_once

CHUNKING = ChunkingConfig(max_chars=70, overlap_chars=10, context_headers=False)
COOKIE = "docpilot_session"
ORIGIN = {"Origin": "http://localhost:3000"}
# The agent's settings for /ask: the key only has to be present, because the
# model client is the scripted fake.
ASK_SETTINGS = Settings(
    agent_model="claude-opus-5-5",
    agent_effort="medium",
    anthropic_api_key="test",
    daily_budget_usd=0,
    ask_rate_limit_per_minute=0,
)


# --- two organizations' data ---------------------------------------------------


@dataclass(frozen=True)
class Tenant:
    """One organization and the ids of everything it owns."""

    organization_id: uuid.UUID
    user_id: uuid.UUID
    document_id: uuid.UUID
    extraction_id: uuid.UUID
    pending_field_ids: tuple[uuid.UUID, ...]
    reconciliation_id: uuid.UUID
    page_id: uuid.UUID
    chunk_ids: tuple[uuid.UUID, ...]
    ask_run_id: uuid.UUID
    chat_run_id: uuid.UUID
    conversation_id: uuid.UUID
    api_key_id: uuid.UUID
    api_token: str
    session_token: str
    file_path: Path
    file_bytes: bytes
    word: str
    vendor: str
    filename: str
    total: Decimal
    currency: str
    extraction_cost: float
    run_cost: float
    latency_ms: int

    @property
    def run_ids(self) -> tuple[uuid.UUID, uuid.UUID]:
        return self.ask_run_id, self.chat_run_id

    @property
    def markers(self) -> list[str]:
        """Strings that appear in a response only if this tenant's data does."""
        ids = (
            self.document_id,
            self.extraction_id,
            self.reconciliation_id,
            self.page_id,
            self.conversation_id,
            *self.pending_field_ids,
            *self.chunk_ids,
            *self.run_ids,
        )
        return [*map(str, ids), self.word, self.vendor, self.filename, f"{self.total:.2f}"]


def _leaks(content: object, tenant: Tenant) -> list[str]:
    """The markers of `tenant` found anywhere in a response, a tool result, or
    any other JSON-shaped value (empty when nothing of it is there)."""
    rendered = content if isinstance(content, str) else json.dumps(content, default=str)
    return [marker for marker in tenant.markers if marker in rendered]


async def _add_extracted_document(
    db: AsyncSession,
    *,
    filename: str,
    vendor: str,
    total: Decimal,
    currency: str,
    cost: float = 0.01,
    latency_ms: int = 100,
    storage_path: str = "/tmp/isolation",
    pending: tuple[str, ...] = (),
) -> tuple[Document, Extraction, list[ExtractedField]]:
    """An extracted document of the organization the session is bound to, with
    the fields the agent's tools read and a pending low-confidence field for
    each name in `pending`."""
    document = Document(
        filename=filename, mime_type="image/png", storage_path=storage_path, status="extracted"
    )
    db.add(document)
    await db.flush()
    extraction = Extraction(
        document_id=document.id,
        prompt_version=SCHEMA_V1.prompt_version,
        model="isolation-test",
        raw_response={},
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
        latency_ms=latency_ms,
    )
    db.add(extraction)
    await db.flush()
    # The total is stored as a JSON number, as the model's answer is.
    leaves = {"vendor": vendor, "total": float(total), "currency": currency}
    fields = [
        ExtractedField(
            extraction_id=extraction.id,
            field_name=name,
            value={"value": value, "confidence": 0.99},
            confidence=0.99,
        )
        for name, value in leaves.items()
    ]
    pending_fields = [
        ExtractedField(
            extraction_id=extraction.id,
            field_name=name,
            value={"value": 1.0, "confidence": 0.4},
            confidence=0.4,
            needs_review=True,
            review_reason="confidence",
        )
        for name in pending
    ]
    db.add_all([*fields, *pending_fields])
    await db.flush()
    return document, extraction, pending_fields


async def _build_tenant(
    db: AsyncSession,
    root: Path,
    tag: str,
    *,
    word: str,
    vendor: str,
    total: Decimal,
    currency: str,
    pending: tuple[str, ...],
    extraction_cost: float,
    run_cost: float,
    latency_ms: int,
    feedback: str | None,
) -> Tenant:
    """A new organization with an owner and one of everything: a document with
    a file, an extraction with a pending field and a reconciliation, a page and
    chunks indexed with the hashing embedder around `word`, an /ask run, a
    chat conversation, an API key, and a signed-in session."""
    unique = uuid.uuid4().hex[:8]
    organization = Organization(name=f"Tenant {tag}", slug=f"isolation-{tag}-{unique}")
    user = User(email=f"{tag}-{unique}@isolation.example", password_hash="!")
    db.add_all([organization, user])
    await db.flush()
    db.add(Membership(organization_id=organization.id, user_id=user.id, role=MEMBERSHIP_ROLE_OWNER))
    await db.flush()
    await bind_tenant(db, organization.id)

    file_bytes = TINY_PNG + f"-{tag}".encode()
    file_path = root / f"{tag}-receipt.png"
    file_path.write_bytes(file_bytes)
    filename = f"{tag}-{word}-receipt.png"
    document, extraction, pending_fields = await _add_extracted_document(
        db,
        filename=filename,
        vendor=vendor,
        total=total,
        currency=currency,
        cost=extraction_cost,
        latency_ms=latency_ms,
        storage_path=str(file_path),
        pending=pending,
    )
    reconciliation = Reconciliation(
        extraction_id=extraction.id, rules_version="rules_v2", status="pass", report={}
    )
    db.add(reconciliation)
    page_text = (
        f"{vendor}\nOrder of {word} for the office\nThank you for shopping with us\n"
        f"Total {total:.2f} {currency}"
    )
    await index_document_pages(
        db, document.id, [PageText(1, page_text, "gold")], EMBEDDER, CHUNKING
    )
    page_id = (await db.execute(select(DocumentPage.id).where(DocumentPage.document_id == document.id))).scalar_one()
    chunk_ids = tuple(
        (await db.execute(select(DocumentChunk.id).where(DocumentChunk.document_id == document.id))).scalars()
    )
    assert len(chunk_ids) > 1

    conversation_id = uuid.uuid4()

    def run(question: str, **kwargs) -> AskRun:
        return AskRun(
            question=question,
            status="answered",
            answer=f"{word} cost {total:.2f}",
            citations=[],
            tool_calls=[],
            evidence=f"evidence about {word}",
            steps=1,
            model="isolation-test",
            prompt_version="agent_v1",
            cost_usd=run_cost,
            **kwargs,
        )

    ask_run = run(f"What did the {word} order cost?", feedback=feedback)
    chat_run = run(f"And the {word} tax?", document_id=document.id, conversation_id=conversation_id)
    db.add_all([ask_run, chat_run])
    await db.flush()

    key, api_token = await api_keys.create_api_key(
        db, organization_id=organization.id, name=f"{tag} key", created_by_user_id=user.id
    )
    session_token = await sessions.create_session(
        db, user_id=user.id, organization_id=organization.id, ttl=timedelta(hours=1)
    )
    await bind_tenant(db, DEFAULT_ORGANIZATION_ID)
    return Tenant(
        organization_id=organization.id,
        user_id=user.id,
        document_id=document.id,
        extraction_id=extraction.id,
        pending_field_ids=tuple(field.id for field in pending_fields),
        reconciliation_id=reconciliation.id,
        page_id=page_id,
        chunk_ids=chunk_ids,
        ask_run_id=ask_run.id,
        chat_run_id=chat_run.id,
        conversation_id=conversation_id,
        api_key_id=key.id,
        api_token=api_token,
        session_token=session_token,
        file_path=file_path,
        file_bytes=file_bytes,
        word=word,
        vendor=vendor,
        filename=filename,
        total=total,
        currency=currency,
        extraction_cost=extraction_cost,
        run_cost=run_cost,
        latency_ms=latency_ms,
    )


@pytest_asyncio.fixture
async def tenants(db_session: AsyncSession, tmp_path: Path) -> tuple[Tenant, Tenant]:
    """(A, B): two organizations that differ in every number /stats and
    /monitoring/daily report, so one tenant's figures cannot pass for the
    other's. `two_tenants` in conftest.py is one bare row of every model for
    the layer tests; these have records, credentials and a file, which only
    an end-to-end test reads."""
    a = await _build_tenant(
        db_session,
        tmp_path,
        "a",
        word="quokkaville",
        vendor="Aardvark Supplies",
        total=Decimal("7777.77"),
        currency="GBP",
        pending=("tax", "subtotal"),
        extraction_cost=5.0,
        run_cost=0.75,
        latency_ms=900,
        feedback="up",
    )
    b = await _build_tenant(
        db_session,
        tmp_path,
        "b",
        word="narwhalton",
        vendor="Bluebird Traders",
        total=Decimal("11.11"),
        currency="EUR",
        pending=("tax",),
        extraction_cost=2.5,
        run_cost=0.25,
        latency_ms=111,
        feedback=None,
    )
    return a, b


# --- who is asking ---------------------------------------------------------------


@pytest.fixture(params=["cookie", "key"])
def credential(request: pytest.FixtureRequest) -> str:
    """How the caller proves who it is: a signed-in browser's session cookie
    (which must also send an Origin on a write), or an API key as a bearer."""
    return request.param


def _credentials(tenant: Tenant, credential: str) -> dict[str, str]:
    if credential == "key":
        return {"Authorization": f"Bearer {tenant.api_token}"}
    return {"Cookie": f"{COOKIE}={tenant.session_token}", **ORIGIN}


class Caller:
    """The test client acting as one tenant, with that tenant's credential on
    every request."""

    def __init__(self, client: AsyncClient, headers: Mapping[str, str]) -> None:
        self._client = client
        self._headers = dict(headers)

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return await self._client.request(method, url, headers=self._headers, **kwargs)

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs):
        return await self.request("POST", url, **kwargs)

    async def delete(self, url: str, **kwargs):
        return await self.request("DELETE", url, **kwargs)


@pytest_asyncio.fixture
async def isolated_client(anon_client: AsyncClient, db_session: AsyncSession) -> AsyncClient:
    """`anon_client` (the real sign-in, cookie, and key checks) with each
    request's session bound to the organization its credential names. The
    `client` fixtures hand every request one session bound to the default
    organization, which would test nothing here. The embedder is the hashing
    one; nothing reaches a model."""

    async def bound_session(principal: Principal = Depends(current_principal)) -> AsyncGenerator[AsyncSession]:
        await bind_tenant(db_session, principal.organization_id)
        yield db_session

    def bound_factory(principal: Principal = Depends(current_principal)):
        @asynccontextmanager
        async def factory() -> AsyncGenerator[AsyncSession]:
            await bind_tenant(db_session, principal.organization_id)
            yield db_session

        return factory

    app.dependency_overrides[get_session] = bound_session
    app.dependency_overrides[get_tenant_session_factory] = bound_factory
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    return anon_client


@pytest.fixture
def as_a(isolated_client: AsyncClient, tenants: tuple[Tenant, Tenant], credential: str) -> Caller:
    return Caller(isolated_client, _credentials(tenants[0], credential))


@pytest.fixture
def as_b(isolated_client: AsyncClient, tenants: tuple[Tenant, Tenant], credential: str) -> Caller:
    return Caller(isolated_client, _credentials(tenants[1], credential))


# --- what must not change -------------------------------------------------------

_SKIPPED_COLUMNS = {"embedding", "tsv"}
_CONTENT_MODELS = (
    Document, Job, Extraction, ExtractedField, Reconciliation, DocumentPage, DocumentChunk, AskRun,
)


async def _snapshot(db: AsyncSession, organization_id: uuid.UUID) -> dict[str, list[tuple]]:
    """Every stored column of every content row of the organization (and its
    API keys' state), as plain values: rows that were read through a mapped
    object would show an in-memory change the database never saw."""
    await bind_tenant(db, organization_id)
    snapshot: dict[str, list[tuple]] = {}
    for model in _CONTENT_MODELS:
        columns = [
            getattr(model, attribute.key)
            for attribute in sa_inspect(model).column_attrs
            if attribute.key not in _SKIPPED_COLUMNS
        ]
        rows = (await db.execute(select(*columns).order_by(model.id))).all()
        snapshot[model.__tablename__] = [tuple(row) for row in rows]
    keys = await db.execute(
        select(ApiKey.id, ApiKey.revoked_at, ApiKey.token_digest)
        .where(ApiKey.organization_id == organization_id)
        .order_by(ApiKey.id)
    )
    snapshot["api_keys"] = [tuple(row) for row in keys]
    return snapshot


@asynccontextmanager
async def _rows_unchanged(db: AsyncSession, *tenants: Tenant) -> AsyncGenerator[None]:
    """Fail unless every row of each tenant is exactly as it was on entry."""
    before = [await _snapshot(db, tenant.organization_id) for tenant in tenants]
    yield
    after = [await _snapshot(db, tenant.organization_id) for tenant in tenants]
    assert after == before


def _scripted_model_never_called() -> SimpleNamespace:
    """The scripted fake model with no reply scripted: any use of it fails the
    request, and `_model_calls` says whether it was used."""
    return _client([])


def _model_calls(fake: SimpleNamespace) -> int:
    return fake.beta.messages.create.await_count


# --- a control: each tenant reads its own data ---------------------------------


async def test_a_tenant_reads_its_own_data_by_every_route(
    tenants: tuple[Tenant, Tenant], as_a: Caller, as_b: Caller
) -> None:
    """If this fails, a 404 below proves nothing: the ids or the credentials are
    wrong, not the isolation."""
    for tenant, caller in zip(tenants, (as_a, as_b), strict=True):
        listed = (await caller.get("/documents")).json()
        assert [item["id"] for item in listed] == [str(tenant.document_id)]
        detail = await caller.get(f"/documents/{tenant.document_id}")
        assert detail.status_code == 200
        assert detail.json()["extraction"]["id"] == str(tenant.extraction_id)
        file = await caller.get(f"/documents/{tenant.document_id}/file")
        assert file.status_code == 200 and file.content == tenant.file_bytes
        queue = (await caller.get("/review/queue")).json()
        assert {item["field_id"] for item in queue} == set(map(str, tenant.pending_field_ids))
        found = (await caller.get("/search", params={"q": tenant.word, "mode": "lexical"})).json()
        assert {hit["document_id"] for hit in found["results"]} == {str(tenant.document_id)}
        runs = (await caller.get("/ask/runs")).json()
        assert {run["id"] for run in runs} == set(map(str, tenant.run_ids))
        conversations = await caller.get(f"/documents/{tenant.document_id}/chat/conversations")
        assert [c["conversation_id"] for c in conversations.json()] == [str(tenant.conversation_id)]
        turns = await caller.get(
            f"/documents/{tenant.document_id}/chat/conversations/{tenant.conversation_id}"
        )
        assert [turn["id"] for turn in turns.json()] == [str(tenant.chat_run_id)]


# --- documents -------------------------------------------------------------------


async def test_documents_of_another_tenant_are_not_listed_read_or_downloaded(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], as_b: Caller
) -> None:
    a, b = tenants
    async with _rows_unchanged(db_session, a, b):
        listed = await as_b.get("/documents", params={"limit": 200})
        detail = await as_b.get(f"/documents/{a.document_id}")
        file = await as_b.get(f"/documents/{a.document_id}/file")

    assert [item["id"] for item in listed.json()] == [str(b.document_id)]
    assert _leaks(listed.text, a) == []
    for refused in (detail, file):
        assert refused.status_code == 404
        assert _leaks(refused.text, a) == []
    assert a.file_bytes not in file.content
    assert a.file_path.read_bytes() == a.file_bytes


# --- the review queue ------------------------------------------------------------


async def test_the_review_queue_and_its_count_hold_only_this_tenants_fields(
    tenants: tuple[Tenant, Tenant], as_a: Caller, as_b: Caller
) -> None:
    a, b = tenants

    queue = await as_b.get("/review/queue", params={"limit": 200})
    count = await as_b.get("/review/queue/count")

    assert {item["field_id"] for item in queue.json()} == set(map(str, b.pending_field_ids))
    assert _leaks(queue.text, a) == []
    assert count.json() == {"pending": len(b.pending_field_ids)}
    # The other tenant's queue is longer, so a merged count could not pass.
    assert (await as_a.get("/review/queue/count")).json() == {"pending": len(a.pending_field_ids)}
    assert len(a.pending_field_ids) != len(b.pending_field_ids)


@pytest.mark.parametrize("action", ["approve", "correct"])
async def test_resolving_another_tenants_field_is_a_404_and_leaves_it_pending(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], as_b: Caller, action: str
) -> None:
    a, b = tenants
    body = {"action": action, **({"corrected_value": "1.00"} if action == "correct" else {})}

    async with _rows_unchanged(db_session, a, b):
        responses = [
            await as_b.post(f"/review/fields/{field_id}/resolve", json=body)
            for field_id in a.pending_field_ids
        ]

    for response in responses:
        assert response.status_code == 404
        assert _leaks(response.text, a) == []
    await bind_tenant(db_session, a.organization_id)
    pending = (
        await db_session.execute(
            select(ExtractedField.id).where(
                ExtractedField.reviewed_at.is_(None), ExtractedField.needs_review.is_(True)
            )
        )
    ).scalars()
    assert set(pending) == set(a.pending_field_ids)


# --- search ------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_search_never_returns_another_tenants_chunk(
    tenants: tuple[Tenant, Tenant], as_a: Caller, as_b: Caller, mode: str
) -> None:
    """k=50 is the API's largest, so the candidates cover both tenants' chunks
    and a ranking that filtered only afterwards would still return A's."""
    a, b = tenants

    # A's own word, asked by B: the query is exactly what A's chunks match.
    theirs = (await as_b.get("/search", params={"q": a.word, "k": 50, "mode": mode})).json()
    ours = (await as_b.get("/search", params={"q": b.word, "k": 50, "mode": mode})).json()

    for body in (theirs, ours):
        assert _leaks(body["results"], a) == []
        assert {hit["document_id"] for hit in body["results"]} <= {str(b.document_id)}
        assert {hit["chunk_id"] for hit in body["results"]} <= set(map(str, b.chunk_ids))
    assert ours["results"], "B finds its own word, or this test cannot tell"
    # The control: the same query as A does find A's chunks.
    found = (await as_a.get("/search", params={"q": a.word, "k": 50, "mode": mode})).json()
    assert {hit["document_id"] for hit in found["results"]} == {str(a.document_id)}


# --- /ask runs -----------------------------------------------------------------------


async def test_another_tenants_ask_runs_are_not_listed_read_or_rated(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], as_b: Caller
) -> None:
    a, b = tenants

    async with _rows_unchanged(db_session, a, b):
        listed = await as_b.get("/ask/runs", params={"limit": 100})
        reads = [await as_b.get(f"/ask/runs/{run_id}") for run_id in a.run_ids]
        rated = [
            await as_b.post(f"/ask/runs/{run_id}/feedback", json={"rating": "down", "note": "no"})
            for run_id in a.run_ids
        ]

    assert {run["id"] for run in listed.json()} == set(map(str, b.run_ids))
    assert _leaks(listed.text, a) == []
    for refused in (*reads, *rated):
        assert refused.status_code == 404
        assert _leaks(refused.text, a) == []
    await bind_tenant(db_session, a.organization_id)
    feedback = {
        run_id: rating
        for run_id, rating in (await db_session.execute(select(AskRun.id, AskRun.feedback))).all()
    }
    assert feedback == {a.ask_run_id: "up", a.chat_run_id: None}


# --- chat ------------------------------------------------------------------------------


async def test_chat_on_another_tenants_document_is_a_404_before_the_model_is_called(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], as_b: Caller
) -> None:
    a, b = tenants
    fake = _scripted_model_never_called()
    app.dependency_overrides[get_anthropic_client] = lambda: fake
    other = f"/documents/{a.document_id}"
    own = f"/documents/{b.document_id}"
    question = {"question": "What is in this document?"}
    followup = {**question, "conversation_id": str(a.conversation_id)}

    async with _rows_unchanged(db_session, a, b):
        responses = [
            await as_b.post(f"{other}/chat", json=question),
            await as_b.post(f"{other}/chat", json=followup),
            await as_b.post(f"{other}/chat/stream", json=question),
            await as_b.post(f"{other}/chat/stream", json=followup),
            await as_b.get(f"{other}/chat/conversations"),
            await as_b.get(f"{other}/chat/conversations/{a.conversation_id}"),
            # B's own document does not make A's conversation B's.
            await as_b.post(f"{own}/chat", json=followup),
            await as_b.post(f"{own}/chat/stream", json=followup),
            await as_b.get(f"{own}/chat/conversations/{a.conversation_id}"),
        ]

    for response in responses:
        assert response.status_code == 404
        assert _leaks(response.text, a) == []
    assert _model_calls(fake) == 0


# --- /stats and /monitoring/daily ---------------------------------------------------------


async def test_stats_are_this_tenants_numbers_only(
    tenants: tuple[Tenant, Tenant], as_a: Caller, as_b: Caller
) -> None:
    a, b = tenants

    stats = (await as_b.get("/stats")).json()
    theirs = (await as_a.get("/stats")).json()

    runs = len(b.run_ids)
    assert stats["documents_total"] == 1
    assert stats["documents_by_status"] == {
        "uploaded": 0, "processing": 0, "extracted": 1, "failed": 0, "refused": 0,
    }
    assert stats["documents_processed"] == stats["extractions_total"] == 1
    assert stats["total_cost_usd"] == pytest.approx(b.extraction_cost)
    assert stats["latency_p50_ms"] == stats["latency_p95_ms"] == b.latency_ms
    assert stats["review"] == {"pending": len(b.pending_field_ids), "approved": 0, "corrected": 0}
    assert stats["spend"]["agent_usd"] == pytest.approx(runs * b.run_cost)
    assert stats["spend"]["total_usd"] == pytest.approx(b.extraction_cost + runs * b.run_cost)
    assert stats["budget"]["spent_today_usd"] == pytest.approx(b.extraction_cost + runs * b.run_cost)
    assert stats["ask"]["runs"] == stats["ask"]["answered"] == runs
    assert stats["ask"]["feedback_up"] == stats["ask"]["feedback_down"] == 0
    # The control, and the figures that differ: A's are not B's.
    assert theirs["total_cost_usd"] == pytest.approx(a.extraction_cost)
    assert theirs["latency_p50_ms"] == a.latency_ms
    assert theirs["review"]["pending"] == len(a.pending_field_ids)
    assert theirs["ask"]["feedback_up"] == 1


async def test_the_daily_series_is_this_tenants_numbers_only(
    tenants: tuple[Tenant, Tenant], as_a: Caller, as_b: Caller
) -> None:
    a, b = tenants

    today = (await as_b.get("/monitoring/daily", params={"days": 1})).json()[-1]
    theirs = (await as_a.get("/monitoring/daily", params={"days": 1})).json()[-1]

    runs = len(b.run_ids)
    assert today["extractions"]["documents"] == 1
    assert today["extractions"]["latency_p50_ms"] == b.latency_ms
    assert today["answers"]["runs"] == today["answers"]["answered"] == runs
    assert today["answers"]["feedback_up"] == today["answers"]["feedback_down"] == 0
    assert today["spend"]["documents_usd"] == pytest.approx(b.extraction_cost)
    assert today["spend"]["answers_usd"] == pytest.approx(runs * b.run_cost)
    assert today["spend"]["total_usd"] == pytest.approx(b.extraction_cost + runs * b.run_cost)
    assert theirs["spend"]["documents_usd"] == pytest.approx(a.extraction_cost)
    assert theirs["answers"]["feedback_up"] == 1


# --- API keys ------------------------------------------------------------------------------


async def test_revoking_another_tenants_api_key_is_a_404_and_the_key_still_works(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], isolated_client: AsyncClient
) -> None:
    a, b = tenants
    as_b = Caller(isolated_client, _credentials(b, "cookie"))
    as_a_with_key = Caller(isolated_client, _credentials(a, "key"))

    async with _rows_unchanged(db_session, a, b):
        revoked = await as_b.delete(f"/api-keys/{a.api_key_id}")
        listed = await as_b.get("/api-keys")

    assert revoked.status_code == 404
    assert _leaks(revoked.text, a) == []
    assert [key["id"] for key in listed.json()["api_keys"]] == [str(b.api_key_id)]
    assert str(a.api_key_id) not in listed.text and a.api_token not in listed.text
    assert (await as_a_with_key.get("/documents")).status_code == 200


async def test_a_key_cannot_even_name_another_tenants_key(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], isolated_client: AsyncClient
) -> None:
    """A key never manages keys (403), whoever's they are, and says no more."""
    a, b = tenants
    as_b_key = Caller(isolated_client, _credentials(b, "key"))

    async with _rows_unchanged(db_session, a, b):
        response = await as_b_key.delete(f"/api-keys/{a.api_key_id}")

    assert response.status_code == 403
    assert _leaks(response.text, a) == []


# --- the agent --------------------------------------------------------------------------------


def _agent_script(a: Tenant, seen: list[dict]) -> SimpleNamespace:
    """A fake model that calls the three tools about A, asking as B, then
    answers. `seen` collects the tool results the model is handed."""

    def finish(**kwargs):
        seen.extend(copy.deepcopy(kwargs["messages"][-1]["content"]))
        return _message([_text("Nothing here is about that.")], "end_turn")

    calls = [
        _tool_use("query_extractions", {}, "toolu_query"),
        _tool_use("search_documents", {"query": a.word}, "toolu_search"),
        _tool_use("get_page", {"document_id": str(a.document_id), "page_number": 1}, "toolu_page"),
    ]
    return _client([_message(calls, "tool_use"), finish])


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
async def test_the_agent_asked_by_b_is_handed_none_of_a(
    db_session: AsyncSession,
    tenants: tuple[Tenant, Tenant],
    isolated_client: AsyncClient,
    credential: str,
    endpoint: str,
) -> None:
    a, b = tenants
    seen: list[dict] = []
    app.dependency_overrides[get_settings] = lambda: ASK_SETTINGS
    app.dependency_overrides[get_anthropic_client] = lambda: _agent_script(a, seen)
    as_b = Caller(isolated_client, _credentials(b, credential))

    async with _rows_unchanged(db_session, a):
        response = await as_b.post(endpoint, json={"question": f"Anything about {a.word}?"})

    assert response.status_code == 200
    if endpoint == "/ask":
        run = response.json()
    else:
        (answer,) = [event for event in _sse_events(response.text) if event["type"] == "answer"]
        run = answer["run"]
    assert run["status"] == "answered"
    # The tool results as the model saw them: B's own records, and nothing of A's.
    by_call = {result["tool_use_id"]: result for result in seen}
    assert list(by_call) == ["toolu_query", "toolu_search", "toolu_page"]
    assert _leaks(seen, a) == []
    assert str(b.document_id) in json.dumps(by_call["toolu_query"]["content"])
    assert by_call["toolu_page"].get("is_error") is True
    assert by_call["toolu_query"].get("is_error") is None and by_call["toolu_search"].get("is_error") is None
    # What the run stored has none of A's either. The inputs of its tool calls
    # are the asker's own words (A's ids, as the script supplied them), so the
    # trail is read by what the calls returned.
    returned = [call["result_summary"] for call in run["tool_calls"]]
    assert _leaks([run["answer"], run["citations"], returned], a) == []
    await bind_tenant(db_session, b.organization_id)
    stored = await db_session.get(AskRun, uuid.UUID(run["id"]))
    assert stored is not None and stored.organization_id == b.organization_id
    assert _leaks(stored.evidence, a) == []
    await bind_tenant(db_session, a.organization_id)
    assert await db_session.get(AskRun, uuid.UUID(run["id"])) is None
    # The run is B's to read and A's not to.
    assert (await as_b.get(f"/ask/runs/{run['id']}")).status_code == 200
    as_a = Caller(isolated_client, _credentials(a, credential))
    assert (await as_a.get(f"/ask/runs/{run['id']}")).status_code == 404


async def test_the_same_agent_asked_by_a_does_see_a(
    tenants: tuple[Tenant, Tenant], isolated_client: AsyncClient, credential: str
) -> None:
    """The control for the test above: the three calls find A's data for A."""
    a, b = tenants
    seen: list[dict] = []
    app.dependency_overrides[get_settings] = lambda: ASK_SETTINGS
    app.dependency_overrides[get_anthropic_client] = lambda: _agent_script(a, seen)
    as_a = Caller(isolated_client, _credentials(a, credential))

    response = await as_a.post("/ask", json={"question": "Anything?"})

    assert response.status_code == 200
    by_call = {result["tool_use_id"]: result for result in seen}
    assert str(a.document_id) in json.dumps(by_call["toolu_query"]["content"])
    assert a.word in json.dumps(by_call["toolu_search"]["content"])
    assert a.word in json.dumps(by_call["toolu_page"]["content"])
    assert by_call["toolu_page"].get("is_error") is None
    assert _leaks(seen, b) == []


# --- MCP ----------------------------------------------------------------------------------------


@pytest.fixture(params=["legacy", "auto"])
def mode(request: pytest.FixtureRequest) -> str:
    """The two protocol eras the SDK serves: the handshake one and the stateless
    2026-07-28 one."""
    return request.param


def _mcp_rendered(result: types.CallToolResult) -> str:
    return json.dumps(
        [block.model_dump() for block in result.content] + [result.structured_content], default=str
    )


async def test_mcp_with_bs_key_returns_none_of_a_in_either_protocol_era(
    db_session: AsyncSession, tenants: tuple[Tenant, Tenant], mode: str
) -> None:
    a, b = tenants

    async def call(endpoint, token: str, name: str, arguments: dict) -> types.CallToolResult:
        http_client = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=endpoint), base_url=BASE_URL, headers=_bearer(token)
        )
        async with Client(streamable_http_client(f"{BASE_URL}/mcp", http_client=http_client), mode=mode) as client:
            return await client.call_tool(name, arguments)

    page = {"document_id": str(a.document_id), "page_number": 1}
    async with _http(db_session, b.api_token) as (endpoint, _):
        records = await call(endpoint, b.api_token, "query_extractions", {})
        found = await call(endpoint, b.api_token, "search_documents", {"query": a.word})
        got = await call(endpoint, b.api_token, "get_page", page)
        # The controls: A's key reads A, and B's own page is readable by B.
        own_records = await call(endpoint, a.api_token, "query_extractions", {})
        own_page = await call(
            endpoint, b.api_token, "get_page", {"document_id": str(b.document_id), "page_number": 1}
        )

    for result in (records, found, got):
        assert _leaks(_mcp_rendered(result), a) == []
    assert records.is_error is False and str(b.document_id) in _mcp_rendered(records)
    assert found.is_error is False
    assert got.is_error is True
    assert str(a.document_id) in _mcp_rendered(own_records)
    assert own_page.is_error is False and b.word in _mcp_rendered(own_page)


# --- every route ---------------------------------------------------------------------------------

PATH_PARAMETER = re.compile(r"\{([^}]+)\}")
# The id each path parameter takes: tenant A's, asked for by B.
ID_PARAMETERS: dict[str, Callable[[Tenant], uuid.UUID]] = {
    "document_id": lambda t: t.document_id,
    "conversation_id": lambda t: t.conversation_id,
    "run_id": lambda t: t.ask_run_id,
    "field_id": lambda t: t.pending_field_ids[0],
    "key_id": lambda t: t.api_key_id,
}
# A request body each write route accepts, so a 404 is the id and not a 422.
BODIES: dict[tuple[str, str], dict] = {
    ("POST", "/review/fields/{field_id}/resolve"): {"action": "approve"},
    ("POST", "/ask/runs/{run_id}/feedback"): {"rating": "down"},
    ("POST", "/documents/{document_id}/chat"): {"question": "What is this?"},
    ("POST", "/documents/{document_id}/chat/stream"): {"question": "What is this?"},
}


def _routes_with_an_id() -> list[tuple[str, str]]:
    """(method, path template) of every route with a path parameter."""
    return sorted(
        {
            (method, route.path)
            for route in iter_route_contexts(app.router.routes)
            if route.path is not None and "{" in route.path
            for method in (route.methods or {"GET"}) - {"HEAD", "OPTIONS"}
        }
    )


def test_the_walk_knows_every_id_route_and_parameter() -> None:
    """A route added with a parameter or a body this module does not know must
    fail here, not slip past the walk below."""
    routes = _routes_with_an_id()
    assert len(routes) >= 10
    for method, template in routes:
        assert set(PATH_PARAMETER.findall(template)) <= set(ID_PARAMETERS), (method, template)
        if method not in {"GET", "DELETE"}:
            assert (method, template) in BODIES, f"{method} {template} needs a body in BODIES"
    assert set(BODIES) <= set(routes)


@pytest.mark.parametrize(("method", "template"), _routes_with_an_id(), ids=lambda value: str(value)[:48])
async def test_every_route_with_an_id_is_a_404_for_another_tenants_id_and_changes_nothing(
    db_session: AsyncSession,
    tenants: tuple[Tenant, Tenant],
    isolated_client: AsyncClient,
    credential: str,
    method: str,
    template: str,
) -> None:
    a, b = tenants
    fake = _scripted_model_never_called()
    app.dependency_overrides[get_anthropic_client] = lambda: fake
    path = PATH_PARAMETER.sub(lambda found: str(ID_PARAMETERS[found.group(1)](a)), template)
    as_b = Caller(isolated_client, _credentials(b, credential))
    # An API key may not manage keys at all, so that route answers 403 to it.
    expected = 403 if credential == "key" and template.startswith("/api-keys") else 404

    async with _rows_unchanged(db_session, a, b):
        response = await as_b.request(method, path, json=BODIES.get((method, template)))

    assert response.status_code == expected, f"{method} {path}: {response.text}"
    assert _leaks(response.text, a) == []
    assert _model_calls(fake) == 0


# --- the worker ----------------------------------------------------------------------------------

WORKER_TENANTS = {
    "a": ("quokkaville", "Aardvark Supplies"),
    "b": ("narwhalton", "Bluebird Traders"),
}
WORKER_TABLES = (
    "jobs", "extracted_fields", "reconciliations", "document_chunks", "document_pages",
    "extractions", "ask_runs", "documents",
)


@dataclass(frozen=True)
class WorkerTenant:
    organization_id: uuid.UUID
    document_id: uuid.UUID
    word: str
    vendor: str


async def _commit_worker_tenant(root: Path, tag: str) -> WorkerTenant:
    """A committed organization with an uploaded document (a real file), its
    extract job, and a stored answer with its judge job. The judge job also
    names the document, which does not change what it does: it is how the claim
    below is scoped to these rows (worker.pending_job_stmt), as in
    tests/test_worker.py."""
    word, vendor = WORKER_TENANTS[tag]
    async with async_session_maker() as db:
        organization = Organization(name=f"Worker {tag}", slug=f"worker-{tag}-{uuid.uuid4().hex[:8]}")
        db.add(organization)
        await db.commit()
        organization_id = organization.id
    stored = root / f"{tag}-worker.png"
    stored.write_bytes(TINY_PNG)
    async with tenant_session(async_session_maker, organization_id) as db:
        document = Document(
            filename=f"{tag}.png", mime_type="image/png", storage_path=str(stored), status="uploaded"
        )
        run = AskRun(
            question=f"What about {word}?",
            status="answered",
            answer=f"{word} is here.",
            citations=[],
            tool_calls=[],
            evidence=f"{vendor} sold some {word}.",
            steps=1,
            model="isolation-test",
            prompt_version="agent_v1",
            judge_sampled=True,
        )
        db.add_all([document, run])
        await db.flush()
        db.add_all(
            [
                Job(document_id=document.id),
                Job(
                    kind=JOB_KIND_JUDGE, document_id=document.id, ask_run_id=run.id
                ),
            ]
        )
        await db.commit()
        document_id = document.id
    return WorkerTenant(organization_id, document_id, word, vendor)


async def _remove_worker_rows(owner_engine: AsyncEngine, organization_ids: list[uuid.UUID]) -> None:
    """Delete what the worker test committed, as the owner (row-level security
    never binds it), children first."""
    async with owner_engine.begin() as connection:
        for table in WORKER_TABLES:
            await connection.execute(
                text(f"DELETE FROM {table} WHERE organization_id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": organization_ids},
            )
        await connection.execute(
            text("DELETE FROM organizations WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": organization_ids},
        )


def _fake_extraction_client(vendor: str) -> SimpleNamespace:
    tool_input = {**REALISTIC_TOOL_INPUT, "vendor": {"value": vendor, "confidence": 0.95}}
    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", name="record_extraction", id="toolu_x", input=tool_input)],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0, cache_read_input_tokens=0
        ),
    )
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


def _fake_grader() -> SimpleNamespace:
    verdict = {"unsupported_claims": [], "grounded": True, "answers_question": True, "explanation": "ok"}
    response = SimpleNamespace(
        id="judge",
        model="claude-sonnet-5-5",
        stop_reason="end_turn",
        usage=_usage(),
        content=[SimpleNamespace(type="text", text=json.dumps(verdict))],
    )
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response))))


async def test_the_worker_writes_each_organizations_jobs_only_into_that_organization(
    owner_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both organizations' extract, index, and judge jobs run through the real
    claim and the real handlers, interleaved, with the model, the transcription,
    and the embedder faked. Each handler runs in a session of its own
    organization, and every row it writes lands there."""
    vendors = {vendor: word for word, vendor in WORKER_TENANTS.values()}
    clients = {vendor: _fake_extraction_client(vendor) for vendor in vendors}
    current: dict[str, str] = {}

    def build_client(settings):
        return clients[current["vendor"]]

    async def transcribe(storage_path: str, mime_type: str) -> list[PageText]:
        vendor = current["vendor"]
        return [PageText(1, f"{vendor}\nOrder of {vendors[vendor]}\nTotal 10.00 USD", "transcription")]

    monkeypatch.setattr(extraction_module, "_build_client", build_client)
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-sonnet-5", review_threshold=0.8),
    )
    monkeypatch.setattr(index_job_module, "transcribe_document", transcribe)
    monkeypatch.setattr(index_job_module, "get_embedder", lambda: EMBEDDER)
    monkeypatch.setattr(online_module, "_build_client", lambda settings: _fake_grader())

    seen: list[tuple[str, uuid.UUID, str, list[uuid.UUID]]] = []
    tenant_by_org: dict[uuid.UUID, WorkerTenant] = {}

    async def handler(session: AsyncSession, job: Job) -> None:
        """The real dispatch, after noting which organization the session is
        set to and which documents it can see."""
        setting = (
            await session.execute(text("SELECT current_setting('docpilot.organization_id', true)"))
        ).scalar_one()
        visible = list((await session.execute(select(Document.id))).scalars())
        seen.append((job.kind, job.organization_id, setting, visible))
        current["vendor"] = tenant_by_org[job.organization_id].vendor
        await dispatch_job(session, job)

    organization_ids: list[uuid.UUID] = []
    try:
        built = []
        for tag in WORKER_TENANTS:
            built.append(await _commit_worker_tenant(tmp_path, tag))
            organization_ids.append(built[-1].organization_id)
        tenant_by_org.update({tenant.organization_id: tenant for tenant in built})

        # Extract, then its index job, and the judge job: three claims each.
        for _ in range(3):
            for tenant in built:
                assert await run_once(handler, document_id=tenant.document_id) is True
        for tenant in built:
            assert await run_once(handler, document_id=tenant.document_id) is False

        assert sorted(kind for kind, *_ in seen) == sorted([JOB_KIND_EXTRACT, "index", JOB_KIND_JUDGE] * 2)
        for kind, organization_id, setting, visible in seen:
            assert setting == str(organization_id), kind
            assert visible == [tenant_by_org[organization_id].document_id], kind
        await _assert_worker_rows_are_in_their_own_organization(owner_engine, built)
    finally:
        await _remove_worker_rows(owner_engine, organization_ids)


async def _assert_worker_rows_are_in_their_own_organization(
    owner_engine: AsyncEngine, built: list[WorkerTenant]
) -> None:
    """Read as the owner, which sees every organization."""
    async with owner_engine.connect() as connection:

        async def rows(sql: str, **parameters) -> list[tuple]:
            return [tuple(row) for row in await connection.execute(text(sql), parameters)]

        for tenant in built:
            org = tenant.organization_id
            assert await rows(
                "SELECT kind, state, organization_id FROM jobs WHERE document_id = :d ORDER BY kind",
                d=tenant.document_id,
            ) == [
                ("extract", "done", org),
                ("index", "done", org),
                ("judge", "done", org),
            ]
            # What the extract job wrote: one extraction with fields and a
            # reconciliation, all in this organization.
            assert await rows(
                "SELECT DISTINCT e.organization_id FROM extractions e WHERE e.document_id = :d",
                d=tenant.document_id,
            ) == [(org,)]
            vendors = await rows(
                "SELECT DISTINCT organization_id FROM extracted_fields "
                "WHERE field_name = 'vendor' AND value->>'value' = :vendor",
                vendor=tenant.vendor,
            )
            assert vendors == [(org,)]
            assert await rows(
                "SELECT count(*) > 0 FROM reconciliations WHERE organization_id = :o", o=org
            ) == [(True,)]
            # What the index job wrote: this organization's page and chunks.
            for table in ("document_pages", "document_chunks"):
                assert await rows(
                    f"SELECT DISTINCT organization_id FROM {table} WHERE text LIKE :needle",
                    needle=f"%{tenant.word}%",
                ) == [(org,)]
            # What the judge job wrote: a verdict on this organization's answer.
            verdicts = await rows(
                "SELECT organization_id, judge_grounded, judge_error IS NULL "
                "FROM ask_runs WHERE judged_at IS NOT NULL AND question LIKE :needle",
                needle=f"%{tenant.word}%",
            )
            assert verdicts == [(org, True, True)]


# --- the property ----------------------------------------------------------------------------------

CURRENCIES = ["USD", "EUR", "GBP", "CAD"]
SUM_LINE = re.compile(r"Sum of totals: (.+?)\.(?: |$)")


def _summed_totals(content: list[dict]) -> dict[str, Decimal]:
    """The per-currency sums in a query_extractions result's match summary."""
    summary = next(
        line["text"]
        for block in content
        for line in (block["content"] if block["type"] == "search_result" else [block])
        if "matching document(s)" in line["text"]
    )
    found = SUM_LINE.search(summary)
    if found is None:
        return {}
    amounts = (part.split(" ") for part in found.group(1).split("; "))
    return {currency: Decimal(amount) for amount, currency in amounts}


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.tuples(st.integers(1, 99_999_999), st.sampled_from(CURRENCIES), st.booleans()),
        min_size=2,
        max_size=6,
    )
)
async def test_query_extractions_returns_exactly_the_organizations_documents_and_sums(
    db_session: AsyncSession, documents: list[tuple[int, str, bool]]
) -> None:
    """However 2 to 6 documents with random totals and currencies are split
    between two organizations, `query_extractions({})` under each returns
    exactly that organization's documents, and its per-currency sums are the
    exact Decimal sums of that organization's totals."""
    savepoint = await db_session.begin_nested()
    try:
        organizations = {}
        for key in (True, False):
            organization = Organization(name=f"Property {key}", slug=f"property-{uuid.uuid4().hex[:12]}")
            db_session.add(organization)
            await db_session.flush()
            organizations[key] = organization.id
        expected_ids: dict[bool, set[uuid.UUID]] = {True: set(), False: set()}
        expected_sums: dict[bool, dict[str, Decimal]] = {True: {}, False: {}}
        for index, (cents, currency, to_a) in enumerate(documents):
            await bind_tenant(db_session, organizations[to_a])
            total = Decimal(cents).scaleb(-2)
            document, _, _ = await _add_extracted_document(
                db_session, filename=f"doc-{index}.png", vendor=f"Vendor {index}", total=total, currency=currency
            )
            expected_ids[to_a].add(document.id)
            expected_sums[to_a][currency] = expected_sums[to_a].get(currency, Decimal(0)) + total

        for key, organization_id in organizations.items():
            await bind_tenant(db_session, organization_id)
            ctx = ToolContext(session=db_session, embedder=EMBEDDER)
            outcome = await execute_tool(ctx, "query_extractions", {}, "toolu_property")

            assert outcome.is_error is False
            assert {source.document_id for source in ctx.sources.values()} == expected_ids[key]
            assert _summed_totals(outcome.content) == expected_sums[key]
            others = expected_ids[not key]
            assert not any(str(document_id) in json.dumps(outcome.content) for document_id in others)
    finally:
        await savepoint.rollback()
        await bind_tenant(db_session, DEFAULT_ORGANIZATION_ID)
        db_session.expunge_all()
