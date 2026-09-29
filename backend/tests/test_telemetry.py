"""Tracing: GenAI span attributes on model calls, and one trace per
document across the job queue.

A process can install its global TracerProvider only once, so this module
installs an SDK provider with an in-memory exporter the first time it
runs (the app itself never installs one unless OTEL_EXPORTER_OTLP_ENDPOINT
is set) and clears the captured spans before each test.
"""

import base64
import uuid
from collections.abc import AsyncGenerator, Callable
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import extraction as extraction_module
from app import telemetry
from app import transcription as transcription_module
from app.config import Settings
from app.db import async_session_maker
from app.extraction import PROMPT_VERSION, extract_document, process_document_job
from app.models import Document, Job
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages
from app.retrieval.search import search
from app.transcription import transcribe_document
from app.worker import run_once

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(autouse=True)
def spans() -> InMemorySpanExporter:
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    if not getattr(provider, "_docpilot_test_exporter", False):
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        provider._docpilot_test_exporter = True  # type: ignore[attr-defined]
    _EXPORTER.clear()
    return _EXPORTER


def _named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int, cache_read: int | None = None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read


class _ToolUse:
    type = "tool_use"
    name = "record_extraction"
    id = "toolu_1"

    def __init__(self, tool_input: dict) -> None:
        self.input = tool_input


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, content, stop_reason, usage, model="claude-sonnet-5") -> None:
        self.id = "msg_test123"
        self.model = model
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


def _fake_client(response) -> object:
    client = type("C", (), {})()
    client.messages = type("M", (), {})()
    client.messages.create = AsyncMock(return_value=response)
    return client


TOOL_INPUT = {
    field: {"value": value, "confidence": 0.95}
    for field, value in {
        "vendor": "Acme",
        "document_date": "2026-01-01",
        "subtotal": 10.0,
        "tax": 0.0,
        "total": 10.0,
        "currency": "USD",
    }.items()
} | {"line_items": {"value": [], "confidence": 0.9}}


# --- plumbing ---------------------------------------------------------------


def test_tracing_is_off_without_an_otlp_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    installed = []
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", installed.append)

    assert telemetry.configure_tracing("svc") is False
    assert installed == []


def test_configure_tracing_installs_an_otlp_provider_when_endpoint_set(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "doc-pilot-test")
    installed = []
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", installed.append)

    assert telemetry.configure_tracing("svc") is True
    (provider,) = installed
    assert provider.resource.attributes["service.name"] == "doc-pilot-test"
    provider.shutdown()


def test_traceparent_round_trips_through_a_job_row() -> None:
    assert telemetry.current_traceparent() is None

    with telemetry.tracer().start_as_current_span("enqueue") as outer:
        traceparent = telemetry.current_traceparent()
    assert traceparent is not None

    ctx = telemetry.context_from_traceparent(traceparent)
    parent = trace.get_current_span(ctx).get_span_context()
    assert parent.trace_id == outer.get_span_context().trace_id
    assert parent.span_id == outer.get_span_context().span_id


def test_missing_traceparent_starts_a_new_root_even_inside_another_span(spans) -> None:
    tracer = telemetry.tracer()
    with (
        tracer.start_as_current_span("unrelated"),
        tracer.start_as_current_span("job", context=telemetry.context_from_traceparent(None)),
    ):
        pass

    (job,) = _named(spans, "job")
    assert job.parent is None


# --- model call spans -------------------------------------------------------


async def test_extraction_call_is_a_genai_chat_span(tmp_path, monkeypatch, spans) -> None:
    monkeypatch.setattr(
        extraction_module, "get_settings", lambda: Settings(extraction_model="claude-sonnet-5")
    )
    response = _Message([_ToolUse(TOOL_INPUT)], "tool_use", _Usage(1500, 300, cache_read=200))
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: _fake_client(response))
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    await extract_document(path, "image/png")

    (span,) = _named(spans, "chat claude-sonnet-5")
    attrs = span.attributes
    assert span.kind == SpanKind.CLIENT
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-5"
    assert attrs["gen_ai.request.max_tokens"] == 4096
    assert attrs["gen_ai.response.id"] == "msg_test123"
    assert attrs["gen_ai.response.finish_reasons"] == ("tool_use",)
    assert attrs["gen_ai.usage.input_tokens"] == 1500
    assert attrs["gen_ai.usage.output_tokens"] == 300
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 200
    assert attrs["docpilot.prompt_version"] == PROMPT_VERSION
    # claude-sonnet-5 at $2/$10 per MTok
    assert attrs["docpilot.cost_usd"] == pytest.approx(0.006)


async def test_failed_model_call_marks_its_span_as_error(tmp_path, monkeypatch, spans) -> None:
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.InternalServerError(
        "boom", response=httpx.Response(500, request=request), body=None
    )
    client = _fake_client(None)
    client.messages.create = AsyncMock(side_effect=error)
    monkeypatch.setattr(
        extraction_module, "get_settings", lambda: Settings(extraction_model="claude-sonnet-5")
    )
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: client)
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    with pytest.raises(extraction_module.ExtractionError):
        await extract_document(path, "image/png")

    (span,) = _named(spans, "chat claude-sonnet-5")
    assert span.status.status_code == StatusCode.ERROR
    assert "gen_ai.usage.input_tokens" not in span.attributes


async def test_transcription_span_records_page_number(tmp_path, monkeypatch, spans) -> None:
    monkeypatch.setattr(
        transcription_module,
        "get_settings",
        lambda: Settings(transcription_model="claude-haiku-4-5"),
    )
    response = _Message([_Text("hello")], "end_turn", _Usage(800, 20), model="claude-haiku-4-5")
    monkeypatch.setattr(transcription_module, "_build_client", lambda s: _fake_client(response))
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    await transcribe_document(path, "image/png")

    (span,) = _named(spans, "chat claude-haiku-4-5")
    assert span.attributes["docpilot.page_number"] == 1
    assert span.attributes["docpilot.prompt_version"] == "transcribe_v1"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("end_turn",)


# --- retrieval spans --------------------------------------------------------


async def test_indexing_and_search_emit_embeddings_and_retrieval_spans(
    db_session: AsyncSession, spans
) -> None:
    document = Document(filename="r.png", mime_type="image/png", storage_path="/tmp/r", status="extracted")
    db_session.add(document)
    await db_session.flush()
    embedder = HashingEmbedder()
    await index_document_pages(
        db_session,
        document.id,
        [PageText(1, "Cobblestone Bakery\nHerbal Tea Sampler  1  $8.05", "gold")],
        embedder,
        ChunkingConfig(0, 0, True),
    )
    await search(db_session, "tea", embedder=embedder, k=3, mode="hybrid", document_ids=[document.id])

    (embeddings,) = _named(spans, "embeddings hashing-v1")
    assert embeddings.attributes["gen_ai.operation.name"] == "embeddings"
    assert embeddings.attributes["docpilot.chunk_count"] == 1
    (retrieval,) = _named(spans, "retrieval document_chunks")
    assert retrieval.attributes["gen_ai.operation.name"] == "retrieval"
    assert retrieval.attributes["docpilot.search.mode"] == "hybrid"
    assert retrieval.attributes["docpilot.search.results"] == 1
    assert "tea" not in str(dict(retrieval.attributes))  # query text is never recorded


# --- one trace across the queue ---------------------------------------------


@pytest_asyncio.fixture
async def committed_document() -> AsyncGenerator[Callable]:
    created: list[uuid.UUID] = []

    async def make() -> Document:
        async with async_session_maker() as session:
            document = Document(
                filename="t.png", mime_type="image/png", storage_path="/tmp/t", status="uploaded"
            )
            session.add(document)
            await session.commit()
        created.append(document.id)
        return document

    yield make

    async with async_session_maker() as session:
        for document_id in created:
            await session.execute(delete(Job).where(Job.document_id == document_id))
            await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


async def _job(document_id: uuid.UUID, traceparent: str | None) -> Job:
    async with async_session_maker() as session:
        job = Job(document_id=document_id, state="pending", traceparent=traceparent)
        session.add(job)
        await session.commit()
        return job


async def test_job_span_continues_the_enqueuers_trace(committed_document, spans) -> None:
    document = await committed_document()
    with telemetry.tracer().start_as_current_span("POST /documents") as upload:
        traceparent = telemetry.current_traceparent()
    job = await _job(document.id, traceparent)

    async def handler(session, job) -> None:
        with telemetry.tracer().start_as_current_span("inner work"):
            pass

    await run_once(handler, document_id=document.id)

    (job_span,) = _named(spans, "job extract")
    (inner,) = _named(spans, "inner work")
    assert job_span.kind == SpanKind.CONSUMER
    assert job_span.context.trace_id == upload.get_span_context().trace_id
    assert job_span.parent.span_id == upload.get_span_context().span_id
    assert inner.parent.span_id == job_span.context.span_id
    assert job_span.attributes["docpilot.job.id"] == str(job.id)
    assert job_span.attributes["docpilot.document.id"] == str(document.id)
    assert job_span.attributes["docpilot.job.attempt"] == 1
    assert job_span.attributes["docpilot.job.outcome"] == "done"


async def test_failed_job_span_is_an_error_with_its_outcome(committed_document, spans) -> None:
    document = await committed_document()
    await _job(document.id, None)

    async def handler(session, job) -> None:
        raise RuntimeError("transient")

    await run_once(handler, document_id=document.id)

    (job_span,) = _named(spans, "job extract")
    assert job_span.parent is None  # no traceparent -> new root
    assert job_span.status.status_code == StatusCode.ERROR
    assert job_span.attributes["docpilot.job.outcome"] == "requeued"
    assert any(event.name == "exception" for event in job_span.events)


async def test_upload_stores_the_request_traceparent_on_its_job(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    with telemetry.tracer().start_as_current_span("request") as request_span:
        response = await client.post(
            "/documents", files={"file": ("r.png", TINY_PNG, "image/png")}
        )
    assert response.status_code == 201

    job = (
        await db_session.execute(
            select(Job).where(Job.document_id == uuid.UUID(response.json()["id"]))
        )
    ).scalar_one()
    trace_id = format(request_span.get_span_context().trace_id, "032x")
    assert job.traceparent is not None and trace_id in job.traceparent


async def test_index_job_inherits_the_extract_jobs_trace(
    db_session: AsyncSession, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        extraction_module, "get_settings", lambda: Settings(extraction_model="claude-sonnet-5")
    )
    response = _Message([_ToolUse(TOOL_INPUT)], "tool_use", _Usage(100, 50))
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: _fake_client(response))
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)
    document = Document(filename="r.png", mime_type="image/png", storage_path=str(path), status="uploaded")
    db_session.add(document)
    await db_session.commit()
    job = Job(document_id=document.id, state="processing")
    db_session.add(job)
    await db_session.commit()

    with telemetry.tracer().start_as_current_span("job extract") as job_span:
        await process_document_job(db_session, job)
    await db_session.flush()

    index_job = (
        await db_session.execute(
            select(Job).where(Job.document_id == document.id, Job.kind == "index")
        )
    ).scalar_one()
    span_id = format(job_span.get_span_context().span_id, "016x")
    assert index_job.traceparent is not None and span_id in index_job.traceparent


async def test_sql_spans_are_emitted_on_the_installed_sqlalchemy(spans) -> None:
    """Guards instrument_engine's skip_dep_check: the upstream instrumentor
    declares SQLAlchemy < 2.1, so if its hooks ever stop working on the
    installed version this fails instead of SQL spans silently vanishing."""
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    engine = create_async_engine(get_settings().database_url)
    telemetry.instrument_engine(engine)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    finally:
        SQLAlchemyInstrumentor().uninstrument()
        await engine.dispose()

    assert any(span.name.startswith("SELECT") for span in spans.get_finished_spans())


async def test_worker_claim_poll_emits_no_spans(spans) -> None:
    """The poll runs every second; traced, each empty poll would be a
    one-span trace of its own."""
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    from app.db import engine

    telemetry.instrument_engine(engine)
    try:
        claimed = await run_once(document_id=uuid.uuid4())  # nothing to claim
    finally:
        SQLAlchemyInstrumentor().uninstrument()

    assert claimed is False
    assert spans.get_finished_spans() == ()


async def test_agent_run_is_an_invoke_agent_span_over_chat_and_tool_spans(
    db_session: AsyncSession, spans
) -> None:
    from types import SimpleNamespace

    from app.agent.loop import answer_question
    from app.agent.tools import ToolContext

    usage = SimpleNamespace(
        input_tokens=900, output_tokens=120, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    tool_use = SimpleNamespace(type="tool_use", id="toolu_9", name="query_extractions", input={})
    responses = [
        SimpleNamespace(id="m1", model="claude-opus-5-5", content=[tool_use], stop_reason="tool_use", stop_details=None, usage=usage),
        SimpleNamespace(id="m2", model="claude-opus-5-5", content=[_Text("None found.")], stop_reason="end_turn", stop_details=None, usage=usage),
    ]
    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    )
    ctx = ToolContext(session=db_session, embedder=HashingEmbedder(), document_ids=[])

    result = await answer_question(ctx, "q", Settings(agent_model="claude-opus-5-5"), client)

    (agent,) = _named(spans, "invoke_agent doc-pilot-ask")
    chats = _named(spans, "chat claude-opus-5-5")
    (tool,) = _named(spans, "execute_tool query_extractions")
    assert len(chats) == 2
    assert all(chat.parent.span_id == agent.context.span_id for chat in chats)
    assert tool.parent.span_id == agent.context.span_id
    assert tool.attributes["gen_ai.tool.call.id"] == "toolu_9"
    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent.attributes["docpilot.agent.status"] == "answered"
    assert agent.attributes["docpilot.agent.tool_calls"] == 1
    assert result.trace_id == format(agent.context.trace_id, "032x")


async def test_judge_verdict_lands_on_the_agent_runs_trace(spans) -> None:
    import json as _json
    from types import SimpleNamespace

    from app.evals.judge import judge_answer

    with telemetry.tracer().start_as_current_span("invoke_agent doc-pilot-ask") as agent_span:
        traceparent = telemetry.current_traceparent()
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    payload = {"unsupported_or_wrong_claims": ["x"], "correct": False, "grounded": True, "score": 2, "explanation": "e"}
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
        id="j", model="claude-sonnet-5-5", stop_reason="end_turn", usage=usage,
        content=[SimpleNamespace(type="text", text=_json.dumps(payload))],
    )))))

    await judge_answer("q", "r", "e", "a", Settings(judge_model="claude-sonnet-5-5"), client=client, traceparent=traceparent)

    (evaluate,) = _named(spans, "evaluate agent_answer")
    assert evaluate.context.trace_id == agent_span.get_span_context().trace_id
    events = {e.attributes["gen_ai.evaluation.name"]: e for e in evaluate.events if e.name == "gen_ai.evaluation.result"}
    assert events["correctness"].attributes["gen_ai.evaluation.score.label"] == "fail"
    assert events["groundedness"].attributes["gen_ai.evaluation.score.value"] == 1.0
    assert _named(spans, "chat claude-sonnet-5-5")[0].parent.span_id == evaluate.context.span_id
