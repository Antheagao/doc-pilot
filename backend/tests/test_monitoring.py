"""The monitoring dashboard's data (app/routers/monitoring.py): eval runs
normalized across suites, and production metrics per UTC day."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals import history
from app.evals.history import (
    agent_points,
    eval_history,
    parse_started_at,
    retrieval_points,
)
from app.models import AskRun, Document, DocumentPage, Extraction
from app.retrieval.indexing import ChunkingConfig

PRODUCTION = ChunkingConfig(max_chars=200, overlap_chars=80, context_headers=True)


def _retrieval_config(mode, chunking, recall, scoring="idf"):
    return {
        "mode": mode,
        "chunking": chunking,
        "lexical_scoring": scoring,
        "overall": {"n": 109, "recall@5": recall, "mrr": recall - 0.1, "ndcg@10": recall - 0.05},
        "query_latency_p50_ms": 12.5,
    }


def _write(path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# --- eval runs ----------------------------------------------------------------


def test_both_artifact_timestamp_forms_parse_to_utc() -> None:
    assert parse_started_at("20260929T061630Z") == datetime(2026, 9, 29, 6, 16, 30, tzinfo=UTC)
    assert parse_started_at("2026-07-31T06:46:48+00:00") == datetime(2026, 7, 31, 6, 46, 48, tzinfo=UTC)
    assert parse_started_at("2026-07-31T06:46:48").tzinfo == UTC


def test_retrieval_follows_the_production_configuration(tmp_path) -> None:
    production = {"max_chars": 200, "overlap_chars": 80, "context_headers": True}
    whole_page = {"max_chars": 0, "overlap_chars": 0, "context_headers": False}
    _write(tmp_path / "20260929T061630Z_fastembed.json", {
        "embedder": "fastembed:BAAI/bge-small-en-v1.5", "query_set_version": "v1",
        "dataset_version": "v1", "started_at_utc": "20260929T061630Z", "configs": [
            _retrieval_config("dense", production, 0.88),
            _retrieval_config("hybrid", whole_page, 0.81),
            _retrieval_config("hybrid", production, 0.90),
        ],
    })
    # A run that never measured production's configuration is left out.
    _write(tmp_path / "20260901T000000Z_old.json", {
        "embedder": "hashing", "query_set_version": "v1", "dataset_version": "v1",
        "started_at_utc": "20260901T000000Z", "configs": [_retrieval_config("hybrid", whole_page, 0.5)],
    })
    _write(tmp_path / "20260930T000000Z_broken.json", {"embedder": "x"})  # no configs
    (tmp_path / "20261001T000000Z_garbage.json").write_text("{not json")

    (point,) = retrieval_points(PRODUCTION, tmp_path)

    assert (point.series, point.accuracy, point.accuracy_metric) == (
        "fastembed:BAAI/bge-small-en-v1.5", 0.90, "recall@5"
    )
    assert point.n == 109 and point.latency_p50_ms == 12.5
    assert point.mean_cost_usd is None  # local embeddings: no model cost
    assert point.extra["mrr"] == pytest.approx(0.80)


def test_agent_runs_are_normalized_and_bad_artifacts_skipped(tmp_path) -> None:
    _write(tmp_path / "20261002T100000Z_claude-opus-5-5_medium.json", {
        "model": "claude-opus-5-5", "effort": "medium", "prompt_version": "agent_v1",
        "question_set_version": "v1", "started_at_utc": "20261002T100000Z",
        "summary": {
            "n": 18, "accuracy": 0.89, "mean_cost_usd": 0.07, "total_cost_usd": 1.26,
            "latency_p50_ms": 11000, "cites_relevant": 0.94, "judge_grounded": 0.83,
        },
    })
    _write(tmp_path / "20261002T110000Z_partial.json", {"model": "m", "summary": {}})

    (point,) = agent_points(tmp_path)

    assert (point.series, point.n, point.accuracy) == ("claude-opus-5-5 (medium)", 18, 0.89)
    assert (point.mean_cost_usd, point.latency_p50_ms, point.latency_p95_ms) == (0.07, 11000, None)
    assert point.extra == {"judge_grounded": 0.83, "cites_relevant": 0.94}


def test_the_committed_runs_load_oldest_first() -> None:
    runs = eval_history(PRODUCTION)

    for suite in ("extraction", "retrieval"):
        assert runs[suite], f"no committed {suite} eval runs found"
        starts = [p.started_at for p in runs[suite]]
        assert starts == sorted(starts)
    extraction = runs["extraction"][-1]
    assert extraction.accuracy_metric == "field accuracy" and 0 <= extraction.accuracy <= 1
    assert extraction.mean_cost_usd > 0 and extraction.latency_p95_ms >= extraction.latency_p50_ms


async def test_eval_endpoint_returns_every_suite(client: AsyncClient, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(history, "AGENT_RESULTS_DIR", tmp_path)  # none yet

    body = (await client.get("/monitoring/evals")).json()

    assert set(body) == {"extraction", "agent", "retrieval"}
    assert body["agent"] == []
    first = body["extraction"][0]
    assert {"started_at", "series", "accuracy", "mean_cost_usd", "latency_p50_ms"} <= set(first)


# --- production, per day ----------------------------------------------------------


def _run(day: datetime, **overrides) -> AskRun:
    fields = {
        "question": "q", "status": "answered", "answer": "a", "citations": [], "tool_calls": [],
        "evidence": "e", "steps": 2, "model": "claude-opus-5-5", "prompt_version": "agent_v1",
        "created_at": day,
    }
    return AskRun(**{**fields, **overrides})


@pytest.fixture
async def empty_db(db_session: AsyncSession) -> AsyncSession:
    """The dev database's own rows, out of the way -- inside the test's
    transaction, so they come back when it rolls back."""
    for table in ("jobs", "ask_runs", "extracted_fields", "extractions", "document_chunks", "document_pages"):
        await db_session.execute(text(f"DELETE FROM {table}"))
    return db_session


async def test_daily_metrics_bucket_by_utc_day(client: AsyncClient, empty_db: AsyncSession) -> None:
    db = empty_db
    today = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    document = Document(filename="r.png", mime_type="image/png", storage_path="/tmp/r", status="extracted")
    db.add(document)
    await db.flush()
    db.add_all([
        _run(yesterday, cost_usd=0.04, latency_ms=4000, feedback="up",
             judge_grounded=True, judge_cost_usd=0.002, judged_at=today),
        _run(yesterday, cost_usd=0.06, latency_ms=8000, feedback="down",
             judge_grounded=False, judge_cost_usd=0.003, judged_at=today),
        _run(yesterday, status="refused", cost_usd=0.02, latency_ms=1000),
        _run(today, cost_usd=0.05, latency_ms=5000),
        Extraction(document_id=document.id, prompt_version="extract_v1", model="claude-sonnet-5",
                   raw_response={}, input_tokens=1, output_tokens=1, latency_ms=3000, cost_usd=0.01,
                   created_at=yesterday),
        DocumentPage(document_id=document.id, page_number=1, text="t", source="transcription",
                     cost_usd=0.004, created_at=yesterday),
        # Gold text (the evals) isn't billed.
        DocumentPage(document_id=document.id, page_number=2, text="t", source="gold",
                     cost_usd=0, created_at=yesterday),
    ])
    await db.commit()

    days = (await client.get("/monitoring/daily", params={"days": 3})).json()

    assert [d["date"] for d in days] == [
        (today - timedelta(days=n)).date().isoformat() for n in (2, 1, 0)
    ]
    quiet, before, now = days
    assert quiet["answers"]["runs"] == 0 and quiet["answers"]["grounded_rate"] is None
    assert quiet["spend"]["total_usd"] == 0 and quiet["extractions"]["mean_cost_usd"] is None

    answers = before["answers"]
    assert (answers["runs"], answers["answered"], answers["judged"]) == (3, 2, 2)
    assert answers["grounded_rate"] == 0.5
    assert (answers["feedback_up"], answers["feedback_down"]) == (1, 1)
    assert answers["mean_cost_usd"] == pytest.approx(0.04)
    assert answers["latency_p50_ms"] == 4000 and answers["latency_p95_ms"] == pytest.approx(7600)
    assert before["extractions"] == {
        "documents": 1, "mean_cost_usd": 0.01, "latency_p50_ms": 3000, "latency_p95_ms": 3000,
    }
    assert before["spend"] == {
        "documents_usd": 0.014, "answers_usd": 0.12, "grading_usd": 0, "total_usd": 0.134,
    }
    # Grading is billed the day it ran, not the day of the answer.
    assert now["spend"]["grading_usd"] == pytest.approx(0.005)
    assert now["spend"]["answers_usd"] == pytest.approx(0.05)
    assert now["answers"]["runs"] == 1


async def test_daily_window_is_validated(client: AsyncClient) -> None:
    assert (await client.get("/monitoring/daily", params={"days": 0})).status_code == 422
    assert (await client.get("/monitoring/daily", params={"days": 367})).status_code == 422
    assert len((await client.get("/monitoring/daily")).json()) == 30
