"""Tests for the aggregate stats endpoint (app/routers/stats.py).

Follows test_review.py's pattern: rows are seeded directly through
db_session (no worker/VLM involvement), and assertions use the delta
pattern -- snapshot /stats, seed, assert the delta -- since tests share
the dev Postgres on port 5434 and must never assert absolute global
counts.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, ExtractedField, Extraction
from app.routers import stats as stats_router

_LAST_EVAL_KEYS = {
    "model",
    "prompt_version",
    "dataset_version",
    "started_at_utc",
    "overall_accuracy",
    "caught_by_review",
    "mean_cost_per_doc",
    "n_scored",
}

# Mirrors app.evals.report._REQUIRED_KEYS -- the exact set of top-level
# keys an eval artifact must have to survive load_results' own filtering.
# Used to build a JSON artifact that passes load_results (so it reaches
# stats.py's projection step) but whose nested "summary" is malformed,
# pinning FINDING 1 (the projection must be inside the try/except too).
_REPORT_REQUIRED_KEYS = (
    "model",
    "prompt_version",
    "dataset_version",
    "started_at_utc",
    "summary",
    "n_scored",
    "n_errors",
    "total_cost_usd",
    "total_error_cost_usd",
    "mean_cost_per_doc",
    "latency_p50_ms",
    "latency_p95_ms",
    "per_doc",
)


async def test_stats_reflects_seeded_document_and_extractions(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """One document, TWO extractions -- pins COUNT(DISTINCT document_id)
    (documents_processed, +1) against COUNT(*) (extractions_total, +2),
    which a naive single-extraction seed can't distinguish. Also covers
    the review.corrected delta and documents_by_status["extracted"]."""
    before = (await client.get("/stats")).json()

    document = Document(
        filename="receipt.png",
        mime_type="image/png",
        storage_path=f"/tmp/{uuid.uuid4()}.png",
        status="extracted",
    )
    db_session.add(document)
    await db_session.flush()

    extraction_1 = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="claude-sonnet-5",
        raw_response={},
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.123456,
        latency_ms=2000,
    )
    extraction_2 = Extraction(
        document_id=document.id,
        prompt_version="extract_v1",
        model="claude-sonnet-5",
        raw_response={},
        input_tokens=200,
        output_tokens=75,
        cost_usd=0.02,
        latency_ms=1500,
    )
    db_session.add_all([extraction_1, extraction_2])
    await db_session.flush()

    db_session.add(
        ExtractedField(
            extraction_id=extraction_1.id,
            field_name="vendor",
            value={"value": "Acme", "confidence": 0.5},
            confidence=0.5,
            needs_review=True,
        )
    )
    db_session.add(
        ExtractedField(
            extraction_id=extraction_1.id,
            field_name="total",
            value={"value": 10.0, "confidence": 0.99},
            confidence=0.99,
            needs_review=False,
            review_action="approved",
            reviewed_at=datetime.now(UTC),
        )
    )
    db_session.add(
        ExtractedField(
            extraction_id=extraction_2.id,
            field_name="tax",
            value={"value": 1.0, "confidence": 0.4},
            confidence=0.4,
            needs_review=False,
            review_action="corrected",
            corrected_value=1.5,
            reviewed_at=datetime.now(UTC),
        )
    )
    await db_session.commit()

    after = (await client.get("/stats")).json()

    assert after["documents_total"] == before["documents_total"] + 1
    assert after["documents_processed"] == before["documents_processed"] + 1
    assert after["extractions_total"] == before["extractions_total"] + 2
    assert after["total_cost_usd"] == pytest.approx(
        before["total_cost_usd"] + 0.143456, abs=1e-6
    )
    assert after["total_input_tokens"] == before["total_input_tokens"] + 300
    assert after["total_output_tokens"] == before["total_output_tokens"] + 125
    assert after["review"]["pending"] == before["review"]["pending"] + 1
    assert after["review"]["approved"] == before["review"]["approved"] + 1
    assert after["review"]["corrected"] == before["review"]["corrected"] + 1
    assert (
        after["documents_by_status"]["extracted"]
        == before["documents_by_status"]["extracted"] + 1
    )
    # total_cost_usd / documents_processed isn't assertable exactly on a
    # shared DB (both sides shift with concurrent rows), so just pin the
    # never-divide-by-zero contract: once documents_processed > 0,
    # mean_cost_per_doc must be a present float, not null.
    assert after["mean_cost_per_doc"] is not None
    assert isinstance(after["mean_cost_per_doc"], float)


async def test_last_eval_is_null_or_has_documented_shape(client: AsyncClient) -> None:
    response = await client.get("/stats")

    assert response.status_code == 200
    last_eval = response.json()["last_eval"]
    if last_eval is not None:
        assert set(last_eval.keys()) == _LAST_EVAL_KEYS


async def test_last_eval_tolerates_corrupt_artifact_dir(
    client: AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    junk = tmp_path / "broken.json"
    junk.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(stats_router, "_RESULTS_DIR", tmp_path)

    response = await client.get("/stats")

    assert response.status_code == 200
    assert response.json()["last_eval"] is None


async def test_last_eval_null_when_loader_raises(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINDING 2(a): _load_last_eval_results itself raising (e.g. a
    filesystem error, not just a bad file load_results already tolerates)
    must still surface as last_eval: null, not a 500."""

    def _boom() -> list[dict]:
        raise RuntimeError("boom")

    monkeypatch.setattr(stats_router, "_load_last_eval_results", _boom)

    response = await client.get("/stats")

    assert response.status_code == 200
    assert response.json()["last_eval"] is None


async def test_last_eval_null_when_summary_is_malformed(
    client: AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINDING 2(b) / pins FINDING 1: an artifact that is valid JSON with
    every key load_results requires (so it survives load_results' own
    filtering and reaches stats.py's projection step) but whose "summary"
    is an empty dict, not the full summary shape -- summary["overall_accuracy"]
    must KeyError, and that KeyError must be caught by the endpoint's
    try/except rather than propagating into a 500."""
    artifact = {
        "model": "claude-sonnet-5",
        "prompt_version": "extract_v1",
        "dataset_version": "v1",
        "started_at_utc": "2026-07-31T00:00:00+00:00",
        "summary": {},
        "n_scored": 0,
        "n_errors": 0,
        "total_cost_usd": 0.0,
        "total_error_cost_usd": 0.0,
        "mean_cost_per_doc": 0.0,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "per_doc": [],
    }
    assert set(artifact.keys()) == set(_REPORT_REQUIRED_KEYS)
    (tmp_path / "malformed.json").write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(stats_router, "_RESULTS_DIR", tmp_path)

    response = await client.get("/stats")

    assert response.status_code == 200
    assert response.json()["last_eval"] is None


async def test_documents_by_status_has_all_known_statuses(client: AsyncClient) -> None:
    response = await client.get("/stats")

    assert response.status_code == 200
    by_status = response.json()["documents_by_status"]
    for known_status in stats_router.KNOWN_STATUSES:
        assert known_status in by_status
