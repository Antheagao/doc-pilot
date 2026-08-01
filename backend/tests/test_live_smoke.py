"""Live smoke suite: hits the REAL Anthropic API. Opt-in via RUN_LIVE_SMOKE=1.

Companion to scripts/smoke.py (a human-readable manual dump printed to the
console) -- this file is the assertion-bearing counterpart, wired into
pytest so a live pass/fail shows up in CI-shaped output. See that script's
docstring for the cross-reference back to here.

Skipped by default (pytestmark below): plain `pytest` never makes a
network call or spends money. Running with `-m live` and
RUN_LIVE_SMOKE=1 set makes at most two extraction round-trips (each may
include one schema-repair call, so up to four billed requests) -- one
round-trip in test_live_extract_document, one inside app.worker.run_once
in test_live_run_once_end_to_end. live_budget (below) fails a test the
moment the running total exceeds MAX_LIVE_SPEND_USD; a later test still
makes its one call before hitting its own budget check, so the true
worst case is one round-trip past the cap -- cents, not dollars.

Both tests use the same first eval case (evals/labels/001-clean-coffee-
receipt.json via app.evals.dataset.load_cases) so there is exactly one
real document involved across the whole file.

Run this suite with the local dev worker STOPPED: a `python -m app.worker`
polling the same dev database can legally claim test 2's pending job
first, failing the test and spending the money itself. Cleanup only ever
deletes rows created by (or under) this test's own document_id.
"""

import os
import uuid

import pytest
from sqlalchemy import delete, select

from app.config import get_settings
from app.db import async_session_maker
from app.evals.dataset import load_cases
from app.evals.scoring import score_case
from app.extraction import TOP_LEVEL_FIELDS, extract_document
from app.models import Document, ExtractedField, Extraction, Job
from app.worker import run_once

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_SMOKE") != "1",
        reason="live API test; set RUN_LIVE_SMOKE=1 to run",
    ),
]

# Hard cap on total spend across every live API call in this module (see
# live_budget below).
MAX_LIVE_SPEND_USD = 0.10


@pytest.fixture(scope="session", autouse=True)
def _require_live_api_key() -> None:
    """RUN_LIVE_SMOKE=1 is an explicit operator request for a live run --
    if ANTHROPIC_API_KEY isn't actually configured, fail loudly instead
    of the tests silently no-op'ing or erroring somewhere less obvious.
    """
    if not get_settings().anthropic_api_key:
        pytest.fail(
            "RUN_LIVE_SMOKE=1 but get_settings().anthropic_api_key is not set; "
            "configure ANTHROPIC_API_KEY in backend/.env before running the "
            "live smoke suite"
        )


@pytest.fixture(scope="session")
def live_budget():
    """Accumulates cost_usd from each successful extraction result,
    failing the recording test the moment the running total exceeds
    MAX_LIVE_SPEND_USD. Per-test fail-fast, not a pre-flight gate: a
    later test still makes its one call before its own check runs.
    Prints each increment (visible under `pytest -s`) so the actual
    observed cost of a live run is easy to read off the console.
    """
    spent = 0.0

    def record(cost_usd: float) -> float:
        nonlocal spent
        spent += cost_usd
        print(f"\n[live_budget] +${cost_usd:.6f} -> running total ${spent:.6f}")
        assert spent <= MAX_LIVE_SPEND_USD, (
            f"live smoke suite exceeded MAX_LIVE_SPEND_USD (${MAX_LIVE_SPEND_USD}): "
            f"running total is ${spent:.6f}"
        )
        return spent

    return record


async def test_live_extract_document(live_budget) -> None:
    """No DB: calls extract_document directly against the real API for
    the first eval case and checks both the raw result shape and its
    eval score against the gold label. The >=6/7 field margin (rather
    than demanding all 7) absorbs ordinary live-model variance -- this
    is a smoke test for the pipeline being wired correctly, not a full
    eval run (see evals/ for that).
    """
    settings = get_settings()
    case = load_cases(limit=1)[0]
    assert case.doc_id == "001-clean-coffee-receipt"

    result = await extract_document(case.image_path, case.mime_type)
    live_budget(result.cost_usd)

    assert all(field_name in result.tool_input for field_name in TOP_LEVEL_FIELDS)
    assert result.cost_usd > 0
    assert result.latency_ms > 0
    assert result.schema_violations == 0

    case_score = score_case(case, result.tool_input, settings.review_threshold)
    assert case_score.fields["vendor"].correct
    assert case_score.fields["total"].correct
    n_correct = sum(1 for field_score in case_score.fields.values() if field_score.correct)
    assert n_correct >= 6, f"only {n_correct}/7 fields correct: {case_score.to_dict()}"


async def test_live_run_once_end_to_end(live_budget) -> None:
    """Needs Postgres on 5434: end-to-end through the real job queue.
    Inserts a Document + pending Job with real commits (mirrors
    scripts/smoke.py's insert_document_and_job -- the rolled-back
    db_session fixture can't be used here since run_once opens its own
    sessions via async_session_maker), runs exactly one worker cycle
    scoped to this document_id (see pending_job_stmt in app/worker.py --
    this is what keeps a live run from claiming a stray real pending job
    elsewhere in the shared dev table), then asserts what landed in the
    DB. Cleanup always runs, mirroring scripts/smoke.py's delete order:
    fields -> extraction -> job -> document.
    """
    case = load_cases(limit=1)[0]

    async with async_session_maker() as session:
        document = Document(
            filename=case.image_path.name,
            mime_type=case.mime_type,
            storage_path=str(case.image_path),
            status="uploaded",
        )
        session.add(document)
        await session.flush()
        job = Job(document_id=document.id, state="pending")
        session.add(job)
        await session.commit()
        document_id: uuid.UUID = document.id
        job_id: uuid.UUID = job.id

    try:
        claimed = await run_once(document_id=document_id)
        assert claimed is True

        async with async_session_maker() as session:
            document = await session.get(Document, document_id)
            assert document.status == "extracted"

            extractions = (
                (
                    await session.execute(
                        select(Extraction).where(Extraction.document_id == document_id)
                    )
                )
                .scalars()
                .all()
            )
            assert len(extractions) == 1
            extraction = extractions[0]
            assert extraction.cost_usd > 0
            assert extraction.input_tokens > 0
            live_budget(float(extraction.cost_usd))

            fields = (
                (
                    await session.execute(
                        select(ExtractedField).where(
                            ExtractedField.extraction_id == extraction.id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(fields) > 0
    finally:
        async with async_session_maker() as session:
            extraction_ids = select(Extraction.id).where(Extraction.document_id == document_id)
            await session.execute(
                delete(ExtractedField).where(ExtractedField.extraction_id.in_(extraction_ids))
            )
            await session.execute(delete(Extraction).where(Extraction.document_id == document_id))
            await session.execute(delete(Job).where(Job.id == job_id))
            await session.execute(delete(Document).where(Document.id == document_id))
            await session.commit()
