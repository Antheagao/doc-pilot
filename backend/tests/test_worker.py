import asyncio
import uuid
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete, text

from app import worker as worker_module
from app.config import Settings
from app.db import async_session_maker
from app.models import Document, Job
from app.worker import MAX_ATTEMPTS, pending_job_stmt, reclaim_orphaned_jobs, run_once


@pytest_asyncio.fixture
async def real_documents() -> AsyncGenerator[Callable]:
    """Creates documents committed for real against the running Postgres
    instance, and cleans them (and any jobs referencing them) up after
    the test.

    The worker always opens its own sessions via async_session_maker
    rather than accepting an injected one, so it never sees data
    inserted through the savepoint-rollback `db_session` fixture used
    elsewhere in the suite -- and the concurrency test specifically needs
    two independently-committed transactions to exercise real row
    locking. These tests commit for real instead.

    Scoping, not isolation: because these rows are committed for real
    against the shared dev database, a stray real pending job (e.g. left
    behind by a manual API upload with no worker running) could
    otherwise get claimed -- and corrupted -- by a test's fake handler,
    since run_once/claim_job normally claim whatever pending job is
    globally oldest. Every test below claims through `document_id=`
    (see pending_job_stmt in app/worker.py) so it only ever touches jobs
    it created itself, regardless of what else is sitting in the table.
    """
    created: list[uuid.UUID] = []

    async def make(status: str = "uploaded") -> Document:
        async with async_session_maker() as session:
            document = Document(
                filename="test.png",
                mime_type="image/png",
                storage_path="/tmp/test.png",
                status=status,
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)
        created.append(document.id)
        return document

    yield make

    async with async_session_maker() as session:
        for document_id in created:
            await session.execute(delete(Job).where(Job.document_id == document_id))
            await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


async def _make_job(document_id: uuid.UUID, **kwargs) -> Job:
    async with async_session_maker() as session:
        job = Job(document_id=document_id, **kwargs)
        session.add(job)
        await session.commit()
        await session.refresh(job)
    return job


async def _refresh_job(job_id: uuid.UUID) -> Job:
    async with async_session_maker() as session:
        return await session.get(Job, job_id)


async def _refresh_document(document_id: uuid.UUID) -> Document:
    async with async_session_maker() as session:
        return await session.get(Document, document_id)


async def test_skip_locked_never_double_claims(real_documents: Callable) -> None:
    """The important test: two real, concurrent sessions running the
    claim query must never end up with the same job. Session1's claim
    transaction is deliberately left open (uncommitted) while session2
    runs the identical query -- SKIP LOCKED must make session2 skip
    session1's locked row and take the other pending job, never block.

    session2's query is wrapped in a timeout: if SKIP LOCKED ever
    regressed to a plain FOR UPDATE, session2 would block on session1's
    held lock indefinitely (since session1 only commits after session2
    returns) -- this test would hang forever instead of failing. The
    timeout turns that into a clear, fast assertion failure.
    """
    document = await real_documents()
    job_a = await _make_job(document.id, state="pending")
    job_b = await _make_job(document.id, state="pending")

    # Scoped to this test's document_id (see the real_documents docstring)
    # so a stray real pending job elsewhere in the table can't be picked
    # up instead of job_a/job_b.
    stmt = pending_job_stmt(datetime.now(UTC), document_id=document.id)

    session1 = async_session_maker()
    session2 = async_session_maker()
    try:
        claimed1 = (await session1.execute(stmt)).scalars().first()
        assert claimed1 is not None
        assert claimed1.id in {job_a.id, job_b.id}

        # session1's row lock is still held (no commit yet) when session2
        # runs the same query on its own connection.
        try:
            claimed2 = (
                await asyncio.wait_for(session2.execute(stmt), timeout=5)
            ).scalars().first()
        except TimeoutError:
            pytest.fail(
                "session2's claim query blocked for 5s while session1's "
                "claim was uncommitted -- SKIP LOCKED appears not to be "
                "in effect (a plain FOR UPDATE would wait instead of "
                "skipping the locked row)"
            )

        assert claimed2 is not None
        assert claimed2.id in {job_a.id, job_b.id}
        assert claimed2.id != claimed1.id

        await session1.commit()
        await session2.commit()
    finally:
        await session1.close()
        await session2.close()


async def test_success_path_marks_job_done(real_documents: Callable) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def fake_handler(session, job) -> None:
        return None

    claimed = await run_once(fake_handler, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "done"
    assert result.finished_at is not None


async def test_retry_path_requeues_with_backoff(real_documents: Callable) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def failing_handler(session, job) -> None:
        raise ValueError("boom")

    claimed = await run_once(failing_handler, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "pending"
    assert result.attempts == 1
    assert result.run_after is not None
    assert result.run_after > datetime.now(UTC)
    assert result.last_error is not None
    assert "boom" in result.last_error


async def test_exhaustion_marks_job_and_document_failed(real_documents: Callable) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending", attempts=MAX_ATTEMPTS - 1)

    async def failing_handler(session, job) -> None:
        raise ValueError("boom")

    claimed = await run_once(failing_handler, document_id=document.id)

    assert claimed is True
    result_job = await _refresh_job(job.id)
    assert result_job.state == "failed"
    assert result_job.finished_at is not None

    result_document = await _refresh_document(document.id)
    assert result_document.status == "failed"


async def test_run_after_gating_skips_future_jobs(real_documents: Callable) -> None:
    document = await real_documents()
    future = datetime.now(UTC) + timedelta(hours=1)
    job = await _make_job(document.id, state="pending", run_after=future)

    claimed = await run_once(document_id=document.id)

    assert claimed is False
    result = await _refresh_job(job.id)
    assert result.state == "pending"


async def test_poisoned_session_survives_and_requeues(real_documents: Callable) -> None:
    """Regression (HIGH, proven live): a handler whose failure is a DB
    error, not just a Python exception, leaves the session's transaction
    aborted -- any further statement on that session (including
    fail_job's own commit) raises InFailedSqlTransactionError unless
    fail_job rolls back first. Before that fix, this would have escaped
    run_once entirely, leaving the job stuck in 'processing'.
    """
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def db_error_handler(session, job) -> None:
        await session.execute(text("SELECT * FROM this_table_does_not_exist"))

    claimed = await run_once(db_error_handler, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "pending"
    assert result.attempts == 1
    assert result.last_error is not None


async def test_run_worker_survives_unexpected_run_once_failure(monkeypatch) -> None:
    """Regression (HIGH): run_worker must not die if run_once itself
    raises -- not just if a handler exception is caught inside it (that
    was the poisoned-session bug above escaping one layer further up).
    A flaky run_once that raises once must not stop the poll loop from
    calling it again.
    """
    calls = 0

    async def flaky_run_once(handler):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated unexpected failure")
        return False

    monkeypatch.setattr(worker_module, "run_once", flaky_run_once)
    monkeypatch.setattr(
        worker_module, "get_settings", lambda: Settings(worker_poll_interval=0.01)
    )

    task = asyncio.ensure_future(worker_module.run_worker())
    try:
        await asyncio.sleep(0.2)
        assert calls >= 2
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_cancelled_handler_requeues_without_burning_attempt(
    real_documents: Callable,
) -> None:
    """Graceful shutdown mid-handler (e.g. worker cancelled while a VLM
    call is in flight) must requeue the job immediately, without
    counting it against MAX_ATTEMPTS -- it wasn't the job's fault.
    """
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def cancelling_handler(session, job) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_once(cancelling_handler, document_id=document.id)

    result = await _refresh_job(job.id)
    assert result.state == "pending"
    assert result.attempts == 0
    assert result.last_error is None


async def test_reclaim_orphaned_jobs_resets_processing_to_pending(
    real_documents: Callable,
) -> None:
    """Startup sweep for jobs orphaned by a process crash mid-handler
    (nothing left to notice the row is abandoned once its claiming
    process is gone)."""
    document = await real_documents()
    job = await _make_job(document.id, state="processing", attempts=1)

    async with async_session_maker() as session:
        await reclaim_orphaned_jobs(session)

    result = await _refresh_job(job.id)
    assert result.state == "pending"
