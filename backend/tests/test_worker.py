import asyncio
import contextlib
import logging
import signal
import tempfile
import uuid
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select, text, update

from app import worker as worker_module
from app.config import Settings
from app.db import async_session_maker
from app.extraction import (
    ExtractionError,
    ModelRefusalError,
    NonRetryableExtractionError,
)
from app.models import DEFAULT_ORGANIZATION_ID, AskRun, Document, Job
from app.tenancy import queue_session, tenant_session
from app.worker import (
    BACKOFF_BASE,
    BACKOFF_MAX,
    MAX_ATTEMPTS,
    RETRY_AFTER_MAX,
    _backoff_delay,
    complete_job,
    fail_job,
    pending_job_stmt,
    reclaim_orphaned_jobs,
    run_once,
)

# --- _backoff_delay (pure function, no DB) ---------------------------------


def test_backoff_delay_base_is_monotonic_non_decreasing_across_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With jitter pinned to its lower bound (random.uniform returns `a`),
    _backoff_delay returns exactly base/2 -- deterministic, so the
    exponential growth across attempts can be asserted directly rather
    than just bounded.
    """
    monkeypatch.setattr(worker_module.random, "uniform", lambda a, b: a)

    delays = [_backoff_delay(attempts, None) for attempts in range(1, 10)]

    assert delays == sorted(delays)
    # Attempt 1: base = BACKOFF_BASE * 2**0 = 2.0 -> delay = 1.0
    assert delays[0] == pytest.approx(BACKOFF_BASE / 2)
    # Growth stops once base hits BACKOFF_MAX (attempt 6: 2*2**5=64 -> capped at 60).
    assert delays[-1] == pytest.approx(BACKOFF_MAX / 2)


def test_backoff_delay_never_exceeds_backoff_max_when_retry_after_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worst case for the cap: jitter pinned to its upper bound (random.uniform
    returns `b`), so delay == base == min(BACKOFF_MAX, BACKOFF_BASE * 2**(n-1)).
    Checked at high attempt counts where the uncapped exponential would be
    enormous.
    """
    monkeypatch.setattr(worker_module.random, "uniform", lambda a, b: b)

    for attempts in (1, 5, 10, 50, 1000):
        assert _backoff_delay(attempts, None) <= BACKOFF_MAX


def test_backoff_delay_always_at_least_retry_after_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retry_after is a floor: even with zero jitter and a low attempt
    count (whose plain exponential base is far smaller than retry_after),
    the returned delay must not go below it.
    """
    monkeypatch.setattr(worker_module.random, "uniform", lambda a, b: a)  # smallest possible delay

    assert _backoff_delay(1, 30.0) == pytest.approx(30.0)
    assert _backoff_delay(1, 0.5) == pytest.approx(BACKOFF_BASE / 2)  # floor below the natural delay is a no-op
    # The floor itself is capped: a misbehaving proxy's retry-after: 9999
    # must not park the job for hours.
    assert _backoff_delay(50, 9999.0) == pytest.approx(RETRY_AFTER_MAX)


def test_backoff_delay_jitter_within_base_half_to_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin random.uniform to its exact arguments and to a known return
    value, so the delay computed from it can be asserted exactly -- this
    both confirms the [0, base/2] range random.uniform is called with and
    that _backoff_delay combines it with base/2 correctly (final delay in
    [base/2, base]).
    """
    captured: dict[str, float] = {}

    def fake_uniform(a: float, b: float) -> float:
        captured["a"] = a
        captured["b"] = b
        return (a + b) / 2  # deterministic midpoint

    monkeypatch.setattr(worker_module.random, "uniform", fake_uniform)

    attempts = 3  # base = BACKOFF_BASE * 2**2 = 8.0
    delay = _backoff_delay(attempts, None)

    assert captured == {"a": 0, "b": pytest.approx(4.0)}  # [0, base/2]
    assert delay == pytest.approx(4.0 + 2.0)  # base/2 + midpoint of [0, base/2]


def _default_organization():
    """A new session bound to the default organization: what a handler's
    session is, and so what these tests read and write through."""
    return tenant_session(async_session_maker, DEFAULT_ORGANIZATION_ID)


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

    Scoping on top of isolation: the suite has its own database (see
    tests/conftest.py), so the compose worker never sees these rows, but
    a pending job left behind by a crashed earlier test run could
    otherwise get claimed -- and corrupted -- by a test's fake handler,
    since run_once/claim_job normally claim whatever pending job is
    globally oldest. Every test below claims through `document_id=`
    (see pending_job_stmt in app/worker.py) so it only ever touches jobs
    it created itself, regardless of what else is sitting in the table.
    """
    created: list[uuid.UUID] = []

    async def make(status: str = "uploaded") -> Document:
        async with _default_organization() as session:
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

    async with _default_organization() as session:
        for document_id in created:
            await session.execute(delete(Job).where(Job.document_id == document_id))
            await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


async def _make_job(document_id: uuid.UUID, **kwargs) -> Job:
    async with _default_organization() as session:
        job = Job(document_id=document_id, **kwargs)
        session.add(job)
        await session.commit()
        await session.refresh(job)
    return job


async def _refresh_job(job_id: uuid.UUID) -> Job:
    async with _default_organization() as session:
        return await session.get(Job, job_id)


async def _refresh_document(document_id: uuid.UUID) -> Document:
    async with _default_organization() as session:
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

    async with queue_session(async_session_maker) as session1, queue_session(
        async_session_maker
    ) as session2:
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


async def test_retry_after_seconds_floors_the_requeue_delay(real_documents: Callable) -> None:
    """When the handler raises an ExtractionError carrying
    retry_after_seconds (set when the VLM call failed with a rate-limit/
    overload response that told us how long to wait -- see
    app.extraction), fail_job must requeue with a run_after at least that
    far out, not just the plain exponential backoff for attempt 1 (which
    on its own would land well under 30s). Small clock slack is allowed
    since `before` is captured just ahead of the real requeue timestamp.
    """
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def rate_limited_handler(session, job) -> None:
        raise ExtractionError("rate limited", retry_after_seconds=30)

    before = datetime.now(UTC)
    claimed = await run_once(rate_limited_handler, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "pending"
    assert result.attempts == 1
    assert result.run_after is not None
    assert result.run_after >= before + timedelta(seconds=30) - timedelta(seconds=1)


async def test_non_retryable_failure_skips_requeue_and_fails_immediately(
    real_documents: Callable,
) -> None:
    """A NonRetryableExtractionError (refusal, bad mime, unpriced model,
    stop_reason == "max_tokens" truncation -- see app/extraction.py) is
    deterministic: retrying the identical job would produce the
    identical outcome, so fail_job must skip the requeue-with-backoff
    path entirely and mark the job (and its document) failed on the very
    first attempt, unlike test_retry_path_requeues_with_backoff's plain
    ValueError (attempt 1 of MAX_ATTEMPTS, requeues).
    """
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def refusing_handler(session, job) -> None:
        raise NonRetryableExtractionError("model refused")

    claimed = await run_once(refusing_handler, document_id=document.id)

    assert claimed is True
    result_job = await _refresh_job(job.id)
    assert result_job.state == "failed"
    assert result_job.attempts == 1  # did not burn through MAX_ATTEMPTS retries
    assert result_job.finished_at is not None
    assert result_job.last_error is not None
    assert "model refused" in result_job.last_error

    result_document = await _refresh_document(document.id)
    assert result_document.status == "failed"


async def test_model_refusal_marks_document_refused(real_documents: Callable) -> None:
    """A ModelRefusalError (see app/extraction.py) is a NonRetryableExtractionError
    subtype, so it gets the same skip-requeue-and-fail-immediately job
    handling as the plain-NonRetryable case above -- but fail_job marks
    the document "refused" instead of "failed" so the UI can tell a
    model refusal apart from every other permanent failure.
    """
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def refusing_handler(session, job) -> None:
        raise ModelRefusalError("model refused")

    claimed = await run_once(refusing_handler, document_id=document.id)

    assert claimed is True
    result_job = await _refresh_job(job.id)
    assert result_job.state == "failed"
    assert result_job.attempts == 1  # did not burn through MAX_ATTEMPTS retries
    assert result_job.finished_at is not None
    assert result_job.last_error is not None
    assert "model refused" in result_job.last_error

    result_document = await _refresh_document(document.id)
    assert result_document.status == "refused"


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


async def test_a_handler_that_rolled_back_then_failed_requeues_with_its_error(
    real_documents: Callable,
) -> None:
    """Regression: every real handler rolls back before its model call,
    which expires the Job in the session, and AsyncSession cannot lazy-load
    it back. fail_job and the log lines used to read job.id from that
    expired object, so a model failure (a 401, say) raised MissingGreenlet
    out of run_once, lost the real error, and left the job 'processing'.
    """
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def rolls_back_then_fails(session, job) -> None:
        await session.rollback()
        raise ExtractionError("model call failed")

    claimed = await run_once(rolls_back_then_fails, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "pending"
    assert result.attempts == 1
    assert result.run_after is not None and result.run_after > datetime.now(UTC)
    assert result.last_error is not None
    assert "model call failed" in result.last_error


async def test_a_handler_that_rolled_back_then_failed_for_good_fails_the_document(
    real_documents: Callable,
) -> None:
    document = await real_documents(status="processing")
    job = await _make_job(document.id, state="pending")

    async def rolls_back_then_fails(session, job) -> None:
        await session.rollback()
        raise NonRetryableExtractionError("invalid api key")

    claimed = await run_once(rolls_back_then_fails, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "failed"
    assert result.finished_at is not None
    assert result.last_error is not None
    assert "invalid api key" in result.last_error
    assert (await _refresh_document(document.id)).status == "failed"


async def test_a_handler_that_rolled_back_then_succeeded_marks_job_done(
    real_documents: Callable,
) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def rolls_back_then_succeeds(session, job) -> None:
        await session.rollback()

    claimed = await run_once(rolls_back_then_succeeds, document_id=document.id)

    assert claimed is True
    result = await _refresh_job(job.id)
    assert result.state == "done"
    assert result.finished_at is not None


async def test_a_handler_cancelled_after_a_rollback_still_requeues(
    real_documents: Callable,
) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def rolls_back_then_is_cancelled(session, job) -> None:
        await session.rollback()
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_once(rolls_back_then_is_cancelled, document_id=document.id)

    result = await _refresh_job(job.id)
    assert result.state == "pending"
    assert result.attempts == 0
    assert result.last_error is None


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

    async with queue_session(async_session_maker) as session:
        await reclaim_orphaned_jobs(session)

    result = await _refresh_job(job.id)
    assert result.state == "pending"


async def test_a_reclaim_lease_leaves_jobs_still_being_worked_alone(
    real_documents: Callable,
) -> None:
    """During a deploy that overlaps two workers, the new one's sweep must
    not steal the job the old one is finishing -- only abandoned ones."""
    document = await real_documents()
    now = datetime.now(UTC)
    fresh = await _make_job(document.id, state="processing", attempts=1, started_at=now - timedelta(seconds=30))
    stale = await _make_job(document.id, state="processing", attempts=1, started_at=now - timedelta(hours=1))
    unstamped = await _make_job(document.id, state="processing", attempts=1)

    async with queue_session(async_session_maker) as session:
        await reclaim_orphaned_jobs(session, older_than=timedelta(minutes=30))

    assert (await _refresh_job(fresh.id)).state == "processing"
    assert (await _refresh_job(stale.id)).state == "pending"
    assert (await _refresh_job(unstamped.id)).state == "pending"


async def test_run_worker_reclaims_periodically_with_a_lease(monkeypatch) -> None:
    sweeps: list[timedelta | None] = []

    async def fake_reclaim(session, older_than=None):
        sweeps.append(older_than)
        return 0

    async def idle(handler):
        return False

    monkeypatch.setattr(worker_module, "reclaim_orphaned_jobs", fake_reclaim)
    monkeypatch.setattr(worker_module, "run_once", idle)
    monkeypatch.setattr(worker_module, "RECLAIM_CHECK_SECONDS", 0.02)
    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: Settings(worker_poll_interval=0.01, worker_reclaim_after_seconds=1800),
    )
    stop = asyncio.Event()

    task = asyncio.ensure_future(worker_module.run_worker(stop=stop))
    await asyncio.sleep(0.15)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert len(sweeps) >= 2 and set(sweeps) == {timedelta(seconds=1800)}


async def test_without_a_lease_the_loop_never_sweeps(monkeypatch) -> None:
    """compose's single worker: the startup sweep in main() is the only one."""
    sweeps = []

    async def fake_reclaim(session, older_than=None):
        sweeps.append(older_than)

    async def idle(handler):
        return False

    monkeypatch.setattr(worker_module, "reclaim_orphaned_jobs", fake_reclaim)
    monkeypatch.setattr(worker_module, "run_once", idle)
    monkeypatch.setattr(worker_module, "RECLAIM_CHECK_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "get_settings", lambda: Settings(worker_poll_interval=0.01))
    stop = asyncio.Event()

    task = asyncio.ensure_future(worker_module.run_worker(stop=stop))
    await asyncio.sleep(0.08)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert sweeps == []


async def test_stop_lets_the_job_in_hand_finish(monkeypatch) -> None:
    """SIGTERM sets `stop`: no new claims, but the running job completes
    rather than being cancelled mid-call."""
    stop = asyncio.Event()
    finished = []

    async def slow_job(handler):
        stop.set()  # the shutdown signal lands mid-job
        await asyncio.sleep(0.05)
        finished.append(1)
        return True

    monkeypatch.setattr(worker_module, "run_once", slow_job)
    monkeypatch.setattr(worker_module, "get_settings", lambda: Settings(worker_poll_interval=0.01))

    await asyncio.wait_for(worker_module.run_worker(stop=stop), timeout=1)

    assert finished == [1]


def test_worker_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        Settings(worker_reclaim_after_seconds=-1)


# --- two workers racing for one job: compare-and-set settling ---------------
#
# Every test here commits for real and uses two independent run_once calls, the
# way two worker processes would: worker A claims a job and is held inside its
# handler after a content write; its lease is backdated; worker B reclaims and
# claims the job. What A does when it is finally released is the subject.


@pytest_asyncio.fixture
async def marked_documents() -> AsyncGenerator[str]:
    """A marker for the filenames of the content rows a handler writes, and
    their removal afterwards."""
    marker = f"cas-{uuid.uuid4().hex[:12]}"
    yield marker
    async with _default_organization() as session:
        await session.execute(delete(Document).where(Document.filename.like(f"{marker}%")))
        await session.commit()


async def _marked_filenames(marker: str) -> list[str]:
    async with _default_organization() as session:
        stmt = select(Document.filename).where(Document.filename.like(f"{marker}%"))
        return sorted((await session.execute(stmt)).scalars())


def _written_row(marker: str, who: str) -> Document:
    return Document(
        filename=f"{marker}-{who}", mime_type="image/png", storage_path="/tmp/cas", status="uploaded"
    )


async def _backdate_lease(job_id: uuid.UUID) -> None:
    async with _default_organization() as session:
        await session.execute(
            update(Job).where(Job.id == job_id).values(started_at=func.now() - timedelta(hours=1))
        )
        await session.commit()


class _TwoWorkers:
    """Worker A held mid-handler, then reclaimed from and replaced by worker B."""

    def __init__(self, document: Document, job: Job, marker: str, how_a_ends: str) -> None:
        self.document, self.job, self.marker, self.how_a_ends = document, job, marker, how_a_ends
        self.a_wrote, self.release_a = asyncio.Event(), asyncio.Event()
        self.b_wrote, self.release_b = asyncio.Event(), asyncio.Event()
        self.release_b.set()
        self.a_task: asyncio.Task | None = None
        self.b_task: asyncio.Task | None = None
        # What each worker's after-done callback ran: only a settled job's does.
        self.posted: list[str] = []

    async def _handler_a(self, session, job) -> None:
        session.add(_written_row(self.marker, "a"))
        await session.flush()
        self.a_wrote.set()
        await self.release_a.wait()
        if self.how_a_ends == "fails":
            raise ExtractionError("a failed late")
        if self.how_a_ends == "fails for good":
            raise NonRetryableExtractionError("a failed late for good")
        if self.how_a_ends == "is cancelled":
            raise asyncio.CancelledError()
        return lambda: self.posted.append("a")

    async def _handler_b(self, session, job) -> None:
        session.add(_written_row(self.marker, "b"))
        await session.flush()
        self.b_wrote.set()
        await self.release_b.wait()
        return lambda: self.posted.append("b")

    async def start_a_then_reclaim_and_start_b(self, *, hold_b: bool) -> None:
        self.a_task = asyncio.ensure_future(run_once(self._handler_a, document_id=self.document.id))
        await asyncio.wait_for(self.a_wrote.wait(), timeout=10)
        await _backdate_lease(self.job.id)
        async with queue_session(async_session_maker) as session:
            await reclaim_orphaned_jobs(session, older_than=timedelta(minutes=30))
        assert (await _refresh_job(self.job.id)).state == "pending"
        if hold_b:
            self.release_b.clear()
        self.b_task = asyncio.ensure_future(run_once(self._handler_b, document_id=self.document.id))
        await asyncio.wait_for(self.b_wrote.wait(), timeout=10)

    async def release_a_and_wait(self) -> None:
        self.release_a.set()
        if self.how_a_ends == "is cancelled":
            with pytest.raises(asyncio.CancelledError):
                await self.a_task
        else:
            assert await asyncio.wait_for(self.a_task, timeout=10) is True

    async def finish_b(self) -> None:
        self.release_b.set()
        assert await asyncio.wait_for(self.b_task, timeout=10) is True

    async def cancel_whatever_is_left(self) -> None:
        for task in (self.a_task, self.b_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


_A_ENDINGS = ["succeeds", "fails", "fails for good", "is cancelled"]


@pytest.mark.parametrize("how_a_ends", _A_ENDINGS)
async def test_a_stale_worker_after_the_new_claim_finished_changes_nothing(
    real_documents: Callable,
    marked_documents: str,
    caplog: pytest.LogCaptureFixture,
    how_a_ends: str,
) -> None:
    """A is reclaimed mid-run, B claims the job and completes it; then A is
    released. A's settling (a completion, a failure, a permanent failure of the
    document, a shutdown requeue) updates no row, its content write is rolled
    back, it is logged, and B's completion and write stand, once."""
    document = await real_documents()
    job = await _make_job(document.id, state="pending")
    workers = _TwoWorkers(document, job, marked_documents, how_a_ends)
    caplog.set_level(logging.INFO, logger=worker_module.logger.name)

    try:
        await workers.start_a_then_reclaim_and_start_b(hold_b=False)
        await workers.finish_b()
        done = await _refresh_job(job.id)
        assert (done.state, done.attempts) == ("done", 2)

        await workers.release_a_and_wait()
    finally:
        await workers.cancel_whatever_is_left()

    settled = await _refresh_job(job.id)
    assert (settled.state, settled.attempts) == ("done", 2)
    assert settled.finished_at == done.finished_at and settled.last_error is None
    assert settled.run_after is None
    assert (await _refresh_document(document.id)).status == "processing"  # never failed by A
    assert await _marked_filenames(marked_documents) == [f"{marked_documents}-b"]
    stale = [r for r in caplog.records if r.levelno == logging.WARNING and str(job.id) in r.getMessage()]
    assert len(stale) == 1 and "dropped" in stale[0].getMessage()
    completions = [r for r in caplog.records if r.getMessage() == f"completed job {job.id}"]
    assert len(completions) == 1
    assert workers.posted == ["b"]  # a stale worker's side effects never ran


@pytest.mark.parametrize("how_a_ends", _A_ENDINGS)
async def test_a_stale_worker_while_the_new_claim_runs_leaves_that_claim_alone(
    real_documents: Callable, marked_documents: str, how_a_ends: str
) -> None:
    """The attempt number is what tells B's claim from A's: the job is
    'processing' both times, so a compare-and-set on state alone would let A's
    late failure requeue B's job, or its late completion finish it."""
    document = await real_documents()
    job = await _make_job(document.id, state="pending")
    workers = _TwoWorkers(document, job, marked_documents, how_a_ends)

    try:
        await workers.start_a_then_reclaim_and_start_b(hold_b=True)
        await workers.release_a_and_wait()

        mid = await _refresh_job(job.id)
        assert (mid.state, mid.attempts) == ("processing", 2)
        assert mid.finished_at is None and mid.last_error is None and mid.run_after is None
        assert (await _refresh_document(document.id)).status == "processing"

        await workers.finish_b()
    finally:
        await workers.cancel_whatever_is_left()

    settled = await _refresh_job(job.id)
    assert (settled.state, settled.attempts) == ("done", 2)
    assert await _marked_filenames(marked_documents) == [f"{marked_documents}-b"]
    assert workers.posted == ["b"]


async def test_a_stale_completion_is_dropped_without_touching_the_job(
    real_documents: Callable, marked_documents: str
) -> None:
    """complete_job and fail_job on their own: a job that is not processing at
    the claimed attempt is left as it is, and the session's flushed writes are
    rolled back."""
    document = await real_documents()
    job = await _make_job(document.id, state="processing", attempts=2, started_at=datetime.now(UTC))

    async with _default_organization() as session:
        session.add(_written_row(marked_documents, "late"))
        await session.flush()
        assert await complete_job(session, job.id, attempts=1) == "stale"
    async with _default_organization() as session:
        session.add(_written_row(marked_documents, "late"))
        await session.flush()
        assert await fail_job(session, job.id, 1, ExtractionError("late")) == "stale"
    async with _default_organization() as session:
        assert await complete_job(session, uuid.uuid4(), attempts=1) == "stale"

    unchanged = await _refresh_job(job.id)
    assert (unchanged.state, unchanged.attempts, unchanged.last_error) == ("processing", 2, None)
    assert await _marked_filenames(marked_documents) == []


async def test_the_claimed_attempt_settles_its_own_claim(
    real_documents: Callable, marked_documents: str
) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="processing", attempts=2, started_at=datetime.now(UTC))

    async with _default_organization() as session:
        session.add(_written_row(marked_documents, "on-time"))
        await session.flush()
        assert await complete_job(session, job.id, attempts=2) == "done"

    settled = await _refresh_job(job.id)
    assert settled.state == "done" and settled.finished_at is not None
    assert await _marked_filenames(marked_documents) == [f"{marked_documents}-on-time"]


async def test_a_job_settled_twice_is_settled_once(real_documents: Callable) -> None:
    """A second settling of the same claim (the first already committed) is a
    no-op, not an overwrite: a job failed for good is not reopened by a stray
    late completion."""
    document = await real_documents()
    job = await _make_job(document.id, state="processing", attempts=MAX_ATTEMPTS)

    async with _default_organization() as session:
        assert await fail_job(session, job.id, MAX_ATTEMPTS, ExtractionError("out")) == "failed"
    async with _default_organization() as session:
        assert await complete_job(session, job.id, MAX_ATTEMPTS) == "stale"

    assert (await _refresh_job(job.id)).state == "failed"


# --- the reclaim lease -------------------------------------------------------


async def test_reclaim_only_resets_jobs_that_are_processing(real_documents: Callable) -> None:
    """The sweep is one UPDATE with 'processing' and the lease in its WHERE,
    not a select followed by a write per job: a job that finished (or was
    requeued) since would otherwise be reset and run twice."""
    document = await real_documents()
    old = datetime.now(UTC) - timedelta(hours=1)
    done = await _make_job(document.id, state="done", attempts=1, started_at=old)
    failed = await _make_job(document.id, state="failed", attempts=3, started_at=old)
    abandoned = await _make_job(document.id, state="processing", attempts=1, started_at=old)

    async with queue_session(async_session_maker) as session:
        assert await reclaim_orphaned_jobs(session, older_than=timedelta(minutes=30)) >= 1

    assert (await _refresh_job(done.id)).state == "done"
    assert (await _refresh_job(failed.id)).state == "failed"
    reclaimed = await _refresh_job(abandoned.id)
    assert (reclaimed.state, reclaimed.attempts) == ("pending", 1)


@pytest.fixture
def stubbed_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """worker.main() against the real database up to its startup sweep, with
    everything after it (the model check, the embedder, the loop) replaced."""

    async def nothing(*args, **kwargs) -> None:
        return None

    def with_settings(**overrides) -> None:
        settings = Settings(
            worker_heartbeat_file=str(tmp_path / "heartbeat"), worker_poll_interval=0.01, **overrides
        )
        monkeypatch.setattr(worker_module, "get_settings", lambda: settings)

    monkeypatch.setattr(worker_module, "name_process", lambda name: None)
    monkeypatch.setattr(worker_module, "verify_tenant_isolation", nothing)
    monkeypatch.setattr(worker_module, "check_model_provider", lambda settings: None)
    monkeypatch.setattr(worker_module, "warm_up_embedder", lambda: None)
    monkeypatch.setattr(worker_module, "run_worker", nothing)
    monkeypatch.setattr(worker_module, "shutdown_tracing", lambda: None)
    monkeypatch.setattr(worker_module, "engine", SimpleNamespace(dispose=nothing))

    async def start() -> None:
        try:
            await asyncio.wait_for(worker_module.main(), timeout=10)
        finally:
            # main() installs a SIGTERM handler on the suite's one event loop.
            asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)

    return with_settings, start


async def test_a_starting_worker_with_a_lease_leaves_another_workers_job_alone(
    real_documents: Callable, stubbed_startup
) -> None:
    with_settings, start = stubbed_startup
    with_settings(worker_reclaim_after_seconds=600)
    document = await real_documents()
    now = datetime.now(UTC)
    live = await _make_job(document.id, state="processing", attempts=1, started_at=now - timedelta(seconds=30))
    abandoned = await _make_job(document.id, state="processing", attempts=1, started_at=now - timedelta(hours=1))

    await start()

    assert (await _refresh_job(live.id)).state == "processing"
    assert (await _refresh_job(abandoned.id)).state == "pending"


async def test_a_starting_worker_without_a_lease_reclaims_every_processing_job(
    real_documents: Callable, stubbed_startup
) -> None:
    """WORKER_RECLAIM_AFTER_SECONDS=0, the single-worker dev setting: unchanged."""
    with_settings, start = stubbed_startup
    with_settings()
    assert worker_module._reclaim_lease() is None
    document = await real_documents()
    just_started = await _make_job(
        document.id, state="processing", attempts=1, started_at=datetime.now(UTC) - timedelta(seconds=5)
    )

    await start()

    assert (await _refresh_job(just_started.id)).state == "pending"


async def test_the_work_after_completion_runs_once_the_job_is_done(real_documents: Callable) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")
    ran: list[str] = []

    async def handler(session, job):
        job_id = job.id

        def after_done() -> None:
            ran.append(f"ran for {job_id}")

        return after_done

    await run_once(handler, document_id=document.id)

    assert ran == [f"ran for {job.id}"]
    assert (await _refresh_job(job.id)).state == "done"


async def test_a_failing_after_done_callback_does_not_undo_the_job(
    real_documents: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    async def handler(session, job):
        def after_done() -> None:
            raise RuntimeError("langfuse is down")

        return after_done

    assert await run_once(handler, document_id=document.id) is True

    assert (await _refresh_job(job.id)).state == "done"
    assert any("the work after its completion failed" in r.getMessage() for r in caplog.records)


# --- the attempts cap at reclaim ---------------------------------------------


async def _committed_judge_job(**job_fields) -> tuple[Job, AskRun]:
    async with _default_organization() as session:
        run = AskRun(
            question="q", status="answered", answer="a", citations=[], tool_calls=[], evidence="e",
            steps=1, model="m", prompt_version="agent_v1", judge_sampled=True,
        )
        session.add(run)
        await session.flush()
        job = Job(ask_run_id=run.id, kind="judge", **job_fields)
        session.add(job)
        await session.commit()
        return job, run


@pytest_asyncio.fixture
async def judge_runs() -> AsyncGenerator[list[uuid.UUID]]:
    run_ids: list[uuid.UUID] = []
    yield run_ids
    async with _default_organization() as session:
        for run_id in run_ids:
            await session.execute(delete(Job).where(Job.ask_run_id == run_id))
            await session.execute(delete(AskRun).where(AskRun.id == run_id))
        await session.commit()


async def test_reclaim_fails_a_job_that_used_up_its_attempts_and_its_subjects(
    real_documents: Callable, judge_runs: list[uuid.UUID], caplog: pytest.LogCaptureFixture
) -> None:
    """A job that kills its worker, or outlasts the lease, every time would
    otherwise be reclaimed and billed forever."""
    document = await real_documents(status="processing")
    old = datetime.now(UTC) - timedelta(hours=1)
    spent = await _make_job(document.id, state="processing", attempts=MAX_ATTEMPTS, started_at=old)
    below = await _make_job(document.id, state="processing", attempts=MAX_ATTEMPTS - 1, started_at=old)
    judge, run = await _committed_judge_job(state="processing", attempts=MAX_ATTEMPTS + 2, started_at=old)
    judge_runs.append(run.id)
    caplog.set_level(logging.INFO, logger=worker_module.logger.name)

    async with queue_session(async_session_maker) as session:
        assert await reclaim_orphaned_jobs(session, older_than=timedelta(minutes=30)) >= 3

    failed = await _refresh_job(spent.id)
    assert failed.state == "failed" and failed.finished_at is not None and failed.attempts == MAX_ATTEMPTS
    assert failed.last_error is not None and f"reclaimed after {MAX_ATTEMPTS} attempts" in failed.last_error
    assert "WORKER_RECLAIM_AFTER_SECONDS" in failed.last_error
    assert (await _refresh_document(document.id)).status == "failed"
    requeued = await _refresh_job(below.id)
    assert (requeued.state, requeued.attempts, requeued.last_error, requeued.finished_at) == (
        "pending", MAX_ATTEMPTS - 1, None, None,
    )
    judged = await _refresh_job(judge.id)
    assert judged.state == "failed" and f"reclaimed after {MAX_ATTEMPTS + 2} attempts" in judged.last_error
    async with _default_organization() as session:
        graded = await session.get(AskRun, run.id)
    assert graded.judged_at is not None and graded.judge_grounded is None
    assert graded.judge_error.startswith("grading failed: reclaimed after")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    for failed_job in (spent, judge):
        assert sum(str(failed_job.id) in message for message in warnings) == 1
    assert not any(str(below.id) in message for message in warnings)

    # The claim query never offers the failed job again.
    async with queue_session(async_session_maker) as session:
        offered = (await session.execute(pending_job_stmt(datetime.now(UTC), document.id))).scalars().all()
    assert {job.id for job in offered} == {below.id}


async def test_reclaim_leaves_a_live_job_at_the_cap_alone_under_a_lease(real_documents: Callable) -> None:
    document = await real_documents(status="processing")
    live = await _make_job(
        document.id, state="processing", attempts=MAX_ATTEMPTS, started_at=datetime.now(UTC) - timedelta(seconds=30)
    )

    async with queue_session(async_session_maker) as session:
        await reclaim_orphaned_jobs(session, older_than=timedelta(minutes=30))

    assert (await _refresh_job(live.id)).state == "processing"
    assert (await _refresh_document(document.id)).status == "processing"


async def test_a_starting_worker_without_a_lease_applies_the_same_cap(
    real_documents: Callable, stubbed_startup
) -> None:
    with_settings, start = stubbed_startup
    with_settings()
    document = await real_documents(status="processing")
    spent = await _make_job(document.id, state="processing", attempts=MAX_ATTEMPTS, started_at=datetime.now(UTC))
    below = await _make_job(document.id, state="processing", attempts=1, started_at=datetime.now(UTC))

    await start()

    assert (await _refresh_job(spent.id)).state == "failed"
    assert (await _refresh_job(below.id)).state == "pending"
    assert (await _refresh_document(document.id)).status == "failed"


# --- job kinds (extract vs index) -------------------------------------------


async def test_index_job_claim_leaves_document_status_alone(real_documents: Callable) -> None:
    """Only an extract job drives the document's status: an index job for
    an already-extracted document must not flip it back to processing."""
    document = await real_documents(status="extracted")
    await _make_job(document.id, state="pending", kind="index")
    seen: list[str] = []

    async def fake_handler(session, job) -> None:
        seen.append((await session.get(Document, job.document_id)).status)

    await run_once(fake_handler, document_id=document.id)

    assert seen == ["extracted"]
    assert (await _refresh_document(document.id)).status == "extracted"


async def test_failed_index_job_leaves_document_extracted(real_documents: Callable) -> None:
    document = await real_documents(status="extracted")
    job = await _make_job(document.id, state="pending", kind="index")

    async def failing_handler(session, job) -> None:
        raise NonRetryableExtractionError("transcription refused")

    await run_once(failing_handler, document_id=document.id)

    assert (await _refresh_job(job.id)).state == "failed"
    assert (await _refresh_document(document.id)).status == "extracted"


async def test_dispatch_routes_by_job_kind(
    real_documents: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await real_documents(status="extracted")
    job = await _make_job(document.id, state="pending", kind="index")
    called: list[str] = []

    async def extract_handler(session, job) -> None:
        called.append("extract")

    async def index_handler(session, job) -> None:
        called.append("index")

    monkeypatch.setattr(
        worker_module, "HANDLERS", {"extract": extract_handler, "index": index_handler}
    )

    await run_once(worker_module.dispatch_job, document_id=document.id)

    assert called == ["index"]
    assert (await _refresh_job(job.id)).state == "done"


async def test_unknown_job_kind_fails_permanently(real_documents: Callable) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending", kind="summarize")

    await run_once(worker_module.dispatch_job, document_id=document.id)

    result = await _refresh_job(job.id)
    assert result.state == "failed"
    assert result.attempts == 1
    assert "no handler for job kind 'summarize'" in result.last_error


async def test_jobs_default_to_extract_kind(real_documents: Callable) -> None:
    document = await real_documents()
    job = await _make_job(document.id, state="pending")

    assert job.kind == "extract"


# --- heartbeat (the compose healthcheck's liveness signal) -------------------


async def test_the_heartbeat_file_is_touched_repeatedly(tmp_path) -> None:
    path = tmp_path / "heartbeat"
    task = asyncio.ensure_future(worker_module._heartbeat_loop(path, 0.01))
    try:
        await asyncio.sleep(0.05)
        first = path.stat().st_mtime_ns
        await asyncio.sleep(0.1)
        assert path.stat().st_mtime_ns > first
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_the_heartbeat_keeps_ticking_while_a_job_runs(tmp_path, monkeypatch) -> None:
    """A long extraction occupies run_once for minutes; the heartbeat must
    not go stale meanwhile or the healthcheck would report a busy worker
    as dead."""
    path = tmp_path / "heartbeat"
    mtimes: list[int] = []

    async def long_job(handler):
        for _ in range(5):
            await asyncio.sleep(0.03)
            mtimes.append(path.stat().st_mtime_ns)
        return False

    monkeypatch.setattr(worker_module, "run_once", long_job)
    monkeypatch.setattr(worker_module, "get_settings", lambda: Settings(worker_poll_interval=0.01))
    stop = asyncio.Event()
    heartbeat = asyncio.ensure_future(worker_module._heartbeat_loop(path, 0.01))
    run = asyncio.ensure_future(worker_module.run_worker(stop=stop))
    try:
        await asyncio.sleep(0.25)
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=1)
        heartbeat.cancel()

    assert len(set(mtimes)) > 1


async def test_an_unwritable_heartbeat_path_does_not_stop_the_loop(tmp_path) -> None:
    path = tmp_path / "missing-directory" / "heartbeat"
    task = asyncio.ensure_future(worker_module._heartbeat_loop(path, 0.01))
    try:
        await asyncio.sleep(0.05)
        assert not task.done()
        path.parent.mkdir()
        await asyncio.sleep(0.05)
        assert path.exists()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_the_heartbeat_path_defaults_under_the_temp_dir() -> None:
    assert Path(Settings().worker_heartbeat_file).parent == Path(tempfile.gettempdir())
