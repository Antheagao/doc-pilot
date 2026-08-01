"""Postgres SKIP LOCKED job worker.

Run as `python -m app.worker`. Polls the `jobs` table for pending work,
claims one job at a time using `SELECT ... FOR UPDATE SKIP LOCKED`, and
hands it to a pluggable handler.

Concurrency model: this process handles exactly ONE job at a time. To
process jobs in parallel, run more instances of this module — SKIP LOCKED
guarantees they never claim the same job. There is no in-process
concurrency here on purpose: it keeps the claim/complete bookkeeping
trivial and lets horizontal scaling do the work.

Claim/commit split: claiming a job (state -> processing) is committed
immediately, *before* the handler runs, rather than holding the row lock
for the duration of the handler. The default handler (process_document_job,
app/extraction.py) makes a VLM call and can take several seconds; holding
a transaction open that long would tie up a connection and block other
workers from even attempting SKIP LOCKED scans against the table. Once
claimed, a job is "owned" by this process via its state, not via a held
lock.

Consequence: if this process dies (crash, kill -9, power loss) after
claiming a job but before it finishes, that job is orphaned in
'processing' forever -- there's no lock left to notice. Recovery:
reclaim_orphaned_jobs() runs once at startup (see main()) and resets any
job it finds in 'processing' back to 'pending'. This is ONLY safe because
deployments are single-worker: it assumes any 'processing' row at startup
must be a leftover from a previous run of *this same process*, not a job
another live worker is actively working on right now. Running two worker
processes concurrently would make this sweep steal in-flight jobs.

Retry/backoff: failed jobs are requeued (state -> pending) up to
MAX_ATTEMPTS times, with an equal-jitter exponential backoff (see
_backoff_delay) applied via the `run_after` column, capped at
BACKOFF_MAX. When the failure carries an ExtractionError.retry_after_seconds
(a 429/529/5xx response with a `retry-after` header -- see
app.extraction.extract_document), that value is a floor on the delay:
the server's own requested wait always wins over a shorter jittered
backoff. The claim query only considers jobs whose `run_after` is null
or in the past, so a backing-off job doesn't get immediately re-claimed
by the next poll.
"""

import asyncio
import logging
import random
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import async_session_maker, engine
from app.extraction import (
    ModelRefusalError,
    NonRetryableExtractionError,
    process_document_job,
)
from app.models import Document, Job

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0
# Ceiling on the deterministic half of _backoff_delay's exponential growth
# (before jitter and before the retry_after floor) -- without this a job
# that's failed many times would otherwise wait longer and longer forever.
BACKOFF_MAX = 60.0
# Ceiling on how far a server-supplied retry-after header can push the
# requeue. A misbehaving proxy sending retry-after: 86400 would otherwise
# park the document in "processing" for a day with no operator signal.
# (The Anthropic SDK's own in-process retry layer ignores retry-after
# values over 60s for the same reason.)
RETRY_AFTER_MAX = 900.0

Handler = Callable[[AsyncSession, Job], Awaitable[None]]


def _backoff_delay(attempts: int, retry_after: float | None) -> float:
    """Equal-jitter exponential backoff delay, in seconds, for the
    `attempts`-th failed attempt (1-indexed, matching Job.attempts).

    Pure function -- no I/O, no clock reads -- so tests can assert exact
    values by monkeypatching random.uniform instead of asserting a loose
    range against a real timestamp.

    `base` is the plain exponential (BACKOFF_BASE * 2**(attempts-1)),
    capped at BACKOFF_MAX so a job that keeps failing doesn't wait longer
    and longer without bound. The returned delay is base/2 plus a uniform
    random draw from [0, base/2]: half deterministic floor, half jitter,
    so many simultaneously-failing jobs don't all retry in lockstep (the
    "equal jitter" strategy -- see
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/).

    `retry_after` -- ExtractionError.retry_after_seconds, when the
    failure was a rate-limit/overload response that told us explicitly
    how long to wait (see app.extraction.extract_document) -- is then
    applied as a floor, capped at RETRY_AFTER_MAX: jitter may extend the
    delay past what the server asked for, but must never shorten it
    below that.
    """
    base = min(BACKOFF_MAX, BACKOFF_BASE * 2 ** (attempts - 1))
    delay = base / 2 + random.uniform(0, base / 2)
    floor = min(retry_after, RETRY_AFTER_MAX) if retry_after else 0.0
    return max(delay, floor)


def pending_job_stmt(now: datetime, document_id: uuid.UUID | None = None):
    """The claim query, factored out so tests can run it directly against
    real, independently-committed sessions to exercise FOR UPDATE SKIP
    LOCKED under genuine concurrency (something a single-connection
    savepoint session can't do).

    document_id narrows the scan to a single document's jobs. Production
    code never passes it (a worker should claim whatever is oldest); it
    exists so tests running against the shared dev database can scope
    themselves to rows they created instead of racing/corrupting a
    stray real pending job left over from manual API use.
    """
    stmt = select(Job).where(
        Job.state == "pending",
        (Job.run_after.is_(None)) | (Job.run_after <= now),
    )
    if document_id is not None:
        stmt = stmt.where(Job.document_id == document_id)
    return stmt.order_by(Job.created_at).with_for_update(skip_locked=True).limit(1)


async def claim_job(session: AsyncSession, document_id: uuid.UUID | None = None) -> Job | None:
    """Claim the oldest eligible pending job, if any, and commit the
    claim. Uses FOR UPDATE SKIP LOCKED so concurrent workers scanning the
    same table never block on or double-claim a row.
    """
    now = datetime.now(UTC)
    job = (await session.execute(pending_job_stmt(now, document_id))).scalars().first()
    if job is None:
        return None

    job.state = "processing"
    job.started_at = now
    job.attempts += 1

    document = await session.get(Document, job.document_id)
    if document is not None:
        document.status = "processing"

    await session.commit()
    logger.info("claimed job %s (attempt %d)", job.id, job.attempts)
    return job


async def complete_job(session: AsyncSession, job: Job) -> None:
    """Mark a job done after its handler succeeded.

    NOTE: unconditional UPDATE, not compare-and-set. Safe today because
    exactly one worker process ever exists (see module docstring) and
    nothing else transitions a job out of 'processing'. If a reaper/
    timeout or multiple concurrent writers ever land, this (and
    fail_job) should become `WHERE id = :id AND state = 'processing'` so
    a stale completion can't clobber a job another writer already
    reclaimed.
    """
    job.state = "done"
    job.finished_at = datetime.now(UTC)
    await session.commit()
    logger.info("completed job %s", job.id)


async def fail_job(session: AsyncSession, job: Job, exc: Exception) -> None:
    """Handle a failed job: requeue with backoff if attempts remain,
    otherwise mark it permanently failed (and fail its document).

    NonRetryableExtractionError (see app/extraction.py) skips the
    requeue path entirely, regardless of attempts remaining: it marks a
    deterministic failure (bad mime, unpriced model, a refusal, output
    truncation) where re-running the identical job would produce the
    identical outcome, so retrying would only spend more money for the
    same result. Every other exception keeps the existing
    requeue-with-backoff-until-MAX_ATTEMPTS behavior.

    Same compare-and-set caveat as complete_job applies here.
    """
    job_id = job.id
    non_retryable = isinstance(exc, NonRetryableExtractionError)

    # Roll back first, for two reasons: (1) a handler that failed with a
    # DB error leaves this session's transaction aborted -- any further
    # statement, including our own commit below, would raise
    # InFailedSqlTransactionError -- and (2) a handler that failed after
    # partial, uncommitted writes shouldn't have those writes flushed
    # alongside the requeue/failure bookkeeping. Rollback may expire
    # `job`'s attributes, and AsyncSession can't do the implicit
    # lazy-load refresh a sync Session would do transparently, so
    # re-fetch it fresh afterward rather than touching the stale object.
    await session.rollback()
    job = await session.get(Job, job_id)
    if job is None:
        logger.warning("job %s vanished during failure handling", job_id)
        return

    error_tail = "".join(traceback.format_exception(exc))[-2000:]

    if not non_retryable and job.attempts < MAX_ATTEMPTS:
        job.state = "pending"
        job.last_error = error_tail
        # retry_after_seconds is only ever set on ExtractionError (see
        # app.extraction) -- getattr defaults to None for any other
        # exception type, same as _backoff_delay's own `retry_after or
        # 0.0` fallback when there's nothing to floor against.
        retry_after = getattr(exc, "retry_after_seconds", None)
        job.run_after = datetime.now(UTC) + timedelta(
            seconds=_backoff_delay(job.attempts, retry_after)
        )
        await session.commit()
        logger.info(
            "requeued job %s after failure (attempt %d/%d, run_after=%s)",
            job.id,
            job.attempts,
            MAX_ATTEMPTS,
            job.run_after,
        )
    else:
        job.state = "failed"
        job.last_error = error_tail
        job.finished_at = datetime.now(UTC)
        document = await session.get(Document, job.document_id)
        refused = isinstance(exc, ModelRefusalError)
        if document is not None:
            document.status = "refused" if refused else "failed"
        await session.commit()
        if refused:
            logger.info("failed job %s permanently (model refusal)", job.id)
        elif non_retryable:
            logger.info(
                "failed job %s permanently (non-retryable: %s)", job.id, type(exc).__name__
            )
        else:
            logger.info("failed job %s permanently after %d attempts", job.id, job.attempts)


async def reclaim_orphaned_jobs(session: AsyncSession) -> int:
    """Reset any job stuck in 'processing' back to 'pending'. Meant to be
    called exactly once, at startup, before the poll loop begins -- see
    the "Consequence" paragraph in the module docstring for why this is
    only safe in a single-worker deployment.
    """
    stmt = select(Job).where(Job.state == "processing")
    jobs = (await session.execute(stmt)).scalars().all()
    for job in jobs:
        job.state = "pending"
    if jobs:
        await session.commit()
        logger.info("reclaimed %d orphaned processing job(s) at startup", len(jobs))
    return len(jobs)


async def run_once(
    handler: Handler = process_document_job, document_id: uuid.UUID | None = None
) -> bool:
    """Claim and run at most one job. Returns True if a job was claimed
    (regardless of success/failure), False if there was no work to do.

    document_id is test-only scoping -- see pending_job_stmt.
    """
    async with async_session_maker() as session:
        job = await claim_job(session, document_id=document_id)
        if job is None:
            return False
        job_id = job.id

    async with async_session_maker() as session:
        job = await session.get(Job, job_id)
        if job is None:
            logger.warning("job %s vanished before handler could run", job_id)
            return True
        try:
            await handler(session, job)
        except asyncio.CancelledError:
            # The process is shutting down mid-handler. Requeue rather
            # than fail: this wasn't the job's fault, so it shouldn't
            # burn one of its MAX_ATTEMPTS. Roll back first for the same
            # reason fail_job does -- don't flush partial handler writes
            # alongside the requeue -- and re-fetch afterward since
            # rollback may have expired `job`.
            logger.info("job %s cancelled mid-handler, requeuing", job.id)
            await session.rollback()
            job = await session.get(Job, job_id)
            if job is not None:
                job.state = "pending"
                job.attempts = max(job.attempts - 1, 0)
                await session.commit()
            raise
        except Exception as exc:
            logger.exception("job %s handler failed", job.id)
            await fail_job(session, job, exc)
        else:
            await complete_job(session, job)

    return True


async def run_worker(handler: Handler = process_document_job) -> None:
    """Poll indefinitely, processing one job at a time.

    Per-iteration exceptions from run_once (e.g. a handler that leaves
    the session poisoned, or any other unexpected failure) are caught,
    logged, and treated like "no work found" -- the loop backs off for
    one poll interval and keeps going. A single bad job must never take
    the whole process down.

    Stops on CancelledError, which is not caught by the `except
    Exception` above (CancelledError is a BaseException, not an
    Exception). The expected trigger is `asyncio.run(main())` seeing a
    KeyboardInterrupt: asyncio.run's own cleanup then cancels all
    outstanding tasks, including the one running this coroutine, before
    the process exits -- we don't install a signal handler ourselves.
    """
    settings = get_settings()
    logger.info("worker started (poll_interval=%.1fs)", settings.worker_poll_interval)
    try:
        while True:
            try:
                claimed = await run_once(handler)
            except Exception:
                logger.exception("run_once failed unexpectedly; backing off and continuing")
                claimed = False
            if not claimed:
                await asyncio.sleep(settings.worker_poll_interval)
    except asyncio.CancelledError:
        logger.info("worker stopping (cancelled)")
        raise


async def main() -> None:
    async with async_session_maker() as session:
        await reclaim_orphaned_jobs(session)

    task = asyncio.ensure_future(run_worker())
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        await engine.dispose()
        logger.info("worker stopped, engine disposed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("worker interrupted, shutting down")
