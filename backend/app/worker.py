"""Postgres SKIP LOCKED job worker.

Run as `python -m app.worker`. Polls the `jobs` table for pending work,
claims one job at a time using `SELECT ... FOR UPDATE SKIP LOCKED`, and
hands it to a pluggable handler.

Concurrency model: this process handles exactly ONE job at a time. To
process jobs in parallel, run more instances of this module — SKIP LOCKED
guarantees they never claim the same job, and scaling is safe provided
WORKER_RECLAIM_AFTER_SECONDS is set (see "Consequence" below). There is no
in-process concurrency here on purpose: it keeps the claim/complete
bookkeeping trivial and lets horizontal scaling do the work.

Claim/commit split: claiming a job (state -> processing) is committed
immediately, *before* the handler runs, rather than holding the row lock
for the duration of the handler. Both job handlers (process_document_job
and process_index_job, see "Job kinds" below) make VLM calls that can take
several seconds; holding
a transaction open that long would tie up a connection and block other
workers from even attempting SKIP LOCKED scans against the table. Once
claimed, a job is "owned" by this process via its state, not via a held
lock.

Job kinds: every job row has a `kind` (app.models.JOB_KIND_*), and
dispatch_job routes it to that kind's handler -- 'extract' runs the VLM
extraction (process_document_job), 'index' transcribes, chunks and embeds
a document for retrieval (app/retrieval/index_job.py), and 'judge' grades
a stored /ask answer (app/evals/online.py). Only an extract
job drives the document's user-visible status (processing / extracted /
failed / refused): a document whose extraction succeeded stays
'extracted' even if its index job later fails, since the extracted data
is still valid -- the failure is on the job row.

Consequence: if this process dies (crash, kill -9, power loss) after
claiming a job but before it finishes, that job is orphaned in
'processing' forever -- there's no lock left to notice. Recovery:
reclaim_orphaned_jobs() runs at startup (see main()) and resets jobs it
finds in 'processing' back to 'pending'.

With WORKER_RECLAIM_AFTER_SECONDS set (the lease), only jobs processing
longer than that are reclaimed -- at startup and every RECLAIM_CHECK_SECONDS
after -- so a starting worker never steals a job another live worker is
working on, and one a dead worker abandoned is picked up within the lease.
That is what makes running several workers (`docker compose up --scale
worker=N`) or overlapping an old and a new worker during a deploy (Cloud Run;
see infra/terraform) safe. SIGTERM (Cloud Run's and docker stop's shutdown
signal) stops the loop claiming new work; the job in hand finishes if the
platform's grace period allows, and is reclaimed if not.

A job claimed MAX_ATTEMPTS times without ever settling as done (its worker
keeps crashing, or it keeps outlasting the lease) is not requeued again by the
sweep: it is failed, with its document or ask run, so a poison job cannot be
billed forever. The lease must exceed the slowest job, since nothing renews it
while a job runs.

With no lease (0, the default and the single-worker dev setup) the startup
sweep resets EVERY 'processing' job, assuming each is a leftover of the
previous run of this same process. Two workers on that setting would steal
each other's in-flight jobs, so scale only with a lease.

A lease can still expire on a live worker whose job runs longer than it, and
the job is then handed to a second one. So settling a job is compare-and-set
(see "Settling"): the first worker's late completion, failure, or
cancellation requeue finds the job no longer its own, and is logged and
dropped.

Settling: complete_job, fail_job and requeue_job update the job row only
WHERE id, state = 'processing' and attempts = the attempt this worker claimed
(carried from the claim like job_id; claim_job increments attempts, so a
reclaimed job's next claim has a different number. Only a cancellation requeue
gives a number back, and only the claim that held it does that). A handler never commits: its content writes are
flushed in the session's one transaction, and the settling UPDATE is in that
same transaction, so they commit together or not at all. If the UPDATE
touches no row the job was reclaimed, and the transaction -- the handler's
writes with it -- is rolled back: a stale worker's rows never land beside the
new claim's. The row lock the UPDATE takes also orders it against a
concurrent reclaim, so exactly one of the two wins.

Tenancy: a job belongs to an organization (jobs.organization_id), and
everything a worker does for it runs as that organization (app/tenancy.py).
The claim is the one step that cannot: it picks the oldest pending job of any
organization, so it runs in a queue_session, which may touch the job queue and
nothing else. The claim then binds the job's organization to its session
before it reads the document, and the handler, complete_job and fail_job run
in a new session bound to the same one.

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
import contextlib
import logging
import random
import signal
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db import async_session_maker, engine, name_process, verify_tenant_isolation
from app.evals.online import process_judge_job
from app.extraction import (
    ModelRefusalError,
    NonRetryableExtractionError,
    check_model_provider,
    process_document_job,
)
from app.models import (
    JOB_KIND_EXTRACT,
    JOB_KIND_INDEX,
    JOB_KIND_JUDGE,
    AskRun,
    Document,
    Job,
)
from app.retrieval.embeddings import warm_up_embedder
from app.retrieval.index_job import process_index_job
from app.telemetry import (
    DOCPILOT_ASK_RUN_ID,
    DOCPILOT_DOCUMENT_ID,
    DOCPILOT_JOB_ATTEMPT,
    DOCPILOT_JOB_ID,
    DOCPILOT_JOB_KIND,
    DOCPILOT_JOB_OUTCOME,
    configure_tracing,
    context_from_traceparent,
    instrument_engine,
    otlp_enabled,
    shutdown_tracing,
    tracer,
    untraced,
)
from app.tenancy import bind_tenant, queue_session, tenant_session

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

# Work a handler wants done only if its job settles as done (a Langfuse score
# for a stored verdict, say): it returns the callable instead of doing the work,
# and _run_handler calls it after the completion commits. A worker whose job was
# reclaimed never calls it, so a stale run leaves no trace outside the database.
AfterDone = Callable[[], None]
Handler = Callable[[AsyncSession, Job], Awaitable[AfterDone | None]]

# How a job was settled. "stale": the job was no longer this worker's claim
# (reclaimed, or already settled), so nothing was written.
JobOutcome = Literal["done", "requeued", "failed", "stale"]

HANDLERS: dict[str, Handler] = {
    JOB_KIND_EXTRACT: process_document_job,
    JOB_KIND_INDEX: process_index_job,
    JOB_KIND_JUDGE: process_judge_job,
}


async def dispatch_job(session: AsyncSession, job: Job) -> AfterDone | None:
    """The default handler: route a job to its kind's handler. An unknown
    kind (a row written by a newer version of the app, say) fails
    permanently rather than retrying -- no amount of retrying teaches this
    process a handler it doesn't have."""
    handler = HANDLERS.get(job.kind)
    if handler is None:
        raise NonRetryableExtractionError(f"no handler for job kind {job.kind!r}")
    return await handler(session, job)


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
    exists so tests can scope themselves to rows they created instead of
    racing/corrupting a pending job left over in the test database by an
    earlier crashed run.
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

    `session` is a queue_session: it finds the job in any organization, then
    binds the job's organization to itself before touching the document, so
    from the first line that reads anything else it is that organization's.
    """
    now = datetime.now(UTC)
    job = (await session.execute(pending_job_stmt(now, document_id))).scalars().first()
    if job is None:
        return None

    job.state = "processing"
    # The database's clock, like the cutoff reclaim_orphaned_jobs compares it
    # with: workers on different hosts never disagree about a lease.
    job.started_at = func.now()
    job.attempts += 1
    await bind_tenant(session, job.organization_id)

    if job.kind == JOB_KIND_EXTRACT:
        document = await session.get(Document, job.document_id)
        if document is not None:
            document.status = "processing"

    await session.commit()
    logger.info("claimed %s job %s (attempt %d)", job.kind, job.id, job.attempts)
    return job


def _still_claimed(job_id: uuid.UUID, attempts: int):
    """The compare-and-set condition on a job: still in 'processing', and still
    the claim this worker made (each claim increments attempts, so a job that
    was reclaimed and claimed again by another worker no longer matches)."""
    return (Job.id == job_id, Job.state == "processing", Job.attempts == attempts)


async def _drop_stale(session: AsyncSession, job_id: uuid.UUID, attempts: int, what: str) -> None:
    """Roll back a worker's transaction -- the handler's writes included -- when
    the job it was settling is no longer its claim."""
    await session.rollback()
    logger.warning(
        "dropped the %s of job %s (attempt %d): it was reclaimed or already settled, "
        "so this worker's writes for it were rolled back",
        what,
        job_id,
        attempts,
    )


async def complete_job(session: AsyncSession, job_id: uuid.UUID, attempts: int) -> JobOutcome:
    """Mark a job done after its handler succeeded, in the same transaction
    that holds the handler's writes (see "Settling" in the module docstring).
    `attempts` is the attempt this worker claimed; if the job is no longer
    processing at that attempt, nothing lands and the outcome is "stale".

    Takes the id, not the Job: a handler's pre-model-call rollback expires
    every object in the session, and AsyncSession cannot lazy-load one
    back, so the job is never touched stale.
    """
    result = await session.execute(
        update(Job)
        .where(*_still_claimed(job_id, attempts))
        .values(state="done", finished_at=datetime.now(UTC))
        .execution_options(synchronize_session="fetch")
    )
    if result.rowcount == 0:
        await _drop_stale(session, job_id, attempts, "completion")
        return "stale"
    await session.commit()
    logger.info("completed job %s", job_id)
    return "done"


async def requeue_job(session: AsyncSession, job_id: uuid.UUID, attempts: int) -> JobOutcome:
    """Put a job this worker claimed back to 'pending' without counting the
    attempt, because the worker is shutting down mid-handler: it wasn't the
    job's fault, so it shouldn't burn one of its MAX_ATTEMPTS. Rolls back
    first for the same reason fail_job does -- don't flush partial handler
    writes alongside the requeue. Compare-and-set like complete_job."""
    await session.rollback()
    result = await session.execute(
        update(Job)
        .where(*_still_claimed(job_id, attempts))
        .values(state="pending", attempts=Job.attempts - 1)
        .execution_options(synchronize_session="fetch")
    )
    if result.rowcount == 0:
        await _drop_stale(session, job_id, attempts, "cancellation requeue")
        return "stale"
    await session.commit()
    return "requeued"


async def _fail_job_subject(
    session: AsyncSession,
    kind: str,
    document_id: uuid.UUID | None,
    ask_run_id: uuid.UUID | None,
    *,
    document_status: str,
    judge_error: str,
) -> None:
    """What a job that failed for good leaves behind: an extract job's document
    takes `document_status` (only an extract job drives it), and a judge job's
    run that was never graded takes `judge_error`, otherwise it stays "queued
    for grading" forever. Writes on `session`; the caller commits."""
    if kind == JOB_KIND_EXTRACT:
        document = await session.get(Document, document_id)
        if document is not None:
            document.status = document_status
    elif kind == JOB_KIND_JUDGE and ask_run_id is not None:
        run = await session.get(AskRun, ask_run_id)
        if run is not None and run.judged_at is None:
            run.judge_error = judge_error
            run.judged_at = datetime.now(UTC)


async def fail_job(
    session: AsyncSession, job_id: uuid.UUID, attempts: int, exc: Exception
) -> JobOutcome:
    """Handle a failed job: requeue with backoff if attempts remain,
    otherwise mark it permanently failed (and fail its document).

    NonRetryableExtractionError (see app/extraction.py) skips the
    requeue path entirely, regardless of attempts remaining: it marks a
    deterministic failure (bad mime, unpriced model, a refusal, output
    truncation) where re-running the identical job would produce the
    identical outcome, so retrying would only spend more money for the
    same result. Every other exception keeps the existing
    requeue-with-backoff-until-MAX_ATTEMPTS behavior.

    Compare-and-set on the claimed `attempts`, like complete_job: a late
    failure from a worker whose job was reclaimed must not requeue or fail
    the new claim, so it returns "stale" and writes nothing. `attempts` is
    also the count the retry decision and the backoff use.
    """
    non_retryable = isinstance(exc, NonRetryableExtractionError)

    # Roll back first, for two reasons: (1) a handler that failed with a
    # DB error leaves this session's transaction aborted -- any further
    # statement, including our own commit below, would raise
    # InFailedSqlTransactionError -- and (2) a handler that failed after
    # partial, uncommitted writes shouldn't have those writes flushed
    # alongside the requeue/failure bookkeeping. Rollback may expire
    # the handler's Job, and AsyncSession can't do the implicit
    # lazy-load refresh a sync Session would do transparently, so the
    # job is only ever touched through the UPDATE below.
    await session.rollback()

    error_tail = "".join(traceback.format_exception(exc))[-2000:]
    requeue = not non_retryable and attempts < MAX_ATTEMPTS
    now = datetime.now(UTC)
    if requeue:
        # retry_after_seconds is only ever set on ExtractionError (see
        # app.extraction) -- getattr defaults to None for any other
        # exception type, same as _backoff_delay's own `retry_after or
        # 0.0` fallback when there's nothing to floor against.
        retry_after = getattr(exc, "retry_after_seconds", None)
        run_after = now + timedelta(seconds=_backoff_delay(attempts, retry_after))
        values = {"state": "pending", "last_error": error_tail, "run_after": run_after}
    else:
        values = {"state": "failed", "last_error": error_tail, "finished_at": now}
    row = (
        await session.execute(
            update(Job)
            .where(*_still_claimed(job_id, attempts))
            .values(**values)
            .returning(Job.kind, Job.document_id, Job.ask_run_id)
            .execution_options(synchronize_session="fetch")
        )
    ).first()
    if row is None:
        await _drop_stale(session, job_id, attempts, "failure")
        return "stale"

    if requeue:
        await session.commit()
        logger.info(
            "requeued job %s after failure (attempt %d/%d, run_after=%s)",
            job_id,
            attempts,
            MAX_ATTEMPTS,
            run_after,
        )
        return "requeued"

    kind, document_id, ask_run_id = row
    refused = isinstance(exc, ModelRefusalError)
    await _fail_job_subject(
        session,
        kind,
        document_id,
        ask_run_id,
        document_status="refused" if refused else "failed",
        judge_error=f"grading failed: {type(exc).__name__}: {exc}"[:500],
    )
    await session.commit()
    if refused:
        logger.info("failed job %s permanently (model refusal)", job_id)
    elif non_retryable:
        logger.info("failed job %s permanently (non-retryable: %s)", job_id, type(exc).__name__)
    else:
        logger.info("failed job %s permanently after %d attempts", job_id, attempts)
    return "failed"


# How often a worker with a reclaim lease looks for abandoned jobs.
RECLAIM_CHECK_SECONDS = 60.0

# jobs.last_error of a job reclaim failed for good; %s is its attempts (a
# Postgres format() template, and the same text for the judge's run).
RECLAIM_EXHAUSTED_ERROR = (
    "reclaimed after %s attempts: the worker that claimed it never settled it "
    "(crashed, or the job outlasted WORKER_RECLAIM_AFTER_SECONDS)"
)


async def reclaim_orphaned_jobs(
    session: AsyncSession, older_than: timedelta | None = None
) -> int:
    """Reset jobs stuck in 'processing' back to 'pending', in every
    organization (`session` is a queue_session), or fail them for good once
    they have used up MAX_ATTEMPTS. Returns how many it settled either way.

    With a lease (`older_than`, WORKER_RECLAIM_AFTER_SECONDS), only jobs that
    started longer ago than that -- the only form that is safe beside another
    live worker. With no lease (older_than=None), every one: meant to be called
    exactly once, at startup, before the poll loop begins, and only in a
    single-worker deployment (see the "Consequence" paragraph in the module
    docstring).

    MAX_ATTEMPTS means "at most that many claims that do not settle as done":
    a job whose worker keeps dying (or that outlasts the lease every time)
    would otherwise be reclaimed, claimed and billed again forever, because
    only a worker that settles a job ever looks at its attempts. A job at the
    cap is failed here with a last_error saying why. The queue scope may write
    only the job queue's columns, so its document (or ask run) is marked
    afterwards, in a session for the job's organization.

    One UPDATE per outcome decides which jobs qualify and settles them, rather
    than loading them and writing each back: a job that completes between a
    select and a write would otherwise be reset to 'pending' and run twice.
    The reclaimed job keeps its attempts, so the next claim's number differs
    from the one its first worker holds and that worker's late settling is
    refused."""
    orphaned = [Job.state == "processing"]
    if older_than is not None:
        # The database's clock on both sides (claim_job stamps started_at with
        # now()): a timedelta binds as an interval.
        cutoff = func.now() - older_than
        orphaned.append(or_(Job.started_at.is_(None), Job.started_at < cutoff))
    exhausted = (
        await session.execute(
            update(Job)
            .where(*orphaned, Job.attempts >= MAX_ATTEMPTS)
            .values(
                state="failed",
                finished_at=func.now(),
                last_error=func.format(RECLAIM_EXHAUSTED_ERROR, Job.attempts),
            )
            .returning(Job.id, Job.attempts, Job.organization_id, Job.kind, Job.document_id, Job.ask_run_id)
            .execution_options(synchronize_session=False)
        )
    ).all()
    requeued = (
        await session.execute(
            update(Job)
            .where(*orphaned, Job.attempts < MAX_ATTEMPTS)
            .values(state="pending")
            .execution_options(synchronize_session=False)
        )
    ).rowcount
    if exhausted or requeued:
        await session.commit()
    if requeued:
        logger.info("reclaimed %d orphaned processing job(s)", requeued)
    for job_id, attempts, organization_id, kind, document_id, ask_run_id in exhausted:
        logger.warning(
            "failed job %s for good at reclaim: it was claimed %d times and never settled "
            "(the worker crashed, or the job outlasted WORKER_RECLAIM_AFTER_SECONDS)",
            job_id,
            attempts,
        )
        async with tenant_session(async_session_maker, organization_id) as subject_session:
            await _fail_job_subject(
                subject_session,
                kind,
                document_id,
                ask_run_id,
                document_status="failed",
                judge_error=f"grading failed: {RECLAIM_EXHAUSTED_ERROR % attempts}"[:500],
            )
            await subject_session.commit()
    return requeued + len(exhausted)


def _reclaim_lease() -> timedelta | None:
    seconds = get_settings().worker_reclaim_after_seconds
    return timedelta(seconds=seconds) if seconds > 0 else None


async def run_once(
    handler: Handler = dispatch_job, document_id: uuid.UUID | None = None
) -> bool:
    """Claim and run at most one job. Returns True if a job was claimed
    (regardless of success/failure), False if there was no work to do.

    document_id is test-only scoping -- see pending_job_stmt.
    """
    # The claim poll is untraced: it runs every poll interval, work or not.
    # A claimed job's own span (below) starts the traced part.
    with untraced():
        async with queue_session(async_session_maker) as session:
            job = await claim_job(session, document_id=document_id)
            if job is None:
                return False
            job_id, organization_id, attempts = job.id, job.organization_id, job.attempts

    async with tenant_session(async_session_maker, organization_id) as session:
        job = await session.get(Job, job_id)
        if job is None:
            logger.warning("job %s vanished before handler could run", job_id)
            return True
        with _job_span(job) as span:
            await _run_handler(session, job, job_id, attempts, handler, span)

    return True


@contextmanager
def _job_span(job: Job) -> Iterator[Span]:
    """The job's span, a child of the context that enqueued it (see
    Job.traceparent) or a new root. The handler's own spans -- model
    calls, embeddings, SQL -- nest under it."""
    attributes = {
        DOCPILOT_JOB_ID: str(job.id),
        DOCPILOT_JOB_KIND: job.kind,
        DOCPILOT_JOB_ATTEMPT: job.attempts,
    }
    # A job targets a document (extract, index) or a stored answer (judge).
    if job.document_id is not None:
        attributes[DOCPILOT_DOCUMENT_ID] = str(job.document_id)
    if job.ask_run_id is not None:
        attributes[DOCPILOT_ASK_RUN_ID] = str(job.ask_run_id)
    with tracer().start_as_current_span(
        f"job {job.kind}",
        context=context_from_traceparent(job.traceparent),
        kind=SpanKind.CONSUMER,
        attributes=attributes,
    ) as span:
        yield span


async def _run_handler(
    session: AsyncSession,
    job: Job,
    job_id: uuid.UUID,
    attempts: int,
    handler: Handler,
    span: Span,
) -> None:
    """Run the handler and settle the job: complete it, requeue/fail it
    via fail_job, or requeue it on cancellation, each compare-and-set on the
    `attempts` this worker claimed. The outcome (done, requeued, failed, or
    stale when the job had been reclaimed) is recorded on the span as
    docpilot.job.outcome, and a handler failure is recorded as the span's
    exception/error status even though it is handled here rather than
    propagated."""
    try:
        after_done = await handler(session, job)
    except asyncio.CancelledError:
        # The process is shutting down mid-handler. Requeue rather
        # than fail: this wasn't the job's fault.
        logger.info("job %s cancelled mid-handler, requeuing", job_id)
        await requeue_job(session, job_id, attempts)
        raise
    except Exception as exc:
        logger.exception("job %s handler failed", job_id)
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:500]))
        span.set_attribute(DOCPILOT_JOB_OUTCOME, await fail_job(session, job_id, attempts, exc))
    else:
        outcome = await complete_job(session, job_id, attempts)
        span.set_attribute(DOCPILOT_JOB_OUTCOME, outcome)
        if outcome == "done" and after_done is not None:
            try:
                after_done()
            except Exception:
                # The job is done and committed; a failed side effect is not its failure.
                logger.exception("job %s: the work after its completion failed", job_id)


async def run_worker(handler: Handler = dispatch_job, stop: asyncio.Event | None = None) -> None:
    """Poll until `stop` is set (never, without one), processing one job
    at a time. Setting `stop` lets the job in hand finish, then returns.
    With a reclaim lease (WORKER_RECLAIM_AFTER_SECONDS), it also reclaims
    abandoned jobs every RECLAIM_CHECK_SECONDS.

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
    stop = stop or asyncio.Event()
    lease = _reclaim_lease()
    next_reclaim = time.monotonic() + RECLAIM_CHECK_SECONDS
    logger.info("worker started (poll_interval=%.1fs)", settings.worker_poll_interval)
    try:
        while not stop.is_set():
            try:
                if lease is not None and time.monotonic() >= next_reclaim:
                    next_reclaim = time.monotonic() + RECLAIM_CHECK_SECONDS
                    with untraced():
                        async with queue_session(async_session_maker) as session:
                            await reclaim_orphaned_jobs(session, older_than=lease)
                claimed = await run_once(handler)
            except Exception:
                logger.exception("run_once failed unexpectedly; backing off and continuing")
                claimed = False
            if not claimed:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_interval)
        logger.info("worker stopping (asked to stop)")
    except asyncio.CancelledError:
        logger.info("worker stopping (cancelled)")
        raise


# How often the heartbeat file is touched. The compose healthcheck calls
# the worker unhealthy once the file is more than six intervals old.
HEARTBEAT_INTERVAL_SECONDS = 10.0


async def _heartbeat_loop(path: Path, interval: float) -> None:
    """Touch `path` every `interval` seconds until cancelled: the worker's
    liveness signal for the compose healthcheck, which reads the file's
    mtime. It runs beside the poll loop rather than inside it, so it keeps
    ticking through a long model call (a chunked PDF can take many
    minutes) and through the embedding model's first download at startup,
    and stops only when the event loop stalls or the process dies. A
    write failure is logged once per outage and never stops the worker."""
    failing = False
    while True:
        try:
            path.touch()
            failing = False
        except OSError:
            if not failing:
                logger.warning("cannot touch worker heartbeat file %s", path, exc_info=True)
            failing = True
        await asyncio.sleep(interval)


async def main() -> None:
    name_process("doc-pilot-worker")
    # Before the heartbeat starts, so a worker whose role skips row-level
    # security never reports healthy.
    await verify_tenant_isolation(engine)
    logger.info(get_settings().model_provider_summary())
    check_model_provider(get_settings())
    heartbeat = asyncio.ensure_future(
        _heartbeat_loop(Path(get_settings().worker_heartbeat_file), HEARTBEAT_INTERVAL_SECONDS)
    )
    if configure_tracing("doc-pilot-worker") and otlp_enabled():
        instrument_engine(engine)
        logger.info("tracing enabled (OTLP export)")

    with untraced():
        async with queue_session(async_session_maker) as session:
            await reclaim_orphaned_jobs(session, older_than=_reclaim_lease())

    # Load the embedding model before claiming work, so the first index
    # job doesn't pay for it (tracing showed a cold first `embeddings`
    # span taking seconds inside the job).
    await run_in_threadpool(warm_up_embedder)

    # SIGTERM -- how Cloud Run and `docker stop` end a container -- stops
    # the loop claiming new work instead of killing the job in hand.
    stop = asyncio.Event()
    with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)

    task = asyncio.ensure_future(run_worker(stop=stop))
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        heartbeat.cancel()
        await engine.dispose()
        shutdown_tracing()
        logger.info("worker stopped, engine disposed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("worker interrupted, shutting down")
