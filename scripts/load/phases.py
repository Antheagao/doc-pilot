"""Phases of a load run: ingest, worker scaling, and the read ramp.

A phase is a plan (what to vary, how often, how long) and a driver (what moves
the stack: reset it, scale the workers, run Locust, query the database). The
plan and the arithmetic here are pure and tested without Docker; load.py supplies
the driver that talks to the load project.

What each phase measures, and how:

- ingest: with no workers running, upload a fixed number of corpus documents at
  stepped concurrency -- the API's upload throughput and latency alone, nothing
  consuming the queue.
- worker scaling: for N workers, upload a backlog with none running, start N
  workers, and let them drain it. Throughput is the jobs completed between the
  10% and the 90% mark of the backlog divided by the time between those two
  completions, so the staggered start of the workers and the thin tail of the
  last jobs do not count (steady_window). Completion times are the jobs' own
  `finished_at`, read from the database, not a sampled approximation.
- read ramp: tenants of fixed size, then a fixed mix of reads at rising numbers
  of users, a minute each. A step fails when more than 1% of its requests fail
  or the /search p95 passes a 1 s SLO written down before any run; the ramp
  stops at the first failing step, and the knee is the last step that passed.

Every measured configuration repeats `repetitions` times, interleaved (all the
worker counts, then again) so slow drift of a laptop spreads over all of them.
A phase is marked invalid, not discarded, when jobs did not complete, and, for
the phases the generator drives (an ingest step, a read step), when its p95 CPU
over the measured window passed 80% of a core (it, and not the server, may be
the limit) or docker stats gave no figures for that window; a read step is also
invalid if a cookie user failed to sign in (it ran at lower concurrency) or the
step was cut short. The scaling drain runs no Locust, so its generator figures
are recorded and never gate. The generator is a closed loop: the users number is
the concurrency, with no think time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import artifact
import samplers
import workload
from samplers import HARNESS_TAG, PhaseTelemetry

PHASES = ("ingest", "scaling", "ramp")

# Pre-registered before any run: a read step fails past either.
SLO_SEARCH_P95_MS = 1000.0
MAX_ERROR_RATE = 0.01

# The completion marks that bound the steady part of a drain, in percent.
WINDOW_LOW_PCT = 10
WINDOW_HIGH_PCT = 90

# Every job a document makes is an extract and then an index job.
JOBS_PER_DOCUMENT = 2
DRAIN_POLL_SECONDS = 1.0


class PhaseError(Exception):
    """A phase could not run (Locust failed, the queue did not drain)."""


# --- the plan ---------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Everything a run varies, recorded in the artifact (and digested into the
    scenario, so a different plan is a different scenario)."""

    preset: str
    repetitions: int
    # True: each repetition of a phase starts from a reset database. False (the
    # quick preset and the smoke) shares one stack between phases and is never
    # publishable.
    fresh_stack: bool
    phases: tuple[str, ...]
    ingest_users: tuple[int, ...]
    ingest_documents: int
    workers: tuple[int, ...]
    backlog_documents: int
    backlog_users: int
    ramp_users: tuple[int, ...]
    step_seconds: int
    warmup_seconds: int
    # (tenant slug, documents) to upload to each tenant before the ramp; empty
    # uploads none.
    tenant_documents: tuple[tuple[str, int], ...]
    populate_users: int
    populate_workers: int
    read_workers: int
    drain_timeout_seconds: int
    # How long a read step waits for the queue (the judge jobs the last step's
    # answers queued) to empty before it starts. Under the replay profile one
    # worker may never catch up, so this is a wait, not a requirement; the jobs
    # still open are recorded.
    step_settle_seconds: int
    # JSON of the user classes and the read mix; None is the defaults.
    user_weights: tuple[tuple[str, int], ...] | None = None
    read_mix: tuple[tuple[str, int], ...] | None = None
    # Question types the agent questions leave out, besides the ones that expect
    # no answer (workload.EXCLUDED_QUESTION_TYPES).
    exclude_question_types: tuple[str, ...] = ()


def full_plan() -> Plan:
    """The measurement of the plan: three repetitions of every configuration."""
    return Plan(
        preset="full",
        repetitions=3,
        fresh_stack=True,
        phases=PHASES,
        ingest_users=(1, 2, 4, 8),
        ingest_documents=100,
        workers=(1, 2, 4, 8),
        backlog_documents=300,
        backlog_users=8,
        ramp_users=(2, 4, 8, 16, 32, 64),
        step_seconds=60,
        warmup_seconds=5,
        tenant_documents=(("load-large", 2000), ("load-medium", 300), ("load-small", 30)),
        populate_users=8,
        populate_workers=4,
        read_workers=1,
        drain_timeout_seconds=3600,
        step_settle_seconds=120,
    )


def quick_plan() -> Plan:
    """A dry run of every phase in a few minutes: one repetition, small
    numbers, one stack shared by the phases. For `--quick`; never published."""
    return replace(
        full_plan(),
        preset="quick",
        repetitions=1,
        fresh_stack=False,
        ingest_users=(2,),
        ingest_documents=10,
        workers=(1,),
        backlog_documents=20,
        backlog_users=4,
        ramp_users=(2, 4),
        step_seconds=15,
        warmup_seconds=3,
        # 25 each: the whole corpus, so every agent question has its document.
        tenant_documents=(("load-large", 25), ("load-medium", 25), ("load-small", 25)),
        populate_users=4,
        populate_workers=2,
        drain_timeout_seconds=600,
        step_settle_seconds=30,
    )


def smoke_plan() -> Plan:
    """The CI smoke: 10 uploads with no workers, then 2 workers draining them,
    then 20 s of reads at 2 users (one key user and one cookie user of the large
    tenant, and every endpoint weighted alike so a short run still meets all of
    them)."""
    return replace(
        full_plan(),
        preset="smoke",
        repetitions=1,
        fresh_stack=False,
        phases=("scaling", "ramp"),
        workers=(2,),
        backlog_documents=10,
        backlog_users=2,
        ramp_users=(2,),
        step_seconds=20,
        warmup_seconds=2,
        tenant_documents=(),
        read_workers=2,
        drain_timeout_seconds=300,
        step_settle_seconds=60,
        user_weights=(("load-large:key", 1), ("load-large:cookie", 1)),
        read_mix=(("search", 1), ("ask", 1), ("ask_stream", 1), ("documents", 1)),
        # "Which store's receipt came to $425.58" needs document 018 and the
        # euro one 024; ten uploads from the start of the corpus hold neither,
        # and an answer with no citation is an error.
        exclude_question_types=("reverse_lookup",),
    )


def plan_description(plan: Plan) -> dict:
    """The plan as JSON-able data for the scenario digest."""
    return asdict(plan)


def weights_json(weights: tuple[tuple[str, int], ...] | None) -> dict[str, int] | None:
    return None if weights is None else dict(weights)


# --- the arithmetic ---------------------------------------------------------


@dataclass(frozen=True)
class SteadyWindow:
    start: float
    end: float
    jobs: int
    seconds: float
    jobs_per_s: float


def steady_window(
    completions: Sequence[float],
    total: int,
    low_pct: int = WINDOW_LOW_PCT,
    high_pct: int = WINDOW_HIGH_PCT,
) -> SteadyWindow | None:
    """Throughput of a drain between its 10% and 90% marks.

    `completions` are the times (seconds) at which jobs finished and `total` is
    how many the backlog should produce. The window runs from the completion
    at rank ceil(10% of total) to the one at rank ceil(90% of total); the jobs
    counted are those completed after the first and up to the second, so the
    rate is (k90 - k10) / (t90 - t10). None when fewer than `total` jobs
    completed (the backlog did not drain), or the window is empty."""
    if total <= 0 or len(completions) < total:
        return None
    ordered = sorted(completions)[:total]
    first = -(-low_pct * total // 100)
    last = -(-high_pct * total // 100)
    if last <= first:
        return None
    start, end = ordered[first - 1], ordered[last - 1]
    if end <= start:
        return None
    return SteadyWindow(start, end, last - first, end - start, (last - first) / (end - start))


@dataclass(frozen=True)
class StepVerdict:
    ok: bool
    error_rate: float
    search_p95_ms: float | None
    reasons: tuple[str, ...]


def judge_step(
    endpoints: dict[str, dict],
    slo_ms: float = SLO_SEARCH_P95_MS,
    max_error_rate: float = MAX_ERROR_RATE,
) -> StepVerdict:
    """Whether a read step passed: at most 1% of its read requests failed and
    the /search p95 is at most the SLO. Requests of the sign-in are not reads
    and are left out; a step with no /search to judge fails, because the SLO
    cannot be shown to hold."""
    reads = [stats for name, stats in endpoints.items() if name in workload.READ_ENDPOINTS]
    count = sum(stats["count"] for stats in reads)
    errors = sum(stats["errors"] for stats in reads)
    error_rate = errors / count if count else 1.0
    latency = (endpoints.get("search") or {}).get("latency_ms")
    p95 = latency["p95"] if latency else None
    reasons = []
    if not count:
        reasons.append("no read requests completed")
    elif error_rate > max_error_rate:
        reasons.append(f"error rate {error_rate:.1%} is over {max_error_rate:.0%}")
    if p95 is None:
        reasons.append("no successful /search to judge the SLO by")
    elif p95 > slo_ms:
        reasons.append(f"/search p95 {p95:.0f} ms is over the {slo_ms:.0f} ms SLO")
    return StepVerdict(not reasons, error_rate, p95, tuple(reasons))


@dataclass(frozen=True)
class StepResult:
    users: int
    verdict: StepVerdict
    valid: bool


def ramp(users: Sequence[int], run_step: Callable[[int], StepResult]) -> tuple[list[StepResult], int | None]:
    """Run the steps in order and stop at the first that fails (the step one
    past the knee, kept for the report) or is invalid (a saturated generator
    makes every higher step no better). Returns the results and the knee: the
    users of the last step that passed, None if the first already failed."""
    results: list[StepResult] = []
    knee = None
    for count in users:
        result = run_step(count)
        results.append(result)
        if not result.valid or not result.verdict.ok:
            break
        knee = count
    return results, knee


def step_window(samples: list[dict], warmup_seconds: float) -> tuple[float, float]:
    """The part of a read step that is measured, on the samples' own clock: from
    `warmup_seconds` after the first request (users are still signing in and the
    first requests meet cold caches) to the last request that started."""
    first = min(sample["start"] for sample in samples)
    last = max(sample["start"] for sample in samples)
    return first + warmup_seconds, last


# A read step must have measured at least this share of the time it was meant
# to (step_seconds less the warmup), or it was cut short.
MIN_WINDOW_SHARE = 0.8


# --- the driver -------------------------------------------------------------


@dataclass(frozen=True)
class LocustSpec:
    """One Locust run: which workload, how many users, for how long (None for
    an ingest that ends at its upload quota)."""

    tag: str
    workload: str
    users: int
    run_seconds: int | None = None
    upload_quota: int = 0
    user_weights: dict[str, int] | None = None
    read_mix: dict[str, int] | None = None
    exclude_question_types: tuple[str, ...] = ()


class Driver(Protocol):
    """What the phases need from the stack. load.py implements it."""

    samples_dir: Path
    # Seconds the stack's clock (the db container's, which stamps every sample
    # and job) is ahead of the host's, measured when the stack started.
    clock_offset_s: float

    def prepare(self, workers: int, *, fresh: bool) -> None:
        """A seeded, preflighted stack with `workers` workers: reset from empty
        volumes when `fresh`, else the stack that is already up (started fresh
        if there is none)."""

    def scale_workers(self, count: int) -> None:
        """Set the number of worker containers and re-run the no-spend check."""

    def run_locust(self, spec: LocustSpec, destination: Path) -> None:
        """Run Locust in the loadgen container and leave its raw samples at
        `destination`."""

    def query(self, sql: str) -> str:
        """psql tuples of a statement run as the owner in the load database."""

    def telemetry(self) -> AbstractContextManager[PhaseTelemetry]:
        """The samplers around a block (samplers.Telemetry.phase)."""

    def check_spend(self) -> None:
        """Assert, from the database, that nothing was spent and only the
        scripted model was recorded; raises otherwise."""


# --- database questions -----------------------------------------------------

QUEUE_OPEN_SQL = (
    f"{HARNESS_TAG} select count(*) from jobs where state in ('pending', 'processing')"
)
FAILED_JOBS_SQL = f"{HARNESS_TAG} select count(*) from jobs where state = 'failed'"
# Jobs still to run, counting the index job every pending extract job will add.
EXPECTED_JOBS_SQL = (
    f"{HARNESS_TAG} select "
    "coalesce(sum(case when kind = 'extract' then 2 when kind = 'index' then 1 else 0 end), 0) "
    "from jobs where state = 'pending'"
)
DONE_JOBS_SQL = f"{HARNESS_TAG} select count(*) from jobs where state = 'done' and kind in ('extract', 'index')"
COMPLETIONS_SQL = (
    f"{HARNESS_TAG} select extract(epoch from finished_at) from jobs "
    "where state = 'done' and kind in ('extract', 'index') order by finished_at, id"
)
# Fresh planner statistics after a bulk load: thousands of rows arrive in
# seconds, faster than autovacuum's analyze notices them (up to a minute), and a
# query planned on stale estimates can be ten times slower. Run as the owner
# before any measured window, so every run is measured on the same plans.
ANALYZE_SQL = f"{HARNESS_TAG} analyze"
TENANT_SIZES_SQL = (
    f"{HARNESS_TAG} select o.slug, "
    "(select count(*) from documents d where d.organization_id = o.id), "
    "(select count(*) from document_chunks c where c.organization_id = o.id) "
    "from organizations o where o.slug like 'load-%' order by o.slug"
)


def wait_until_drained(
    driver: Driver, timeout_seconds: float, sleep: Callable[[float], None] | None = None
) -> None:
    """Block until no job is pending or processing. PhaseError past the
    timeout: a queue that does not drain means a stuck worker or none at all."""
    sleep = sleep or time.sleep
    deadline = time.monotonic() + timeout_seconds
    while int(driver.query(QUEUE_OPEN_SQL).strip()) > 0:
        if time.monotonic() > deadline:
            raise PhaseError(f"the job queue did not drain within {timeout_seconds:.0f} s")
        sleep(DRAIN_POLL_SECONDS)


def settle_queue(driver: Driver, seconds: float, sleep: Callable[[float], None] | None = None) -> int:
    """Wait up to `seconds` for the queue to empty and return how many jobs are
    still open (0 when it emptied). Unlike wait_until_drained it never fails."""
    sleep = sleep or time.sleep
    deadline = time.monotonic() + seconds
    while (open_jobs := int(driver.query(QUEUE_OPEN_SQL).strip())) > 0:
        if time.monotonic() > deadline:
            break
        sleep(DRAIN_POLL_SECONDS)
    return open_jobs


def tenant_sizes(driver: Driver) -> dict[str, dict[str, int]]:
    """Documents and chunks per load tenant, as the owner sees them."""
    sizes = {}
    for line in driver.query(TENANT_SIZES_SQL).splitlines():
        slug, documents, chunks = line.split("|")
        sizes[slug] = {"documents": int(documents), "chunks": int(chunks)}
    return sizes


# --- a phase record ---------------------------------------------------------


def phase_record(
    name: str,
    repetition: int,
    params: dict,
    results: dict,
    telemetry: PhaseTelemetry,
    started: float,
    ended: float,
    endpoints: dict | None = None,
    invalid_reasons: Sequence[str] = (),
    generator_window: tuple[float, float] | None = None,
) -> dict:
    """One entry of the artifact's `phases`: what was asked, what was measured,
    what the samplers saw, and whether it can be believed.

    `generator_window` (host clock seconds) is the measured window of a phase
    Locust drives. The generator's CPU is taken over it and gates the phase:
    a p95 past 80% of a core, or no figures at all, makes it invalid. Without a
    window (the scaling drain) the CPU of the whole span is recorded and gates
    nothing."""
    reasons = list(invalid_reasons)
    if generator_window is None:
        cpu = samplers.generator_cpu(telemetry.result["container_series"])
        gated = False
    else:
        cpu = samplers.generator_cpu_in_window(telemetry.container_samples, *generator_window)
        gated = True
        if cpu is None:
            reasons.append("docker stats gave no generator CPU figures for the measured window")
        elif cpu["p95_pct"] > artifact.GENERATOR_CPU_LIMIT_PCT:
            reasons.append(
                f"the generator's p95 CPU was {cpu['p95_pct']:.0f}% of a core "
                f"(max {cpu['max_pct']:.0f}%, limit {artifact.GENERATOR_CPU_LIMIT_PCT:.0f}%)"
            )
    return {
        "name": name,
        "repetition": repetition,
        "params": params,
        "started_at": artifact.utc_iso(started),
        "ended_at": artifact.utc_iso(ended),
        "valid": not reasons,
        "invalid_reasons": reasons,
        "results": results,
        "endpoints": endpoints or {},
        "generator_cpu": None if cpu is None else {**cpu, "gated": gated},
        **telemetry.result,
    }


@dataclass
class RunResult:
    phases: list[dict]
    sample_files: list[Path]


# --- ingest -----------------------------------------------------------------


def upload_step(
    driver: Driver,
    tag: str,
    users: int,
    documents: int,
    weights: dict[str, int] | None,
    sample_files: list[Path],
) -> dict:
    """Upload `documents` corpus documents with `users` concurrent users and
    summarize: the span from the first request to the end of the last, and the
    exact statistics of every request."""
    destination = driver.samples_dir / f"{tag}.jsonl"
    spec = LocustSpec(tag, "ingest", users, upload_quota=documents, user_weights=weights)
    driver.run_locust(spec, destination)
    sample_files.append(destination)
    samples = artifact.read_samples(destination)
    if not samples:
        raise PhaseError(f"{tag}: Locust recorded no requests")
    start = min(sample["start"] for sample in samples)
    end = max(sample["start"] + sample["ms"] / 1000 for sample in samples)
    seconds = end - start
    uploaded = sum(1 for sample in samples if sample["name"] == "upload" and sample["ok"])
    return {
        "users": users,
        "documents": documents,
        "uploaded": uploaded,
        "start": start,
        "end": end,
        "seconds": seconds,
        "uploads_per_s": uploaded / seconds if seconds > 0 else None,
        "samples_file": destination.name,
        "endpoints": artifact.endpoint_stats(samples, seconds),
    }


def run_ingest(driver: Driver, plan: Plan, repetition: int, sample_files: list[Path]) -> list[dict]:
    """One record per concurrency step, with no worker running."""
    driver.prepare(0, fresh=plan.fresh_stack)
    records = []
    for step, users in enumerate(plan.ingest_users, start=1):
        tag = f"ingest-r{repetition}-u{users}"
        with driver.telemetry() as telemetry:
            started = time.time()
            result = upload_step(
                driver, tag, users, plan.ingest_documents, weights_json(plan.user_weights), sample_files
            )
            ended = time.time()
        endpoints = result.pop("endpoints")
        reasons = []
        if result["uploaded"] != plan.ingest_documents:
            reasons.append(f"{result['uploaded']} of {plan.ingest_documents} uploads succeeded")
        params = {"users": users, "documents": plan.ingest_documents, "step": step}
        window = (result["start"] - driver.clock_offset_s, result["end"] - driver.clock_offset_s)
        records.append(
            phase_record(
                "ingest", repetition, params, result, telemetry, started, ended, endpoints, reasons, window
            )
        )
    driver.check_spend()
    return records


# --- worker scaling ---------------------------------------------------------


def run_scaling(driver: Driver, plan: Plan, workers: int, repetition: int, sample_files: list[Path]) -> dict:
    """Upload a backlog with no worker running, start `workers` workers, and
    measure the drain."""
    driver.prepare(0, fresh=plan.fresh_stack)
    tag = f"scaling-w{workers}-r{repetition}-backlog"
    backlog = upload_step(
        driver, tag, plan.backlog_users, plan.backlog_documents, weights_json(plan.user_weights), sample_files
    )
    backlog_endpoints = backlog.pop("endpoints")
    driver.query(ANALYZE_SQL)
    expected = int(driver.query(EXPECTED_JOBS_SQL).strip())
    already_done = int(driver.query(DONE_JOBS_SQL).strip())
    with driver.telemetry() as telemetry:
        started = time.time()
        driver.scale_workers(workers)
        wait_until_drained(driver, plan.drain_timeout_seconds)
        ended = time.time()
    completions = [float(line) for line in driver.query(COMPLETIONS_SQL).split()][already_done:]
    failed = int(driver.query(FAILED_JOBS_SQL).strip())
    window = steady_window(completions, expected)
    reasons = []
    if backlog["uploaded"] != plan.backlog_documents:
        reasons.append(f"{backlog['uploaded']} of {plan.backlog_documents} backlog uploads succeeded")
    if window is None:
        reasons.append(f"{len(completions)} of {expected} jobs completed: no steady window")
    if failed:
        reasons.append(f"{failed} job(s) failed")
    results = {
        "workers": workers,
        "backlog_documents": plan.backlog_documents,
        "expected_jobs": expected,
        "completed_jobs": len(completions),
        "failed_jobs": failed,
        "drain_seconds": ended - started,
        "window": None if window is None else asdict(window),
        # Jobs are extract and index jobs, two for each document.
        "jobs_counted": "extract+index",
        "jobs_per_s": None if window is None else window.jobs_per_s,
        "documents_per_s": None if window is None else window.jobs_per_s / JOBS_PER_DOCUMENT,
        "backlog": {**backlog, "endpoints": backlog_endpoints},
    }
    params = {"workers": workers, "backlog_documents": plan.backlog_documents}
    driver.check_spend()
    return phase_record("scaling", repetition, params, results, telemetry, started, ended, None, reasons)


# --- the read ramp ----------------------------------------------------------


def populate_tenants(driver: Driver, plan: Plan, repetition: int, sample_files: list[Path]) -> list[dict]:
    """Upload each tenant's planned number of documents, with the populate
    workers running, and wait for the queue to empty.

    A tenant gets exactly its planned count, walking the corpus from its first
    document, so a tenant of 25 or more holds all 25 and every agent question
    has its document (an answer with no citation is an error). Documents left
    by an earlier phase on a shared stack (the quick preset) are extra, and the
    artifact records the sizes the tenants really had. An upload that fails
    ends the run: the tenant would be short of the plan."""
    steps = []
    if not plan.tenant_documents:
        return steps
    driver.scale_workers(plan.populate_workers)
    for slug, documents in plan.tenant_documents:
        weights = {
            workload.user_class_name(slug, "key"): workload.KEY_USER_WEIGHT,
            workload.user_class_name(slug, "cookie"): workload.COOKIE_USER_WEIGHT,
        }
        tag = f"ramp-r{repetition}-populate-{slug}"
        step = upload_step(driver, tag, plan.populate_users, documents, weights, sample_files)
        if step["uploaded"] != documents:
            raise PhaseError(f"{tag}: {step['uploaded']} of {documents} uploads succeeded")
        steps.append({"tenant": slug, **step})
    wait_until_drained(driver, plan.drain_timeout_seconds)
    return steps


def run_ramp(driver: Driver, plan: Plan, repetition: int, sample_files: list[Path]) -> list[dict]:
    """Fill the tenants, then raise the users until a step fails. One record per
    step; the first holds the populate steps in its results."""
    driver.prepare(plan.populate_workers if plan.tenant_documents else plan.read_workers, fresh=plan.fresh_stack)
    setup = populate_tenants(driver, plan, repetition, sample_files)
    driver.scale_workers(plan.read_workers)
    wait_until_drained(driver, plan.drain_timeout_seconds)
    driver.query(ANALYZE_SQL)
    sizes = tenant_sizes(driver)
    records: list[dict] = []

    def run_step(users: int) -> StepResult:
        # Each step starts from an empty queue where it can: the judge jobs the
        # last step's answers queued would otherwise compete with this one's.
        open_at_start = settle_queue(driver, plan.step_settle_seconds)
        tag = f"ramp-r{repetition}-u{users}"
        destination = driver.samples_dir / f"{tag}.jsonl"
        spec = LocustSpec(
            tag,
            "read",
            users,
            run_seconds=plan.step_seconds,
            user_weights=weights_json(plan.user_weights),
            read_mix=weights_json(plan.read_mix),
            exclude_question_types=plan.exclude_question_types,
        )
        with driver.telemetry() as telemetry:
            started = time.time()
            driver.run_locust(spec, destination)
            ended = time.time()
        sample_files.append(destination)
        samples = artifact.read_samples(destination)
        if not samples:
            raise PhaseError(f"{tag}: Locust recorded no requests")
        window_start, last_start = step_window(samples, plan.warmup_seconds)
        measured = artifact.in_window(samples, window_start, last_start)
        # Requests per second over the measured span, to the last completion.
        window_end = max((s["start"] + s["ms"] / 1000 for s in measured), default=last_start)
        seconds = window_end - window_start
        endpoints = artifact.endpoint_stats(measured, seconds)
        reasons = []
        login_errors = sum(1 for s in samples if s["name"] == "login" and not s["ok"])
        if login_errors:
            reasons.append(f"{login_errors} cookie user(s) failed to sign in and ran at lower concurrency")
        wanted = plan.step_seconds - plan.warmup_seconds
        if last_start - window_start < MIN_WINDOW_SHARE * wanted:
            reasons.append(
                f"the step measured {max(last_start - window_start, 0):.1f} s of the {wanted:.0f} s it was meant to"
            )
        verdict = judge_step(endpoints)
        params = {
            "users": users,
            "step_seconds": plan.step_seconds,
            "warmup_seconds": plan.warmup_seconds,
            "workers": plan.read_workers,
            "slo_search_p95_ms": SLO_SEARCH_P95_MS,
            "max_error_rate": MAX_ERROR_RATE,
        }
        results = {
            "verdict": {
                "ok": verdict.ok,
                "error_rate": verdict.error_rate,
                "search_p95_ms": verdict.search_p95_ms,
                "reasons": list(verdict.reasons),
            },
            "window": {"start": window_start, "last_start": last_start, "end": window_end, "seconds": seconds},
            "login_errors": login_errors,
            "samples_file": destination.name,
            "queue_open_at_start": open_at_start,
            "tenants": sizes,
            "setup": setup if not records else [],
        }
        generator_window = (window_start - driver.clock_offset_s, window_end - driver.clock_offset_s)
        record = phase_record(
            "ramp", repetition, params, results, telemetry, started, ended, endpoints, reasons, generator_window
        )
        records.append(record)
        return StepResult(users, verdict, record["valid"])

    _results, knee = ramp(plan.ramp_users, run_step)
    for record in records:
        record["results"]["knee_users"] = knee
    driver.check_spend()
    return records


# --- the run ----------------------------------------------------------------


def run_plan(driver: Driver, plan: Plan, log: Callable[[str], None] = print) -> RunResult:
    """Every selected phase, repetition by repetition. Returns the phase
    records in the order they ran and the raw sample files."""
    records: list[dict] = []
    sample_files: list[Path] = []
    for repetition in range(1, plan.repetitions + 1):
        if "ingest" in plan.phases:
            log(f"ingest, repetition {repetition}")
            records += run_ingest(driver, plan, repetition, sample_files)
        if "scaling" in plan.phases:
            for workers in plan.workers:
                log(f"scaling to {workers} worker(s), repetition {repetition}")
                records.append(run_scaling(driver, plan, workers, repetition, sample_files))
        if "ramp" in plan.phases:
            log(f"read ramp, repetition {repetition}")
            records += run_ramp(driver, plan, repetition, sample_files)
    return RunResult(records, sample_files)


def summarize(phases: list[dict]) -> dict:
    """The headline numbers across repetitions, median with the min-max range,
    from the valid phases only (`invalid` counts the ones left out)."""

    def grouped(name: str, key: Callable[[dict], object], value: Callable[[dict], float | None]) -> dict:
        groups: dict[str, list[float]] = {}
        invalid: dict[str, int] = {}
        for phase in phases:
            if phase["name"] != name:
                continue
            label = str(key(phase))
            if not phase["valid"]:
                invalid[label] = invalid.get(label, 0) + 1
            elif (number := value(phase)) is not None:
                groups.setdefault(label, []).append(number)
        return {
            label: {**artifact.spread(numbers), "invalid": invalid.get(label, 0)}
            for label, numbers in groups.items()
        }

    ramp_knees: dict[str, list[float]] = {}
    for phase in phases:
        if phase["name"] == "ramp" and phase["valid"] and phase["results"].get("knee_users") is not None:
            ramp_knees[str(phase["repetition"])] = [phase["results"]["knee_users"]]
    knees = [values[0] for values in ramp_knees.values()]
    return {
        "ingest_uploads_per_s_by_users": grouped(
            "ingest", lambda p: p["params"]["users"], lambda p: p["results"]["uploads_per_s"]
        ),
        "scaling_jobs_per_s_by_workers": grouped(
            "scaling", lambda p: p["params"]["workers"], lambda p: p["results"]["jobs_per_s"]
        ),
        "scaling_documents_per_s_by_workers": grouped(
            "scaling", lambda p: p["params"]["workers"], lambda p: p["results"].get("documents_per_s")
        ),
        "ramp_knee_users": artifact.spread(knees) if knees else None,
    }
