"""Tests for the phase arithmetic and the phase flow (run: python -m pytest
../scripts -q). No Docker: a fake driver plays the stack."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import artifact
import phases
import pytest
from samplers import ContainerSample, PhaseTelemetry, fold_container_samples

# --- the steady window ----------------------------------------------------------------


def uniform(count: int, gap: float = 1.0, start: float = 0.0) -> list[float]:
    return [start + gap * i for i in range(count)]


def test_a_steady_drain_has_its_own_rate() -> None:
    # 100 jobs, one a second: marks at the 10th (t=9) and the 90th (t=89).
    window = phases.steady_window(uniform(100), 100)
    assert window == phases.SteadyWindow(start=9.0, end=89.0, jobs=80, seconds=80.0, jobs_per_s=1.0)


def test_a_slow_start_and_a_thin_tail_are_left_out() -> None:
    """Workers that start late and a last job that straggles do not count: only
    the 10% to 90% marks do."""
    steady = uniform(80, gap=0.1, start=50.0)  # 10 jobs/s from t=50
    completions = uniform(10, gap=5.0) + steady + [500.0, 900.0] + uniform(8, gap=0.1, start=60.0)
    window = phases.steady_window(completions, 100)
    assert window is not None
    ordered = sorted(completions)
    assert window.start == ordered[9] and window.end == ordered[89]
    assert window.jobs == 80
    assert window.jobs_per_s == pytest.approx(80 / (ordered[89] - ordered[9]))


def test_the_marks_round_up() -> None:
    # 25 jobs: 10% -> rank ceil(2.5) = 3, 90% -> rank ceil(22.5) = 23.
    window = phases.steady_window(uniform(25), 25)
    assert window is not None and (window.start, window.end, window.jobs) == (2.0, 22.0, 20)


def test_an_unfinished_drain_has_no_window() -> None:
    assert phases.steady_window(uniform(99), 100) is None
    assert phases.steady_window([], 0) is None


def test_extra_completions_beyond_the_backlog_are_ignored() -> None:
    assert phases.steady_window(uniform(150), 100) == phases.steady_window(uniform(100), 100)


def test_a_window_with_no_elapsed_time_has_no_rate() -> None:
    assert phases.steady_window([5.0] * 20, 20) is None  # every job finished at once
    assert phases.steady_window([1.0], 1) is None  # both marks are the same job


def test_unsorted_completions_are_sorted() -> None:
    shuffled = [3.0, 1.0, 2.0, 5.0, 4.0, 10.0, 9.0, 8.0, 7.0, 6.0]
    assert phases.steady_window(shuffled, 10) == phases.steady_window(sorted(shuffled), 10)


# --- the stop rule ---------------------------------------------------------------------


def read_stats(count: int = 1000, errors: int = 0, search_p95: float | None = 100.0) -> dict:
    latency = None if search_p95 is None else {"p95": search_p95}
    return {
        "search": {"count": count, "errors": errors, "latency_ms": latency},
        "ask": {"count": 100, "errors": 0, "latency_ms": {"p95": 5000.0}},
        "login": {"count": 3, "errors": 3, "latency_ms": None},  # not a read
    }


def test_a_step_that_meets_both_limits_passes() -> None:
    verdict = phases.judge_step(read_stats())
    assert verdict.ok and verdict.reasons == () and verdict.search_p95_ms == 100.0


def test_exactly_one_percent_errors_and_exactly_the_slo_still_pass() -> None:
    # 11 errors of 1100 reads is exactly 1%.
    assert phases.judge_step(read_stats(count=1000, errors=11)).ok
    assert phases.judge_step(read_stats(search_p95=1000.0)).ok


def test_more_than_one_percent_errors_fails_the_step() -> None:
    verdict = phases.judge_step(read_stats(count=1000, errors=12))
    assert not verdict.ok and "error rate" in verdict.reasons[0]


def test_a_search_p95_over_the_slo_fails_the_step() -> None:
    verdict = phases.judge_step(read_stats(search_p95=1000.1))
    assert not verdict.ok and "SLO" in verdict.reasons[0]


def test_logins_are_not_counted_as_read_errors() -> None:
    assert phases.judge_step(read_stats()).error_rate == 0.0


def test_a_step_with_nothing_to_judge_fails() -> None:
    assert not phases.judge_step(read_stats(search_p95=None)).ok
    verdict = phases.judge_step({})
    assert not verdict.ok and verdict.error_rate == 1.0 and len(verdict.reasons) == 2


def step_result(users: int, ok: bool = True, valid: bool = True) -> phases.StepResult:
    return phases.StepResult(users, phases.StepVerdict(ok, 0.0, 1.0, () if ok else ("slow",)), valid)


def test_the_ramp_stops_at_the_first_failing_step_and_keeps_it() -> None:
    ran: list[int] = []

    def run_step(users: int) -> phases.StepResult:
        ran.append(users)
        return step_result(users, ok=users < 8)

    results, knee = phases.ramp([2, 4, 8, 16, 32, 64], run_step)
    assert ran == [2, 4, 8]  # 16 and above never run
    assert [r.users for r in results] == [2, 4, 8]
    assert knee == 4  # the last step that met both limits


def test_a_ramp_whose_first_step_fails_has_no_knee() -> None:
    results, knee = phases.ramp([2, 4], lambda users: step_result(users, ok=False))
    assert len(results) == 1 and knee is None


def test_a_ramp_that_never_fails_runs_every_step() -> None:
    results, knee = phases.ramp([2, 4, 8], lambda users: step_result(users))
    assert len(results) == 3 and knee == 8


def test_a_saturated_generator_ends_the_ramp_without_moving_the_knee() -> None:
    results, knee = phases.ramp(
        [2, 4, 8, 16], lambda users: step_result(users, valid=users < 8)
    )
    assert [r.users for r in results] == [2, 4, 8]
    assert knee == 4


def test_the_measured_part_of_a_step_starts_after_the_warmup() -> None:
    samples = [{"start": 100.0}, {"start": 101.0}, {"start": 130.0}, {"start": 110.0}]
    assert phases.step_window(samples, 5.0) == (105.0, 130.0)


# --- plans -------------------------------------------------------------------------------


def test_the_full_plan_is_the_plan_of_record() -> None:
    plan = phases.full_plan()
    assert plan.preset == "full" and plan.repetitions == 3 and plan.fresh_stack
    assert plan.workers == (1, 2, 4, 8) and plan.ingest_users == (1, 2, 4, 8)
    assert plan.ramp_users == (2, 4, 8, 16, 32, 64) and plan.step_seconds == 60
    assert plan.backlog_documents == 300
    assert dict(plan.tenant_documents) == {"load-large": 2000, "load-medium": 300, "load-small": 30}
    assert plan.phases == ("ingest", "scaling", "ramp")
    assert plan.read_workers == 1


def test_the_smoke_plan_is_ten_uploads_two_workers_twenty_seconds_at_two_users() -> None:
    plan = phases.smoke_plan()
    assert plan.preset == "smoke" and plan.backlog_documents == 10 and plan.workers == (2,)
    assert plan.ramp_users == (2,) and plan.step_seconds == 20 and plan.read_workers == 2
    assert plan.phases == ("scaling", "ramp") and plan.tenant_documents == ()
    assert dict(plan.user_weights) == {"load-large:key": 1, "load-large:cookie": 1}
    assert set(dict(plan.read_mix)) == {"search", "ask", "ask_stream", "documents"}


def test_the_quick_plan_is_small_and_never_publishable() -> None:
    plan = phases.quick_plan()
    assert plan.preset == "quick" and plan.repetitions == 1 and not plan.fresh_stack
    assert all(documents >= 25 for _, documents in plan.tenant_documents)  # every agent question has its document


def test_the_plan_description_is_json_and_follows_the_plan() -> None:
    full = phases.plan_description(phases.full_plan())
    assert json.loads(json.dumps(full)) == json.loads(json.dumps(full))
    assert artifact.scenario_digest(full) != artifact.scenario_digest(phases.plan_description(phases.quick_plan()))


# --- the fake stack --------------------------------------------------------------------------


class FakeDriver:
    """Plays the stack: records what the phases ask for, writes the samples a
    Locust run would, and answers the database questions."""

    def __init__(self, directory: Path) -> None:
        self.samples_dir = directory
        self.events: list[tuple] = []
        # The generator's CPU at a time on the host clock (docker stats prints a
        # figure each second here); None means docker stats printed nothing.
        self.cpu_at: Callable[[float], float] | None = lambda t: 10.0
        self.clock_offset_s = 0.0
        self.read_samples = 500
        self.login_errors = 0
        self.read_p95 = lambda users: 50.0
        self.read_errors = lambda users: 0
        self.uploads_short_by = 0
        self.failed_jobs = 0
        self.completed_fraction = 1.0
        self.open_jobs = [0]
        self.backlog = 0

    def prepare(self, workers: int, *, fresh: bool) -> None:
        self.events.append(("prepare", workers, fresh))

    def scale_workers(self, count: int) -> None:
        self.events.append(("scale", count))

    def run_locust(self, spec: phases.LocustSpec, destination: Path) -> None:
        self.events.append(("locust", spec.workload, spec.users, spec.upload_quota, spec.tag))
        lines = []
        if spec.workload == "ingest":
            self.backlog = spec.upload_quota
            for n in range(spec.upload_quota - self.uploads_short_by):
                lines.append(self.sample("upload", 1000.0 + n * 0.01, 20.0, document_id=f"doc-{n}"))
        else:
            for n in range(self.login_errors):
                lines.append(self.sample("login", 2000.0 + n * 0.01, 100.0, "http_401"))
            for n in range(self.read_samples):
                name = ("search", "ask", "ask_stream", "documents")[n % 4]
                error = "http_500" if n < self.read_errors(spec.users) else None
                lines.append(self.sample(name, 2000.0 + n * 0.1, self.read_p95(spec.users), error))
        destination.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    @staticmethod
    def sample(name: str, start: float, ms: float, error: str | None = None, **extra: object) -> dict:
        return {"name": name, "tenant": "load-large", "auth": "key", "start": start, "ms": ms,
                "status": 200 if error is None else 500, "ok": error is None, "error": error, **extra}

    def query(self, sql: str) -> str:
        if sql == phases.QUEUE_OPEN_SQL:
            return str(self.open_jobs.pop(0) if len(self.open_jobs) > 1 else self.open_jobs[0])
        if sql == phases.EXPECTED_JOBS_SQL:
            return str(2 * self.backlog)
        if sql == phases.DONE_JOBS_SQL:
            return "0"
        if sql == phases.FAILED_JOBS_SQL:
            return str(self.failed_jobs)
        if sql == phases.COMPLETIONS_SQL:
            count = int(2 * self.backlog * self.completed_fraction)
            return "\n".join(str(3000.0 + 0.1 * i) for i in range(count)) + "\n"
        if sql == phases.TENANT_SIZES_SQL:
            return "load-large|25|100\nload-medium|25|100\nload-small|25|100\n"
        raise AssertionError(f"unexpected query: {sql}")

    @contextmanager
    def telemetry(self) -> Iterator[PhaseTelemetry]:
        self.events.append(("telemetry-start",))
        telemetry = PhaseTelemetry()
        try:
            yield telemetry
        finally:
            self.events.append(("telemetry-stop",))
            if self.cpu_at is not None:
                telemetry.container_samples = [
                    ContainerSample(float(t), "loadgen", 1, self.cpu_at(float(t)), 100.0) for t in range(900, 3300)
                ]
            telemetry.result = {
                "queue_series": [[1.0, {"extract.pending": 1}]],
                "activity_series": [],
                "container_series": fold_container_samples(telemetry.container_samples),
                "pg_stat_statements": [],
            }

    def check_spend(self) -> None:
        self.events.append(("spend",))

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


@pytest.fixture
def driver(tmp_path: Path) -> FakeDriver:
    return FakeDriver(tmp_path)


def small_plan(**changes: object) -> phases.Plan:
    base = {
        "phases": ("ingest", "scaling", "ramp"),
        "ingest_users": (2, 4),
        "ingest_documents": 10,
        "workers": (1, 2),
        "backlog_documents": 10,
        "backlog_users": 2,
        "ramp_users": (2, 4, 8),
        "step_seconds": 20,
        "warmup_seconds": 2,
        "tenant_documents": (),
    }
    return replace(phases.smoke_plan(), **{**base, **changes})


# --- ingest ------------------------------------------------------------------------------------


def test_ingest_steps_upload_with_no_worker_and_record_each_step(driver: FakeDriver) -> None:
    records = phases.run_ingest(driver, small_plan(), 1, [])
    assert [r["params"]["users"] for r in records] == [2, 4]
    assert driver.events[0] == ("prepare", 0, False)
    assert driver.names().count("scale") == 0  # no worker is ever started
    first = records[0]
    assert first["name"] == "ingest" and first["valid"]
    assert first["results"]["uploaded"] == 10 and first["results"]["uploads_per_s"] > 0
    assert first["endpoints"]["upload"]["count"] == 10 and first["endpoints"]["upload"]["errors"] == 0
    assert "endpoints" not in first["results"]
    assert driver.events[-1] == ("spend",)


def test_a_short_ingest_step_is_invalid_not_dropped(driver: FakeDriver) -> None:
    driver.uploads_short_by = 3
    [record, _] = phases.run_ingest(driver, small_plan(), 1, [])
    assert not record["valid"] and "7 of 10 uploads" in record["invalid_reasons"][0]


# --- worker scaling ---------------------------------------------------------------------------


def test_a_scaling_phase_uploads_a_backlog_then_starts_the_workers(driver: FakeDriver) -> None:
    record = phases.run_scaling(driver, small_plan(), 2, 1, [])
    assert driver.events == [
        ("prepare", 0, False),
        ("locust", "ingest", 2, 10, "scaling-w2-r1-backlog"),
        ("telemetry-start",),
        ("scale", 2),
        ("telemetry-stop",),
        ("spend",),
    ]
    assert record["name"] == "scaling" and record["valid"] and record["invalid_reasons"] == []
    results = record["results"]
    assert results["expected_jobs"] == 20 and results["completed_jobs"] == 20
    # 20 completions 0.1 s apart: marks at the 2nd and 18th, 16 jobs in 1.6 s.
    assert results["jobs_per_s"] == pytest.approx(10.0)
    assert results["documents_per_s"] == pytest.approx(5.0) and results["jobs_counted"] == "extract+index"
    assert results["window"]["jobs"] == 16
    assert results["backlog"]["uploaded"] == 10
    assert record["params"] == {"workers": 2, "backlog_documents": 10}


def test_a_drain_that_did_not_finish_is_invalid(driver: FakeDriver) -> None:
    driver.completed_fraction = 0.5
    record = phases.run_scaling(driver, small_plan(), 1, 1, [])
    assert not record["valid"]
    assert "10 of 20 jobs completed" in record["invalid_reasons"][0]
    assert record["results"]["jobs_per_s"] is None


def test_failed_jobs_make_the_phase_invalid(driver: FakeDriver) -> None:
    driver.failed_jobs = 2
    record = phases.run_scaling(driver, small_plan(), 1, 1, [])
    assert not record["valid"] and "2 job(s) failed" in record["invalid_reasons"]


def test_the_drain_runs_no_generator_so_its_cpu_is_recorded_and_never_gates(driver: FakeDriver) -> None:
    """The smoke showed the generator at 32% inside a drain: the tail of the upload step
    that built the backlog. That is not a reason to call the drain invalid."""
    driver.cpu_at = lambda t: 150.0  # the upload tail, over the limit
    record = phases.run_scaling(driver, small_plan(), 1, 1, [])
    assert record["valid"] and record["invalid_reasons"] == []
    assert record["generator_cpu"]["max_pct"] == 150.0 and record["generator_cpu"]["gated"] is False
    driver.cpu_at = None  # and docker stats printing nothing does not invalidate it either
    record = phases.run_scaling(driver, small_plan(), 1, 1, [])
    assert record["valid"] and record["generator_cpu"] is None


def test_an_ingest_step_is_judged_on_its_own_upload_window(driver: FakeDriver) -> None:
    """The step's window is from its first upload to its last, on the sample clock (1000.0
    to about 1000.1 here), plus the one interval a figure describes."""
    driver.cpu_at = lambda t: 95.0 if t > 1010 else 10.0  # busy only long after the step
    records = phases.run_ingest(driver, small_plan(), 1, [])
    assert all(record["valid"] for record in records)
    assert records[0]["generator_cpu"]["gated"] is True
    assert records[0]["generator_cpu"]["max_pct"] == 10.0
    driver.cpu_at = lambda t: 95.0 if 999 < t < 1010 else 10.0
    assert all(not record["valid"] for record in phases.run_ingest(driver, small_plan(), 1, []))


def test_an_ingest_window_uses_the_stacks_clock_offset(driver: FakeDriver) -> None:
    """Samples are stamped by the stack's clock and docker stats by the host's; a stack
    clock 60 s fast puts the upload at host time 940."""
    driver.clock_offset_s = 60.0
    driver.cpu_at = lambda t: 95.0 if 939 < t < 950 else 10.0
    [record, _] = phases.run_ingest(driver, small_plan(), 1, [])
    assert not record["valid"]


@pytest.mark.parametrize(
    ("cpu", "fragment"),
    [(lambda t: 81.0, "p95 CPU was 81%"), (lambda t: 150.0, "150%"), (None, "no generator CPU")],
)
def test_a_generator_past_eighty_percent_or_unmeasured_makes_a_driven_phase_invalid(
    driver: FakeDriver, cpu: Callable[[float], float] | None, fragment: str
) -> None:
    driver.cpu_at = cpu
    [record, *_] = phases.run_ingest(driver, small_plan(), 1, [])
    assert not record["valid"] and fragment in record["invalid_reasons"][0]
    [step, *_] = phases.run_ramp(FakeDriverWith(driver, cpu), small_plan(ramp_users=(2,)), 1, [])
    assert not step["valid"] and fragment in step["invalid_reasons"][0]


def FakeDriverWith(driver: FakeDriver, cpu: Callable[[float], float] | None) -> FakeDriver:
    fresh = FakeDriver(driver.samples_dir)
    fresh.cpu_at = cpu
    return fresh


def test_the_gate_is_the_p95_of_the_windows_samples_not_one_spike(driver: FakeDriver) -> None:
    # One 95% figure among many quiet ones in the ramp window (2002 to 2050) is under p95.
    driver.cpu_at = lambda t: 95.0 if t == 2020 else 10.0
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert record["valid"]
    assert record["generator_cpu"]["max_pct"] == 95.0 and record["generator_cpu"]["p95_pct"] == 10.0


def test_exactly_eighty_percent_is_still_valid(driver: FakeDriver) -> None:
    driver.cpu_at = lambda t: 80.0
    [record, *_] = phases.run_ingest(driver, small_plan(), 1, [])
    assert record["valid"]


# --- the read ramp -----------------------------------------------------------------------------


def test_the_ramp_stops_one_step_past_the_knee(driver: FakeDriver) -> None:
    driver.read_p95 = lambda users: 100.0 if users < 8 else 1500.0
    records = phases.run_ramp(driver, small_plan(), 1, [])
    assert [r["params"]["users"] for r in records] == [2, 4, 8]
    assert [r["results"]["verdict"]["ok"] for r in records] == [True, True, False]
    assert {r["results"]["knee_users"] for r in records} == {4}
    assert records[2]["results"]["verdict"]["reasons"] == ["/search p95 1500 ms is over the 1000 ms SLO"]
    assert driver.events.count(("spend",)) == 1


def test_a_ramp_that_meets_the_limits_at_every_step_runs_them_all(driver: FakeDriver) -> None:
    records = phases.run_ramp(driver, small_plan(), 1, [])
    assert [r["params"]["users"] for r in records] == [2, 4, 8]
    assert records[0]["results"]["knee_users"] == 8


def test_errors_over_one_percent_end_the_ramp(driver: FakeDriver) -> None:
    driver.read_errors = lambda users: 0 if users < 4 else 30  # 30 of 500
    records = phases.run_ramp(driver, small_plan(), 1, [])
    assert [r["params"]["users"] for r in records] == [2, 4]
    assert records[1]["endpoints"]["search"]["errors"] >= 0
    assert "error rate" in records[1]["results"]["verdict"]["reasons"][0]


def test_a_step_starts_even_if_the_queue_will_not_empty_and_says_how_many_jobs_were_open(
    driver: FakeDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver.open_jobs = [0, 7]  # the strict wait after scaling sees an empty queue, then jobs appear
    clock = iter(range(0, 100000, 50))
    monkeypatch.setattr(phases.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(phases.time, "sleep", lambda _: None)
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,), tenant_documents=()), 1, [])
    assert record["results"]["queue_open_at_start"] == 7


def test_each_step_waits_for_an_empty_queue_and_runs_under_telemetry(driver: FakeDriver) -> None:
    phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert driver.events[:1] == [("prepare", 2, False)]  # read_workers of the smoke plan
    assert ("locust", "read", 2, 0, "ramp-r1-u2") in driver.events
    locust_at = driver.events.index(("locust", "read", 2, 0, "ramp-r1-u2"))
    assert driver.events[locust_at - 1] == ("telemetry-start",)
    assert driver.events[locust_at + 1] == ("telemetry-stop",)


def test_the_measured_window_leaves_out_the_warmup(driver: FakeDriver) -> None:
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    window = record["results"]["window"]
    assert window["start"] == pytest.approx(2002.0) and window["last_start"] == pytest.approx(2049.9)
    # The span runs to the last completion (the last request took 50 ms), and rates use it.
    assert window["end"] == pytest.approx(2049.95) and window["seconds"] == pytest.approx(47.95)
    assert record["endpoints"]["search"]["rps"] == pytest.approx(record["endpoints"]["search"]["count"] / 47.95)
    counted = sum(entry["count"] for entry in record["endpoints"].values())
    assert counted == 500 - 20  # the samples of the first two seconds


def test_a_step_where_a_cookie_user_could_not_sign_in_is_invalid(driver: FakeDriver) -> None:
    """That user never ran, so the step measured lower concurrency than it says."""
    driver.login_errors = 1
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert not record["valid"] and "1 cookie user(s) failed to sign in" in record["invalid_reasons"][0]
    assert record["results"]["login_errors"] == 1


def test_a_step_that_was_cut_short_is_invalid(driver: FakeDriver) -> None:
    driver.read_samples = 100  # 10 s of requests where 20 s less a 2 s warmup were meant
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert not record["valid"] and "measured 7.9 s of the 18 s it was meant to" in record["invalid_reasons"][0]
    driver.read_samples = 150  # 14.9 s - 2 s = 12.9 s is under 80% of 18 s (14.4 s)
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert not record["valid"]
    driver.read_samples = 170  # 16.9 - 2 = 14.9 s, over it
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert record["valid"]


def test_populating_uploads_each_tenant_its_planned_count_from_the_start_of_the_corpus(
    driver: FakeDriver,
) -> None:
    plan = small_plan(tenant_documents=(("load-large", 30), ("load-small", 5)), populate_workers=3, populate_users=4)
    steps = phases.populate_tenants(driver, plan, 1, [])
    assert [(s["tenant"], s["documents"], s["uploaded"]) for s in steps] == [("load-large", 30, 30), ("load-small", 5, 5)]
    assert driver.events[0] == ("scale", 3)
    assert ("locust", "ingest", 4, 30, "ramp-r1-populate-load-large") in driver.events


def test_a_populate_upload_that_fails_ends_the_run(driver: FakeDriver) -> None:
    driver.uploads_short_by = 1
    plan = small_plan(tenant_documents=(("load-large", 30),))
    with pytest.raises(phases.PhaseError, match="29 of 30 uploads succeeded"):
        phases.populate_tenants(driver, plan, 1, [])


def test_a_ramp_without_populate_starts_the_read_workers(driver: FakeDriver) -> None:
    phases.run_ramp(driver, small_plan(ramp_users=(2,), read_workers=2), 1, [])
    assert driver.events[0] == ("prepare", 2, False)
    assert ("scale", 2) in driver.events


def test_the_ramp_records_the_tenant_sizes(driver: FakeDriver) -> None:
    [record] = phases.run_ramp(driver, small_plan(ramp_users=(2,)), 1, [])
    assert record["results"]["tenants"]["load-large"] == {"documents": 25, "chunks": 100}


# --- a whole run ---------------------------------------------------------------------------------


def test_repetitions_interleave_the_worker_counts_and_reset_each_time_when_fresh(driver: FakeDriver) -> None:
    plan = small_plan(repetitions=2, fresh_stack=True, phases=("scaling",), workers=(1, 2))
    result = phases.run_plan(driver, plan, log=lambda _: None)
    assert [(r["repetition"], r["params"]["workers"]) for r in result.phases] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [e for e in driver.events if e[0] == "prepare"] == [("prepare", 0, True)] * 4
    assert len(result.sample_files) == 4 and all(path.exists() for path in result.sample_files)


def test_a_run_goes_through_the_selected_phases_in_order(driver: FakeDriver) -> None:
    result = phases.run_plan(driver, small_plan(), log=lambda _: None)
    assert [r["name"] for r in result.phases] == ["ingest", "ingest", "scaling", "scaling", "ramp", "ramp", "ramp"]
    only_ramp = phases.run_plan(FakeDriver(driver.samples_dir), small_plan(phases=("ramp",)), log=lambda _: None)
    assert {r["name"] for r in only_ramp.phases} == {"ramp"}


def test_the_summary_is_the_median_and_range_of_the_valid_repetitions() -> None:
    def scaling(rep: int, workers: int, rate: float, valid: bool = True) -> dict:
        return {"name": "scaling", "repetition": rep, "params": {"workers": workers}, "valid": valid,
                "results": {"jobs_per_s": rate}}

    def ramp(rep: int, knee: int | None) -> dict:
        return {"name": "ramp", "repetition": rep, "params": {}, "valid": True, "results": {"knee_users": knee}}

    records = [
        scaling(1, 1, 5.0), scaling(2, 1, 6.0), scaling(3, 1, 4.0),
        scaling(1, 2, 9.0), scaling(2, 2, 100.0, valid=False), scaling(3, 2, 11.0),
        ramp(1, 16), ramp(1, 16), ramp(2, 8), ramp(3, None),
    ]
    summary = phases.summarize(records)
    assert summary["scaling_jobs_per_s_by_workers"]["1"] == {"n": 3, "median": 5.0, "min": 4.0, "max": 6.0, "invalid": 0}
    assert summary["scaling_jobs_per_s_by_workers"]["2"] == {"n": 2, "median": 10.0, "min": 9.0, "max": 11.0, "invalid": 1}
    assert summary["ramp_knee_users"] == {"n": 2, "median": 12.0, "min": 8, "max": 16}
    assert summary["ingest_uploads_per_s_by_users"] == {}
    assert phases.summarize([])["ramp_knee_users"] is None
    assert phases.summarize([])["scaling_documents_per_s_by_workers"] == {}


# --- waiting for the queue -----------------------------------------------------------------------


def test_waiting_for_the_queue_polls_until_it_is_empty(driver: FakeDriver) -> None:
    driver.open_jobs = [5, 3, 1, 0]
    sleeps: list[float] = []
    phases.wait_until_drained(driver, 60, sleep=sleeps.append)
    assert sleeps == [phases.DRAIN_POLL_SECONDS] * 3


def test_a_queue_that_never_empties_is_an_error(driver: FakeDriver, monkeypatch: pytest.MonkeyPatch) -> None:
    driver.open_jobs = [1]
    clock = iter(range(0, 1000, 20))
    monkeypatch.setattr(phases.time, "monotonic", lambda: next(clock))
    with pytest.raises(phases.PhaseError, match="did not drain within 30 s"):
        phases.wait_until_drained(driver, 30, sleep=lambda _: None)


def test_settling_returns_the_open_jobs_and_never_fails(driver: FakeDriver, monkeypatch: pytest.MonkeyPatch) -> None:
    driver.open_jobs = [3, 1, 0]
    sleeps: list[float] = []
    assert phases.settle_queue(driver, 60, sleep=sleeps.append) == 0 and len(sleeps) == 2
    driver.open_jobs = [5]
    clock = iter(range(0, 1000, 20))
    monkeypatch.setattr(phases.time, "monotonic", lambda: next(clock))
    assert phases.settle_queue(driver, 30, sleep=lambda _: None) == 5


def test_tenant_sizes_parse(driver: FakeDriver) -> None:
    assert phases.tenant_sizes(driver)["load-small"] == {"documents": 25, "chunks": 100}


def test_every_harness_statement_is_tagged_so_the_statement_snapshot_can_skip_it() -> None:
    for sql in (
        phases.QUEUE_OPEN_SQL, phases.FAILED_JOBS_SQL, phases.EXPECTED_JOBS_SQL, phases.DONE_JOBS_SQL,
        phases.COMPLETIONS_SQL, phases.TENANT_SIZES_SQL,
    ):
        assert sql.startswith("/* load-harness */")
