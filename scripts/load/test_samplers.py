"""Tests for the samplers' parsers and the sampler wiring (run: python -m pytest
../scripts -q). No Docker: the processes are Python one-liners."""

from __future__ import annotations

import json
import sys

import pytest
import samplers

# --- the db sampler -----------------------------------------------------------------


def test_a_queue_line_parses() -> None:
    row = samplers.parse_db_line("Q|1792956966.586|index|processing|2\n")
    assert row == samplers.DbRow("Q", 1792956966.586, "index", "processing", 2)


def test_an_activity_line_with_a_spaced_state_parses() -> None:
    row = samplers.parse_db_line("A|1792956973.792|doc-pilot-api|idle in transaction|1")
    assert row == samplers.DbRow("A", 1792956973.792, "doc-pilot-api", "idle in transaction", 1)


def test_a_tick_line_parses_with_empty_fields() -> None:
    assert samplers.parse_db_line("T|1792956973.792|||0") == samplers.DbRow("T", 1792956973.792, "", "", 0)


@pytest.mark.parametrize(
    "line",
    ["", "\n", "NOTICE: something", "Q|1.0|index|pending", "Q|1.0|index|pending|2|extra", "X|1.0|a|b|1", "Q|late|a|b|1", "Q|1.0|a|b|many"],
)
def test_anything_else_is_not_a_row(line: str) -> None:
    assert samplers.parse_db_line(line) is None


def test_ticks_fold_into_a_queue_series_and_an_activity_series() -> None:
    rows = [
        samplers.DbRow("T", 10.0, "", "", 0),
        samplers.DbRow("Q", 10.0, "extract", "pending", 3),
        samplers.DbRow("Q", 10.0, "extract", "done", 1),
        samplers.DbRow("A", 10.0, "doc-pilot-api", "idle", 2),
        samplers.DbRow("T", 11.0, "", "", 0),
        samplers.DbRow("A", 11.0, "doc-pilot-worker", "active", 1),
        samplers.DbRow("T", 12.0, "", "", 0),
        samplers.DbRow("Q", 12.0, "index", "pending", 4),
    ]
    queue, activity = samplers.fold_db_rows(rows)
    assert queue == [
        [10.0, {"extract.pending": 3, "extract.done": 1}],
        [11.0, {}],  # an empty queue is still a sample
        [12.0, {"index.pending": 4}],
    ]
    assert activity == [
        [10.0, {"doc-pilot-api|idle": 2}],
        [11.0, {"doc-pilot-worker|active": 1}],
        [12.0, {}],
    ]


def test_no_rows_no_series() -> None:
    assert samplers.fold_db_rows([]) == ([], [])


def test_the_sampling_sql_survives_sh_c_and_is_tagged() -> None:
    assert samplers.SAMPLE_SQL.startswith(samplers.HARNESS_TAG)
    assert '"' not in samplers.SAMPLE_SQL and "$" not in samplers.SAMPLE_SQL and "`" not in samplers.SAMPLE_SQL
    assert "application_name <> 'psql'" in samplers.SAMPLE_SQL
    assert samplers.STATEMENTS_SQL.startswith(samplers.HARNESS_TAG)
    assert f"not like '{samplers.HARNESS_TAG}%'" in samplers.STATEMENTS_SQL
    assert samplers.RESET_STATEMENTS_SQL.startswith(samplers.HARNESS_TAG)


def test_the_loop_stops_on_a_file_and_on_a_broken_pipe() -> None:
    script = samplers.db_loop_script("/tmp/x.stop", user="docpilot", database="docpilot_load")
    assert '[ ! -e "/tmp/x.stop" ]' in script
    assert "|| break" in script
    assert "psql -U docpilot -d docpilot_load" in script
    assert script.endswith('rm -f "/tmp/x.stop"')


# --- docker stats ---------------------------------------------------------------------

REAL_LINE = (
    '\x1b[H{"BlockIO":"33.6MB / 123kB","CPUPerc":"12.50%","Container":"6c013001e8c1e24d",'
    '"ID":"6c013001e8c1","MemPerc":"4.83%","MemUsage":"383MiB / 7.75GiB",'
    '"Name":"doc-pilot-load-api-1","NetIO":"99.6kB / 136kB","PIDs":"32"}\x1b[K\n'
)


def test_a_load_container_line_parses_through_docker_s_escape_codes() -> None:
    sample = samplers.parse_stats_line(REAL_LINE, 100.5)
    assert sample == samplers.ContainerSample(100.5, "api", 1, 12.5, 383.0)
    assert sample.key == "api-1"


def test_a_scaled_worker_keeps_its_replica_number() -> None:
    line = REAL_LINE.replace("doc-pilot-load-api-1", "doc-pilot-load-worker-3").replace("383MiB", "1.5GiB")
    sample = samplers.parse_stats_line(line, 1.0)
    assert (sample.service, sample.replica, sample.mem_mib) == ("worker", 3, 1536.0)


@pytest.mark.parametrize("name", ["doc-pilot-api-1", "doc-pilot-demo-worker-1", "doc-pilot-loadx-api-1", "other-load-api-1"])
def test_another_project_s_container_is_dropped_unread(name: str) -> None:
    assert samplers.parse_stats_line(REAL_LINE.replace("doc-pilot-load-api-1", name), 1.0) is None


@pytest.mark.parametrize(
    "line",
    ["", "\x1b[K\n", "\x1b[J\x1b[H", "not json", "{", '{"Name": "doc-pilot-load-api-1"}', "[1]"],
)
def test_screen_control_and_junk_lines_are_not_samples(line: str) -> None:
    assert samplers.parse_stats_line(line, 1.0) is None


def test_a_container_without_figures_is_dropped() -> None:
    line = REAL_LINE.replace("12.50%", "--").replace("383MiB / 7.75GiB", "-- / --")
    assert samplers.parse_stats_line(line, 1.0) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("123.4MiB", 123.4 * 1024**2), ("1.5GiB", 1.5 * 1024**3), ("512kB", 512e3), ("0B", 0), ("4.1KB", 4100.0), (" 2MB ", 2e6)],
)
def test_sizes_parse(text: str, expected: float) -> None:
    assert samplers.parse_size(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "--", "12", "MiB", "12 parsecs", "1.2.3MiB"])
def test_a_size_that_is_not_one_is_none(text: str) -> None:
    assert samplers.parse_size(text) is None


def test_samples_fold_into_a_series_per_container() -> None:
    samples = [
        samplers.ContainerSample(1.0, "api", 1, 10.0, 300.0),
        samplers.ContainerSample(1.0, "worker", 1, 50.0, 200.0),
        samplers.ContainerSample(3.0, "api", 1, 20.0, 310.0),
    ]
    assert samplers.fold_container_samples(samples) == {
        "api-1": [[1.0, 10.0, 300.0], [3.0, 20.0, 310.0]],
        "worker-1": [[1.0, 50.0, 200.0]],
    }


def test_generator_cpu_is_the_loadgen_alone() -> None:
    series = {
        "api-1": [[1.0, 400.0, 1.0]],
        "loadgen-1": [[1.0, 10.0, 1.0], [2.0, 70.0, 1.0], [3.0, 40.0, 1.0]],
    }
    assert samplers.generator_cpu(series) == {"max_pct": 70.0, "mean_pct": 40.0, "p95_pct": 70.0, "samples": 3}


def test_cpu_is_summarized_by_max_mean_and_a_nearest_rank_p95() -> None:
    values = [float(n) for n in range(1, 21)]  # 1 .. 20
    assert samplers.cpu_summary(values) == {"max_pct": 20.0, "mean_pct": 10.5, "p95_pct": 19.0, "samples": 20}
    assert samplers.cpu_summary([5.0]) == {"max_pct": 5.0, "mean_pct": 5.0, "p95_pct": 5.0, "samples": 1}
    assert samplers.cpu_summary([]) is None


def loadgen_sample(t: float, cpu: float, service: str = "loadgen") -> samplers.ContainerSample:
    return samplers.ContainerSample(t, service, 1, cpu, 1.0)


def test_the_window_takes_the_samples_that_describe_it() -> None:
    """A figure printed at t describes the interval before t: one printed after the
    window's start up to an interval past its end overlaps it; earlier or later ones do not."""
    samples = [
        loadgen_sample(99.0, 95.0),  # printed before the window began: describes idle time
        loadgen_sample(101.0, 10.0),
        loadgen_sample(105.0, 30.0),
        loadgen_sample(112.5, 20.0),  # within one interval past the end: its interval overlaps
        loadgen_sample(113.5, 99.0),  # past it
        loadgen_sample(103.0, 99.0, service="api"),  # not the generator
    ]
    cpu = samplers.generator_cpu_in_window(samples, 100.0, 110.0)
    assert cpu == {"max_pct": 30.0, "mean_pct": 20.0, "p95_pct": 30.0, "samples": 3}
    assert samplers.generator_cpu_in_window(samples, 200.0, 210.0) is None


def test_a_busy_drain_after_a_window_does_not_count() -> None:
    samples = [loadgen_sample(101.0, 10.0), loadgen_sample(120.0, 90.0)]
    assert samplers.generator_cpu_in_window(samples, 100.0, 110.0)["max_pct"] == 10.0


def test_no_generator_figures_is_none_so_the_phase_cannot_be_called_valid() -> None:
    assert samplers.generator_cpu({"api-1": [[1.0, 1.0, 1.0]]}) is None
    assert samplers.generator_cpu({}) is None


# --- the statement snapshot ---------------------------------------------------------


def test_the_snapshot_is_a_json_array() -> None:
    rows = [{"query": "SELECT 1", "calls": 2, "mean_ms": 0.5, "total_ms": 1.0, "rows": 2}]
    assert samplers.parse_statements(json.dumps(rows) + "\n") == rows
    assert samplers.parse_statements("") == []
    with pytest.raises(TypeError):
        samplers.parse_statements("{}")
    with pytest.raises(ValueError):
        samplers.parse_statements("not json")


# --- the process wiring ---------------------------------------------------------------


def test_a_line_reader_hands_over_each_line_with_its_arrival_time() -> None:
    seen: list[tuple[str, float]] = []
    reader = samplers.LineReader([sys.executable, "-c", "print('a'); print('b')"], lambda line, t: seen.append((line, t)))
    reader.start()
    reader.stop()
    assert [line for line, _ in seen] == ["a\n", "b\n"]
    assert all(t > 1.7e9 for _, t in seen)


def test_a_reader_that_will_not_end_is_terminated_after_the_grace_period() -> None:
    reader = samplers.LineReader([sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"], lambda *_: None)
    reader.start()
    reader.stop(grace_seconds=0.5)  # returns, having terminated the process


def test_stopping_a_reader_that_never_started_is_harmless() -> None:
    samplers.LineReader(["true"], lambda *_: None).stop()


# --- the telemetry around a phase --------------------------------------------------------------------

DB_PRINTER = "print('T|1.0|||0'); print('Q|1.0|extract|pending|3'); print('A|1.0|doc-pilot-api|idle|1')"
STATS_PRINTER = (
    "import json, time\n"
    "line = json.dumps({'Name': 'doc-pilot-load-loadgen-1', 'CPUPerc': '90.00%', 'MemUsage': '50MiB / 7GiB'})\n"
    "other = json.dumps({'Name': 'doc-pilot-api-1', 'CPUPerc': '10.00%', 'MemUsage': '50MiB / 7GiB'})\n"
    "print(line, flush=True); print(other, flush=True); time.sleep(60)"
)


def fake_telemetry(psql_calls: list[str] | None = None) -> samplers.Telemetry:
    def exec_db(argv: list[str]) -> list[str]:
        return [sys.executable, "-c", DB_PRINTER] if argv[0] == "sh" else ["true"]

    def psql(sql: str) -> str:
        if psql_calls is not None:
            psql_calls.append(sql)
        return "[]" if "json_agg" in sql else ""

    return samplers.Telemetry(
        exec_db, psql, owner="docpilot", database="docpilot_load",
        stats_command=[sys.executable, "-c", STATS_PRINTER],
    )


def test_a_phase_resets_the_statements_samples_and_snapshots() -> None:
    import time

    calls: list[str] = []
    telemetry = fake_telemetry(calls)
    telemetry.ensure_stats()
    try:
        time.sleep(0.5)  # the stream's first figures arrive before the phase
        with telemetry.phase() as phase:
            pass
    finally:
        telemetry.close()
    assert calls[0] == samplers.RESET_STATEMENTS_SQL and calls[-1] == samplers.STATEMENTS_SQL
    result = phase.result
    assert result["queue_series"] == [[1.0, {"extract.pending": 3}]]
    assert result["activity_series"] == [[1.0, {"doc-pilot-api|idle": 1}]]
    # The sample taken just before the phase counts; another project's container is dropped.
    assert sorted(result["container_series"]) == ["loadgen-1"]
    assert "generator_cpu" not in result  # a phase takes it over its own measured window
    assert [(x.service, x.cpu_pct) for x in phase.container_samples] == [("loadgen", 90.0)]
    assert result["pg_stat_statements"] == []


def test_a_sample_older_than_the_carry_is_not_in_the_phase() -> None:
    telemetry = fake_telemetry()
    telemetry._stats_samples.append(samplers.ContainerSample(1.0, "loadgen", 1, 99.0, 1.0))
    with telemetry.phase() as phase:
        pass
    telemetry.close()
    assert all(sample.cpu_pct != 99.0 for sample in phase.container_samples)
    assert "loadgen-1" not in phase.result["container_series"] or all(
        point[1] != 99.0 for point in phase.result["container_series"]["loadgen-1"]
    )


def test_a_phase_that_raises_still_stops_its_samplers() -> None:
    telemetry = fake_telemetry()
    with pytest.raises(RuntimeError, match="boom"), telemetry.phase():
        raise RuntimeError("boom")
    telemetry.close()


def test_the_stats_stream_restarts_if_it_died_and_closes_once() -> None:
    telemetry = fake_telemetry()
    telemetry.ensure_stats()
    first = telemetry._stats_reader
    telemetry.ensure_stats()
    assert telemetry._stats_reader is first  # still alive: not started twice
    first.terminate()
    first.stop(grace_seconds=1.0)
    telemetry.ensure_stats()
    assert telemetry._stats_reader is not first and telemetry._stats_reader.alive()
    telemetry.close()
    telemetry.close()  # harmless
