"""Samplers: what the stack is doing while a phase runs.

Three sources, all read-only, all stamped with the host's UTC wall clock when
the line arrives (every request sample carries its own UTC stamp too, and Docker
Desktop's VM clock follows the host's):

- One streaming `psql` loop in the db container, once a second, as the database
  owner (the runtime role cannot see across tenants, and the owner is how the
  queue is read): the job queue by kind and state, and pg_stat_activity by
  application_name and state. The loop runs in the container, so a host
  process that dies cannot leave it running: it stops when its stdout closes or
  when a stop file appears, and the container is removed with the stack.
- `docker stats`, streaming, one record per running container per interval:
  CPU as a percentage of one core, and memory. One stream serves the whole run
  (its first figures take a second or two to arrive). It streams every
  container on the Docker host, so lines are kept only for the load project's
  own containers (named `doc-pilot-load-<service>-<n>`); everything else is
  dropped unread.
- pg_stat_statements: reset before a phase and a snapshot of the top statements
  by total time after it.

The parsers are pure functions of one line each; the classes only wire a
process to them.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# The load project's container names: `<project>-<service>-<replica>`.
CONTAINER_NAME = re.compile(r"^doc-pilot-load-(?P<service>.+)-(?P<replica>\d+)$")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SIZE_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3}
# Every statement the harness runs through the db container starts with this
# comment, so the pg_stat_statements snapshot can leave the harness out.
HARNESS_TAG = "/* load-harness */"
TOP_STATEMENTS = 15
SAMPLE_INTERVAL_SECONDS = 1

# One row per queue (kind, state), one per activity group, and a tick row so an
# empty queue still produces a sample. `now()` is the statement's start, shared
# by all its rows. The sampler's own connections (application_name psql) are
# left out of the activity counts. No double quote or dollar sign in it: it is
# passed through `sh -c`.
SAMPLE_SQL = (
    f"{HARNESS_TAG} "
    "select 'T', extract(epoch from now()), '', '', 0 "
    "union all "
    "select 'Q', extract(epoch from now()), kind, state, count(*) from jobs group by kind, state "
    "union all "
    "select 'A', extract(epoch from now()), application_name, coalesce(state, '-'), count(*) "
    "from pg_stat_activity where datname = current_database() "
    "and backend_type = 'client backend' and application_name <> 'psql' "
    "group by application_name, state"
)

STATEMENTS_SQL = (
    f"{HARNESS_TAG} "
    "select coalesce(json_agg(t), '[]'::json) from ("
    "select left(query, 300) as query, calls, "
    "round(mean_exec_time::numeric, 3) as mean_ms, "
    "round(total_exec_time::numeric, 3) as total_ms, rows "
    "from pg_stat_statements "
    "where dbid = (select oid from pg_database where datname = current_database()) "
    f"and query not like '{HARNESS_TAG}%' "
    f"order by total_exec_time desc limit {TOP_STATEMENTS}) t"
)
RESET_STATEMENTS_SQL = f"{HARNESS_TAG} select pg_stat_statements_reset()"


# --- parsers ----------------------------------------------------------------


@dataclass(frozen=True)
class DbRow:
    """One line of the db sampler. tag `T` is a tick, `Q` a queue count (a=job
    kind, b=job state) and `A` an activity count (a=application_name,
    b=connection state)."""

    tag: str
    t: float
    a: str
    b: str
    n: int


def parse_db_line(line: str) -> DbRow | None:
    """A DbRow from one `tag|epoch|a|b|n` line; None for anything else (a blank
    line, a notice, a half-written line)."""
    parts = line.rstrip("\r\n").split("|")
    if len(parts) != 5 or parts[0] not in ("T", "Q", "A"):
        return None
    try:
        return DbRow(parts[0], float(parts[1]), parts[2], parts[3], int(parts[4]))
    except ValueError:
        return None


def fold_db_rows(rows: list[DbRow]) -> tuple[list[list], list[list]]:
    """The queue series and the activity series from the sampler's rows, one
    entry per tick: `[t, {"extract.pending": 3, ...}]` and
    `[t, {"doc-pilot-api|idle in transaction": 2, ...}]`. A tick with an empty
    queue has an empty dict."""
    queue: dict[float, dict[str, int]] = {}
    activity: dict[float, dict[str, int]] = {}
    for row in rows:
        queue.setdefault(row.t, {})
        activity.setdefault(row.t, {})
        if row.tag == "Q":
            queue[row.t][f"{row.a}.{row.b}"] = row.n
        elif row.tag == "A":
            activity[row.t][f"{row.a}|{row.b}"] = row.n
    ticks = sorted(queue)
    return (
        [[round(t, 3), queue[t]] for t in ticks],
        [[round(t, 3), activity[t]] for t in ticks],
    )


def parse_size(text: str) -> float | None:
    """Bytes in a docker size such as `123.4MiB` or `1.5GB`; None when it is
    not one."""
    match = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]+)\s*", text)
    if not match or match[2].upper() not in SIZE_UNITS:
        return None
    try:
        return float(match[1]) * SIZE_UNITS[match[2].upper()]
    except ValueError:
        return None


@dataclass(frozen=True)
class ContainerSample:
    t: float
    service: str
    replica: int
    cpu_pct: float
    mem_mib: float

    @property
    def key(self) -> str:
        return f"{self.service}-{self.replica}"


def parse_stats_line(line: str, now: float) -> ContainerSample | None:
    """A ContainerSample from one `docker stats --format '{{json .}}'` line, or
    None: for another project's container, a blank or half-written line, the
    screen-clearing escape sequences docker prints between refreshes, or a
    container whose figures are not numbers (`--`)."""
    text = ANSI.sub("", line).strip()
    if not text.startswith("{"):
        return None
    try:
        record = json.loads(text)
        name = CONTAINER_NAME.match(record["Name"])
        if not name:
            return None
        used = parse_size(record["MemUsage"].split("/")[0])
        cpu = float(record["CPUPerc"].rstrip("%"))
    except (ValueError, KeyError, AttributeError, TypeError):
        return None
    if used is None:
        return None
    return ContainerSample(now, name["service"], int(name["replica"]), cpu, round(used / 1024**2, 2))


def fold_container_samples(samples: list[ContainerSample]) -> dict[str, list[list]]:
    """Per container (`api-1`, `worker-3`): `[t, cpu_pct, mem_mib]` in time
    order."""
    series: dict[str, list[list]] = {}
    for sample in samples:
        series.setdefault(sample.key, []).append([round(sample.t, 3), sample.cpu_pct, sample.mem_mib])
    return series


def cpu_summary(values: list[float]) -> dict | None:
    """Max, mean and nearest-rank p95 of CPU percentages (of one core); None
    when there are none."""
    if not values:
        return None
    ordered = sorted(values)
    rank = -(-95 * len(ordered) // 100)
    return {
        "max_pct": ordered[-1],
        "mean_pct": round(sum(ordered) / len(ordered), 2),
        "p95_pct": ordered[rank - 1],
        "samples": len(ordered),
    }


def generator_cpu(series: dict[str, list[list]]) -> dict | None:
    """The load generator's CPU over everything a series holds: max, mean and
    p95 of its samples; None when docker stats gave none."""
    return cpu_summary(
        [point[1] for key, points in series.items() if key.startswith("loadgen-") for point in points]
    )


# docker stats prints each container about every two seconds, and a figure
# describes the interval that ended when it was printed.
STATS_INTERVAL_SECONDS = 3.0


def generator_cpu_in_window(samples: list[ContainerSample], start: float, end: float) -> dict | None:
    """The generator's CPU over a measured window (host clock seconds): the
    samples whose interval overlaps it, that is those printed after `start` and
    up to one interval after `end`. None when there are none."""
    return cpu_summary(
        [
            sample.cpu_pct
            for sample in samples
            if sample.service == "loadgen" and start < sample.t <= end + STATS_INTERVAL_SECONDS
        ]
    )


def parse_statements(text: str) -> list[dict]:
    """The rows of the pg_stat_statements snapshot (a JSON array)."""
    rows = json.loads(text.strip() or "[]")
    if not isinstance(rows, list):
        raise TypeError("the statements snapshot is not a JSON array")
    return rows


# --- the processes ----------------------------------------------------------


def db_loop_script(stop_file: str, *, user: str, database: str) -> str:
    """The shell loop run in the db container: sample, sleep a second, and
    stop when psql fails (its stdout closed), or when `stop_file` exists."""
    return (
        f'while [ ! -e "{stop_file}" ]; do '
        f'psql -U {user} -d {database} -X -tA -c "{SAMPLE_SQL}" || break; '
        f"sleep {SAMPLE_INTERVAL_SECONDS}; done; rm -f \"{stop_file}\""
    )


class LineReader:
    """Runs a command and hands each line of its stdout, with the time it
    arrived, to `handle` on a thread of its own."""

    def __init__(self, command: list[str], handle: Callable[[str, float], None]) -> None:
        self._command = command
        self._handle = handle
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            self._handle(line, time.time())

    def stop(self, grace_seconds: float = 5.0) -> None:
        """Wait for the process to end on its own for a grace period, then
        terminate it (kill if it will not go), and join the reader."""
        process = self._process
        if process is None:
            return
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if process.stdout is not None:
            process.stdout.close()

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def terminate(self) -> None:
        if self.alive():
            self._process.terminate()


class PhaseTelemetry:
    """What the samplers saw during one phase. `result` is filled when the
    phase ends: the queue and activity series, the per-container series and the
    statement snapshot. `container_samples` keeps the raw docker stats samples
    (host clock) so a phase can take the generator's CPU over its own measured
    window."""

    def __init__(self) -> None:
        self.db_rows: list[DbRow] = []
        self.container_samples: list[ContainerSample] = []
        self.result: dict = {}


# docker stats refreshes about every two seconds, so a phase's figures include
# the last sample taken just before it began: a short phase (an upload step of
# a fraction of a second) would otherwise see none at all.
STATS_CARRY_SECONDS = 3.0


class Telemetry:
    """Starts and stops the samplers around a phase.

    `exec_db(argv)` is the host command line that runs argv in the db
    container; `psql(sql)` runs a statement as the owner and returns its
    tuples. Both belong to load.py, which pins them to the load project.

    The db sampler lives for one phase. `docker stats` lives from the first
    `ensure_stats()` to `close()`, across phases and stack resets, because its
    first figures arrive a second or two after it starts; a phase takes the
    samples from its own span."""

    def __init__(
        self,
        exec_db: Callable[[list[str]], list[str]],
        psql: Callable[[str], str],
        *,
        owner: str,
        database: str,
        stats_command: list[str] | None = None,
    ) -> None:
        self._exec_db = exec_db
        self._psql = psql
        self._owner = owner
        self._database = database
        self._stats_command = stats_command or ["docker", "stats", "--format", "{{json .}}"]
        self._stats_samples: list[ContainerSample] = []
        self._stats_reader: LineReader | None = None

    def _on_stats(self, line: str, arrived: float) -> None:
        sample = parse_stats_line(line, arrived)
        if sample is not None:
            self._stats_samples.append(sample)

    def ensure_stats(self) -> None:
        """Start the `docker stats` stream if it is not running (it ends if the
        Docker daemon restarts)."""
        if self._stats_reader is None or not self._stats_reader.alive():
            self._stats_reader = LineReader(self._stats_command, self._on_stats)
            self._stats_reader.start()

    def close(self) -> None:
        """Stop the `docker stats` stream."""
        if self._stats_reader is not None:
            self._stats_reader.terminate()
            self._stats_reader.stop(grace_seconds=1.0)
            self._stats_reader = None

    @contextmanager
    def phase(self) -> Iterator[PhaseTelemetry]:
        """Reset the statement counters, sample while the body runs, and fill
        `result` on the way out -- also when the body raises, so a failed phase
        leaves no sampler behind."""
        telemetry = PhaseTelemetry()
        self._psql(RESET_STATEMENTS_SQL)
        self.ensure_stats()
        stop_file = f"/tmp/load-sampler-{uuid.uuid4().hex}.stop"
        script = db_loop_script(stop_file, user=self._owner, database=self._database)

        def on_db(line: str, _arrived: float) -> None:
            row = parse_db_line(line)
            if row is not None:
                telemetry.db_rows.append(row)

        db = LineReader(self._exec_db(["sh", "-c", script]), on_db)
        started = time.time()
        db.start()
        try:
            yield telemetry
        finally:
            ended = time.time()
            subprocess.run(self._exec_db(["touch", stop_file]), capture_output=True, check=False)
            db.stop()
        queue, activity = fold_db_rows(telemetry.db_rows)
        telemetry.container_samples = [
            sample
            for sample in list(self._stats_samples)
            if started - STATS_CARRY_SECONDS <= sample.t <= ended
        ]
        containers = fold_container_samples(telemetry.container_samples)
        telemetry.result = {
            "queue_series": queue,
            "activity_series": activity,
            "container_series": containers,
            "pg_stat_statements": parse_statements(self._psql(STATEMENTS_SQL)),
        }
