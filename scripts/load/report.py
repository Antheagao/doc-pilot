"""README Performance region and charts, rendered from committed load artifacts.

Stdlib only, like artifact.py. Every number in the README's Performance section
comes from here: `render` turns the `load_v1` artifacts in a directory into the
markdown between the PERF_TABLE markers and two SVGs (light and dark) of the
worker scaling curve and the read latency against users. Nothing is typed by
hand, so a number can only change by committing a new artifact, and the sync
test (test_report.py) fails until the README and the SVGs are regenerated.

What is published, and what is refused:

- Only artifacts that pass `artifact.validate(publishable=True)`: a clean tree,
  a full plan, no invalid phase, zero spend, the scripted model. One bad file in
  scripts/load/results/ stops the report, it is never skipped, because a
  silently missing run reads as a smaller claim than the one that was measured.
  A development render (`--from DIR`) checks the same rules except the
  publishable ones, skips what fails, says so, and writes only to a scratch
  directory, never the README.
- An artifact whose label starts with "after" is an after; any other is a
  baseline. A baseline is identified by the scripted model's profile and
  latency scale, so zero and replay runs are separate groups. Each after of a
  group is compared with the run immediately before it, in order of
  `started_at` (the baseline for the first after), so a second change is not
  credited with the first one's gain; a pair is refused when its two runs
  cannot be compared (`comparability_problems`): a number that moved because
  the machine, Docker's share of it, the traffic, the model profile or an
  override moved says nothing about the code.

Medians and ranges come from the valid repetitions only, as artifact.spread
computes them. The read latency tables use each repetition's own knee, the last
ramp step that met the SLO, and the step after it.

Output is deterministic: no clock, no environment, fixed number formats, sorted
input, so the same artifacts give byte-identical markdown and SVG.
"""

from __future__ import annotations

import json
import math
import shlex
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import artifact
import phases

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
README_PATH = REPO_ROOT / "README.md"
CHART_DIR = REPO_ROOT / "screenshots" / "performance"

START_MARKER = "<!-- PERF_TABLE:START -->"
END_MARKER = "<!-- PERF_TABLE:END -->"
LIGHT_CHART = "load-light.svg"
DARK_CHART = "load-dark.svg"

# The profile each chart panel draws, first group with data wins, so a panel
# falls to the next profile when the first did not measure it. Worker throughput
# under `zero` is the application's own cost, while under `replay` it is mostly
# the scripted sleeps overlapping, so the scaling panel prefers `zero`. The read
# latency panel prefers `replay`, whose agent and judge calls put the model path
# under the reads that `zero` skips.
PANEL_PROFILES = {"scaling": ("zero", "replay"), "latency": ("replay", "zero")}
PROFILE_ORDER = ("zero", "replay")

PROFILE_BLURB = {
    "zero": "every scripted model call returns at once, so the numbers are the application's own cost",
    "replay": (
        "scripted model calls sleep for the latency the committed extraction runs recorded "
        "(a proxy, multiplied by the latency scale) and for the assumed agent step and judge durations"
    ),
}

# The read endpoints in table order: (artifact name, label, stats key).
LATENCY_SERIES = (
    ("search", "GET /search", "latency_ms"),
    ("ask", "POST /ask", "latency_ms"),
    ("ask_stream", "POST /ask/stream, first event", "first_event_ms"),
    ("ask_stream", "POST /ask/stream, answer", "answer_ms"),
    ("documents", "GET /documents", "latency_ms"),
)
CHART_ENDPOINT = "search"

# Chart colors. The series are the dashboard's slots 1 and 2 (frontend/src/app/
# globals.css, --series-1 and --series-2, light and dark), validated for color
# vision deficiency against these surfaces; test_report.py reads globals.css and
# fails if they drift. The surfaces and inks are the dashboard's --background,
# --border, --foreground and --muted.


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    border: str
    ink: str
    muted: str
    baseline: str
    after: str


LIGHT = Theme("light", "#ffffff", "#e4e4e7", "#18181b", "#52525b", "#2a78d6", "#eb6834")
DARK = Theme("dark", "#0b0b0e", "#26262c", "#f4f4f5", "#a1a1aa", "#3987e5", "#d95926")

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


class ReportError(Exception):
    """The artifacts cannot be reported: one is not publishable, a pair cannot
    be compared, or a baseline is ambiguous."""


# --- reading the artifacts ----------------------------------------------------


@dataclass(frozen=True)
class Run:
    name: str  # the file name
    doc: dict

    @property
    def is_after(self) -> bool:
        return self.doc["label"].lower().startswith("after")

    @property
    def group_key(self) -> tuple[str, float]:
        model = self.doc["model"]
        return model["profile"], float(model["latency_scale"])

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.doc["run_id"], self.name


@dataclass(frozen=True)
class Group:
    """A baseline and the afters that followed it, oldest `started_at` first."""

    baseline: Run
    afters: tuple[Run, ...]

    @property
    def pairs(self) -> tuple[tuple[Run, Run], ...]:
        """(before, after) for every after: the run just before it, which is
        the baseline for the first."""
        return tuple(zip((self.baseline, *self.afters), self.afters, strict=False))

    @property
    def profile(self) -> str:
        return self.baseline.group_key[0]

    @property
    def scale(self) -> float:
        return self.baseline.group_key[1]


def load_runs(directory: Path, *, publishable: bool) -> tuple[list[Run], list[str]]:
    """The `load_v1` artifacts (`*.json`) in `directory`, oldest first, and a
    note for each file left out.

    `publishable` is the committed-results mode: a file that is not JSON, not
    `load_v1`, or fails `artifact.validate(publishable=True)` raises
    ReportError naming every problem. Otherwise (a development render) a file
    that fails the plain validation is skipped with a note. A directory that
    does not exist holds no runs."""
    if not directory.is_dir():
        return [], []
    runs: list[Run] = []
    problems: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: not readable JSON ({exc.__class__.__name__})")
            continue
        if not isinstance(document, dict):
            problems.append(f"{path.name}: not a JSON object")
            continue
        if document.get("schema") != artifact.SCHEMA:
            problems.append(f"{path.name}: not a {artifact.SCHEMA} artifact")
            continue
        found = artifact.validate(document, publishable=publishable)
        if found:
            problems.extend(f"{path.name}: {problem}" for problem in found)
            continue
        runs.append(Run(path.name, document))
    if problems and publishable:
        raise ReportError(
            "these artifacts cannot be published (fix or remove them):\n  " + "\n  ".join(problems)
        )
    notes = [f"skipped {problem}" for problem in problems]
    return sorted(runs, key=lambda run: run.sort_key), notes


def _differences(label: str, left: dict, right: dict) -> list[str]:
    return [
        f"{label} {key} differs: before {left.get(key)!r}, after {right.get(key)!r}"
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]


def comparability_problems(before: dict, after: dict) -> list[str]:
    """Why two artifacts cannot be set side by side; empty when they can.

    Same machine, same Docker resources, the same scripted-model profile, scale
    and durations, the same scenario digest (the traffic and the plan), the same
    overrides and embedder, and the same methodology. Commit, label, times and
    results are the point of a comparison and are not compared, and neither are
    the parameters of the phases: a change under test may record one the before
    run never had (`embedding_threads`)."""
    problems = _differences("machine", before["machine"], after["machine"])
    docker_keys = ("ncpu", "memory_bytes", "operating_system", "architecture")
    problems += _differences(
        "Docker",
        {key: before["docker"].get(key) for key in docker_keys},
        {key: after["docker"].get(key) for key in docker_keys},
    )
    model_keys = ("provider", "profile", "latency_scale", "agent_step_ms", "judge_ms", "models")
    problems += _differences(
        "model",
        {key: before["model"].get(key) for key in model_keys},
        {key: after["model"].get(key) for key in model_keys},
    )
    if before["scenario"]["digest"] != after["scenario"]["digest"]:
        problems.append(
            f"scenario digest differs: before {before['scenario']['digest'][:12]}, "
            f"after {after['scenario']['digest'][:12]}"
        )
    problems += _differences("override", before["stack"]["overrides"], after["stack"]["overrides"])
    if before["stack"].get("embedder") != after["stack"].get("embedder"):
        problems.append(
            f"embedder differs: before {before['stack'].get('embedder')!r}, after {after['stack'].get('embedder')!r}"
        )
    if before.get("methodology") != after.get("methodology"):
        problems.append("methodology differs")
    return problems


def build_groups(runs: list[Run], *, strict: bool) -> tuple[list[Group], list[str]]:
    """Order the afters of each group after its baseline and check every
    consecutive pair.

    Raises ReportError when the runs disagree on the methodology, when an after
    has no baseline of its model profile and scale, or when a pair (an after and
    the run before it) fails `comparability_problems`. Two baselines of one
    group are an error when `strict` (the committed results must say which one
    is the baseline) and
    otherwise the newest wins, with a note."""
    notes: list[str] = []
    if len({run.doc.get("methodology") for run in runs}) > 1:
        raise ReportError("the artifacts were measured with different methodologies and cannot be reported together")
    baselines: dict[tuple[str, float], list[Run]] = {}
    afters: dict[tuple[str, float], list[Run]] = {}
    for run in runs:
        (afters if run.is_after else baselines).setdefault(run.group_key, []).append(run)
    for key, candidates in baselines.items():
        if len(candidates) > 1:
            names = ", ".join(run.name for run in candidates)
            if strict:
                raise ReportError(
                    f"{len(candidates)} baselines for model profile {key[0]!r} at scale {key[1]:g} ({names}): "
                    "remove the superseded one"
                )
            notes.append(f"using the newest of {names} as the {key[0]} baseline")
    for key, extra in afters.items():
        if key not in baselines:
            have = ", ".join(f"{p!r} at scale {s:g}" for p, s in sorted(baselines)) or "none"
            raise ReportError(
                f"{extra[0].name} has no baseline of model profile {key[0]!r} at scale {key[1]:g} "
                f"(baselines: {have}); a before and after must use the same scripted model"
            )
    groups = []
    for key in sorted(baselines, key=_group_order):
        baseline = baselines[key][-1]
        group_afters = tuple(sorted(afters.get(key, ()), key=lambda run: (run.doc["started_at"], run.name)))
        group = Group(baseline, group_afters)
        for before, after in group.pairs:
            problems = comparability_problems(before.doc, after.doc)
            if problems:
                raise ReportError(
                    f"{after.name} cannot be compared with {before.name}:\n  " + "\n  ".join(problems)
                )
        groups.append(group)
    return groups, notes


def _group_order(key: tuple[str, float]) -> tuple[int, float]:
    profile, scale = key
    return (PROFILE_ORDER.index(profile) if profile in PROFILE_ORDER else len(PROFILE_ORDER), scale)


# --- numbers out of an artifact -----------------------------------------------


def _valid(doc: dict, name: str) -> list[dict]:
    return [phase for phase in doc["phases"] if phase["name"] == name and phase["valid"]]


def _spreads(pairs: Iterable[tuple[object, float | None]]) -> dict:
    """`{key: artifact.spread}` of the values by key, keys sorted, None left out."""
    values: dict = {}
    for key, value in pairs:
        if value is not None:
            values.setdefault(key, []).append(value)
    return {key: artifact.spread(values[key]) for key in sorted(values)}


def ingest_spreads(doc: dict) -> dict[int, dict]:
    """Uploads per second by concurrent users."""
    return _spreads((p["params"]["users"], p["results"]["uploads_per_s"]) for p in _valid(doc, "ingest"))


def ingest_latency(doc: dict, key: str) -> dict[int, dict]:
    """The upload latency quantile `key` (milliseconds) by concurrent users."""

    def quantile(phase: dict) -> float | None:
        latency = phase["endpoints"].get("upload", {}).get("latency_ms")
        return latency[key] if latency else None

    return _spreads((p["params"]["users"], quantile(p)) for p in _valid(doc, "ingest"))


def _documents_per_s(phase: dict) -> float | None:
    results = phase["results"]
    if results.get("documents_per_s") is not None:
        return results["documents_per_s"]
    if results.get("jobs_per_s") is None:
        return None
    return results["jobs_per_s"] / phases.JOBS_PER_DOCUMENT


def scaling_documents(doc: dict) -> dict[int, dict]:
    """Documents per second by worker count."""
    return _spreads((p["params"]["workers"], _documents_per_s(p)) for p in _valid(doc, "scaling"))


def scaling_jobs(doc: dict) -> dict[int, dict]:
    """Jobs (extract and index) per second by worker count."""
    return _spreads((p["params"]["workers"], p["results"]["jobs_per_s"]) for p in _valid(doc, "scaling"))


def _ramp_by_repetition(doc: dict) -> dict[int, list[dict]]:
    steps: dict[int, list[dict]] = {}
    for phase in doc["phases"]:
        if phase["name"] == "ramp":
            steps.setdefault(phase["repetition"], []).append(phase)
    return steps


def knee_spread(doc: dict) -> dict | None:
    """The knee in users across the repetitions that found one."""
    knees = [
        steps[0]["results"]["knee_users"]
        for steps in _ramp_by_repetition(doc).values()
        if steps[0]["results"].get("knee_users") is not None
    ]
    return artifact.spread(knees) if knees else None


def knee_steps(doc: dict) -> tuple[list[dict], list[dict]]:
    """The valid ramp steps at each repetition's knee and one step past it. A
    repetition whose first step already failed has no knee, and its first step
    is the one past it."""
    at_knee: list[dict] = []
    past: list[dict] = []
    for steps in _ramp_by_repetition(doc).values():
        knee = steps[0]["results"].get("knee_users")
        users = [step["params"]["users"] for step in steps]
        if knee is None:
            chosen_past = steps[0]
            chosen_knee = None
        else:
            index = users.index(knee) if knee in users else None
            if index is None:
                continue
            chosen_knee = steps[index]
            chosen_past = steps[index + 1] if index + 1 < len(steps) else None
        if chosen_knee is not None and chosen_knee["valid"]:
            at_knee.append(chosen_knee)
        if chosen_past is not None and chosen_past["valid"]:
            past.append(chosen_past)
    return at_knee, past


def ramp_value(doc: dict, endpoint: str, stats_key: str, quantile: str) -> dict[int, dict]:
    """A read latency quantile (milliseconds) of the valid ramp steps by users."""

    def value(phase: dict) -> float | None:
        latency = phase["endpoints"].get(endpoint, {}).get(stats_key)
        return latency[quantile] if latency else None

    return _spreads((p["params"]["users"], value(p)) for p in _valid(doc, "ramp"))


# --- number formats ------------------------------------------------------------


def fmt_number(value: float) -> str:
    """Three significant places or so: 1,234 / 56.7 / 8.91."""
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_g(value: float) -> str:
    return f"{value:g}"


def fmt_range(spread: dict, fmt=fmt_number) -> str:
    """The min-max of the repetitions, "-" when they print alike."""
    low, high = fmt(spread["min"]), fmt(spread["max"])
    return "-" if low == high else f"{low} to {high}"


def fmt_with_range(spread: dict, fmt=fmt_number) -> str:
    """The median, and the range in parentheses when there is one."""
    spread_text = fmt_range(spread, fmt)
    return fmt(spread["median"]) if spread_text == "-" else f"{fmt(spread['median'])} ({spread_text})"


def fmt_percent(value: float) -> str:
    return f"{value:.1%}"


def fmt_gib(size_bytes: int) -> str:
    return f"{size_bytes / 2**30:.2f}".rstrip("0").rstrip(".")


def fmt_change(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


# --- markdown --------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def machine_line(doc: dict) -> str:
    machine, docker, postgres, stack = doc["machine"], doc["docker"], doc["postgres"], doc["stack"]
    cores = f"{machine['cores']} cores"
    if machine.get("cores_performance") and machine.get("cores_efficiency"):
        cores += f" ({machine['cores_performance']} performance, {machine['cores_efficiency']} efficiency)"
    version = postgres["version"].split(" on ")[0].split(" (")[0]
    api_pool = stack["api_pool"]
    return (
        f"**Machine.** {machine['cpu']}, {cores}, {fmt_gib(machine['memory_bytes'])} GiB, {machine['os']}. "
        f"**Docker.** {docker['operating_system']} {docker['version']} with {docker['ncpu']} CPUs and "
        f"{fmt_gib(docker['memory_bytes'])} GiB for every container, the load generator included. "
        f"**Stack.** {version} with pgvector {postgres['pgvector']} (max_connections {postgres['max_connections']}), "
        f"API pool {api_pool['size']} plus {api_pool['max_overflow']} overflow, "
        f"{stack['uvicorn_workers']} uvicorn worker{'' if stack['uvicorn_workers'] == 1 else 's'}, "
        f"embedder {stack['embedder']}."
    )


def command_block(label: str, doc: dict) -> str:
    return f"{label} (commit `{doc['git']['commit'][:7]}`, run `{doc['run_id']}`):\n\n```\n{shlex.join(doc['command'])}\n```"


def model_line(group: Group) -> str:
    model = group.baseline.doc["model"]
    text = f"**Scripted model.** Profile `{group.profile}` at latency scale {group.scale:g}: {PROFILE_BLURB.get(group.profile, '')}."
    if group.profile == "replay":
        text += f" Agent step {fmt_g(model['agent_step_ms'])} ms and judge call {fmt_g(model['judge_ms'])} ms are assumptions."
    return text


def ingest_section(doc: dict) -> str | None:
    rates = ingest_spreads(doc)
    if not rates:
        return None
    p50, p95 = ingest_latency(doc, "p50"), ingest_latency(doc, "p95")
    rows = [
        [
            fmt_g(users),
            fmt_number(spread["median"]),
            fmt_range(spread),
            fmt_number(p50[users]["median"]) if users in p50 else "-",
            fmt_number(p95[users]["median"]) if users in p95 else "-",
            str(spread["n"]),
        ]
        for users, spread in rates.items()
    ]
    table = _table(
        ["Concurrent users", "Uploads/s (median)", "Range", "Upload p50 (ms)", "Upload p95 (ms)", "Runs"], rows
    )
    return f"#### Ingest\n\n{table}\n\nUploads through `POST /documents` with no worker running, so nothing consumes the queue."


def scaling_section(doc: dict) -> str | None:
    documents, jobs = scaling_documents(doc), scaling_jobs(doc)
    if not documents:
        return None
    smallest = min(documents)
    base = documents[smallest]["median"]
    rows = []
    for workers, spread in documents.items():
        job_spread = jobs.get(workers)
        rows.append(
            [
                fmt_g(workers),
                fmt_number(spread["median"]),
                fmt_range(spread),
                fmt_number(job_spread["median"]) if job_spread else "-",
                fmt_range(job_spread) if job_spread else "-",
                f"{spread['median'] / base:.2f}x" if base else "-",
                str(spread["n"]),
            ]
        )
    table = _table(
        [
            "Workers",
            "Documents/s (median)",
            "Range",
            "Jobs (extract+index)/s (median)",
            "Range",
            f"Speedup vs {fmt_g(smallest)} worker{'' if smallest == 1 else 's'}",
            "Runs",
        ],
        rows,
    )
    return (
        f"#### Worker scaling\n\n{table}\n\nA document is two jobs, an extract and an index. "
        "Throughput is measured between the 10% and 90% marks of a backlog uploaded with no worker running."
    )


def _spread_of(values: Iterable[float | None]) -> dict | None:
    present = [value for value in values if value is not None]
    return artifact.spread(present) if present else None


def _latency_cells(steps: list[dict], endpoint: str, stats_key: str) -> list[str] | None:
    """The cells of one endpoint's row: the median across the steps of p50, p95
    (with its range) and p99, the request rate and the error rate; None when
    the endpoint has no successful request in any of the steps."""
    stats = [step["endpoints"].get(endpoint) for step in steps]
    stats = [entry for entry in stats if entry]

    def quantile(name: str) -> dict | None:
        return _spread_of((entry.get(stats_key) or {}).get(name) for entry in stats)

    p50, p95, p99 = quantile("p50"), quantile("p95"), quantile("p99")
    if not (p50 and p95 and p99):
        return None
    rate = _spread_of(entry["rps"] for entry in stats)
    errors = _spread_of(entry["error_rate"] for entry in stats)
    return [
        fmt_number(p50["median"]),
        fmt_with_range(p95),
        fmt_number(p99["median"]),
        fmt_number(rate["median"]) if rate else "-",
        fmt_percent(errors["median"]) if errors else "-",
        str(p50["n"]),
    ]


def latency_section(doc: dict) -> str | None:
    at_knee, past = knee_steps(doc)
    if not (at_knee or past):
        return None
    rows = []
    for title, steps in (("At the knee", at_knee), ("One step past", past)):
        if not steps:
            continue
        users = artifact.spread([step["params"]["users"] for step in steps])
        users_text = fmt_with_range(users, fmt_g)
        for endpoint, label, stats_key in LATENCY_SERIES:
            cells = _latency_cells(steps, endpoint, stats_key)
            if cells:
                rows.append([title, users_text, label, *cells])
    if not rows:
        return None
    table = _table(
        ["Step", "Users", "Endpoint", "p50 (ms)", "p95 (ms, range)", "p99 (ms)", "Requests/s", "Errors", "Runs"],
        rows,
    )
    params = (at_knee or past)[0]["params"]
    knee = knee_spread(doc)
    knee_text = (
        f"The knee, the last step that met both limits, was {fmt_with_range(knee, fmt_g)} users"
        f" over {knee['n']} run{'s' if knee['n'] != 1 else ''}"
        if knee
        else "No step met both limits"
    )
    sizes = (at_knee or past)[0]["results"].get("tenants") or {}
    tenants = ", ".join(
        f"`{slug}` {size['documents']:,} documents and {size['chunks']:,} chunks" for slug, size in sorted(sizes.items())
    )
    return (
        f"#### Read latency\n\n{table}\n\n"
        f"A step fails when more than {params['max_error_rate']:.0%} of its reads fail or the `/search` p95 is over "
        f"{fmt_number(params['slo_search_p95_ms'])} ms. {knee_text}. "
        f"Each run's own knee and the step after it, then the median across runs. Tenants: {tenants or 'not recorded'}."
    )


@dataclass(frozen=True)
class Metric:
    name: str
    before: dict
    after: dict
    unit: str = ""  # "users" formats the values as whole counts


def _plural(count: float, noun: str) -> str:
    return f"{fmt_g(count)} {noun}" + ("" if count == 1 else "s")


def comparison_metrics(before: dict, after: dict) -> list[Metric]:
    """The measurements both artifacts have, for the before and after table:
    scaling documents/s per worker count, ingest uploads/s per user count, the
    knee, and each read endpoint's p95 at the baseline's knee (the largest ramp
    step at or below its median) and at the step after."""
    metrics: list[Metric] = []
    left, right = scaling_documents(before), scaling_documents(after)
    for workers in sorted(set(left) & set(right)):
        metrics.append(Metric(f"Documents/s at {_plural(workers, 'worker')}", left[workers], right[workers]))
    left, right = ingest_spreads(before), ingest_spreads(after)
    for users in sorted(set(left) & set(right)):
        metrics.append(Metric(f"Uploads/s at {_plural(users, 'user')}", left[users], right[users]))
    left_knee, right_knee = knee_spread(before), knee_spread(after)
    if not (left_knee and right_knee):
        return metrics
    metrics.append(Metric("Knee (users)", left_knee, right_knee, unit="users"))
    steps = sorted({step["params"]["users"] for step in _valid(before, "ramp")})
    at_or_below = [users for users in steps if users <= left_knee["median"]]
    if not at_or_below:
        return metrics
    index = steps.index(at_or_below[-1])
    for users in steps[index : index + 2]:
        for endpoint, label, stats_key in LATENCY_SERIES:
            if stats_key == "first_event_ms":
                continue  # the answer time carries the stream's story; the table stays short
            left = ramp_value(before, endpoint, stats_key, "p95")
            right = ramp_value(after, endpoint, stats_key, "p95")
            if users in left and users in right:
                metrics.append(Metric(f"{label} p95 (ms) at {_plural(users, 'user')}", left[users], right[users]))
    return metrics


def runs_separated(before: dict, after: dict) -> bool:
    """Every after run lies beyond every before run. With three runs a side, an
    after median just past a tight before range is noise, not a change."""
    return after["min"] > before["max"] or after["max"] < before["min"]


def embedding_threads_by_workers(doc: dict) -> dict[int, int | None]:
    """The ONNX threads each worker count ran with, from the phases that
    recorded `embedding_threads` (None is the default of every core); empty for
    a run made before it was recorded."""
    return {
        phase["params"]["workers"]: phase["params"]["embedding_threads"]
        for phase in doc["phases"]
        if "embedding_threads" in phase["params"] and "workers" in phase["params"]
    }


def embedding_threads_note(before: dict, after: dict) -> str | None:
    """A sentence on how the workers' threads changed between two runs, when the
    after recorded them and the before did not. It is read from the phases, not
    written into the compared methodology text, so a run's methodology stays
    comparable with every earlier artifact."""
    sized = embedding_threads_by_workers(after)
    if not sized or embedding_threads_by_workers(before):
        return None
    items = []
    unit_said = False  # only the first count of threads says what the number is
    for workers in sorted(sized):
        threads = sized[workers]
        if threads is None:
            value = "onnxruntime's default of every core"
        else:
            value = str(threads) if unit_said else f"{threads} threads"
            unit_said = True
        items.append(f"{value} at {_plural(workers, 'worker')}")
    listed = " and ".join(items) if len(items) < 3 else f"{', '.join(items[:-1])}, and {items[-1]}"
    return (
        "The after run sized each worker's ONNX threads to the Docker CPUs divided by the number of "
        f"workers, at least 1: {listed}; the before run left the default in every worker."
    )


def comparison_title(before: Run, after: Run, group: Group) -> str:
    """The section heading. Against the baseline it names only the after: the
    group's baseline block, with its run id, sits right above. Against an earlier
    after it names that run too, so a second change is not read as the first."""
    title = f"#### Before and after: {after_title(after)}"
    if before is group.baseline:
        return title
    return f"{title} against {after_title(before)} (run `{before.doc['run_id']}`)"


def comparison_section(group: Group) -> str | None:
    sections = []
    for before, after in group.pairs:
        metrics = comparison_metrics(before.doc, after.doc)
        if not metrics:
            continue
        rows = []
        for metric in metrics:
            earlier, now = metric.before, metric.after
            fmt = fmt_g if metric.unit == "users" else fmt_number
            rows.append(
                [
                    metric.name,
                    fmt_with_range(earlier, fmt),
                    fmt_with_range(now, fmt),
                    fmt_change(earlier["median"], now["median"]),
                    "yes" if runs_separated(earlier, now) else "no",
                ]
            )
        table = _table(["Measure", "Before (median, range)", "After (median, range)", "Change", "Runs separated"], rows)
        threads = embedding_threads_note(before.doc, after.doc)
        sections.append(
            f"{comparison_title(before, after, group)}\n\n"
            f"{command_block('After', after.doc)}\n\n{table}\n\n"
            "Both runs used the same machine, Docker resources, scripted model, traffic and overrides. "
            + (f"{threads} " if threads else "")
            + "A change counts as evidence only when the runs are separated: every after run lies beyond every before run."
        )
    return "\n\n".join(sections) if sections else None


def after_slug(run: Run) -> str:
    label = run.doc["label"]
    return label[len("after") :].lstrip("-_ ")


def after_title(run: Run) -> str:
    slug = after_slug(run)
    return f"`{slug}`" if slug else "the change"


def group_markdown(group: Group) -> str:
    doc = group.baseline.doc
    parts = [
        f"### Scripted model `{group.profile}`, latency scale {group.scale:g}",
        model_line(group),
        machine_line(doc),
        command_block("Baseline", doc),
    ]
    for section in (ingest_section(doc), scaling_section(doc), latency_section(doc), comparison_section(group)):
        if section:
            parts.append(section)
    return "\n\n".join(parts)


NOTICE = (
    "*Generated by `python scripts/load/load.py report --update-readme` from the committed artifacts in "
    "`scripts/load/results/`; do not edit by hand.*"
)
NO_RUN = (
    "**No published run yet.** The tables and charts appear here once a full run is committed to "
    "`scripts/load/results/`."
)


def render_markdown(groups: list[Group], notes: Iterable[str] = (), *, development: bool = False) -> str:
    """The text between the PERF_TABLE markers (without the blank lines that
    separate it from them). A development render opens with a banner and its
    notes."""
    parts = [NOTICE]
    if development:
        banner = "**Development render, not publishable.**"
        parts.append(" ".join([banner, *(f"{note}." for note in notes)]))
    if not groups:
        parts.append(NO_RUN)
        return "\n\n".join(parts)
    parts.append(f"**Method.** {groups[0].baseline.doc['methodology']}")
    parts += [group_markdown(group) for group in groups]
    return "\n\n".join(parts)


# --- charts -------------------------------------------------------------------

WIDTH, HEIGHT = 960, 440
PLOT_TOP, PLOT_BOTTOM = 148, 360
PLOTS = ((64, 384), (534, 854))  # x extent of the left and right plot
MIN_LABEL_GAP = 14.0


@dataclass(frozen=True)
class Point:
    x: float
    median: float
    low: float
    high: float


@dataclass(frozen=True)
class Series:
    name: str
    role: str  # "baseline" or "after"
    points: tuple[Point, ...]


@dataclass(frozen=True)
class PanelData:
    source: str  # "replay profile, latency scale 0.1"
    after: str  # "After: onnx-threads, on top of query-extractions", "" with no after drawn
    series: tuple[Series, ...]
    ideal: tuple[tuple[float, float], ...] = ()
    limit: float | None = None


def _series_name(run: Run) -> str:
    if not run.is_after:
        return "Baseline"
    slug = after_slug(run)
    return f"After ({slug})" if slug else "After"


def _points(spreads: dict) -> tuple[Point, ...]:
    return tuple(Point(float(key), s["median"], s["min"], s["max"]) for key, s in spreads.items())


def _source(group: Group) -> str:
    return f"{group.profile} profile, latency scale {group.scale:g}"


def _panel_groups(groups: list[Group], panel: str) -> list[Group]:
    """The groups in the order `panel` prefers their profiles (PANEL_PROFILES)."""
    preferred = PANEL_PROFILES[panel]
    rank = {profile: index for index, profile in enumerate(preferred)}
    return sorted(groups, key=lambda g: (rank.get(g.profile, len(rank)), g.scale))


def _after_note(group: Group, series: tuple[Series, ...]) -> str:
    """Which after the panel draws, and the earlier ones its numbers include:
    each after is measured on top of the run before it."""
    if not any(s.role == "after" for s in series):
        return ""
    *earlier, latest = (after_slug(run) or "the change" for run in group.afters)
    return f"After: {latest}" + (f", on top of {', '.join(earlier)}" if earlier else "")


def _panel_series(group: Group, measured: Callable[[dict], dict]) -> tuple[Series, ...]:
    """The group's baseline and its latest after, each with the points
    `measured` reads from its artifact (none for a run that lacks them)."""
    return tuple(
        Series(_series_name(run), "after" if run.is_after else "baseline", _points(measured(run.doc)))
        for run in (group.baseline, *group.afters[-1:])
        if measured(run.doc)
    )


def scaling_panel(groups: list[Group]) -> PanelData | None:
    for group in _panel_groups(groups, "scaling"):
        series = _panel_series(group, scaling_documents)
        if not series or series[0].role != "baseline":
            continue
        first = series[0].points[0]
        ideal = tuple((p.x, first.median * p.x / first.x) for p in series[0].points) if len(series[0].points) > 1 else ()
        return PanelData(_source(group), _after_note(group, series), series, ideal)
    return None


def latency_panel(groups: list[Group]) -> PanelData | None:
    for group in _panel_groups(groups, "latency"):
        series = _panel_series(group, lambda doc: ramp_value(doc, CHART_ENDPOINT, "latency_ms", "p95"))
        if not series or series[0].role != "baseline":
            continue
        steps = _valid(group.baseline.doc, "ramp")
        limit = steps[0]["params"]["slo_search_p95_ms"] if steps else None
        return PanelData(_source(group), _after_note(group, series), series, limit=limit)
    return None


def _f(value: float) -> str:
    return f"{value:.1f}"


def _linear_axis(maximum: float) -> tuple[float, list[float]]:
    """A top and ticks from zero: the first step of 1, 2 or 5 times a power of
    ten that makes at most five intervals."""
    maximum = max(maximum, 1e-9)
    magnitude = 10 ** math.floor(math.log10(maximum))
    for step in (magnitude * m for m in (0.2, 0.5, 1, 2, 5, 10)):
        if maximum / step <= 5:
            count = math.ceil(maximum / step - 1e-9)
            return count * step, [i * step for i in range(count + 1)]
    raise AssertionError("a step of ten magnitudes always fits")


def _tick_text(value: float) -> str:
    if value >= 1:
        return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"
    return f"{value:.2f}" if value else "0"


class _Svg:
    """A list of SVG elements with the attributes every chart element needs."""

    def __init__(self, theme: Theme) -> None:
        self.theme = theme
        self.parts: list[str] = []

    def line(self, x1, y1, x2, y2, color, width=1.0, dash=None, opacity=None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' stroke-opacity="{opacity}"' if opacity else ""
        self.parts.append(
            f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{color}" '
            f'stroke-width="{width}"{extra}/>'
        )

    def text(self, x, y, content, *, color=None, size=12, anchor="start", weight="400") -> None:
        self.parts.append(
            f'<text x="{_f(x)}" y="{_f(y)}" fill="{color or self.theme.muted}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(content)}</text>'
        )

    def marker(self, x, y, role: str, color: str) -> None:
        ring = f'stroke="{self.theme.surface}" stroke-width="2"'
        if role == "after":
            self.parts.append(f'<rect x="{_f(x - 4.5)}" y="{_f(y - 4.5)}" width="9" height="9" fill="{color}" {ring}/>')
        else:
            self.parts.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="4.5" fill="{color}" {ring}/>')

    def polyline(self, points: list[tuple[float, float]], color: str, dash: str | None = None) -> None:
        coords = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{extra}/>'
        )


def _series_color(theme: Theme, role: str) -> str:
    return theme.after if role == "after" else theme.baseline


def _draw_end_labels(svg: _Svg, labels: list[tuple[float, float, str]]) -> None:
    """Direct labels at the right end of each series, top to bottom, skipping a
    label that would sit within MIN_LABEL_GAP of the one above it: the legend
    names every series, so a dropped label loses nothing."""
    last = -1e9
    for x, y, content in sorted(labels, key=lambda item: (item[1], item[2])):
        if y - last >= MIN_LABEL_GAP:
            svg.text(x + 10, y + 4, content, color=svg.theme.ink, size=12)
            last = y


def _draw_ideal(
    svg: _Svg,
    points: list[tuple[float, float]],
    end: tuple[float, float],
    labels: list[tuple[float, float, str]],
) -> None:
    """The ideal line, cut where it leaves the top of the plot (a line that rises
    for the whole plot is drawn to the edge, with a label that says where it
    would end: `end` is its last point in data units). A line that fits is
    labelled at its end like the series."""
    if points[-1][1] >= PLOT_TOP:
        svg.polyline(points, svg.theme.muted, dash="6 4")
        labels.append((points[-1][0], points[-1][1], "Ideal"))
        return
    inside = [point for point in points if point[1] >= PLOT_TOP]
    (x1, y1), (x2, y2) = points[len(inside) - 1], points[len(inside)]
    exit_x = x1 + (x2 - x1) * (y1 - PLOT_TOP) / (y1 - y2)
    svg.polyline([*inside, (exit_x, PLOT_TOP)], svg.theme.muted, dash="6 4")
    svg.text(exit_x + 10, PLOT_TOP + 16, f"Ideal: {end[1]:,.0f} at {end[0]:g} workers", color=svg.theme.ink, size=12)


def _draw_panel(svg: _Svg, index: int, data: PanelData | None, *, kind: str) -> None:
    theme = svg.theme
    left, right = PLOTS[index]
    title = "Throughput vs worker containers" if kind == "scaling" else "/search p95 latency vs concurrent users"
    svg.text(left, 96, title, color=theme.ink, size=14, weight="600")
    if data is None:
        svg.text(left, 190, "Not measured in the published runs.", size=12)
        return
    svg.text(left, 114, data.source, size=11)
    if data.after:
        svg.text(left, 130, data.after, size=11)
    xs = sorted({p.x for s in data.series for p in s.points})
    if kind == "scaling":
        x_max = xs[-1] if len(xs) > 1 else xs[-1] * 2  # a lone point sits mid-plot, not on the edge
        # The axis fits the measurements, not the ideal line: where scaling
        # falls far short, ideal would squash every series into the bottom.
        top, ticks = _linear_axis(max(p.high for s in data.series for p in s.points))

        def x_of(value: float) -> float:
            return left + (right - left) * value / x_max

        def y_of(value: float) -> float:
            return PLOT_BOTTOM - (PLOT_BOTTOM - PLOT_TOP) * value / top

        tick_labels = [(t, _tick_text(t)) for t in ticks]
        y_title = "Documents per second"
        x_title = "Worker containers"
    else:
        values = [p.low for s in data.series for p in s.points] + [p.high for s in data.series for p in s.points]
        if data.limit:
            values.append(data.limit)
        low_exp = math.floor(math.log10(max(min(values), 1e-3)))
        high_exp = math.ceil(math.log10(max(values)))
        if high_exp <= low_exp:
            high_exp = low_exp + 1
        positions = {x: i for i, x in enumerate(xs)}

        def x_of(value: float) -> float:
            if len(xs) == 1:
                return (left + right) / 2
            return left + (right - left) * positions[value] / (len(xs) - 1)

        def y_of(value: float) -> float:
            fraction = (math.log10(max(value, 10**low_exp)) - low_exp) / (high_exp - low_exp)
            return PLOT_BOTTOM - (PLOT_BOTTOM - PLOT_TOP) * fraction

        tick_labels = [(10.0**e, f"{10**e:,}" if e >= 0 else f"{10.0**e:g}") for e in range(low_exp, high_exp + 1)]
        y_title = "p95 latency (ms, log scale)"
        x_title = "Concurrent users (closed loop)"
    for value, label in tick_labels:
        y = y_of(value)
        svg.line(left, y, right, y, theme.border, 1)
        svg.text(left - 8, y + 4, label, size=11, anchor="end")
    svg.line(left, PLOT_BOTTOM, right, PLOT_BOTTOM, theme.muted, 1)
    for x in xs:
        svg.text(x_of(x), PLOT_BOTTOM + 18, f"{x:g}", size=11, anchor="middle")
    svg.text((left + right) / 2, PLOT_BOTTOM + 40, x_title, size=12, anchor="middle")
    svg.parts.append(
        f'<text transform="translate({_f(left - 48)} {_f((PLOT_TOP + PLOT_BOTTOM) / 2)}) rotate(-90)" '
        f'fill="{theme.muted}" font-size="12" text-anchor="middle">{escape(y_title)}</text>'
    )
    labels: list[tuple[float, float, str]] = []
    if data.limit:
        y = y_of(data.limit)
        svg.line(left, y, right, y, theme.muted, 1.5, dash="6 4")
        svg.text(left + 6, y + 16, f"SLO {data.limit:,.0f} ms", size=11)
    if data.ideal:
        _draw_ideal(svg, [(x_of(x), y_of(y)) for x, y in data.ideal], data.ideal[-1], labels)
    for series in data.series:
        color = _series_color(theme, series.role)
        coords = [(x_of(p.x), y_of(p.median)) for p in series.points]
        for point in series.points:
            if point.low != point.high:
                x, top_y, low_y = x_of(point.x), y_of(point.high), y_of(point.low)
                svg.line(x, top_y, x, low_y, color, 1.5, opacity="0.7")
                svg.line(x - 4, top_y, x + 4, top_y, color, 1.5, opacity="0.7")
                svg.line(x - 4, low_y, x + 4, low_y, color, 1.5, opacity="0.7")
        if len(coords) > 1:
            svg.polyline(coords, color)
        for x, y in coords:
            svg.marker(x, y, series.role, color)
        labels.append((coords[-1][0], coords[-1][1], "After" if series.role == "after" else "Baseline"))
    _draw_end_labels(svg, labels)


def _legend(svg: _Svg, panels: list[PanelData | None]) -> None:
    """The roles the panels draw, named generically: each panel's subtitle says
    which after it shows, and the panels may draw different ones."""
    theme = svg.theme
    drawn = {series.role for panel in panels for series in (panel.series if panel else ())}
    entries: list[tuple[str, str, str | None]] = [  # (label, role, color)
        (label, role, _series_color(theme, role))
        for label, role in (("Baseline", "baseline"), ("After", "after"))
        if role in drawn
    ]
    if any(panel and panel.ideal for panel in panels):
        entries.append(("Ideal linear scaling", "ideal", None))
    x = 64.0
    for label, role, color in entries:
        if role == "ideal":
            svg.line(x, 60, x + 22, 60, theme.muted, 2, dash="6 4")
        else:
            svg.line(x, 60, x + 22, 60, color, 2)
            svg.marker(x + 11, 60, role, color)
        svg.text(x + 30, 64, label, color=theme.ink, size=12)
        x += 30 + 7 * len(label) + 28


def _describe(scaling: PanelData | None, latency: PanelData | None) -> str:
    parts = []
    if scaling:
        for series in scaling.series:
            points = ", ".join(f"{p.x:g} workers {fmt_number(p.median)}" for p in series.points)
            parts.append(f"{series.name} documents per second: {points}.")
    if latency:
        for series in latency.series:
            points = ", ".join(f"{p.x:g} users {fmt_number(p.median)} ms" for p in series.points)
            parts.append(f"{series.name} /search p95: {points}.")
    return " ".join(parts) or "No published run yet."


def render_chart(groups: list[Group], theme: Theme, note: str = "") -> str:
    """One SVG of the two panels (scaling, read latency) in a theme. The surface
    is drawn into the image, so it reads the same on either GitHub theme."""
    scaling, latency = scaling_panel(groups), latency_panel(groups)
    svg = _Svg(theme)
    if not groups:
        description = "No published run yet."
        svg.text(WIDTH / 2, 54, "No published run yet", color=theme.ink, size=16, weight="600", anchor="middle")
        svg.text(WIDTH / 2, 78, "The charts are drawn from the artifacts committed to scripts/load/results/.", anchor="middle")
        height = 112
    else:
        description = _describe(scaling, latency)
        doc = groups[0].baseline.doc
        svg.text(64, 28, "Load test: worker scaling and read latency", color=theme.ink, size=16, weight="600")
        svg.text(64, 46, f"{doc['machine']['cpu']}, Docker with {doc['docker']['ncpu']} CPUs, scripted model", size=12)
        _legend(svg, [scaling, latency])
        _draw_panel(svg, 0, scaling, kind="scaling")
        _draw_panel(svg, 1, latency, kind="latency")
        height = HEIGHT
        if note:
            svg.text(64, height - 12, note, size=11)
    title = "Load test results"
    body = "\n".join(svg.parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
        f'font-family="{FONT}" role="img" aria-labelledby="title desc">\n'
        f"<title id=\"title\">{escape(title)}</title>\n"
        f'<desc id="desc">{escape(description)}</desc>\n'
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="12" fill="{theme.surface}" '
        f'stroke="{theme.border}"/>\n{body}\n</svg>\n'
    )


# --- the whole report ----------------------------------------------------------


@dataclass(frozen=True)
class Rendered:
    markdown: str
    charts: dict[str, str]  # file name to SVG text
    notes: tuple[str, ...] | list[str] = ()


def render(directory: Path, *, publishable: bool) -> Rendered:
    """Markdown and both charts from the artifacts in `directory`. ReportError
    when they cannot be published (see the module docstring). Not `publishable`
    is the development render: the output says so, in the markdown and on the
    charts, and lists what was skipped or would be refused."""
    runs, notes = load_runs(directory, publishable=publishable)
    groups, group_notes = build_groups(runs, strict=publishable)
    notes += group_notes
    if not publishable:
        for run in runs:
            problems = artifact.validate(run.doc, publishable=True)
            if problems:
                notes.append(f"{run.name} would be refused: {'; '.join(problems)}")
    chart_note = "" if publishable else "development render, not publishable"
    return Rendered(
        render_markdown(groups, notes, development=not publishable),
        {LIGHT_CHART: render_chart(groups, LIGHT, chart_note), DARK_CHART: render_chart(groups, DARK, chart_note)},
        notes,
    )


# --- the README ----------------------------------------------------------------


def marked_region(text: str) -> str:
    """What sits between the markers, blank lines included."""
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    if start == -1 or end == -1 or start > end:
        raise ReportError(f"README: expected {START_MARKER} before {END_MARKER}")
    return text[start + len(START_MARKER) : end]


def region_for(markdown: str) -> str:
    return f"\n\n{markdown}\n\n"


def splice(text: str, markdown: str) -> str:
    """`text` with the marked region replaced by `markdown`."""
    marked_region(text)  # raises when a marker is missing or out of order
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    return text[: start + len(START_MARKER)] + region_for(markdown) + text[end:]


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def update_published(
    readme: Path | None = None, results_dir: Path | None = None, chart_dir: Path | None = None
) -> list[Path]:
    """Regenerate the README region and both charts from the committed
    artifacts, and return the files that changed. Everything is rendered, and
    so every refusal raised, before the first file is written. The paths
    default to the repository's, read at call time."""
    readme = readme or README_PATH
    results_dir = results_dir or RESULTS_DIR
    chart_dir = chart_dir or CHART_DIR
    rendered = render(results_dir, publishable=True)
    updated = splice(readme.read_text(encoding="utf-8"), rendered.markdown)
    changed = []
    for name, svg in rendered.charts.items():
        if _write_if_changed(chart_dir / name, svg):
            changed.append(chart_dir / name)
    if _write_if_changed(readme, updated):
        changed.append(readme)
    return changed


def render_development(source: Path, out: Path | None) -> tuple[list[Path], list[str]]:
    """Render `source` (a directory of artifacts) into a scratch directory as
    report.md and the two charts, and return the files written and the notes
    (what was skipped, what would be refused). Never touches the README, and
    refuses an `out` inside the committed results or the published charts."""
    if not source.is_dir():
        raise ReportError(f"{source} is not a directory")
    out_dir = Path(tempfile.mkdtemp(prefix="doc-pilot-load-report-")) if out is None else out.resolve()
    for protected in (RESULTS_DIR.resolve(), CHART_DIR.resolve()):
        if out_dir == protected or protected in out_dir.parents:
            raise ReportError(f"--from never writes under {protected}: give another --out")
    rendered = render(source, publishable=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"report.md": rendered.markdown + "\n", **rendered.charts}
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    return [out_dir / name for name in files], list(rendered.notes)


def report_command(*, update_readme: bool, source: str | None, out: str | None) -> int:
    """`load.py report`: exactly one of `--update-readme` and `--from DIR`."""
    if update_readme:
        if out is not None:
            raise ReportError("--out only applies with --from")
        changed = update_published()
        for path in changed:
            print(f"updated {path.relative_to(REPO_ROOT)}")
        if not changed:
            print("already current")
        return 0
    if source is None:
        raise ReportError("give --update-readme or --from DIR")
    written, notes = render_development(Path(source), Path(out) if out else None)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    for path in written:
        print(f"wrote {path}")
    return 0
