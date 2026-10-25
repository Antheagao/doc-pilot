"""The `load_v1` results artifact: what a load run records, how it is
summarized, and when it is valid.

Stdlib only (the CI smoke runs on a bare runner). The pure parts -- exact
quantiles, the per-endpoint statistics, the scenario digest, validation -- take
data and return data; the capture functions take a command runner, so tests
feed them canned output and nothing here needs Docker.

An artifact is a claim about a machine, so it records what the claim rests on:
the commit and whether the tree was clean, the exact command, the machine and
Docker's share of it, Postgres and its settings, the pools, every load-only
override, the scripted model's profile and scale, a digest of the scenario, and
the spend (which must be zero). Raw samples are not in it: they go to the
gitignored scripts/load/out/ and the artifact keeps exact quantiles computed
from them, not Locust's bucketed ones.

Quantiles are nearest-rank: the p-th percentile of n samples is the value at
rank ceil(p/100 * n) of the sorted samples, an observed latency and never an
interpolation. The rank is computed in integer permille, because 99.9/100*1000
is 999.0000000000001 in floating point and would round up to the maximum.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "load_v1"
# full: the plan as written; quick and smoke: the small dry runs; custom: a full
# plan with a number changed on the command line. Only full is publishable.
PRESETS = ("full", "quick", "smoke", "custom")
MODEL_NAME = "scripted"
# The most CPU, as a percentage of one core, the load generator may use in a
# phase. Past it the generator, not the server, may be the bottleneck.
GENERATOR_CPU_LIMIT_PCT = 80.0
# The most the stack's clock (which stamps every sample and job) may differ from
# the host's (which stamps docker stats), in seconds.
CLOCK_OFFSET_LIMIT_S = 1.0

# What the numbers are, for whoever renders them (README "Performance").
METHODOLOGY = (
    "Closed-loop load: N concurrent virtual users, each sending its next request as soon as "
    "the last one finished, with no think time, so N is the concurrency and the request rate "
    "is a result, not an input. Latencies are exact nearest-rank percentiles of the "
    "successful requests; errors are counted and not timed. This is not an open-loop "
    "measurement at a fixed arrival rate, and a closed loop slows down when the server does, "
    "which hides the queueing an open loop would show. Worker throughput is jobs "
    "(extract and index, two per document) completed per second between the 10% and 90% marks "
    "of the backlog."
)

# (label, permille). Rank = ceil(permille * n / 1000).
QUANTILES = (
    ("p1", 10),
    ("p5", 50),
    ("p10", 100),
    ("p25", 250),
    ("p50", 500),
    ("p75", 750),
    ("p90", 900),
    ("p95", 950),
    ("p99", 990),
    ("p99.9", 999),
)

# The settings of api and worker that the artifact records, by name. A list of
# names, so a secret in the environment can never be copied in by accident.
OVERRIDE_NAMES = (
    "MODEL_PROVIDER",
    "SCRIPTED_MODEL_PROFILE",
    "SCRIPTED_MODEL_LATENCY_SCALE",
    "SCRIPTED_AGENT_STEP_MS",
    "SCRIPTED_JUDGE_MS",
    "EMBEDDING_BACKEND",
    "ASK_RATE_LIMIT_PER_MINUTE",
    "UPLOAD_RATE_LIMIT_PER_MINUTE",
    "ASK_JUDGE_SAMPLE_RATE",
    "WORKER_RECLAIM_AFTER_SECONDS",
    "DAILY_BUDGET_USD",
    "ORGANIZATION_DAILY_BUDGET_USD",
    "MCP_RATE_LIMIT_PER_MINUTE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "DB_POOL_SIZE",
    "DB_MAX_OVERFLOW",
    "DB_POOL_TIMEOUT_SECONDS",
)
POSTGRES_SETTINGS = (
    "max_connections",
    "shared_buffers",
    "work_mem",
    "effective_cache_size",
    "maintenance_work_mem",
    "max_wal_size",
    "shared_preload_libraries",
    "synchronous_commit",
    "fsync",
)


# --- statistics -------------------------------------------------------------


def quantile(ordered: list[float], permille: int) -> float:
    """The nearest-rank quantile of an ascending list; `permille` is 1 to 1000
    (p99.9 is 999). ValueError for an empty list."""
    if not ordered:
        raise ValueError("no samples")
    rank = -(-permille * len(ordered) // 1000)  # ceil in integers
    return ordered[max(rank, 1) - 1]


def summarize(values: Iterable[float]) -> dict | None:
    """Count, min, mean, max and the QUANTILES of the values (milliseconds);
    None when there are none."""
    ordered = sorted(values)
    if not ordered:
        return None
    result = {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 3),
        "max": ordered[-1],
    }
    for label, permille in QUANTILES:
        result[label] = quantile(ordered, permille)
    return result


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def spread(values: list[float]) -> dict:
    """The median with the min-max range, for the repetitions of one
    measurement. ValueError for none."""
    if not values:
        raise ValueError("no repetitions")
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values)}


def read_samples(path: Path) -> list[dict]:
    """The raw samples file: one JSON object per line. A last line cut short by
    a killed process is dropped."""
    samples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                samples.append(json.loads(line))
            except ValueError:
                continue
    return samples


def in_window(samples: Iterable[dict], start: float, end: float) -> list[dict]:
    """The samples that started in [start, end] (UTC seconds). Selecting by
    start, not by end, keeps a slow request that began inside the window and
    ended after it: dropping it would bias the percentiles low."""
    return [sample for sample in samples if start <= sample["start"] <= end]


def endpoint_stats(samples: list[dict], seconds: float) -> dict[str, dict]:
    """Per request name: count, errors by code, error rate, rate per second over
    `seconds`, and the exact quantiles of the successful requests' durations
    (errors are counted, not timed: a refusal is fast and would flatter the
    percentiles). An /ask/stream also gets the quantiles of its time to first
    event and to the answer."""
    names = sorted({sample["name"] for sample in samples})
    stats = {}
    for name in names:
        mine = [sample for sample in samples if sample["name"] == name]
        good = [sample for sample in mine if sample["ok"]]
        entry = {
            "count": len(mine),
            "errors": len(mine) - len(good),
            "error_rate": (len(mine) - len(good)) / len(mine),
            "errors_by_code": dict(Counter(sample["error"] for sample in mine if not sample["ok"])),
            "rps": len(mine) / seconds if seconds > 0 else None,
            "latency_ms": summarize(sample["ms"] for sample in good),
        }
        if name == "ask_stream":
            entry["first_event_ms"] = summarize(
                sample["first_event_ms"] for sample in good if sample.get("first_event_ms") is not None
            )
            entry["answer_ms"] = summarize(
                sample["answer_ms"] for sample in good if sample.get("answer_ms") is not None
            )
        stats[name] = entry
    return stats


def scenario_digest(description: dict) -> str:
    """The SHA-256 of a scenario description in canonical JSON: two runs with
    the same digest sent the same traffic, which is what lets a before and an
    after be compared."""
    canonical = json.dumps(description, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def utc_iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run_id_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --- capturing the machine ---------------------------------------------------

Runner = Callable[[list[str]], str]


def run_command(cmd: list[str]) -> str:
    """stdout of a command; RuntimeError when it fails or is not installed."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"{cmd[0]}: {exc}") from exc
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} failed: {done.stderr.strip()[:200]}")
    return done.stdout


def _sysctl(run: Runner, name: str) -> str:
    return run(["sysctl", "-n", name]).strip()


def capture_machine(run: Runner = run_command, system: str | None = None) -> dict:
    """The host: CPU model, core counts, memory and OS. macOS reads `sysctl`
    (the performance and efficiency cores of an Apple chip apart), Linux reads
    /proc; anything else gets what `platform` knows and 0 cores, which
    validation refuses."""
    system = system or platform.system()
    machine = {"system": system, "arch": platform.machine()}
    if system == "Darwin":
        machine.update(
            cpu=_sysctl(run, "machdep.cpu.brand_string"),
            cores=int(_sysctl(run, "hw.ncpu")),
            memory_bytes=int(_sysctl(run, "hw.memsize")),
            os=f"macOS {run(['sw_vers', '-productVersion']).strip()}",
        )
        for key, name in (("cores_performance", "hw.perflevel0.physicalcpu"), ("cores_efficiency", "hw.perflevel1.physicalcpu")):
            try:
                machine[key] = int(_sysctl(run, name))
            except (RuntimeError, ValueError):
                pass  # an Intel Mac has one core level
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        model = re.search(r"^(?:model name|Model)\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        total = re.search(r"^MemTotal:\s*(\d+) kB", meminfo, re.MULTILINE)
        release = re.search(r'^PRETTY_NAME="?([^"\n]+)', Path("/etc/os-release").read_text(encoding="utf-8"), re.MULTILINE)
        machine.update(
            cpu=model[1].strip() if model else "unknown",
            cores=int(run(["nproc"]).strip()),
            memory_bytes=int(total[1]) * 1024 if total else 0,
            os=release[1] if release else "Linux",
        )
    else:
        machine.update(cpu=platform.processor() or "unknown", cores=0, memory_bytes=0, os=platform.platform())
    return machine


def capture_docker(run: Runner = run_command) -> dict:
    """The Docker engine and the share of the host it was given: the
    generator, the API, the workers and Postgres all compete for these CPUs."""
    info = json.loads(run(["docker", "info", "--format", "{{json .}}"]))
    return {
        "version": info.get("ServerVersion"),
        "ncpu": info.get("NCPU"),
        "memory_bytes": info.get("MemTotal"),
        "operating_system": info.get("OperatingSystem"),
        "architecture": info.get("Architecture"),
    }


def capture_git(run: Runner = run_command, cwd: Path | None = None) -> dict:
    """The commit and whether tracked files differ from it. Untracked files do
    not count: the working tree always has a few (agent notes) that no
    measurement depends on, and every scenario file is committed."""
    prefix = ["git", "-C", str(cwd)] if cwd else ["git"]
    try:
        commit = run([*prefix, "rev-parse", "HEAD"]).strip()
        dirty = bool(run([*prefix, "status", "--porcelain", "--untracked-files=no"]).strip())
    except RuntimeError:
        return {"commit": "unknown", "dirty": True}
    return {"commit": commit, "dirty": dirty}


def capture_postgres(psql: Callable[[str], str]) -> dict:
    """Version, pgvector, and the settings that shape a load: `psql(sql)` runs
    a statement as the owner in the load database and returns its tuples."""
    names = ", ".join(f"'{name}'" for name in POSTGRES_SETTINGS)
    rows = psql(f"select name, setting, coalesce(unit, '') from pg_settings where name in ({names}) order by name")
    settings = {}
    for line in rows.splitlines():
        name, _, rest = line.partition("|")
        value, _, unit = rest.partition("|")
        settings[name] = f"{value}{unit}"
    return {
        "version": psql("select version()").strip(),
        "pgvector": psql("select extversion from pg_extension where extname = 'vector'").strip(),
        "max_connections": int(settings.get("max_connections", "0")),
        "settings": settings,
    }


def capture_stack(config: dict, image_command: list[str]) -> dict:
    """The stack as the resolved compose config and the image define it: the
    pool of each process, the uvicorn worker count, the embedder, and every
    load-only override (a fixed list of names, values only). The Anthropic key
    is recorded as blank or not, never as a value."""
    services = config.get("services", {})
    envs = {name: (services.get(name) or {}).get("environment") or {} for name in ("api", "worker")}
    workers = 1
    if "--workers" in image_command:
        workers = int(image_command[image_command.index("--workers") + 1])

    def pool(env: dict) -> dict:
        return {
            "size": int(env.get("DB_POOL_SIZE", 5)),
            "max_overflow": int(env.get("DB_MAX_OVERFLOW", 10)),
            "timeout_seconds": float(env.get("DB_POOL_TIMEOUT_SECONDS", 30)),
        }

    return {
        "api_pool": pool(envs["api"]),
        "worker_pool": pool(envs["worker"]),
        "uvicorn_workers": workers,
        "embedder": envs["api"].get("EMBEDDING_BACKEND", "fastembed"),
        "anthropic_key_blank": all(not env.get("ANTHROPIC_API_KEY") for env in envs.values()),
        "overrides": {
            service: {name: env[name] for name in OVERRIDE_NAMES if name in env}
            for service, env in envs.items()
        },
    }


def capture_model(config: dict) -> dict:
    """The scripted model's profile and scale as the API runs them."""
    env = ((config.get("services") or {}).get("api") or {}).get("environment") or {}

    def number(name: str) -> float | None:
        try:
            return float(env[name])
        except (KeyError, ValueError):
            return None

    return {
        "provider": env.get("MODEL_PROVIDER"),
        "profile": env.get("SCRIPTED_MODEL_PROFILE"),
        "latency_scale": number("SCRIPTED_MODEL_LATENCY_SCALE"),
        "agent_step_ms": number("SCRIPTED_AGENT_STEP_MS"),
        "judge_ms": number("SCRIPTED_JUDGE_MS"),
        "models": {
            name: env.get(name)
            for name in ("EXTRACTION_MODEL", "TRANSCRIPTION_MODEL", "AGENT_MODEL", "JUDGE_MODEL")
        },
    }


# --- the artifact ------------------------------------------------------------


def build_artifact(
    *,
    run_id: str,
    label: str,
    preset: str,
    command: list[str],
    git: dict,
    machine: dict,
    docker: dict,
    postgres: dict,
    stack: dict,
    model: dict,
    scenario: dict,
    tenants: dict,
    generator: dict,
    phases: list[dict],
    summary: dict,
    clock_offset_s: float,
    spend_usd: float,
    models_seen: list[str],
    started_at: str,
    ended_at: str,
) -> dict:
    """Assemble a load_v1 artifact and attach its validation. The scenario is
    the description (workload.scenario_description plus the run's phase plan);
    its digest is computed here."""
    artifact = {
        "schema": SCHEMA,
        "run_id": run_id,
        "label": label,
        "preset": preset,
        "started_at": started_at,
        "ended_at": ended_at,
        "command": command,
        "git": git,
        "machine": machine,
        "docker": docker,
        "postgres": postgres,
        "stack": stack,
        "model": model,
        "scenario": {"digest": scenario_digest(scenario), "description": scenario},
        "tenants": tenants,
        "generator": generator,
        "methodology": METHODOLOGY,
        "clock_offset_s": clock_offset_s,
        "phases": phases,
        "summary": summary,
        "spend_usd": spend_usd,
        "models_seen": sorted(models_seen),
    }
    artifact["validation"] = {
        "problems": validate(artifact),
        "publishable_problems": validate(artifact, publishable=True),
    }
    return artifact


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _positive(record: object, key: str) -> bool:
    return isinstance(record, dict) and _is_number(record.get(key)) and record[key] > 0


def validate(artifact: dict, *, publishable: bool = False) -> list[str]:
    """Why an artifact cannot be trusted; empty when it can.

    Always: the schema, a record of the machine and of Docker's resources, the
    Postgres version, the git commit, the scripted model, a scenario digest that
    matches its description, and a spend of exactly zero with no model but the
    scripted one in any recorded row (a run that wrote no model row, an ingest
    with no workers, sees none). `publishable` (what the README report and a committed baseline
    need) also refuses a dirty tree, an unpinned commit, a phase marked invalid
    (the generator passed its CPU limit; the last step of a ramp that found its knee is
    the exception), and a quick or smoke preset."""
    problems = []
    if artifact.get("schema") != SCHEMA:
        problems.append(f"schema is {artifact.get('schema')!r}, not {SCHEMA!r}")
    for key in ("run_id", "label", "started_at", "ended_at"):
        if not isinstance(artifact.get(key), str) or not artifact[key]:
            problems.append(f"{key} is missing")
    if artifact.get("preset") not in PRESETS:
        problems.append(f"preset is {artifact.get('preset')!r}, not one of {PRESETS}")
    command = artifact.get("command")
    if not isinstance(command, list) or not command:
        problems.append("the exact command is missing")
    git = artifact.get("git")
    if not isinstance(git, dict) or not git.get("commit") or not isinstance(git.get("dirty"), bool):
        problems.append("git commit and dirty flag are missing")
    machine = artifact.get("machine")
    if not isinstance(machine, dict) or not machine.get("cpu") or not machine.get("os"):
        problems.append("the machine record is missing its cpu or os")
    elif not _positive(machine, "cores") or not _positive(machine, "memory_bytes"):
        problems.append("the machine record is missing its core count or memory")
    docker = artifact.get("docker")
    if not isinstance(docker, dict) or not docker.get("version"):
        problems.append("the Docker record is missing")
    elif not _positive(docker, "ncpu") or not _positive(docker, "memory_bytes"):
        problems.append("the Docker record is missing its CPUs or memory")
    postgres = artifact.get("postgres")
    if not isinstance(postgres, dict) or not postgres.get("version") or not _positive(postgres, "max_connections"):
        problems.append("the Postgres record is missing its version or max_connections")
    stack = artifact.get("stack")
    if not isinstance(stack, dict) or not isinstance(stack.get("overrides"), dict) or not stack.get("api_pool"):
        problems.append("the stack record (pools, overrides) is missing")
    elif stack.get("anthropic_key_blank") is not True:
        problems.append("the stack record does not show a blank Anthropic key")
    model = artifact.get("model")
    if not isinstance(model, dict) or model.get("provider") != MODEL_NAME:
        problems.append(f"the model provider is not {MODEL_NAME!r}")
    elif model.get("profile") not in ("zero", "replay") or not _is_number(model.get("latency_scale")):
        problems.append("the model profile or latency scale is missing")
    scenario = artifact.get("scenario")
    if (
        not isinstance(scenario, dict)
        or not isinstance(scenario.get("description"), dict)
        or scenario.get("digest") != scenario_digest(scenario["description"])
    ):
        problems.append("the scenario digest is missing or does not match its description")
    phases = artifact.get("phases")
    if not isinstance(phases, list) or not phases:
        problems.append("the artifact has no phases")
        phases = []
    for phase in phases:
        if not isinstance(phase, dict) or not phase.get("name") or not isinstance(phase.get("valid"), bool):
            problems.append("a phase is missing its name or validity")
    offset = artifact.get("clock_offset_s")
    if not _is_number(offset) or abs(offset) > CLOCK_OFFSET_LIMIT_S:
        problems.append(
            f"clock_offset_s is {offset!r}: the stack's clock must be within {CLOCK_OFFSET_LIMIT_S:g} s of the host's"
        )
    spend = artifact.get("spend_usd")
    if not _is_number(spend) or spend != 0:
        problems.append(f"spend_usd is {spend!r}, not 0")
    models_seen = artifact.get("models_seen")
    if not isinstance(models_seen, list) or any(name != MODEL_NAME for name in models_seen):
        problems.append(f"models seen are {models_seen!r}, not only {MODEL_NAME!r}")
    if publishable:
        if isinstance(git, dict) and git.get("dirty") is not False:
            problems.append("the git tree was dirty")
        if isinstance(git, dict) and git.get("commit") == "unknown":
            problems.append("the git commit is unknown")
        if artifact.get("preset") != "full":
            problems.append(f"preset {artifact.get('preset')!r} is not publishable")
        last_ramp_step = {
            phase.get("repetition"): index
            for index, phase in enumerate(phases)
            if isinstance(phase, dict) and phase.get("name") == "ramp"
        }
        for index, phase in enumerate(phases):
            if not isinstance(phase, dict) or phase.get("valid") is not False:
                continue
            # A ramp is built to end on a step that fails or is invalid (the
            # generator saturating is how it finds its limit), so the last step
            # of a repetition may be invalid once the repetition found a knee.
            knee = (phase.get("results") or {}).get("knee_users")
            if phase.get("name") == "ramp" and last_ramp_step.get(phase.get("repetition")) == index and knee is not None:
                continue
            reasons = "; ".join(phase.get("invalid_reasons") or ["no reason recorded"])
            problems.append(f"phase {phase.get('name')} repetition {phase.get('repetition')} is invalid: {reasons}")
    return problems


def artifact_filename(run_id: str, label: str) -> str:
    """`<run_id>_<label>.json`, the label reduced to lowercase letters, digits
    and hyphens so it is safe as a file name."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not slug:
        raise ValueError("the label has no letters or digits")
    return f"{run_id}_{slug}.json"


def dumps(artifact: dict) -> str:
    """The artifact as JSON with one top-level key per line (and one phase per
    line), compact inside: readable in a diff without a series of a thousand
    ticks taking a thousand lines."""

    def compact(value: object) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    lines = []
    for key, value in artifact.items():
        if key == "phases":
            body = ",\n  ".join(compact(phase) for phase in value)
            lines.append(f'"phases": [\n  {body}\n]')
        else:
            lines.append(f"{compact(key)}: {compact(value)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write_artifact(artifact: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / artifact_filename(artifact["run_id"], artifact["label"])
    path.write_text(dumps(artifact), encoding="utf-8")
    return path
