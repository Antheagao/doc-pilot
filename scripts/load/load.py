r"""Load-test CLI: manage the isolated load stack (docker-compose.load.yml).

Run from anywhere with the backend venv (stdlib only, so no install step):

    backend/.venv/bin/python scripts/load/load.py stack up
    backend/.venv/bin/python scripts/load/load.py stack status
    backend/.venv/bin/python scripts/load/load.py stack down [--volumes] [--dry-run]
    backend/.venv/bin/python scripts/load/load.py stack reset
    backend/.venv/bin/python scripts/load/load.py seed
    backend/.venv/bin/python scripts/load/load.py preflight
    backend/.venv/bin/python scripts/load/load.py run --profile zero --scale 1 --label L [--quick]
    backend/.venv/bin/python scripts/load/load.py smoke
    backend/.venv/bin/python scripts/load/load.py validate FILE [--publishable]
    backend/.venv/bin/python scripts/load/load.py report --update-readme
    backend/.venv/bin/python scripts/load/load.py report --from DIR [--out DIR]

The stack runs the production image against a scripted model (no model API is
ever called) in its own compose project, with its own database `docpilot_load`
and no host database port. `seed` creates three organizations, an owner for
each, and an API key for each, through scripts/accounts.py inside the api
container; the keys and passwords go to the gitignored scripts/load/out/
tenants.json (mode 600) for the harness and are never printed. Documents are
uploaded by the harness, not here. `preflight` refuses to let traffic start
unless every api and worker container reports the scripted provider and holds
no Anthropic key; it also creates the pg_stat_statements extension.

`run` and `smoke` own a whole measurement: they reset the load database (never
the models volume), start the stack with the load generator (profile `loadgen`,
Locust in its own small image, inside the load network), seed, preflight, run
the phases in phases.py with the samplers in samplers.py running, re-run the
no-spend check after every change in the number of workers, check spend and the
recorded model names from the database after every phase, write a `load_v1`
artifact (artifact.py), and stop the stack (volumes kept). `run` writes under
scripts/load/out/ unless told otherwise; only a full-plan run may write to
scripts/load/results/, where committed artifacts live. `smoke` is the small
version CI runs: it writes to a temporary directory and refuses results/.

Every docker call goes through STACK below, which pins the compose project to
PROJECT. The project name is a constant, never an argument, so this CLI cannot
address the dev stack (project `doc-pilot`) or the demo's, or their volumes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import secrets
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

# compose_project.py sits in scripts/, beside this directory and scripts/demos/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import artifact
import phases
import report
import samplers
import workload
from compose_project import (
    REPO_ROOT,
    CliError,
    ComposeProject,
    check_one_shots_and_worker,
    declared_volume_names,
)
from phases import PhaseError
from report import ReportError

PROJECT = "doc-pilot-load"
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.load.yml")
STACK = ComposeProject(PROJECT, COMPOSE_FILES)
API_URL = "http://127.0.0.1:8200"
DATABASE = "docpilot_load"
OWNER_ROLE = "docpilot"
RUNTIME_ROLE = "docpilot_runtime"
SCRIPTED = "scripted"
# The volumes `stack reset` removes: the database and the uploads. The models
# volume (the embedding model, about 135 MB) is kept.
RESET_VOLUMES = ("db_data", "uploads")
# The services that run the backend image: each must carry an explicitly blank
# ANTHROPIC_API_KEY in the resolved config, not merely lack one.
BACKEND_SERVICES = ("api", "worker", "migrate", "loadgen")
# The services whose containers answer model calls and so are asked, live.
MODEL_SERVICES = ("api", "worker")
# The overlay's daily budget: a cent, the tripwire that turns any recorded
# spend into 429s.
TRIPWIRE_BUDGET_USD = 0.01
MODEL_SETTINGS = (
    "MODEL_PROVIDER",
    "EXTRACTION_MODEL",
    "TRANSCRIPTION_MODEL",
    "AGENT_MODEL",
    "JUDGE_MODEL",
)
OUT_DIR = Path(__file__).resolve().parent / "out"
TENANTS_FILE = OUT_DIR / "tenants.json"
# Committed artifacts live here; `smoke` and the quick and custom plans may not
# write to it.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVALS_DIR = REPO_ROOT / "evals"
LOADGEN = "loadgen"
API_INTERNAL_URL = "http://api:8000"
BACKEND_IMAGE = "doc-pilot-load-backend"
# `dpk_<12 lowercase base32>_<43 url-safe>`: backend/app/auth/tokens.py.
API_KEY_PATTERN = re.compile(r"dpk_[a-z2-7]{12}_[A-Za-z0-9_-]{43}")


@dataclass(frozen=True)
class Tenant:
    """One seeded organization: a large, a medium and a small one, so row-level
    security, the per-tenant scans and a small tenant among large ones are all
    exercised. The sizes (documents) are the harness's to fill."""

    slug: str
    name: str
    owner_email: str


TENANTS = (
    Tenant("load-large", "Load Large", "owner-large@load.doc-pilot.local"),
    Tenant("load-medium", "Load Medium", "owner-medium@load.doc-pilot.local"),
    Tenant("load-small", "Load Small", "owner-small@load.doc-pilot.local"),
)


# --- the no-spend checks (pure) ---------------------------------------------


def config_problems(config: dict) -> list[str]:
    """Why a resolved compose config could reach a real model; empty when it
    cannot. Reads values and reports only names, never a value: the config
    holds the key when the overlay forgot to blank it.

    api and worker must run the scripted provider with all four model names
    scripted and the one-cent budget; the worker must hold a lease (a positive
    WORKER_RECLAIM_AFTER_SECONDS), because with none each starting worker resets
    the others' jobs and the throughput would be measured on stolen work; every
    backend service must carry an explicitly empty ANTHROPIC_API_KEY, and no
    service at all a non-empty one."""
    problems = []
    services = config.get("services", {})
    if config.get("name") != PROJECT:
        problems.append(f"compose project is {config.get('name')!r}, not {PROJECT!r}")
    for name, service in sorted(services.items()):
        env = service.get("environment") or {}
        if env.get("ANTHROPIC_API_KEY"):
            problems.append(f"{name}: ANTHROPIC_API_KEY is not empty")
        elif name in BACKEND_SERVICES and "ANTHROPIC_API_KEY" not in env:
            problems.append(f"{name}: ANTHROPIC_API_KEY is not blanked explicitly")
    for name in MODEL_SERVICES:
        env = (services.get(name) or {}).get("environment") or {}
        for setting in MODEL_SETTINGS:
            if env.get(setting) != SCRIPTED:
                problems.append(f"{name}: {setting} is not {SCRIPTED!r}")
    api_env = (services.get("api") or {}).get("environment") or {}
    try:
        budget = float(api_env.get("DAILY_BUDGET_USD", ""))
    except ValueError:
        budget = None
    if budget is None or not 0 < budget <= TRIPWIRE_BUDGET_USD:
        problems.append(f"api: DAILY_BUDGET_USD is not a tripwire of at most ${TRIPWIRE_BUDGET_USD}")
    worker_env = (services.get("worker") or {}).get("environment") or {}
    try:
        lease = float(worker_env.get("WORKER_RECLAIM_AFTER_SECONDS", ""))
    except ValueError:
        lease = None
    if lease is None or not lease > 0:  # also refuses nan
        problems.append("worker: WORKER_RECLAIM_AFTER_SECONDS is not a positive lease")
    return problems


@dataclass(frozen=True)
class Probe:
    """What one running container reports about itself."""

    container: str
    provider: str
    has_key: bool
    database: str
    role: str


# Run inside a container by `python -c`: the settings the process really
# resolved, and the database and role its own engine connects as. The key
# leaves as a yes or no only.
PROBE_SCRIPT = """\
import asyncio, json, os
from sqlalchemy import text
from app.config import get_settings
from app.db import engine

async def connection():
    async with engine.connect() as c:
        row = (await c.execute(text("select current_database(), current_user"))).one()
    await engine.dispose()
    return row

settings = get_settings()
database, role = asyncio.run(connection())
print(json.dumps({
    "provider": settings.model_provider,
    "has_key": bool(settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")),
    "database": database,
    "role": role,
}))
"""


def parse_probe(container: str, output: str) -> Probe:
    """The Probe a container printed; CliError when the output is not it."""
    try:
        report = json.loads(output.strip().splitlines()[-1])
        return Probe(
            container,
            str(report["provider"]),
            bool(report["has_key"]),
            str(report["database"]),
            str(report["role"]),
        )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise CliError(f"{container}: the probe printed something unexpected") from exc


def probe_problems(probe: Probe) -> list[str]:
    """Why a container must not get traffic; empty when it is scripted, keyless
    and connected to the load database as the runtime role."""
    problems = []
    if probe.provider != SCRIPTED:
        problems.append(f"{probe.container}: model provider is {probe.provider!r}, not {SCRIPTED!r}")
    if probe.has_key:
        problems.append(f"{probe.container}: holds an Anthropic API key")
    if probe.database != DATABASE:
        problems.append(f"{probe.container}: connected to database {probe.database!r}, not {DATABASE!r}")
    if probe.role != RUNTIME_ROLE:
        problems.append(f"{probe.container}: connected as role {probe.role!r}, not {RUNTIME_ROLE!r}")
    return problems


def running_model_containers(records: list[dict]) -> list[dict]:
    """The running api and worker containers among `compose ps` records."""
    return [
        record
        for record in records
        if record.get("Service") in MODEL_SERVICES and record.get("State") == "running"
    ]


# --- the preflight -----------------------------------------------------------


def probe_container(record: dict) -> Probe:
    name = record["Name"]
    output = STACK.docker_output(["docker", "exec", name, "python", "-c", PROBE_SCRIPT])
    return parse_probe(name, output)


def owner_psql(*args: str) -> str:
    """psql in the db container as the database owner, in the load database."""
    return STACK.docker_output(
        STACK.compose_args(
            *("exec", "-T", "db", "psql", "-U", OWNER_ROLE, "-d", DATABASE),
            *("-X", "-v", "ON_ERROR_STOP=1", "-tA", *args),
        )
    )


def create_statements_extension() -> None:
    """CREATE EXTENSION IF NOT EXISTS pg_stat_statements as the owner, in the
    load database only, and check that the library is preloaded so it counts."""
    current = owner_psql("-c", "select current_database()").strip()
    if current != DATABASE:
        raise CliError(f"the db container answered database {current!r}, not {DATABASE!r}")
    owner_psql("-c", "CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
    preloaded = owner_psql("-c", "show shared_preload_libraries")
    if "pg_stat_statements" not in preloaded:
        raise CliError("pg_stat_statements is not in shared_preload_libraries of the db container")
    owner_psql("-c", "select count(*) from pg_stat_statements")


def check_stack() -> tuple[list[str], int]:
    """Every reason traffic must not start (empty when the stack is scripted,
    keyless, and on the load database), and how many containers were asked.
    Checks the resolved config of every service (so a container that is not
    running yet is covered), then asks each running api and worker container
    what it resolved. Public so the harness can re-run it after each scale step."""
    problems = config_problems(STACK.resolved_config(all_profiles=True))
    containers = running_model_containers(STACK.container_records(include_stopped=False))
    if not any(record["Service"] == "api" for record in containers):
        problems.append("no api container is running: run `load.py stack up`")
    for record in containers:
        try:
            problems += probe_problems(probe_container(record))
        except CliError as exc:
            problems.append(f"{record.get('Name')}: {exc}")
    return problems, len(containers)


def preflight(_args: argparse.Namespace) -> int:
    problems, asked = check_stack()
    if problems:
        print("Preflight failed, no traffic may be sent:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"ok   config: every service blank-keyed, api and worker {SCRIPTED}, budget a tripwire")
    print(f"ok   {asked} api and worker container(s): {SCRIPTED}, no key, {DATABASE} as {RUNTIME_ROLE}")
    create_statements_extension()
    print(f"ok   pg_stat_statements created in {DATABASE} and preloaded")
    print("Preflight passed.")
    return 0


# --- the stack ---------------------------------------------------------------


def bring_up(workers: int | None = None, *, loadgen: bool = False) -> int:
    """Start the stack. `workers` sets the number of worker containers (0 is
    allowed: a backlog is built with none); `loadgen` adds the load generator.
    Nothing starts from a config that could reach a real model."""
    problems = config_problems(STACK.resolved_config(all_profiles=True))
    if problems:
        raise CliError("refusing to start: " + "; ".join(problems))
    profile = ["--profile", LOADGEN] if loadgen else []
    scale = [] if workers is None else ["--scale", f"worker={workers}"]
    code = STACK.compose(*profile, "up", "-d", "--build", "--wait", *scale)
    if code != 0:
        return code
    states = STACK.service_states()
    if workers == 0:
        migrate = states.get("migrate", {})
        if migrate.get("State") != "exited" or migrate.get("ExitCode") != 0:
            raise CliError(f"migrate did not complete: state {migrate.get('State')!r}")
    else:
        check_one_shots_and_worker(states)
    return 0


def stack_up(_args: argparse.Namespace) -> int:
    code = bring_up()
    if code == 0:
        print(f"Load stack is up: API {API_URL}")
    return code


def stack_status(_args: argparse.Namespace) -> int:
    return STACK.compose("ps", "--all")


def compose_down(*args: str) -> int:
    """`down` for every service, profile or not: a container started with
    `--profile loadgen` would otherwise keep the network and its volumes."""
    return STACK.compose("--profile", "*", "down", *args)


def stack_down(args: argparse.Namespace) -> int:
    if not args.volumes:
        return compose_down()
    # Everything is checked before anything is removed.
    if STACK.report_volumes(STACK.resolved_config(), STACK.existing_volume_projects()):
        return 1
    if args.dry_run:
        print("Dry run: nothing removed.")
        return 0
    code = compose_down("--volumes")
    if code == 0:
        TENANTS_FILE.unlink(missing_ok=True)  # its keys died with the database
    return code


def reset_targets(config: dict) -> list[str]:
    """The volumes `stack reset` removes, as declared in the config."""
    wanted = [f"{PROJECT}_{volume}" for volume in RESET_VOLUMES]
    declared = declared_volume_names(config)
    missing = [name for name in wanted if name not in declared]
    if missing:
        raise CliError(f"the config does not declare {', '.join(missing)}")
    return wanted


def stack_reset(_args: argparse.Namespace) -> int:
    """Stop the stack and remove its database and uploads volumes (not the
    models volume), after the same guard `down --volumes` uses. The next
    `stack up` migrates an empty database; the seeded tenants are gone with
    it, so their keys are deleted too."""
    config = STACK.resolved_config()
    existing = STACK.existing_volume_projects()
    problems = STACK.volume_problems(config, existing)
    if problems:
        print("Refusing to reset:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    targets = reset_targets(config)
    code = compose_down()
    if code != 0:
        return code
    for name in targets:
        if name in existing:
            STACK.docker_output(["docker", "volume", "rm", name])
            print(f"removed {name}")
        else:
            print(f"{name} was not created yet")
    TENANTS_FILE.unlink(missing_ok=True)
    return 0


# --- tenants and keys --------------------------------------------------------


def parse_api_key(output: str) -> str:
    """The key `accounts.py create-api-key` printed alone on stdout. A mismatch
    is a CliError that does not echo the output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1 or not API_KEY_PATTERN.fullmatch(lines[0]):
        raise CliError("accounts.py create-api-key did not print exactly one API key")
    return lines[0]


def run_accounts(*args: str, stdin: str | None = None) -> str:
    """scripts/accounts.py inside the load api container; returns its stdout."""
    return STACK.docker_output(
        STACK.compose_args("exec", "-T", "api", "python", "scripts/accounts.py", *args),
        stdin=stdin,
    )


# What accounts.py says (on stderr, exit 1) when a thing it was asked to create
# is already there.
EXISTS_MARKERS = ("is already taken", "already has an account", "is already a member")


def run_accounts_if_new(*args: str, stdin: str | None = None) -> bool:
    """run_accounts that treats "already exists" as done: True when the command
    changed something, False when accounts.py refused because the organization,
    user or membership is already there. Any other failure is raised."""
    try:
        run_accounts(*args, stdin=stdin)
    except CliError as exc:
        if not any(marker in str(exc) for marker in EXISTS_MARKERS):
            raise
        return False
    return True


def create_tenant(tenant: Tenant) -> dict[str, str]:
    """Create the organization, its owner (the password goes in on stdin, never
    as an argument) and a fresh API key; returns the record the harness reads.
    Safe to repeat after a failure part-way: what already exists is kept, an
    owner that exists gets the new password (so the stored one is right), and
    the key is always newly minted, since an earlier one cannot be recovered."""
    password = secrets.token_urlsafe(24)
    run_accounts_if_new("create-organization", "--name", tenant.name, "--slug", tenant.slug)
    if not run_accounts_if_new(
        "create-user", "--email", tenant.owner_email, "--password-stdin", stdin=password + "\n"
    ):
        run_accounts("set-password", "--email", tenant.owner_email, "--password-stdin", stdin=password + "\n")
    run_accounts_if_new(
        "add-member", "--organization", tenant.slug, "--email", tenant.owner_email, "--role", "owner"
    )
    api_key = parse_api_key(run_accounts("create-api-key", "--organization", tenant.slug, "--name", "load"))
    return {
        "organization": tenant.slug,
        "email": tenant.owner_email,
        "password": password,
        "api_key": api_key,
    }


def repo_path(path: Path) -> Path:
    """`path` relative to the repository when it lies in it, for messages."""
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def write_private_json(path: Path, data: dict) -> None:
    """Write `data` as JSON readable by this user only, from the first byte:
    the file is created with mode 600 and moved into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    descriptor = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.chmod(scratch, 0o600)  # a pre-existing scratch file keeps its old mode
    os.replace(scratch, path)


def read_tenants(path: Path | None = None) -> dict[str, dict[str, str]]:
    """The seeded tenants by organization slug, each with `email`, `password`
    and `api_key`; CliError when `seed` has not written them. The harness reads
    the keys here instead of taking them as arguments or from the environment."""
    path = path or TENANTS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tenants = data["tenants"]
    except (OSError, ValueError, KeyError) as exc:
        raise CliError(f"{path} is missing or unreadable: run `load.py seed`") from exc
    return tenants


def api_status(path: str, api_key: str) -> int:
    """The HTTP status of an authenticated GET; 0 when the API is unreachable.
    The key rides in the Authorization header and is not part of any message."""
    request = urllib.request.Request(f"{API_URL}{path}", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:  # URLError is an OSError
        return 0


def seeded_keys_work() -> bool:
    """Whether tenants.json names every planned tenant and each key is accepted
    by the API (GET /documents answers 200)."""
    try:
        tenants = read_tenants()
    except CliError:
        return False
    if sorted(tenants) != sorted(tenant.slug for tenant in TENANTS):
        return False
    return all(api_status("/documents", record["api_key"]) == 200 for record in tenants.values())


def stored_tenants() -> dict[str, dict[str, str]]:
    """The tenants of an earlier (possibly interrupted) seed; empty when there
    is no readable file."""
    try:
        return read_tenants()
    except CliError:
        return {}


def seed(_args: argparse.Namespace) -> int:
    """Create the tenants, writing tenants.json after each one so an
    interruption keeps the keys already minted. Re-running resumes: a tenant
    whose stored key the API still accepts is kept, the others are (re)created."""
    if seeded_keys_work():
        print(f"Already seeded: {len(TENANTS)} tenants, keys in {repo_path(TENANTS_FILE)}")
        return 0
    records = stored_tenants()
    for tenant in TENANTS:
        kept = records.get(tenant.slug)
        if kept is not None and api_status("/documents", kept.get("api_key", "")) == 200:
            print(f"kept {tenant.slug}: its stored API key still works")
            continue
        try:
            records[tenant.slug] = create_tenant(tenant)
        except CliError as exc:
            raise CliError(
                f"seeding {tenant.slug} failed ({exc}); fix that and run `load.py seed` again, "
                "which resumes with the tenants already stored"
            ) from exc
        write_private_json(TENANTS_FILE, {"project": PROJECT, "api_url": API_URL, "tenants": records})
        print(f"seeded {tenant.slug}: owner {tenant.owner_email}, API key stored")
    print(f"Keys and passwords written (mode 600) to {repo_path(TENANTS_FILE)}")
    return 0


# --- spend and the recorded model names, from the database ------------------

SPEND_SQL = (
    f"{samplers.HARNESS_TAG} select "
    "coalesce((select sum(cost_usd) from extractions), 0) "
    "+ coalesce((select sum(cost_usd) from document_pages), 0) "
    "+ coalesce((select sum(cost_usd) from ask_runs), 0) "
    "+ coalesce((select sum(judge_cost_usd) from ask_runs), 0)"
)
MODELS_SQL = (
    f"{samplers.HARNESS_TAG} select model from extractions "
    "union select model from document_pages where model is not null "
    "union select model from ask_runs "
    "union select judge_model from ask_runs where judge_model is not null order by 1"
)


def spend_problems(spend: Decimal, models: set[str]) -> list[str]:
    """Why recorded rows mean a real model may have been reached: any cost at
    all, or a model name other than the scripted one."""
    found = []
    if spend != 0:
        found.append(f"recorded model spend is {spend} USD, not 0")
    others = sorted(models - {SCRIPTED})
    if others:
        found.append(f"rows name a model other than {SCRIPTED!r}: {', '.join(others)}")
    return found


class Ledger:
    """What the database says was spent, added up over the stack's lives (a
    reset between repetitions forgets the rows, so each phase reports before
    the next reset)."""

    def __init__(self) -> None:
        self.spend_usd = Decimal(0)
        self.models: set[str] = set()

    def add(self, spend: Decimal, models: set[str]) -> None:
        self.spend_usd += spend
        self.models |= models

    def problems(self) -> list[str]:
        return spend_problems(self.spend_usd, self.models)


def read_spend() -> tuple[Decimal, set[str]]:
    """Total recorded model cost (extractions, page transcriptions, answers and
    their judgments) and the model names those rows carry, as the owner sees
    every organization's rows."""
    spend = Decimal(owner_psql("-c", SPEND_SQL).strip() or "0")
    models = {line for line in owner_psql("-c", MODELS_SQL).splitlines() if line}
    return spend, models


# --- the driver ---------------------------------------------------------------


def scrub(text: str) -> str:
    """`text` for output that may be shown (a Locust failure's last lines):
    every line that mentions a password is dropped and anything shaped like an
    API key is replaced."""
    kept = [line for line in text.splitlines() if "password" not in line.lower()]
    return API_KEY_PATTERN.sub("dpk_***", "\n".join(kept))


def parse_clock_offset(container_output: str, before: float, after: float) -> float:
    """Seconds the db container's clock is ahead of the host's, from the
    container's `date +%s.%N` and the host's time just before and after asking
    (the midpoint, so the error is at most half the round trip)."""
    try:
        container = float(container_output.strip())
    except ValueError as exc:
        raise CliError("the db container did not print its time as seconds since the epoch") from exc
    return container - (before + after) / 2


def measure_clock_offset() -> float:
    """The clock offset between the db container (the Docker VM's clock, which
    stamps every sample and job) and the host (which stamps docker stats)."""
    before = time.time()
    output = STACK.docker_output(STACK.compose_args("exec", "-T", "db", "date", "+%s.%N"))
    return parse_clock_offset(output, before, time.time())


class StackDriver:
    """The load stack as phases.py drives it: every docker call goes through
    STACK, which is pinned to the load project."""

    def __init__(self, samples_dir: Path, telemetry: samplers.Telemetry | None = None) -> None:
        self.samples_dir = samples_dir
        samples_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger()
        self.clock_offset_s = 0.0
        self._up = False
        self._telemetry = telemetry or samplers.Telemetry(
            lambda argv: STACK.compose_args("exec", "-T", "db", *argv),
            lambda sql: owner_psql("-c", sql),
            owner=OWNER_ROLE,
            database=DATABASE,
        )

    def prepare(self, workers: int, *, fresh: bool) -> None:
        # Started early so the generator's first CPU figures exist by the time
        # a short upload step needs them.
        self._telemetry.ensure_stats()
        if not fresh and self._up:
            self.scale_workers(workers)
            return
        if stack_reset(argparse.Namespace()) != 0:
            raise CliError("could not reset the load stack")
        if bring_up(workers, loadgen=True) != 0:
            raise CliError("could not start the load stack")
        if seed(argparse.Namespace()) != 0 or preflight(argparse.Namespace()) != 0:
            raise CliError("the load stack did not pass its seed and preflight")
        self._check_clock()
        self._up = True

    def _check_clock(self) -> None:
        """Refuse to measure if the stack's clock is more than a second from the
        host's: windows on the sample clock are compared with docker stats on the
        host's. The largest offset seen is the one recorded."""
        offset = measure_clock_offset()
        if abs(offset) > artifact.CLOCK_OFFSET_LIMIT_S:
            raise CliError(
                f"the stack's clock is {offset:+.2f} s from the host's (limit {artifact.CLOCK_OFFSET_LIMIT_S:g} s)"
            )
        if abs(offset) >= abs(self.clock_offset_s):
            self.clock_offset_s = offset

    def scale_workers(self, count: int) -> None:
        """`up -d --scale worker=N` for the worker service alone, then the
        no-spend check again: the new containers must be scripted and keyless
        like the first, and there must be as many as were asked for."""
        code = STACK.compose(
            "up", "-d", "--no-build", "--no-recreate", "--no-deps", "--scale", f"worker={count}", "worker"
        )
        if code != 0:
            raise CliError(f"could not scale the workers to {count}")
        problems, _asked = check_stack()
        if problems:
            raise CliError("the stack failed its no-spend check after scaling: " + "; ".join(problems))
        running = [
            record
            for record in running_model_containers(STACK.container_records(include_stopped=False))
            if record["Service"] == "worker"
        ]
        if len(running) != count:
            raise CliError(f"asked for {count} worker(s), {len(running)} running")

    def run_locust(self, spec: phases.LocustSpec, destination: Path) -> None:
        """Run Locust headless in the loadgen container with the spec's
        settings, then copy its raw samples to `destination` (and delete the
        copy in the container)."""
        remote = f"/tmp/samples-{uuid.uuid4().hex}.jsonl"
        settings = {
            "LOAD_SAMPLES_FILE": remote,
            "LOAD_WORKLOAD": spec.workload,
            "LOAD_UPLOAD_QUOTA": str(spec.upload_quota),
        }
        if spec.user_weights is not None:
            settings["LOAD_USER_WEIGHTS"] = json.dumps(spec.user_weights)
        if spec.read_mix is not None:
            settings["LOAD_READ_MIX"] = json.dumps(spec.read_mix)
        if spec.exclude_question_types:
            settings["LOAD_EXCLUDE_QUESTION_TYPES"] = ",".join(spec.exclude_question_types)
        flags = [flag for key, value in settings.items() for flag in ("-e", f"{key}={value}")]
        locust = [
            *("locust", "-f", "/load/locustfile.py", "--headless", "--only-summary"),
            *("--host", API_INTERNAL_URL, "-u", str(spec.users), "-r", str(spec.users)),
            # Users mid-request get a minute to finish: no sample is cut short.
            *("--stop-timeout", "60", "--exit-code-on-error", "0", "--loglevel", "WARNING"),
            *(("-t", f"{spec.run_seconds}s") if spec.run_seconds else ()),
        ]
        result = STACK.run(
            STACK.compose_args("--profile", LOADGEN, "exec", "-T", *flags, LOADGEN, *locust), capture=True
        )
        if result.returncode != 0:
            tail = scrub("\n".join((result.stdout + result.stderr).strip().splitlines()[-15:]))
            raise PhaseError(f"{spec.tag}: Locust exited with {result.returncode}:\n{tail}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        STACK.docker_output(
            STACK.compose_args("--profile", LOADGEN, "cp", f"{LOADGEN}:{remote}", str(destination))
        )
        STACK.docker_output(
            STACK.compose_args("--profile", LOADGEN, "exec", "-T", LOADGEN, "rm", "-f", remote)
        )

    def query(self, sql: str) -> str:
        return owner_psql("-c", sql)

    def telemetry(self) -> AbstractContextManager[samplers.PhaseTelemetry]:
        return self._telemetry.phase()

    def close(self) -> None:
        """Stop the `docker stats` stream."""
        self._telemetry.close()

    def check_spend(self) -> None:
        """Add what the database holds to the ledger and refuse to go on if a
        cent was spent or a model other than the scripted one was recorded."""
        spend, models = read_spend()
        self.ledger.add(spend, models)
        problems = spend_problems(spend, models)
        if problems:
            raise CliError("a real model may have been reached: " + "; ".join(problems))


# --- a run ---------------------------------------------------------------------


class DocumentFacts(NamedTuple):
    """What the database holds about one uploaded document."""

    id: str
    status: str
    extractions: int
    chunks: int
    jobs_done: int
    jobs_open: int


def document_facts_sql(document_ids: list[str]) -> str:
    """The query for DocumentFacts of these ids. Each id is parsed as a UUID
    first, so nothing but a UUID can reach the statement."""
    ids = ", ".join(f"'{uuid.UUID(document_id)}'" for document_id in document_ids)
    return (
        f"{samplers.HARNESS_TAG} select d.id, d.status, "
        "(select count(*) from extractions e where e.document_id = d.id), "
        "(select count(*) from document_chunks c where c.document_id = d.id), "
        "(select count(*) from jobs j where j.document_id = d.id and j.state = 'done'), "
        "(select count(*) from jobs j where j.document_id = d.id and j.state <> 'done') "
        f"from documents d where d.id in ({ids})"
    )


def parse_document_facts(output: str) -> list[DocumentFacts]:
    facts = []
    for line in output.splitlines():
        if not line:
            continue
        document_id, status, *counts = line.split("|")
        facts.append(DocumentFacts(document_id, status, *(int(count) for count in counts)))
    return facts


def smoke_failures(
    samples: list[dict],
    facts: list[DocumentFacts],
    ledger: Ledger,
    phase_records: list[dict],
    uploads: int,
) -> list[str]:
    """Everything the smoke asserts, as a list of failures (empty is a pass):
    no failed request; the planned uploads all made; each uploaded document
    extracted, indexed, and its extract and index jobs done; every kind of read
    seen at least once, with an answer among them (a failed answer is a failed
    request: no citation is an error); spend zero from the database with only
    the scripted model recorded; and the samplers saw the queue."""
    failures = []
    for (name, error), count in sorted(Counter((s["name"], s["error"]) for s in samples if not s["ok"]).items()):
        failures.append(f"{count} {name} request(s) failed: {error}")
    made = [s for s in samples if s["name"] == "upload" and s["ok"]]
    if len(made) != uploads:
        failures.append(f"{len(made)} uploads succeeded, {uploads} were planned")
    by_id = {fact.id: fact for fact in facts}
    for document_id in sorted({s.get("document_id") for s in made}):
        fact = by_id.get(document_id)
        if fact is None:
            failures.append(f"document {document_id} is not in the database")
        elif fact.extractions < 1:
            failures.append(f"document {document_id} was not extracted (status {fact.status})")
        elif fact.chunks < 1:
            failures.append(f"document {document_id} was not indexed (no chunks)")
        elif fact.jobs_done != phases.JOBS_PER_DOCUMENT or fact.jobs_open:
            failures.append(
                f"document {document_id}: {fact.jobs_done} jobs done and {fact.jobs_open} not done, "
                f"expected {phases.JOBS_PER_DOCUMENT} and 0"
            )
    for name in workload.READ_ENDPOINTS:
        if not any(s["name"] == name and s["ok"] for s in samples):
            failures.append(f"no {name} request completed")
    failures += ledger.problems()
    if ledger.models != {SCRIPTED}:
        failures.append(f"recorded models are {sorted(ledger.models)}, expected only {SCRIPTED!r}")
    if not any(record.get("queue_series") for record in phase_records):
        failures.append("the database sampler recorded no queue depth")
    return failures


def command_record(argv: list[str]) -> list[str]:
    """The exact command as recorded: python, this script's path from the
    repository root, and the arguments as given."""
    return ["python", str(repo_path(Path(__file__).resolve())), *argv]


def refuse_results_dir(directory: Path, *, what: str) -> Path:
    """`directory`, unless it is scripts/load/results or inside it: only the
    committed baselines go there, and `what` may not."""
    resolved = directory.resolve()
    if resolved == RESULTS_DIR or RESULTS_DIR in resolved.parents:
        raise CliError(f"{what} may not write to {repo_path(RESULTS_DIR)}: that holds committed artifacts")
    return resolved


def apply_run_environment(profile: str, scale: float, embedder: str | None) -> None:
    """Pass the run's settings to compose through the environment, where the
    overlay reads them (docker-compose.load.yml). The artifact records what the
    containers resolved, not what was asked for here."""
    os.environ["SCRIPTED_MODEL_PROFILE"] = profile
    os.environ["SCRIPTED_MODEL_LATENCY_SCALE"] = repr(float(scale))
    if embedder:
        os.environ["LOAD_EMBEDDING_BACKEND"] = embedder


def generator_version() -> str:
    try:
        output = STACK.docker_output(
            STACK.compose_args("--profile", LOADGEN, "exec", "-T", LOADGEN, "locust", "--version")
        )
    except CliError:
        return "unknown"
    match = re.search(r"locust (\d+\.\d+\.\d+)", output)
    return match[1] if match else "unknown"


def assemble_artifact(
    driver: StackDriver,
    plan: phases.Plan,
    result: phases.RunResult,
    *,
    run_id: str,
    label: str,
    argv: list[str],
    started_at: str,
) -> dict:
    """Capture the machine, Docker, Postgres and the stack while the stack is
    still up, and build the artifact from the phases."""
    config = STACK.resolved_config(all_profiles=True)
    image_command = json.loads(
        STACK.docker_output(["docker", "image", "inspect", "--format", "{{json .Config.Cmd}}", BACKEND_IMAGE])
    )
    scenario = {
        "workload": workload.scenario_description(
            EVALS_DIR,
            read_mix=dict(plan.read_mix) if plan.read_mix else None,
            user_weights=dict(plan.user_weights) if plan.user_weights else None,
            exclude_question_types=plan.exclude_question_types,
        ),
        "plan": phases.plan_description(plan),
    }
    return artifact.build_artifact(
        run_id=run_id,
        label=label,
        preset=plan.preset,
        command=command_record(argv),
        git=artifact.capture_git(cwd=REPO_ROOT),
        machine=artifact.capture_machine(),
        docker=artifact.capture_docker(),
        postgres=artifact.capture_postgres(lambda sql: owner_psql("-c", sql)),
        stack=artifact.capture_stack(config, image_command),
        model=artifact.capture_model(config),
        scenario=scenario,
        tenants=phases.tenant_sizes(driver),
        generator={
            "tool": "locust",
            "version": generator_version(),
            "http_user": "HttpUser",
            "loop": "closed",
            "placement": f"inside the {PROJECT} compose network, against {API_INTERNAL_URL}",
            "cpu_limit_pct": artifact.GENERATOR_CPU_LIMIT_PCT,
        },
        phases=result.phases,
        summary=phases.summarize(result.phases),
        clock_offset_s=driver.clock_offset_s,
        spend_usd=float(driver.ledger.spend_usd),
        models_seen=sorted(driver.ledger.models),
        started_at=started_at,
        ended_at=artifact.utc_iso(time.time()),
    )


def execute(
    plan: phases.Plan,
    *,
    label: str,
    out_dir: Path,
    argv: list[str],
    keep_up: bool = False,
    after: Callable[[StackDriver, phases.RunResult], None] | None = None,
) -> tuple[dict, Path, phases.RunResult]:
    """Run a plan on a freshly reset stack and write its artifact to `out_dir`.
    `after(driver, result)` runs while the stack is still up (the smoke reads
    the database there). The stack is stopped on the way out whatever happens,
    unless `keep_up`; its volumes are kept either way."""
    run_id = artifact.run_id_now()
    driver = StackDriver(out_dir / "samples" / run_id)
    started_at = artifact.utc_iso(time.time())
    try:
        result = phases.run_plan(driver, plan)
        document = assemble_artifact(
            driver, plan, result, run_id=run_id, label=label, argv=argv, started_at=started_at
        )
        path = artifact.write_artifact(document, out_dir)
        if after is not None:
            after(driver, result)
        return document, path, result
    finally:
        driver.close()
        if not keep_up:
            try:
                compose_down()
            except CliError as exc:
                print(f"warning: could not stop the stack: {exc}", file=sys.stderr)


def print_summary(document: dict, path: Path) -> None:
    summary = document["summary"]
    print(f"Artifact: {path}")
    for users, entry in summary["ingest_uploads_per_s_by_users"].items():
        print(f"  ingest at {users} user(s): {entry['median']:.1f} uploads/s (n={entry['n']})")
    for workers, entry in summary["scaling_jobs_per_s_by_workers"].items():
        print(f"  {workers} worker(s): {entry['median']:.2f} jobs (extract+index)/s ({entry['min']:.2f} to {entry['max']:.2f}, n={entry['n']})")
    knee = summary["ramp_knee_users"]
    if knee:
        print(f"  read ramp knee: {knee['median']:g} users (range {knee['min']:g} to {knee['max']:g})")
    invalid = [phase for phase in document["phases"] if not phase["valid"]]
    for phase in invalid:
        print(f"  INVALID {phase['name']} r{phase['repetition']}: {'; '.join(phase['invalid_reasons'])}")
    problems = document["validation"]["problems"]
    if problems:
        print("The artifact does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
    else:
        print("The artifact validates.")
    notes = [p for p in document["validation"]["publishable_problems"] if p not in problems]
    if notes:
        print("Not publishable: " + "; ".join(notes))


def parse_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise CliError(f"{text!r} is not a comma-separated list of integers") from exc
    if not values or any(value < 0 for value in values):
        raise CliError(f"{text!r} must list non-negative integers")
    return values


def parse_tenant_documents(text: str) -> tuple[tuple[str, int], ...]:
    pairs = []
    for part in text.split(","):
        slug, _, count = part.partition("=")
        if slug not in workload.TENANT_SLUGS or not count.isdigit():
            raise CliError(f"{part!r} is not <tenant>=<documents> for one of {', '.join(workload.TENANT_SLUGS)}")
        pairs.append((slug, int(count)))
    return tuple(pairs)


def plan_from_args(args: argparse.Namespace) -> phases.Plan:
    """The plan a `run` was asked for: the full plan or `--quick`'s, with any
    number changed on the command line. A changed full plan is `custom`, which
    is not publishable."""
    plan = phases.quick_plan() if args.quick else phases.full_plan()
    changes: dict = {}
    if args.phases is not None:
        chosen = tuple(part for part in args.phases.split(",") if part)
        unknown = sorted(set(chosen) - set(phases.PHASES))
        if unknown or not chosen:
            raise CliError(f"--phases takes some of {', '.join(phases.PHASES)}")
        changes["phases"] = chosen
    if args.reps is not None:
        changes["repetitions"] = args.reps
    if args.workers is not None:
        changes["workers"] = parse_int_list(args.workers)
    if args.backlog is not None:
        changes["backlog_documents"] = args.backlog
    if args.ingest_users is not None:
        changes["ingest_users"] = parse_int_list(args.ingest_users)
    if args.ingest_documents is not None:
        changes["ingest_documents"] = args.ingest_documents
    if args.ramp_users is not None:
        changes["ramp_users"] = parse_int_list(args.ramp_users)
    if args.step_seconds is not None:
        changes["step_seconds"] = args.step_seconds
    if args.tenant_documents is not None:
        changes["tenant_documents"] = parse_tenant_documents(args.tenant_documents)
    if changes and plan.preset == "full":
        changes["preset"] = "custom"
    plan = dataclasses.replace(plan, **changes)
    if plan.repetitions < 1 or plan.backlog_documents < 1 or 0 in plan.workers + plan.ramp_users + plan.ingest_users:
        raise CliError("repetitions, backlog, workers and users must all be at least 1")
    return plan


def run(args: argparse.Namespace) -> int:
    plan = plan_from_args(args)
    out_dir = Path(args.out).resolve() if args.out else OUT_DIR
    if plan.preset != "full":
        out_dir = refuse_results_dir(out_dir, what=f"a {plan.preset} run")
    if args.scale < 0:
        raise CliError("--scale may not be negative")
    apply_run_environment(args.profile, args.scale, None)
    document, path, _result = execute(
        plan, label=args.label, out_dir=out_dir, argv=args.argv, keep_up=args.keep_up
    )
    print_summary(document, path)
    return 1 if document["validation"]["problems"] else 0


def smoke(args: argparse.Namespace) -> int:
    """10 uploads with no worker, 2 workers draining them, 20 s of reads at 2
    users, on the hashing embedder; then the assertions of smoke_failures. The
    samples and the artifact go to a temporary directory (or `--out`), never to
    scripts/load/results. The stack is stopped afterwards."""
    out_dir = refuse_results_dir(
        Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="doc-pilot-load-smoke-")),
        what="the smoke",
    )
    plan = phases.smoke_plan()
    apply_run_environment("zero", 1.0, "hashing")
    failures: list[str] = []

    def check(driver: StackDriver, result: phases.RunResult) -> None:
        samples = [s for path in result.sample_files for s in artifact.read_samples(path)]
        ids = sorted({s["document_id"] for s in samples if s["name"] == "upload" and s["ok"]})
        facts = parse_document_facts(driver.query(document_facts_sql(ids))) if ids else []
        failures.extend(smoke_failures(samples, facts, driver.ledger, result.phases, plan.backlog_documents))

    document, path, _result = execute(
        plan, label="smoke", out_dir=out_dir, argv=args.argv, after=check
    )
    print_summary(document, path)
    failures.extend(document["validation"]["problems"])
    if failures:
        print("Smoke failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("Smoke passed.")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    try:
        document = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CliError(f"{args.file} is not a readable JSON file") from exc
    problems = artifact.validate(document, publishable=args.publishable)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    if problems:
        return 1
    print("valid" + (" and publishable" if args.publishable else ""))
    return 0


def report_command(args: argparse.Namespace) -> int:
    return report.report_command(update_readme=args.update_readme, source=args.source, out=args.out)


# --- the command line --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    stack = commands.add_parser("stack", help="manage the isolated load stack")
    stack_commands = stack.add_subparsers(dest="stack_command", required=True)

    up = stack_commands.add_parser("up", help="build and start the stack, wait until healthy")
    up.set_defaults(run=stack_up)

    down = stack_commands.add_parser("down", help="stop and remove the stack's containers")
    down.add_argument(
        "--volumes",
        action="store_true",
        help=f"also delete the {PROJECT} volumes (db, uploads, models) after listing and checking them",
    )
    down.add_argument(
        "--dry-run",
        action="store_true",
        help="with --volumes: list and check the volumes, remove nothing",
    )
    down.set_defaults(run=stack_down)

    status = stack_commands.add_parser("status", help="show the stack's containers")
    status.set_defaults(run=stack_status)

    reset = stack_commands.add_parser(
        "reset",
        help="stop the stack and remove its database and uploads volumes (not the models volume)",
    )
    reset.set_defaults(run=stack_reset)

    seed_parser = commands.add_parser(
        "seed", help="create the three tenants, their owners and API keys (keys go to out/tenants.json)"
    )
    seed_parser.set_defaults(run=seed)

    preflight_parser = commands.add_parser(
        "preflight",
        help="refuse unless every api and worker is scripted and keyless; create pg_stat_statements",
    )
    preflight_parser.set_defaults(run=preflight)

    run_parser = commands.add_parser(
        "run",
        help="reset the load stack, run the phases, write a load_v1 artifact, stop the stack",
    )
    run_parser.add_argument("--profile", choices=("zero", "replay"), default="zero", help="the scripted model's latency profile")
    run_parser.add_argument("--scale", type=float, default=1.0, help="multiplier on the scripted latencies")
    run_parser.add_argument("--label", required=True, help="names the artifact: <run id>_<label>.json")
    run_parser.add_argument("--quick", action="store_true", help="a dry run of every phase in a few minutes; not publishable")
    run_parser.add_argument("--phases", help=f"comma-separated subset of {','.join(phases.PHASES)}")
    run_parser.add_argument("--reps", type=int, help="repetitions of each configuration (the plan says 3)")
    run_parser.add_argument("--workers", help="worker counts for the scaling phase, e.g. 1,2,4,8")
    run_parser.add_argument("--backlog", type=int, help="documents in the scaling backlog")
    run_parser.add_argument("--ingest-users", help="concurrency steps of the ingest phase, e.g. 1,2,4,8")
    run_parser.add_argument("--ingest-documents", type=int, help="uploads per ingest step")
    run_parser.add_argument("--ramp-users", help="user steps of the read ramp, e.g. 2,4,8")
    run_parser.add_argument("--step-seconds", type=int, help="seconds per read step")
    run_parser.add_argument("--tenant-documents", help="documents uploaded to each tenant before the ramp, e.g. load-large=2000,load-medium=300 (25 or more each, so every agent question has its document)")
    run_parser.add_argument("--out", help=f"where the artifact and samples go (default {repo_path(OUT_DIR)}; {repo_path(RESULTS_DIR)} takes full plans only)")
    run_parser.add_argument("--keep-up", action="store_true", help="leave the stack running afterwards")
    run_parser.set_defaults(run=run)

    smoke_parser = commands.add_parser(
        "smoke",
        help="10 uploads, 2 workers, 20 s of reads at 2 users, then assert zero errors, zero spend",
    )
    smoke_parser.add_argument("--out", help="a directory for the artifact and samples (default a temporary one)")
    smoke_parser.set_defaults(run=smoke)

    validate_parser = commands.add_parser("validate", help="check a load_v1 artifact")
    validate_parser.add_argument("file")
    validate_parser.add_argument("--publishable", action="store_true", help="also refuse a dirty tree, an invalid phase, a quick run")
    validate_parser.set_defaults(run=validate_command)

    report_parser = commands.add_parser(
        "report",
        help="render the README Performance tables and charts from the load artifacts",
    )
    which = report_parser.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--update-readme",
        action="store_true",
        help=f"rewrite the PERF_TABLE region of README.md and {repo_path(report.CHART_DIR)} from "
        f"{repo_path(RESULTS_DIR)}; refuses an artifact that is not publishable",
    )
    which.add_argument(
        "--from",
        dest="source",
        metavar="DIR",
        help="development: render the artifacts in DIR (dry runs included) to a scratch directory; never the README",
    )
    report_parser.add_argument("--out", help="with --from: the scratch directory (default a new temporary one)")
    report_parser.set_defaults(run=report_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "dry_run", False) and not args.volumes:
        print("--dry-run only applies with --volumes", file=sys.stderr)
        return 2
    args.argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return args.run(args)
    except (CliError, PhaseError, ReportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
