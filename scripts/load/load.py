r"""Load-test CLI: manage the isolated load stack (docker-compose.load.yml).

Run from anywhere with the backend venv (stdlib only, so no install step):

    backend/.venv/bin/python scripts/load/load.py stack up
    backend/.venv/bin/python scripts/load/load.py stack status
    backend/.venv/bin/python scripts/load/load.py stack down [--volumes] [--dry-run]
    backend/.venv/bin/python scripts/load/load.py stack reset
    backend/.venv/bin/python scripts/load/load.py seed
    backend/.venv/bin/python scripts/load/load.py preflight

The stack runs the production image against a scripted model (no model API is
ever called) in its own compose project, with its own database `docpilot_load`
and no host database port. `seed` creates three organizations, an owner for
each, and an API key for each, through scripts/accounts.py inside the api
container; the keys and passwords go to the gitignored scripts/load/out/
tenants.json (mode 600) for the harness and are never printed. Documents are
uploaded by the harness, not here. `preflight` refuses to let traffic start
unless every api and worker container reports the scripted provider and holds
no Anthropic key; it also creates the pg_stat_statements extension.

Every docker call goes through STACK below, which pins the compose project to
PROJECT. The project name is a constant, never an argument, so this CLI cannot
address the dev stack (project `doc-pilot`) or the demo's, or their volumes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# compose_project.py sits in scripts/, beside this directory and scripts/demos/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compose_project import (
    REPO_ROOT,
    CliError,
    ComposeProject,
    check_one_shots_and_worker,
    declared_volume_names,
)

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


def stack_up(_args: argparse.Namespace) -> int:
    # Nothing starts from a config that could reach a real model.
    problems = config_problems(STACK.resolved_config(all_profiles=True))
    if problems:
        raise CliError("refusing to start: " + "; ".join(problems))
    code = STACK.compose("up", "-d", "--build", "--wait")
    if code != 0:
        return code
    check_one_shots_and_worker(STACK.service_states())
    print(f"Load stack is up: API {API_URL}")
    return 0


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "dry_run", False) and not args.volumes:
        print("--dry-run only applies with --volumes", file=sys.stderr)
        return 2
    try:
        return args.run(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
