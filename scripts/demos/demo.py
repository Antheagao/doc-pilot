r"""Demo pipeline CLI: manage the isolated demo stack (docker-compose.demo.yml).

Run from anywhere with the backend venv (stdlib only, so no install step):

    backend\.venv\Scripts\python scripts\demos\demo.py stack up
    backend\.venv\Scripts\python scripts\demos\demo.py stack status
    backend\.venv\Scripts\python scripts\demos\demo.py stack down [--volumes] [--dry-run]
    backend\.venv\Scripts\python scripts\demos\demo.py seed [--reset]
    backend\.venv\Scripts\python scripts\demos\demo.py record <slug>... | all
    backend\.venv\Scripts\python scripts\demos\demo.py check

`record` and `check` need the `demo` extra (`pip install -e ".[dev,demo]"` in
backend/, then `python -m playwright install chromium`) and ffmpeg on PATH;
`record` also needs the demo stack up and seeded.

Every docker call goes through compose() below, which pins the compose
project to PROJECT. The project name is a constant, never an argument, so
this CLI cannot address the dev stack (project `doc-pilot`) or its volumes:
the dev database volume was wiped once by a careless `down -v`.

Recording is split across modules: scenarios.py (what to do), harness.py
(Playwright, the only module that imports it), encode.py (ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import encode
from scenarios import SCENARIOS, SMOKE_SLUG

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "doc-pilot-demo"
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.demo.yml")
PROJECT_LABEL = "com.docker.compose.project"
API_URL = "http://localhost:8100"
FRONTEND_URL = "http://localhost:3100"
FRONTEND_WAIT_SECONDS = 60
FEATURES_DIR = REPO_ROOT / "screenshots" / "features"
OUT_DIR = Path(__file__).resolve().parent / "out"


class CliError(Exception):
    """A failure to report on stderr and turn into a non-zero exit code."""


def compose_args(*args: str) -> list[str]:
    """The full `docker compose` command line for this project."""
    cmd = ["docker", "compose", "-p", PROJECT]
    for compose_file in COMPOSE_FILES:
        cmd += ["-f", compose_file]
    return [*cmd, *args]


def compose_env() -> dict[str, str]:
    """The environment compose runs with: the caller's, minus
    COMPOSE_PROFILES. A profile exported in the shell (e.g. `tracing`)
    would start the demo project's own jaeger, whose published ports
    collide with the dev stack's jaeger. Profiles this CLI needs are
    passed as --profile flags instead."""
    return {k: v for k, v in os.environ.items() if k != "COMPOSE_PROFILES"}


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a docker command from the repo root (the compose files and the
    build contexts are relative to it), turning a missing docker into a
    CliError instead of a traceback."""
    try:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=compose_env(),
            capture_output=capture,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CliError("docker was not found on PATH") from exc


def compose(*args: str) -> int:
    """Run `docker compose` for the demo project, output to the terminal."""
    return run(compose_args(*args)).returncode


def docker_output(cmd: list[str]) -> str:
    """stdout of a docker command; CliError with its stderr if it fails
    (typically Docker Desktop not running)."""
    result = run(cmd, capture=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise CliError(f"`{' '.join(cmd[:3])} ...` failed: {detail}")
    return result.stdout


def resolved_config() -> dict:
    """The demo project's fully merged compose config."""
    return json.loads(docker_output(compose_args("config", "--format", "json")))


def existing_volume_projects() -> dict[str, str]:
    """Every volume on this Docker host mapped to its compose project label
    ("" when it has none)."""
    out = docker_output(
        [
            "docker",
            "volume",
            "ls",
            "--format",
            f'{{{{.Name}}}}\t{{{{.Label "{PROJECT_LABEL}"}}}}',
        ]
    )
    projects = {}
    for line in out.splitlines():
        name, _, project = line.partition("\t")
        projects[name] = project
    return projects


def declared_volume_names(config: dict) -> list[str]:
    """Names of the volumes `down --volumes` deletes: compose removes every
    non-external volume in the resolved config by its resolved name, with no
    check of the volume's labels."""
    return [
        volume["name"]
        for volume in config.get("volumes", {}).values()
        if not volume.get("external")
    ]


def volume_problems(config: dict, existing: dict[str, str]) -> list[str]:
    """Reasons `down --volumes` must not run; empty when it is safe.

    config is `compose config --format json` and existing maps every volume
    on the host to its compose project label. Two independent views of the
    same set must agree: what compose would delete (config) and what Docker
    labels as this project's (existing)."""
    problems = []
    if config.get("name") != PROJECT:
        problems.append(f"compose project is {config.get('name')!r}, not {PROJECT!r}")
    declared = declared_volume_names(config)
    for name in declared:
        if not name.startswith(f"{PROJECT}_"):
            problems.append(f"{name}: would be removed but is not named {PROJECT}_*")
        elif name in existing and existing[name] != PROJECT:
            problems.append(f"{name}: exists with project label {existing[name]!r}")
    for name, project in existing.items():
        if project == PROJECT and name not in declared:
            problems.append(
                f"{name}: labelled {PROJECT} but not declared in the config"
            )
    return problems


def service_states() -> dict[str, dict]:
    """`compose ps --all` as {service: container record}."""
    out = docker_output(compose_args("ps", "--all", "--format", "json")).strip()
    # Compose prints one JSON object per line (older versions: one array).
    if out.startswith("["):
        records = json.loads(out)
    else:
        records = [json.loads(line) for line in out.splitlines()]
    return {record["Service"]: record for record in records}


def check_one_shots_and_worker(states: dict[str, dict]) -> None:
    """`up --wait` passes for a one-shot that exited, whatever its exit code,
    so check the migrate one-shot here, and that the worker's heartbeat
    healthcheck reports healthy."""
    migrate = states.get("migrate", {})
    if migrate.get("State") != "exited" or migrate.get("ExitCode") != 0:
        raise CliError(
            f"migrate did not complete: state {migrate.get('State')!r}, "
            f"exit code {migrate.get('ExitCode')!r}"
        )
    worker = states.get("worker", {})
    if worker.get("State") != "running" or worker.get("Health") != "healthy":
        raise CliError(
            f"worker is not healthy: state {worker.get('State')!r}, "
            f"health {worker.get('Health')!r}"
        )


def wait_for_frontend() -> None:
    """Poll the frontend until it serves 200: it has no healthcheck, so
    `up --wait` returns as soon as the container starts."""
    deadline = time.monotonic() + FRONTEND_WAIT_SECONDS
    while True:
        try:
            with urllib.request.urlopen(FRONTEND_URL, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:  # URLError is an OSError
            pass
        if time.monotonic() >= deadline:
            raise CliError(
                f"{FRONTEND_URL} did not return 200 within {FRONTEND_WAIT_SECONDS}s"
            )
        time.sleep(1)


def stack_up(_args: argparse.Namespace) -> int:
    code = compose("up", "-d", "--build", "--wait")
    if code != 0:
        return code
    check_one_shots_and_worker(service_states())
    wait_for_frontend()
    print(f"Demo stack is up: API {API_URL}, frontend {FRONTEND_URL}")
    return 0


def stack_status(_args: argparse.Namespace) -> int:
    return compose("ps", "--all")


def stack_down(args: argparse.Namespace) -> int:
    if not args.volumes:
        return compose("down")
    # Everything is checked before anything is removed.
    config = resolved_config()
    existing = existing_volume_projects()
    declared = declared_volume_names(config)
    print(f"Volumes that `down --volumes` would remove (project {PROJECT}):")
    for name in declared:
        print(f"  {name}  ({'exists' if name in existing else 'not created yet'})")
    if not declared:
        print("  (none)")
    problems = volume_problems(config, existing)
    if problems:
        print("Refusing to remove volumes:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("Dry run: nothing removed.")
        return 0
    return compose("down", "--volumes")


def seed_command(reset: bool) -> list[str]:
    """The compose arguments that run the seed one-shot. The seed profile is
    a flag here, never COMPOSE_PROFILES, for the reason compose_env gives."""
    # -T: no pseudo-TTY, so it works from automation as well as a terminal.
    command = ["--profile", "seed", "run", "--rm", "-T", "seed"]
    if reset:
        command += ["python", "scripts/seed_demo.py", "--reset"]
    return command


def seed(args: argparse.Namespace) -> int:
    # Build first so the seed image carries the current backend code (the
    # layers are cached when nothing changed); `run` alone would reuse a
    # stale image.
    code = compose("--profile", "seed", "build", "seed")
    if code != 0:
        return code
    return compose(*seed_command(args.reset))


def http_get(url: str) -> bytes | None:
    """The body of a GET that answers 200, or None when it does not."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read() if response.status == 200 else None
    except OSError:  # URLError is an OSError
        return None


def readiness_problems(
    api_ok: bool, frontend_ok: bool, document_count: int
) -> list[str]:
    """What stands between a recording and a usable demo stack, each with
    its fix; empty when the stack is up and seeded."""
    problems = []
    if not api_ok:
        problems.append(f"the API at {API_URL} is not healthy: run `demo.py stack up`")
    if not frontend_ok:
        problems.append(
            f"the frontend at {FRONTEND_URL} is not serving: run `demo.py stack up`"
        )
    if api_ok and document_count == 0:
        problems.append("the demo database is empty: run `demo.py seed --reset`")
    return problems


def ensure_stack_ready() -> None:
    api_ok = http_get(f"{API_URL}/healthz") is not None
    documents = http_get(f"{API_URL}/documents") if api_ok else None
    document_count = len(json.loads(documents)) if documents else 0
    frontend_ok = http_get(FRONTEND_URL) is not None
    problems = readiness_problems(api_ok, frontend_ok, document_count)
    if problems:
        raise CliError("; ".join(problems))


def resolve_slugs(requested: list[str]) -> list[str]:
    """The scenarios to record. `all` is every feature scenario: smoke only
    proves the harness and is recorded by name."""
    if requested == ["all"]:
        slugs = [slug for slug in SCENARIOS if slug != SMOKE_SLUG]
        if not slugs:
            raise CliError(
                "no feature scenarios are registered yet; try `record smoke`"
            )
        return slugs
    unknown = [slug for slug in requested if slug not in SCENARIOS]
    if unknown:
        known = ", ".join(sorted(SCENARIOS))
        raise CliError(f"unknown scenario {', '.join(unknown)} (known: {known})")
    return requested


def gif_destination(slug: str) -> Path | None:
    """Where a recorded GIF is committed to; None for smoke, which stays in
    the gitignored out/ directory."""
    return None if slug == SMOKE_SLUG else FEATURES_DIR / f"{slug}.gif"


def record(args: argparse.Namespace) -> int:
    slugs = resolve_slugs(args.slugs)
    encode.require_tools()
    ensure_stack_ready()
    for slug in slugs:
        print(f"Recording {slug} ...")
        try:
            from harness import recording

            with recording(slug, FRONTEND_URL, OUT_DIR) as rec:
                SCENARIOS[slug](rec.page, rec)
        except ModuleNotFoundError as exc:
            if exc.name != "playwright":
                raise
            raise CliError(
                'playwright is not installed: run `pip install -e ".[dev,demo]"` '
                "in backend/ and `python -m playwright install chromium`"
            ) from exc
        paths = encode.encode_all(
            OUT_DIR / f"{slug}.frames" / "frames.txt", OUT_DIR / f"{slug}.json", OUT_DIR
        )
        problems = encode.gif_problems(encode.probe(paths["gif"]))
        if problems:
            raise CliError(f"{slug}.gif breaks the limits: " + "; ".join(problems))
        destination = gif_destination(slug)
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(paths["gif"], destination)
            print(f"  copied the GIF to {destination.relative_to(REPO_ROOT)}")
        for kind, path in paths.items():
            print(
                f"  {kind}: {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)"
            )
    return 0


def check(_args: argparse.Namespace) -> int:
    """Every committed feature GIF against the limits (encode.gif_problems)."""
    encode.require_tools()
    gifs = sorted(FEATURES_DIR.glob("*.gif"))
    if not gifs:
        print(f"No GIFs in {FEATURES_DIR}: nothing to check.")
        return 0
    failed = False
    for gif in gifs:
        info = encode.probe(gif)
        problems = encode.gif_problems(info)
        summary = f"{info.duration:.1f}s, {info.size} bytes, {info.width}px"
        if problems:
            failed = True
            print(
                f"FAIL {gif.name}: {'; '.join(problems)} ({summary})", file=sys.stderr
            )
        else:
            print(f"ok   {gif.name}: {summary}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    stack = commands.add_parser("stack", help="manage the isolated demo stack")
    stack_commands = stack.add_subparsers(dest="stack_command", required=True)

    up = stack_commands.add_parser(
        "up", help="build and start the stack, wait until healthy"
    )
    up.set_defaults(run=stack_up)

    down = stack_commands.add_parser(
        "down", help="stop and remove the stack's containers"
    )
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

    seed_parser = commands.add_parser(
        "seed", help="load the deterministic demo dataset (replayed Sonnet run)"
    )
    seed_parser.add_argument(
        "--reset",
        action="store_true",
        help="wipe the demo database and uploads first (required if it has documents)",
    )
    seed_parser.set_defaults(run=seed)

    record_parser = commands.add_parser(
        "record",
        help="record scenarios against the seeded demo stack and encode them",
    )
    record_parser.add_argument(
        "slugs",
        nargs="+",
        metavar="slug",
        help="scenario slugs, or `all` for every feature scenario",
    )
    record_parser.set_defaults(run=record)

    check_parser = commands.add_parser(
        "check", help="check the committed GIFs against the size and length limits"
    )
    check_parser.set_defaults(run=check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "dry_run", False) and not args.volumes:
        print("--dry-run only applies with --volumes", file=sys.stderr)
        return 2
    try:
        return args.run(args)
    except (CliError, encode.EncodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
