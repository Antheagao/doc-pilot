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
`record` also needs the demo stack up and seeded; it signs in as the demo user
(the seed prints the sign-in), and so do the checks of the API's documents and
spend, so a stack seeded before auth needs `seed --reset` once.

Every docker call goes through compose() below, which pins the compose
project to PROJECT (the shared helpers are scripts/compose_project.py). The project name is a constant, never an argument, so
this CLI cannot address the dev stack (project `doc-pilot`) or its volumes:
the dev database volume was wiped once by a careless `down -v`.

Recording is split across modules: scenarios.py (what to do), harness.py
(Playwright, the only module that imports it), encode.py (ffmpeg).
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# compose_project.py sits in scripts/, beside this directory and scripts/load/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compose_project
import encode
from compose_project import (
    REPO_ROOT,
    CliError,
    ComposeProject,
    check_one_shots_and_worker,
)
from scenarios import LIVE_SLUGS, SCENARIOS, SMOKE_SLUG, ScenarioError

PROJECT = "doc-pilot-demo"
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.demo.yml")
STACK = ComposeProject(PROJECT, COMPOSE_FILES)
# The shared helpers under the names this module has always had.
compose_env = compose_project.compose_env
declared_volume_names = compose_project.declared_volume_names
API_URL = "http://localhost:8100"
FRONTEND_URL = "http://localhost:3100"
# The demo's sign-in, public by design. Repeated from backend/app/demo_seed.py
# (this module cannot import the backend); test_demo.py compares the two.
DEMO_USER_EMAIL = "demo@doc-pilot.local"
DEMO_USER_PASSWORD = "docpilot-demo!"
FRONTEND_WAIT_SECONDS = 60
# The most a set of live takes may spend: the demo overlay's default for
# DAILY_BUDGET_USD. A stack running with more is refused, not trusted.
MAX_LIVE_BUDGET_USD = 1.50
FEATURES_DIR = REPO_ROOT / "screenshots" / "features"
OUT_DIR = Path(__file__).resolve().parent / "out"


def compose_args(*args: str) -> list[str]:
    """The full `docker compose` command line for this project."""
    return STACK.compose_args(*args)


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a docker command from the repo root; see ComposeProject.run."""
    return STACK.run(cmd, capture=capture)


def compose(*args: str) -> int:
    """Run `docker compose` for the demo project, output to the terminal."""
    return STACK.compose(*args)


def docker_output(cmd: list[str]) -> str:
    """stdout of a docker command; CliError with its stderr if it fails
    (typically Docker Desktop not running)."""
    return STACK.docker_output(cmd)


def resolved_config() -> dict:
    """The demo project's fully merged compose config."""
    return STACK.resolved_config()


def existing_volume_projects() -> dict[str, str]:
    """Every volume on this Docker host mapped to its compose project label
    ("" when it has none)."""
    return STACK.existing_volume_projects()


def volume_problems(config: dict, existing: dict[str, str]) -> list[str]:
    """Reasons `down --volumes` must not run; empty when it is safe (the
    shared guard, bound to the demo project)."""
    return STACK.volume_problems(config, existing)


def service_states() -> dict[str, dict]:
    """`compose ps --all` as {service: container record}."""
    return STACK.service_states()


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
    if STACK.report_volumes(resolved_config(), existing_volume_projects()):
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


def http_get(url: str, opener: urllib.request.OpenerDirector | None = None) -> bytes | None:
    """The body of a GET that answers 200, or None when it does not. With an
    `opener` (authenticated_opener) the request carries its cookies."""
    open_url = opener.open if opener is not None else urllib.request.urlopen
    try:
        with open_url(url, timeout=5) as response:
            return response.read() if response.status == 200 else None
    except OSError:  # URLError is an OSError
        return None


SIGN_IN_HINTS = {
    401: "the demo user does not exist or has another password: run `demo.py seed --reset`",
    403: "the API's CORS_ORIGINS does not list the frontend's origin",
    429: "too many sign-in attempts: wait a minute",
}


def sign_in_opener() -> urllib.request.OpenerDirector:
    """An opener that signed in to the demo API as the demo user and keeps the
    session cookie in a CookieJar. The login is a cookie-authenticated write,
    so it names the frontend's origin like the browser does; the API refuses
    one that is not in its CORS_ORIGINS."""
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    request = urllib.request.Request(
        f"{API_URL}/auth/login",
        data=json.dumps({"email": DEMO_USER_EMAIL, "password": DEMO_USER_PASSWORD}).encode(),
        headers={"Content-Type": "application/json", "Origin": FRONTEND_URL},
        method="POST",
    )
    try:
        with opener.open(request, timeout=10):
            pass
    except urllib.error.HTTPError as exc:
        hint = SIGN_IN_HINTS.get(exc.code, "see the API's log")
        raise CliError(f"signing in to {API_URL} as the demo user failed (HTTP {exc.code}): {hint}") from exc
    except OSError as exc:
        raise CliError(f"could not reach {API_URL}/auth/login: {exc}") from exc
    return opener


_opener: urllib.request.OpenerDirector | None = None


def authenticated_opener() -> urllib.request.OpenerDirector:
    """The signed-in opener, made on first use and kept for the run: a session
    lasts 12 hours, and the API allows five sign-ins a minute per account."""
    global _opener
    if _opener is None:
        _opener = sign_in_opener()
    return _opener


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
    documents = http_get(f"{API_URL}/documents", authenticated_opener()) if api_ok else None
    document_count = len(json.loads(documents)) if documents else 0
    frontend_ok = http_get(FRONTEND_URL) is not None
    problems = readiness_problems(api_ok, frontend_ok, document_count)
    if problems:
        raise CliError("; ".join(problems))


def api_has_key() -> bool:
    """Whether the demo API container holds an Anthropic key. Only a yes or
    no leaves the container: the key itself is never printed."""
    out = docker_output(
        compose_args(
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            "import os; print(bool(os.environ.get('ANTHROPIC_API_KEY')))",
        )
    )
    return out.strip() == "True"


def budget_state() -> tuple[float | None, float]:
    """(daily budget, spent today) in USD from the demo API's /stats; the
    budget is None when the cap is off."""
    body = http_get(f"{API_URL}/stats", authenticated_opener())
    if body is None:
        raise CliError(f"could not read {API_URL}/stats")
    budget = json.loads(body)["budget"]
    return budget["daily_budget_usd"], budget["spent_today_usd"]


def live_problems(has_key: bool, budget: float | None, spent: float) -> list[str]:
    """What stands between a live take and a safe spend, each with its fix;
    empty when the key is there, the budget is capped at MAX_LIVE_BUDGET_USD
    or less, and some of it is left."""
    problems = []
    if not has_key:
        problems.append("the demo API has no ANTHROPIC_API_KEY (it would answer 503)")
    if budget is None or budget > MAX_LIVE_BUDGET_USD:
        shown = "off" if budget is None else f"${budget:.2f}"
        problems.append(
            f"the demo's daily budget is {shown}, over the ${MAX_LIVE_BUDGET_USD:.2f} "
            "cap for live takes: restart the stack without DEMO_DAILY_BUDGET_USD"
        )
    elif spent >= budget:
        problems.append(
            f"today's spend ${spent:.4f} has reached the ${budget:.2f} daily budget "
            "(the API would answer 429)"
        )
    return problems


def ensure_live_ready() -> float:
    """Refuse a live take unless live_problems is empty; returns the spend
    so far today, the baseline for the take's cost."""
    budget, spent = budget_state()
    problems = live_problems(api_has_key(), budget, spent)
    if problems:
        raise CliError("; ".join(problems))
    print(f"  live take: ${spent:.4f} of ${budget:.2f} spent today")
    return spent


def report_spend(before: float) -> None:
    """Print what the take cost and today's total, from /stats."""
    budget, spent = budget_state()
    limit = "no cap" if budget is None else f"${budget:.2f}"
    print(f"  spend: this take ${spent - before:.4f}, today ${spent:.4f} of {limit}")


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
        # A live take is checked before it spends, and its spend reported
        # after it whether or not it came out (a failed take still billed).
        spent_before = ensure_live_ready() if slug in LIVE_SLUGS else None
        try:
            from harness import recording

            with recording(
                slug,
                FRONTEND_URL,
                OUT_DIR,
                email=DEMO_USER_EMAIL,
                password=DEMO_USER_PASSWORD,
            ) as rec:
                SCENARIOS[slug](rec.page, rec)
        except ModuleNotFoundError as exc:
            if exc.name != "playwright":
                raise
            raise CliError(
                'playwright is not installed: run `pip install -e ".[dev,demo]"` '
                "in backend/ and `python -m playwright install chromium`"
            ) from exc
        except ScenarioError as exc:
            raise CliError(f"{slug} take failed: {exc}") from exc
        finally:
            if spent_before is not None:
                # Never let a failed spend report replace the take's error.
                try:
                    report_spend(spent_before)
                except CliError as exc:
                    print(f"warning: could not report the spend: {exc}", file=sys.stderr)
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
