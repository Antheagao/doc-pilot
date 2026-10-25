"""Tests for the load CLI's guards, seed, and preflight (run: python -m pytest ../scripts -q).

No Docker is needed: subprocess.run is replaced by a fake that plays a load
stack, so these exercise the real command paths without starting anything.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import load
import pytest

CliError = load.CliError

SECRET_KEY = "dpk_" + "abcdefghijkl" + "_" + "A" * 43
OTHER_KEYS = [
    "dpk_" + "mnopqrstuvwx" + "_" + "B" * 43,
    "dpk_" + "yz2345abcdef" + "_" + "C" * 43,
]
REAL_LOOKING_KEY = "sk-ant-api03-not-a-real-key"
VOLUMES = {
    "db_data": {"name": "doc-pilot-load_db_data"},
    "uploads": {"name": "doc-pilot-load_uploads"},
    "models": {"name": "doc-pilot-load_models"},
}


def good_config() -> dict:
    blank = {"ANTHROPIC_API_KEY": ""}
    scripted = {**blank, **{setting: "scripted" for setting in load.MODEL_SETTINGS}}
    return {
        "name": "doc-pilot-load",
        "services": {
            "db": {"environment": {"POSTGRES_DB": "docpilot_load"}},
            "migrate": {"environment": dict(blank)},
            "api": {"environment": {**scripted, "DAILY_BUDGET_USD": "0.01"}},
            "worker": {
                "environment": {
                    **scripted,
                    "DAILY_BUDGET_USD": "0.01",
                    "WORKER_RECLAIM_AFTER_SECONDS": "600",
                }
            },
            "loadgen": {"environment": dict(blank)},
            "frontend": {},
        },
        "volumes": dict(VOLUMES),
    }


def probe_output(**changes: object) -> str:
    report = {
        "provider": "scripted",
        "has_key": False,
        "database": "docpilot_load",
        "role": "docpilot_runtime",
    }
    return json.dumps({**report, **changes}) + "\n"


def ps_output(*records: dict) -> str:
    return "\n".join(json.dumps(record) for record in records)


RUNNING = (
    {"Service": "api", "Name": "doc-pilot-load-api-1", "State": "running"},
    {"Service": "worker", "Name": "doc-pilot-load-worker-1", "State": "running"},
    {"Service": "db", "Name": "doc-pilot-load-db-1", "State": "running"},
)


Answer = str | tuple[int, str, str] | Callable[[list[str]], str]


class FakeDocker:
    """A stand-in for subprocess.run that plays a load stack. `handlers` maps a
    command-line predicate to the stdout (or a (returncode, stdout, stderr)
    tuple) it answers with; the first match wins and an unmatched command
    fails the test. Every call is recorded with its stdin."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []
        self.handlers: list[tuple[Callable[[list[str]], bool], Answer]] = []

    def on(self, predicate: Callable[[list[str]], bool], answer: Answer) -> None:
        self.handlers.append((predicate, answer))

    def __call__(self, cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        self.stdins.append(kwargs.get("input"))
        for predicate, answer in self.handlers:
            if predicate(cmd):
                if callable(answer):
                    answer = answer(cmd)
                returncode, stdout, stderr = answer if isinstance(answer, tuple) else (0, answer, "")
                return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
        raise AssertionError(f"unexpected command: {cmd}")

    def ran(self, *words: str) -> list[list[str]]:
        return [cmd for cmd in self.calls if all(word in cmd for word in words)]


def has(*words: str) -> Callable[[list[str]], bool]:
    return lambda cmd: all(word in cmd for word in words)


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


@pytest.fixture
def stack(docker: FakeDocker) -> FakeDocker:
    """A healthy load stack: the good config, api and worker running, and every
    container probing clean."""
    docker.on(has("config"), json.dumps(good_config()))
    docker.on(has("ps"), ps_output(*RUNNING))
    docker.on(has("exec", "doc-pilot-load-api-1"), probe_output())
    docker.on(has("exec", "doc-pilot-load-worker-1"), probe_output())
    docker.on(has("volume", "ls"), "")
    return docker


@pytest.fixture
def tenants_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "out" / "tenants.json"
    monkeypatch.setattr(load, "TENANTS_FILE", path)
    return path


# --- the project is a constant -----------------------------------------------


def test_the_project_is_the_load_project() -> None:
    assert load.PROJECT == "doc-pilot-load"
    assert load.STACK.compose_args("ps")[:4] == ["docker", "compose", "-p", "doc-pilot-load"]
    assert load.COMPOSE_FILES[-1] == "docker-compose.load.yml"


def test_no_command_takes_a_project_argument() -> None:
    parser = load.build_parser()
    for argv in (["stack", "up", "--project", "doc-pilot"], ["preflight", "-p", "doc-pilot"]):
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(argv)
        assert raised.value.code == 2


def test_the_parser_reads_every_command() -> None:
    parser = load.build_parser()
    assert parser.parse_args(["stack", "down", "--volumes", "--dry-run"]).dry_run is True
    assert parser.parse_args(["stack", "down"]).volumes is False
    for argv in (["stack", "up"], ["stack", "status"], ["stack", "reset"], ["seed"], ["preflight"]):
        assert callable(parser.parse_args(argv).run)
    with pytest.raises(SystemExit):
        parser.parse_args(["stack"])


def test_dry_run_needs_volumes(capsys: pytest.CaptureFixture) -> None:
    assert load.main(["stack", "down", "--dry-run"]) == 2
    assert "--dry-run only applies with --volumes" in capsys.readouterr().err


def test_a_cli_error_is_reported_as_an_exit_code(
    docker: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    docker.on(has("config"), (1, "", "Cannot connect to the Docker daemon"))
    assert load.main(["stack", "up"]) == 1
    assert "Cannot connect to the Docker daemon" in capsys.readouterr().err


def test_the_tenant_output_directory_is_ignored_by_git() -> None:
    gitignore = (load.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "scripts/load/out/" in gitignore.splitlines()


# --- the volume guard through the CLI ------------------------------------------


def dev_volume_config() -> dict:
    config = good_config()
    config["volumes"]["db_data"] = {"name": "doc-pilot_db_data"}
    return config


def test_down_volumes_dry_run_lists_only_the_load_volumes(
    stack: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    existing = "".join(f"{v['name']}\tdoc-pilot-load\n" for v in VOLUMES.values())
    existing += "doc-pilot_db_data\tdoc-pilot\ndoc-pilot-demo_db_data\tdoc-pilot-demo\n"
    stack.handlers.insert(0, (has("volume", "ls"), existing))
    assert load.main(["stack", "down", "--volumes", "--dry-run"]) == 0
    out = capsys.readouterr().out
    listed = [line.split()[0] for line in out.splitlines() if line.startswith("  doc-pilot")]
    assert listed == ["doc-pilot-load_db_data", "doc-pilot-load_uploads", "doc-pilot-load_models"]
    assert "Dry run: nothing removed." in out
    assert not stack.ran("down")


def test_down_volumes_refuses_a_dev_or_demo_volume(
    stack: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    for foreign in ("doc-pilot_db_data", "doc-pilot-demo_uploads"):
        config = good_config()
        config["volumes"]["db_data"] = {"name": foreign}
        stack.handlers.insert(0, (has("config"), json.dumps(config)))
        assert load.main(["stack", "down", "--volumes"]) == 1
        assert foreign in capsys.readouterr().err
    assert not stack.ran("down")


def test_down_volumes_refuses_a_config_of_another_project(
    stack: FakeDocker, tenants_file: Path
) -> None:
    tenants_file.parent.mkdir(parents=True)
    tenants_file.write_text("{}")
    stack.handlers.insert(0, (has("config"), json.dumps({**good_config(), "name": "doc-pilot"})))
    assert load.main(["stack", "down", "--volumes"]) == 1
    assert not stack.ran("down")
    assert tenants_file.exists()


def test_down_volumes_removes_after_the_guard_passes(stack: FakeDocker, tenants_file: Path) -> None:
    tenants_file.parent.mkdir(parents=True)
    tenants_file.write_text("{}")
    stack.on(has("down"), "")
    assert load.main(["stack", "down", "--volumes"]) == 0
    assert not tenants_file.exists()  # its keys died with the database
    [down] = stack.ran("down")
    assert down[down.index("-p") + 1] == "doc-pilot-load"
    assert down[-2:] == ["down", "--volumes"]
    assert down[down.index("--profile") + 1] == "*"  # profile containers go too


def test_down_without_volumes_checks_nothing_and_keeps_them(stack: FakeDocker) -> None:
    stack.on(has("down"), "")
    assert load.main(["stack", "down"]) == 0
    [down] = stack.ran("down")
    assert "--volumes" not in down
    assert not stack.ran("volume")


# --- reset ---------------------------------------------------------------------


def test_reset_removes_only_the_database_and_uploads(
    stack: FakeDocker, tenants_file: Path
) -> None:
    tenants_file.parent.mkdir(parents=True)
    tenants_file.write_text("{}")
    existing = "".join(f"{v['name']}\tdoc-pilot-load\n" for v in VOLUMES.values())
    stack.handlers.insert(0, (has("volume", "ls"), existing))
    stack.on(has("down"), "")
    stack.on(has("volume", "rm"), "")
    assert load.main(["stack", "reset"]) == 0
    removed = [cmd[-1] for cmd in stack.ran("volume", "rm")]
    assert removed == ["doc-pilot-load_db_data", "doc-pilot-load_uploads"]
    assert "--volumes" not in stack.ran("down")[0]
    assert not tenants_file.exists()  # its keys died with the database


def test_reset_skips_volumes_that_do_not_exist_yet(stack: FakeDocker, tenants_file: Path) -> None:
    stack.on(has("down"), "")
    assert load.main(["stack", "reset"]) == 0
    assert not stack.ran("volume", "rm")


@pytest.mark.parametrize("foreign", ["doc-pilot_db_data", "doc-pilot-demo_uploads"])
def test_reset_refuses_when_the_guard_does_and_removes_nothing(
    stack: FakeDocker, tenants_file: Path, foreign: str, capsys: pytest.CaptureFixture
) -> None:
    tenants_file.parent.mkdir(parents=True)
    tenants_file.write_text("{}")
    config = good_config()
    config["volumes"]["uploads"] = {"name": foreign}
    stack.handlers.insert(0, (has("config"), json.dumps(config)))
    assert load.main(["stack", "reset"]) == 1
    assert foreign in capsys.readouterr().err
    assert not stack.ran("down")
    assert not stack.ran("volume", "rm")
    assert tenants_file.exists()


def test_reset_needs_the_volumes_declared() -> None:
    config = good_config()
    del config["volumes"]["uploads"]
    with pytest.raises(CliError, match="doc-pilot-load_uploads"):
        load.reset_targets(config)
    assert load.reset_targets(good_config()) == ["doc-pilot-load_db_data", "doc-pilot-load_uploads"]


# --- the no-spend config check -----------------------------------------------------


def test_the_overlays_resolved_config_passes() -> None:
    assert load.config_problems(good_config()) == []


def test_a_key_in_any_service_refuses_and_is_never_echoed() -> None:
    for service in ("api", "worker", "migrate", "loadgen", "frontend"):
        config = good_config()
        config["services"][service].setdefault("environment", {})["ANTHROPIC_API_KEY"] = REAL_LOOKING_KEY
        problems = load.config_problems(config)
        assert problems == [f"{service}: ANTHROPIC_API_KEY is not empty"]
        assert REAL_LOOKING_KEY not in " ".join(problems)


def test_a_backend_service_must_blank_the_key_explicitly() -> None:
    for service in load.BACKEND_SERVICES:
        config = good_config()
        del config["services"][service]["environment"]["ANTHROPIC_API_KEY"]
        assert load.config_problems(config) == [f"{service}: ANTHROPIC_API_KEY is not blanked explicitly"]


def test_api_and_worker_must_run_the_scripted_provider_and_models() -> None:
    for service in load.MODEL_SERVICES:
        for setting in load.MODEL_SETTINGS:
            config = good_config()
            config["services"][service]["environment"][setting] = "anthropic"
            assert load.config_problems(config) == [f"{service}: {setting} is not 'scripted'"]
    config = good_config()
    del config["services"]["worker"]["environment"]["MODEL_PROVIDER"]
    assert load.config_problems(config) == ["worker: MODEL_PROVIDER is not 'scripted'"]


def test_the_budget_must_stay_a_tripwire() -> None:
    for budget in ("5", "0", "0.02", "", "off"):
        config = good_config()
        config["services"]["api"]["environment"]["DAILY_BUDGET_USD"] = budget
        assert len(load.config_problems(config)) == 1
    config = good_config()
    del config["services"]["api"]["environment"]["DAILY_BUDGET_USD"]
    assert len(load.config_problems(config)) == 1


def test_the_worker_must_hold_a_lease() -> None:
    for lease in ("0", "-5", "", "soon", "nan"):
        config = good_config()
        config["services"]["worker"]["environment"]["WORKER_RECLAIM_AFTER_SECONDS"] = lease
        assert load.config_problems(config) == ["worker: WORKER_RECLAIM_AFTER_SECONDS is not a positive lease"]
    config = good_config()
    del config["services"]["worker"]["environment"]["WORKER_RECLAIM_AFTER_SECONDS"]
    assert len(load.config_problems(config)) == 1
    config["services"]["worker"]["environment"]["WORKER_RECLAIM_AFTER_SECONDS"] = "0.5"
    assert load.config_problems(config) == []


def test_another_project_refuses() -> None:
    assert load.config_problems({**good_config(), "name": "doc-pilot"})


# --- the live probe ----------------------------------------------------------------


def test_a_clean_probe_has_no_problems() -> None:
    probe = load.parse_probe("c", probe_output())
    assert probe == load.Probe("c", "scripted", False, "docpilot_load", "docpilot_runtime")
    assert load.probe_problems(probe) == []


def test_the_probe_reads_the_last_line_of_the_output() -> None:
    assert load.parse_probe("c", "a warning\n" + probe_output()).provider == "scripted"


@pytest.mark.parametrize("output", ["", "not json\n", '{"provider": "scripted"}\n', "[]\n"])
def test_an_unreadable_probe_is_a_cli_error(output: str) -> None:
    with pytest.raises(CliError, match="probe printed something unexpected"):
        load.parse_probe("c", output)


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"provider": "anthropic"}, "model provider is 'anthropic'"),
        ({"has_key": True}, "holds an Anthropic API key"),
        ({"database": "docpilot"}, "database 'docpilot'"),
        ({"role": "docpilot"}, "role 'docpilot'"),
    ],
)
def test_each_probe_failure_is_named(changes: dict, fragment: str) -> None:
    problems = load.probe_problems(load.parse_probe("c", probe_output(**changes)))
    assert len(problems) == 1
    assert fragment in problems[0]


def test_only_running_api_and_worker_containers_are_asked() -> None:
    records = [
        *RUNNING,
        {"Service": "worker", "Name": "w2", "State": "exited"},
        {"Service": "migrate", "Name": "m", "State": "running"},
    ]
    assert [r["Name"] for r in load.running_model_containers(records)] == [
        "doc-pilot-load-api-1",
        "doc-pilot-load-worker-1",
    ]


# --- the preflight --------------------------------------------------------------------


def test_a_healthy_stack_has_no_problems(stack: FakeDocker) -> None:
    assert load.check_stack() == ([], 2)
    asked = [cmd[:3] for cmd in stack.ran("exec")]
    assert asked == [["docker", "exec", "doc-pilot-load-api-1"], ["docker", "exec", "doc-pilot-load-worker-1"]]


def test_a_worker_with_a_key_fails_the_preflight(stack: FakeDocker) -> None:
    stack.handlers.insert(0, (has("exec", "doc-pilot-load-worker-1"), probe_output(has_key=True)))
    problems, _ = load.check_stack()
    assert problems == ["doc-pilot-load-worker-1: holds an Anthropic API key"]


def test_every_scaled_worker_is_asked(stack: FakeDocker) -> None:
    extra = {"Service": "worker", "Name": "doc-pilot-load-worker-2", "State": "running"}
    stack.handlers.insert(0, (has("ps"), ps_output(*RUNNING, extra)))
    stack.handlers.insert(0, (has("exec", "doc-pilot-load-worker-2"), probe_output(provider="anthropic")))
    problems, asked = load.check_stack()
    assert asked == 3
    assert len(problems) == 1
    assert "doc-pilot-load-worker-2" in problems[0]


def test_a_stack_with_no_api_container_fails(stack: FakeDocker) -> None:
    stack.handlers.insert(0, (has("ps"), ps_output(RUNNING[1], RUNNING[2])))
    problems, _ = load.check_stack()
    assert any("no api container is running" in problem for problem in problems)


def test_a_container_that_cannot_be_probed_fails(stack: FakeDocker) -> None:
    stack.handlers.insert(
        0, (has("exec", "doc-pilot-load-api-1"), (1, "", "pydantic ValidationError: scripted needs"))
    )
    problems, _ = load.check_stack()
    assert len(problems) == 1
    assert "doc-pilot-load-api-1" in problems[0]
    assert "ValidationError" in problems[0]


def test_a_config_with_a_key_fails_even_when_the_containers_are_clean(stack: FakeDocker) -> None:
    config = good_config()
    config["services"]["loadgen"]["environment"]["ANTHROPIC_API_KEY"] = REAL_LOOKING_KEY
    stack.handlers.insert(0, (has("config"), json.dumps(config)))
    problems, _ = load.check_stack()
    assert problems == ["loadgen: ANTHROPIC_API_KEY is not empty"]


def test_the_config_check_covers_every_profile(stack: FakeDocker) -> None:
    load.check_stack()
    [config_call] = stack.ran("config")
    assert config_call[config_call.index("--profile") + 1] == "*"


def psql_answers(stack: FakeDocker, database: str = "docpilot_load", preload: str = "pg_stat_statements") -> None:
    stack.on(has("psql", "select current_database()"), database + "\n")
    stack.on(has("psql", "CREATE EXTENSION IF NOT EXISTS pg_stat_statements"), "CREATE EXTENSION\n")
    stack.on(has("psql", "show shared_preload_libraries"), preload + "\n")
    stack.on(has("psql", "select count(*) from pg_stat_statements"), "0\n")


def test_preflight_creates_the_extension_as_the_owner_in_the_load_database(
    stack: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    psql_answers(stack)
    assert load.main(["preflight"]) == 0
    assert "Preflight passed." in capsys.readouterr().out
    statements = [cmd for cmd in stack.calls if "psql" in cmd]
    assert len(statements) == 4
    for cmd in statements:
        assert cmd[cmd.index("-U") + 1] == "docpilot"
        assert cmd[cmd.index("-d") + 1] == "docpilot_load"
        assert "db" in cmd[cmd.index("exec") : cmd.index("psql")]
    assert any("CREATE EXTENSION IF NOT EXISTS pg_stat_statements" in cmd for cmd in statements)


def test_preflight_creates_nothing_when_a_check_fails(
    stack: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    stack.handlers.insert(0, (has("exec", "doc-pilot-load-api-1"), probe_output(has_key=True)))
    psql_answers(stack)
    assert load.main(["preflight"]) == 1
    assert "holds an Anthropic API key" in capsys.readouterr().err
    assert not [cmd for cmd in stack.calls if "psql" in cmd]


def test_the_extension_is_not_created_in_another_database(stack: FakeDocker) -> None:
    psql_answers(stack, database="docpilot")
    with pytest.raises(CliError, match="'docpilot', not 'docpilot_load'"):
        load.create_statements_extension()
    assert not stack.ran("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")


def test_the_extension_needs_the_library_preloaded(stack: FakeDocker) -> None:
    psql_answers(stack, preload="")
    with pytest.raises(CliError, match="shared_preload_libraries"):
        load.create_statements_extension()


# --- up -------------------------------------------------------------------------------


def test_up_refuses_a_config_that_could_reach_a_model(
    stack: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    config = good_config()
    config["services"]["api"]["environment"]["ANTHROPIC_API_KEY"] = REAL_LOOKING_KEY
    stack.handlers.insert(0, (has("config"), json.dumps(config)))
    assert load.main(["stack", "up"]) == 1
    err = capsys.readouterr().err
    assert "api: ANTHROPIC_API_KEY is not empty" in err
    assert REAL_LOOKING_KEY not in err
    assert not stack.ran("up")


def test_up_builds_and_waits_then_checks_the_one_shots(
    stack: FakeDocker, capsys: pytest.CaptureFixture
) -> None:
    stack.on(has("up"), "")
    stack.handlers.insert(
        0,
        (
            has("ps", "--all"),
            ps_output(
                {"Service": "migrate", "State": "exited", "ExitCode": 0},
                {"Service": "worker", "State": "running", "Health": "healthy"},
            ),
        ),
    )
    assert load.main(["stack", "up"]) == 0
    [up] = stack.ran("up")
    assert up[-4:] == ["up", "-d", "--build", "--wait"]
    assert "127.0.0.1:8200" in capsys.readouterr().out


# --- tenants and keys --------------------------------------------------------------


def accounts_calls(stack: FakeDocker) -> list[list[str]]:
    return stack.ran("scripts/accounts.py")


def test_a_key_is_parsed_from_stdout_alone() -> None:
    assert load.parse_api_key(SECRET_KEY + "\n") == SECRET_KEY
    assert load.parse_api_key(f"\n  {SECRET_KEY}  \n\n") == SECRET_KEY


@pytest.mark.parametrize(
    "output",
    ["", "created API key\n", f"noise\n{SECRET_KEY}\n", f"{SECRET_KEY}\n{SECRET_KEY}\n", SECRET_KEY + "x"],
)
def test_anything_but_one_key_is_refused_without_echoing_it(output: str) -> None:
    with pytest.raises(CliError) as raised:
        load.parse_api_key(output)
    assert SECRET_KEY not in str(raised.value)
    assert "noise" not in str(raised.value)


def test_a_tenant_is_made_through_accounts_with_the_password_on_stdin(
    monkeypatch: pytest.MonkeyPatch, docker: FakeDocker
) -> None:
    docker.on(has("create-organization"), "created\n")
    docker.on(has("create-user"), "created\n")
    docker.on(has("add-member"), "added\n")
    docker.on(has("create-api-key"), SECRET_KEY + "\n")
    tenant = load.TENANTS[0]
    record = load.create_tenant(tenant)
    assert record["api_key"] == SECRET_KEY
    assert record["organization"] == "load-large"
    assert len(record["password"]) >= 24
    commands = [cmd[cmd.index("scripts/accounts.py") + 1] for cmd in accounts_calls(docker)]
    assert commands == ["create-organization", "create-user", "add-member", "create-api-key"]
    for cmd in accounts_calls(docker):
        assert cmd[cmd.index("exec") + 1 :][:3] == ["-T", "api", "python"]
        assert record["password"] not in " ".join(cmd)  # never an argument
        assert SECRET_KEY not in " ".join(cmd)
    create_user = docker.ran("create-user")[0]
    assert "--password-stdin" in create_user
    assert docker.stdins[docker.calls.index(create_user)] == record["password"] + "\n"
    assert [cmd[cmd.index("--role") + 1] for cmd in docker.ran("add-member")] == ["owner"]


def test_the_three_planned_tenants() -> None:
    assert [tenant.slug for tenant in load.TENANTS] == ["load-large", "load-medium", "load-small"]
    assert len({tenant.owner_email for tenant in load.TENANTS}) == 3


def test_a_private_file_is_mode_600_and_replaces_atomically(tmp_path: Path) -> None:
    path = tmp_path / "out" / "tenants.json"
    load.write_private_json(path, {"a": 1})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # A scratch file left behind with a wide mode does not leak into the result.
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text("old")
    scratch.chmod(0o644)
    load.write_private_json(path, {"a": 2})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"a": 2}
    assert not scratch.exists()


def test_reading_tenants_that_were_never_seeded_says_how_to_seed(tmp_path: Path) -> None:
    with pytest.raises(CliError, match=r"load\.py seed"):
        load.read_tenants(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    with pytest.raises(CliError, match=r"load\.py seed"):
        load.read_tenants(broken)


def test_seed_creates_three_tenants_and_stores_their_keys_privately(
    docker: FakeDocker,
    tenants_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    keys = iter([SECRET_KEY, *OTHER_KEYS])
    docker.on(has("create-organization"), "created\n")
    docker.on(has("create-user"), "created\n")
    docker.on(has("add-member"), "added\n")
    docker.on(has("create-api-key"), lambda cmd: next(keys) + "\n")
    monkeypatch.setattr(load, "seeded_keys_work", lambda: False)
    assert load.main(["seed"]) == 0
    tenants = load.read_tenants(tenants_file)
    assert list(tenants) == ["load-large", "load-medium", "load-small"]
    assert tenants["load-large"]["api_key"] == SECRET_KEY
    assert stat.S_IMODE(tenants_file.stat().st_mode) == 0o600
    shown = capsys.readouterr()
    everything = shown.out + shown.err
    stored = [t["api_key"] for t in tenants.values()] + [t["password"] for t in tenants.values()]
    assert len(set(stored)) == 6
    assert not any(secret in everything for secret in stored)
    assert "Keys and passwords written (mode 600)" in everything
    assert len(accounts_calls(docker)) == 12  # four commands for each of three tenants


def test_seed_does_nothing_when_the_stored_keys_still_work(
    docker: FakeDocker, tenants_file: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = {
        tenant.slug: {"organization": tenant.slug, "email": tenant.owner_email, "password": "p", "api_key": key}
        for tenant, key in zip(load.TENANTS, [SECRET_KEY, *OTHER_KEYS], strict=True)
    }
    load.write_private_json(tenants_file, {"tenants": records})
    seen = []
    monkeypatch.setattr(load, "api_status", lambda path, key: seen.append((path, key)) or 200)
    assert load.main(["seed"]) == 0
    assert "Already seeded" in capsys.readouterr().out
    assert [path for path, _ in seen] == ["/documents"] * 3
    assert not docker.calls


def test_seed_runs_again_when_a_stored_key_is_rejected(
    tenants_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load.write_private_json(tenants_file, {"tenants": {"load-large": {"api_key": SECRET_KEY}}})
    monkeypatch.setattr(load, "api_status", lambda path, key: 200)
    assert load.seeded_keys_work() is False  # one tenant of three is not a seeded stack
    records = {t.slug: {"api_key": SECRET_KEY} for t in load.TENANTS}
    load.write_private_json(tenants_file, {"tenants": records})
    assert load.seeded_keys_work() is True
    monkeypatch.setattr(load, "api_status", lambda path, key: 401)
    assert load.seeded_keys_work() is False


def test_a_failed_seed_says_to_run_it_again_and_writes_no_keys(
    docker: FakeDocker, tenants_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker.on(has("create-organization"), (1, "", "error: the database refused the change (ck_x)"))
    monkeypatch.setattr(load, "seeded_keys_work", lambda: False)
    with pytest.raises(CliError, match=r"seeding load-large failed.*load\.py seed. again") as raised:
        load.seed(None)
    assert "ck_x" in str(raised.value)
    assert not tenants_file.exists()


class FakeAccounts:
    """A stand-in for accounts.py behind the fake docker: it remembers what
    exists, refuses a duplicate the way accounts.py words it, mints a distinct
    key per call, and can be told to fail one command once."""

    def __init__(self, docker: FakeDocker) -> None:
        self.organizations: set[str] = set()
        self.users: set[str] = set()
        self.members: set[tuple[str, str]] = set()
        self.minted: list[str] = []
        self.passwords: dict[str, str] = {}
        self.fail_once: tuple[str, str] | None = None  # (command, argument value)
        docker.on(has("scripts/accounts.py"), self.answer)
        self.docker = docker

    def answer(self, cmd: list[str]) -> tuple[int, str, str]:
        command = cmd[cmd.index("scripts/accounts.py") + 1]
        arguments = cmd[cmd.index("scripts/accounts.py") + 2 :]

        def value(flag: str) -> str:
            return arguments[arguments.index(flag) + 1]

        subject = value("--slug") if command == "create-organization" else value("--email") if command in (
            "create-user", "set-password", "add-member") else value("--organization")
        if self.fail_once == (command, subject):
            self.fail_once = None
            return 1, "", "error: the database refused the change (boom)"
        if command == "create-organization":
            if subject in self.organizations:
                return 1, "", f"error: the slug {subject!r} is already taken"
            self.organizations.add(subject)
        elif command == "create-user":
            if subject in self.users:
                return 1, "", f"error: the email {subject!r} already has an account"
            self.users.add(subject)
            self.passwords[subject] = self.stdin_of(cmd)
        elif command == "set-password":
            self.passwords[subject] = self.stdin_of(cmd)
        elif command == "add-member":
            pair = (value("--organization"), subject)
            if pair in self.members:
                return 1, "", "error: the user is already a member of the organization"
            self.members.add(pair)
        elif command == "create-api-key":
            key = "dpk_" + f"{len(self.minted):012d}".translate(str.maketrans("0189", "abcd")) + "_" + "K" * 43
            self.minted.append(key)
            return 0, key + "\n", ""
        return 0, "ok\n", ""

    def stdin_of(self, cmd: list[str]) -> str:
        return self.docker.stdins[len(self.docker.calls) - 1].rstrip("\n")


def test_a_tenant_that_already_exists_is_kept_and_gets_a_new_key_and_password(
    docker: FakeDocker,
) -> None:
    accounts = FakeAccounts(docker)
    tenant = load.TENANTS[0]
    first = load.create_tenant(tenant)
    again = load.create_tenant(tenant)
    assert again["api_key"] != first["api_key"]
    assert again["password"] != first["password"]
    assert accounts.passwords[tenant.owner_email] == again["password"]  # set-password ran
    assert accounts.organizations == {tenant.slug}
    assert accounts.members == {(tenant.slug, tenant.owner_email)}


def test_a_refusal_that_is_not_about_existing_is_raised() -> None:
    docker = FakeDocker()
    docker.on(has("create-organization"), (1, "", "error: the database refused the change (ck_x)"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "run", docker)
        with pytest.raises(CliError, match="ck_x"):
            load.run_accounts_if_new("create-organization", "--name", "n", "--slug", "s")


def test_seed_resumes_after_a_failure_part_way_without_a_reset(
    docker: FakeDocker, tenants_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    accounts = FakeAccounts(docker)
    live_keys: set[str] = set()
    monkeypatch.setattr(load, "api_status", lambda path, key: 200 if key in live_keys else 401)
    accounts.fail_once = ("add-member", "owner-medium@load.doc-pilot.local")

    with pytest.raises(CliError, match=r"seeding load-medium failed.*load\.py seed. again"):
        load.seed(None)
    # The first tenant's key was stored before the second one failed.
    stored = load.read_tenants(tenants_file)
    assert list(stored) == ["load-large"]
    assert stat.S_IMODE(tenants_file.stat().st_mode) == 0o600
    live_keys.add(stored["load-large"]["api_key"])
    minted_before = len(accounts.minted)

    assert load.seed(None) == 0
    tenants = load.read_tenants(tenants_file)
    assert list(tenants) == ["load-large", "load-medium", "load-small"]
    assert tenants["load-large"]["api_key"] == stored["load-large"]["api_key"]  # kept, not re-minted
    assert len(accounts.minted) - minted_before == 2  # medium and small only
    assert accounts.organizations == {t.slug for t in load.TENANTS}
    assert {m for _, m in accounts.members} == {t.owner_email for t in load.TENANTS}
    shown = capsys.readouterr()
    keys = [t["api_key"] for t in tenants.values()]
    assert not any(key in shown.out + shown.err for key in keys)
    assert "kept load-large" in shown.out


def test_seed_recreates_a_tenant_whose_stored_key_is_rejected(
    docker: FakeDocker, tenants_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accounts = FakeAccounts(docker)
    monkeypatch.setattr(load, "api_status", lambda path, key: 200 if key in live else 401)
    live: set[str] = set()
    assert load.seed(None) == 0
    first = load.read_tenants(tenants_file)
    live.update(t["api_key"] for t in first.values() if t["organization"] != "load-small")
    assert load.seed(None) == 0
    again = load.read_tenants(tenants_file)
    assert again["load-large"]["api_key"] == first["load-large"]["api_key"]
    assert again["load-small"]["api_key"] != first["load-small"]["api_key"]
    assert len(accounts.minted) == 4


@pytest.fixture
def keyed_api() -> Iterator[str]:
    """A local API that accepts one key on GET /documents."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            ok = self.path == "/documents" and self.headers["Authorization"] == f"Bearer {SECRET_KEY}"
            self.send_response(200 if ok else 401)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_api_status_sends_the_key_as_a_bearer_token(
    keyed_api: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "API_URL", keyed_api)
    assert load.api_status("/documents", SECRET_KEY) == 200
    assert load.api_status("/documents", OTHER_KEYS[0]) == 401


def test_api_status_of_a_down_api_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load, "API_URL", "http://127.0.0.1:9")
    assert load.api_status("/documents", SECRET_KEY) == 0


def test_the_file_mode_survives_a_permissive_umask(tmp_path: Path) -> None:
    old = os.umask(0)
    try:
        load.write_private_json(tmp_path / "k.json", {})
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "k.json").stat().st_mode) == 0o600
