"""Tests for the shared compose helpers (run: python -m pytest ../scripts -q).

No Docker is needed: subprocess.run is replaced by a fake that records the
command lines it was given.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import compose_project
import pytest
from compose_project import CliError, ComposeProject

LOAD = ComposeProject("doc-pilot-load", ("docker-compose.yml", "docker-compose.load.yml"))


def config(volumes: dict, name: str = "doc-pilot-load") -> dict:
    return {"name": name, "volumes": volumes}


LOAD_VOLUMES = {
    "db_data": {"name": "doc-pilot-load_db_data"},
    "uploads": {"name": "doc-pilot-load_uploads"},
    "models": {"name": "doc-pilot-load_models"},
}


class FakeRun:
    """A stand-in for subprocess.run: answers with the given stdout (or the
    return code and stderr) and records every call."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.result = (stdout, returncode, stderr)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((cmd, kwargs))
        stdout, returncode, stderr = self.result
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(compose_project.subprocess, "run", fake)
    return fake


# --- the volume guard ---------------------------------------------------------


def test_the_projects_own_volumes_pass() -> None:
    existing = {name["name"]: "doc-pilot-load" for name in LOAD_VOLUMES.values()}
    existing["doc-pilot_db_data"] = "doc-pilot"
    assert LOAD.volume_problems(config(LOAD_VOLUMES), existing) == []


def test_volumes_not_created_yet_pass() -> None:
    assert LOAD.volume_problems(config(LOAD_VOLUMES), {}) == []


def test_the_dev_and_demo_volume_names_refuse() -> None:
    for foreign in ("doc-pilot_db_data", "doc-pilot-demo_db_data"):
        volumes = {**LOAD_VOLUMES, "db_data": {"name": foreign}}
        problems = LOAD.volume_problems(config(volumes), {})
        assert any(foreign in problem for problem in problems)


def test_external_volumes_are_not_declared() -> None:
    volumes = {**LOAD_VOLUMES, "shared": {"name": "doc-pilot_db_data", "external": True}}
    assert compose_project.declared_volume_names(config(volumes)) == [
        "doc-pilot-load_db_data",
        "doc-pilot-load_uploads",
        "doc-pilot-load_models",
    ]
    assert LOAD.volume_problems(config(volumes), {}) == []


def test_a_declared_volume_with_a_foreign_label_refuses() -> None:
    assert LOAD.volume_problems(config(LOAD_VOLUMES), {"doc-pilot-load_db_data": "doc-pilot"})


def test_a_labelled_but_undeclared_volume_refuses() -> None:
    assert LOAD.volume_problems(config(LOAD_VOLUMES), {"doc-pilot-load_extra": "doc-pilot-load"})


def test_a_config_of_another_project_refuses() -> None:
    for name in ("doc-pilot", "doc-pilot-demo", None):
        assert LOAD.volume_problems(config(LOAD_VOLUMES, name=name), {})


def test_the_guard_binds_to_the_project_it_was_built_with() -> None:
    demo = ComposeProject("doc-pilot-demo", ())
    assert demo.volume_problems(config(LOAD_VOLUMES), {})  # load names, demo project
    assert LOAD.volume_problems(config(LOAD_VOLUMES), {}) == []


def test_report_volumes_lists_them_and_returns_the_problems(
    capsys: pytest.CaptureFixture,
) -> None:
    assert LOAD.report_volumes(config(LOAD_VOLUMES), {"doc-pilot-load_db_data": "doc-pilot-load"}) == []
    out = capsys.readouterr()
    assert "doc-pilot-load_db_data  (exists)" in out.out
    assert "doc-pilot-load_uploads  (not created yet)" in out.out
    assert out.err == ""
    volumes = {**LOAD_VOLUMES, "db_data": {"name": "doc-pilot_db_data"}}
    assert LOAD.report_volumes(config(volumes), {})
    assert "Refusing to remove volumes" in capsys.readouterr().err


# --- running docker ----------------------------------------------------------


def test_compose_args_pin_the_project_and_files() -> None:
    assert LOAD.compose_args("ps", "--all") == [
        "docker",
        "compose",
        "-p",
        "doc-pilot-load",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.load.yml",
        "ps",
        "--all",
    ]


def test_compose_env_drops_compose_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "tracing")
    monkeypatch.setenv("KEEP_ME", "1")
    env = compose_project.compose_env()
    assert "COMPOSE_PROFILES" not in env
    assert env["KEEP_ME"] == "1"


def test_run_uses_the_repo_root_and_the_profile_free_environment(
    fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "tracing")
    assert LOAD.compose("up", "-d") == 0
    [(cmd, kwargs)] = fake_run.calls
    assert cmd[:4] == ["docker", "compose", "-p", "doc-pilot-load"]
    assert kwargs["cwd"] == compose_project.REPO_ROOT
    assert "COMPOSE_PROFILES" not in kwargs["env"]
    assert kwargs["input"] is None


def test_stdin_goes_to_the_command_not_the_command_line(fake_run: FakeRun) -> None:
    LOAD.docker_output(["docker", "exec", "-i", "c", "cat"], stdin="secret-password\n")
    [(cmd, kwargs)] = fake_run.calls
    assert kwargs["input"] == "secret-password\n"
    assert "secret-password" not in " ".join(cmd)


def test_a_missing_docker_is_a_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(compose_project.subprocess, "run", missing)
    with pytest.raises(CliError, match="docker was not found"):
        LOAD.compose("ps")


def test_a_failing_command_is_a_cli_error_with_its_stderr(fake_run: FakeRun) -> None:
    fake_run.result = ("", 1, "Cannot connect to the Docker daemon\n")
    with pytest.raises(CliError, match="Cannot connect to the Docker daemon") as raised:
        LOAD.docker_output(["docker", "exec", "c", "python", "-c", "hidden-argument"])
    assert "hidden-argument" not in str(raised.value)
    fake_run.result = ("", 3, "")
    with pytest.raises(CliError, match="exit code 3"):
        LOAD.docker_output(["docker", "ps"])


def test_the_resolved_config_is_parsed_and_can_include_every_profile(fake_run: FakeRun) -> None:
    fake_run.result = (json.dumps(config(LOAD_VOLUMES)), 0, "")
    assert LOAD.resolved_config()["name"] == "doc-pilot-load"
    assert "--profile" not in fake_run.calls[-1][0]
    LOAD.resolved_config(all_profiles=True)
    cmd = fake_run.calls[-1][0]
    assert cmd[cmd.index("--profile") + 1] == "*"
    assert cmd[-3:] == ["config", "--format", "json"]


def test_existing_volumes_are_read_with_their_project_labels(fake_run: FakeRun) -> None:
    fake_run.result = (
        "doc-pilot-load_db_data\tdoc-pilot-load\ncosgift_pgdata\tcosgift\nloose\t\n",
        0,
        "",
    )
    assert LOAD.existing_volume_projects() == {
        "doc-pilot-load_db_data": "doc-pilot-load",
        "cosgift_pgdata": "cosgift",
        "loose": "",
    }


# --- container records --------------------------------------------------------


def test_ps_output_is_lines_or_an_array() -> None:
    lines = '{"Service": "api"}\n{"Service": "db"}\n'
    assert compose_project.parse_ps_records(lines) == [{"Service": "api"}, {"Service": "db"}]
    assert compose_project.parse_ps_records('[{"Service": "api"}]') == [{"Service": "api"}]
    assert compose_project.parse_ps_records(" \n") == []


def test_container_records_ask_for_stopped_containers_only_when_told(fake_run: FakeRun) -> None:
    fake_run.result = ('{"Service": "api", "State": "running"}\n', 0, "")
    assert LOAD.container_records(include_stopped=False) == [{"Service": "api", "State": "running"}]
    assert "--all" not in fake_run.calls[-1][0]
    LOAD.container_records()
    assert "--all" in fake_run.calls[-1][0]


def test_service_states_map_a_service_to_its_record(fake_run: FakeRun) -> None:
    records = [
        '{"Service": "migrate", "State": "exited", "ExitCode": 0}',
        '{"Service": "worker", "State": "running", "Health": "healthy"}',
    ]
    fake_run.result = ("\n".join(records), 0, "")
    states = LOAD.service_states()
    assert states["migrate"]["ExitCode"] == 0
    assert states["worker"]["Health"] == "healthy"


def test_the_one_shot_and_worker_checks() -> None:
    ok = {
        "migrate": {"State": "exited", "ExitCode": 0},
        "worker": {"State": "running", "Health": "healthy"},
    }
    compose_project.check_one_shots_and_worker(ok)
    with pytest.raises(CliError, match="migrate"):
        compose_project.check_one_shots_and_worker({**ok, "migrate": {"State": "exited", "ExitCode": 1}})
    with pytest.raises(CliError, match="worker"):
        compose_project.check_one_shots_and_worker({**ok, "worker": {"State": "running"}})


def test_repo_root_holds_the_compose_files() -> None:
    root = compose_project.REPO_ROOT
    assert root == Path(__file__).resolve().parents[1]
    assert (root / "docker-compose.yml").is_file()
