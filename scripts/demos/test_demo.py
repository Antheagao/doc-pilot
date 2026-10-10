"""Tests for the demo CLI's safety guards (run: python -m pytest ../scripts/demos -q)."""

from __future__ import annotations

import demo
import pytest


def config(volumes: dict, name: str = "doc-pilot-demo") -> dict:
    return {"name": name, "volumes": volumes}


DEMO_VOLUMES = {
    "db_data": {"name": "doc-pilot-demo_db_data"},
    "uploads": {"name": "doc-pilot-demo_uploads"},
}


def test_all_demo_names_pass() -> None:
    existing = {
        "doc-pilot-demo_db_data": "doc-pilot-demo",
        "doc-pilot-demo_uploads": "doc-pilot-demo",
        "doc-pilot_db_data": "doc-pilot",
    }
    assert demo.volume_problems(config(DEMO_VOLUMES), existing) == []


def test_volumes_not_created_yet_pass() -> None:
    assert demo.volume_problems(config(DEMO_VOLUMES), {}) == []


def test_dev_volume_name_refuses() -> None:
    volumes = {**DEMO_VOLUMES, "db_data": {"name": "doc-pilot_db_data"}}
    existing = {"doc-pilot_db_data": "doc-pilot"}
    problems = demo.volume_problems(config(volumes), existing)
    assert any("doc-pilot_db_data" in p for p in problems)


def test_external_volumes_are_ignored() -> None:
    volumes = {
        **DEMO_VOLUMES,
        "shared": {"name": "doc-pilot_db_data", "external": True},
    }
    assert demo.declared_volume_names(config(volumes)) == [
        "doc-pilot-demo_db_data",
        "doc-pilot-demo_uploads",
    ]
    assert demo.volume_problems(config(volumes), {}) == []


def test_declared_volume_with_foreign_label_refuses() -> None:
    existing = {"doc-pilot-demo_db_data": "doc-pilot"}
    assert demo.volume_problems(config(DEMO_VOLUMES), existing)


def test_labelled_but_undeclared_volume_refuses() -> None:
    existing = {"doc-pilot-demo_extra": "doc-pilot-demo"}
    assert demo.volume_problems(config(DEMO_VOLUMES), existing)


def test_wrong_project_name_refuses() -> None:
    assert demo.volume_problems(config(DEMO_VOLUMES, name="doc-pilot"), {})


def test_compose_env_drops_compose_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "tracing")
    assert "COMPOSE_PROFILES" not in demo.compose_env()


def test_worker_or_migrate_failure_is_reported() -> None:
    ok = {
        "migrate": {"State": "exited", "ExitCode": 0},
        "worker": {"State": "running", "Health": "healthy"},
    }
    demo.check_one_shots_and_worker(ok)
    with pytest.raises(demo.CliError, match="worker"):
        demo.check_one_shots_and_worker({**ok, "worker": {"State": "exited"}})
    with pytest.raises(demo.CliError, match="worker"):
        demo.check_one_shots_and_worker(
            {**ok, "worker": {"State": "running", "Health": "unhealthy"}}
        )
    with pytest.raises(demo.CliError, match="migrate"):
        demo.check_one_shots_and_worker(
            {**ok, "migrate": {"State": "exited", "ExitCode": 1}}
        )
