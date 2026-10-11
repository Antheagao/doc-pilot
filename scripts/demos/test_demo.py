"""Tests for the demo CLI's safety guards (run: python -m pytest ../scripts/demos -q)."""

from __future__ import annotations

import argparse
from pathlib import Path

import demo
import encode
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


def test_seed_runs_the_one_shot_in_the_seed_profile() -> None:
    command = demo.seed_command(reset=False)
    assert command[:2] == ["--profile", "seed"]
    assert command[2:4] == ["run", "--rm"]
    assert command[-1] == "seed"
    assert "--reset" not in command


def test_seed_reset_passes_reset_to_the_script() -> None:
    assert demo.seed_command(reset=True)[-3:] == [
        "python",
        "scripts/seed_demo.py",
        "--reset",
    ]


def test_seed_parser_reads_reset() -> None:
    parser = demo.build_parser()
    assert parser.parse_args(["seed"]).reset is False
    assert parser.parse_args(["seed", "--reset"]).reset is True


def test_record_all_skips_the_smoke_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "SCENARIOS", {"smoke": None, "search-hybrid": None})
    assert demo.resolve_slugs(["all"]) == ["search-hybrid"]


def test_record_all_puts_the_dashboard_after_the_live_takes() -> None:
    # The dashboard shows today's real runs, which the live takes create.
    slugs = demo.resolve_slugs(["all"])
    assert slugs.index("monitoring-dashboard") > slugs.index("document-chat")
    assert "monitoring-dashboard" not in demo.LIVE_SLUGS


def test_record_all_with_only_smoke_registered_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo, "SCENARIOS", {"smoke": None})
    with pytest.raises(demo.CliError, match="no feature scenarios"):
        demo.resolve_slugs(["all"])


def test_record_refuses_an_unknown_slug() -> None:
    with pytest.raises(demo.CliError, match="unknown scenario nope"):
        demo.resolve_slugs(["smoke", "nope"])
    assert demo.resolve_slugs(["smoke"]) == ["smoke"]


def test_smoke_never_reaches_the_committed_gifs() -> None:
    assert demo.gif_destination("smoke") is None
    assert demo.gif_destination("search-hybrid") == (
        demo.REPO_ROOT / "screenshots" / "features" / "search-hybrid.gif"
    )


def test_a_ready_stack_has_no_problems() -> None:
    assert demo.readiness_problems(True, True, 25) == []


def test_a_down_or_empty_stack_says_how_to_fix_it() -> None:
    problems = demo.readiness_problems(False, False, 0)
    assert len(problems) == 2
    assert all("demo.py stack up" in p for p in problems)
    (empty,) = demo.readiness_problems(True, True, 0)
    assert "demo.py seed --reset" in empty


def test_record_parser_takes_slugs() -> None:
    parser = demo.build_parser()
    assert parser.parse_args(["record", "smoke", "x"]).slugs == ["smoke", "x"]
    assert parser.parse_args(["record", "all"]).slugs == ["all"]


def feature_gifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, infos: dict[str, encode.MediaInfo]
) -> None:
    for name in infos:
        (tmp_path / name).write_bytes(b"gif")
    monkeypatch.setattr(demo, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(encode, "require_tools", lambda: None)
    monkeypatch.setattr(encode, "probe", lambda path: infos[path.name])


def test_check_passes_gifs_within_the_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_gifs(tmp_path, monkeypatch, {"a.gif": encode.MediaInfo(9.0, 960, 800_000)})
    assert demo.check(argparse.Namespace()) == 0


def test_check_fails_when_any_gif_is_over_a_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    feature_gifs(
        tmp_path,
        monkeypatch,
        {
            "ok.gif": encode.MediaInfo(9.0, 960, 800_000),
            "big.gif": encode.MediaInfo(16.0, 960, 1_200_000),
        },
    )
    assert demo.check(argparse.Namespace()) == 1
    err = capsys.readouterr().err
    assert "big.gif" in err and "ok.gif" not in err


def test_check_with_no_gifs_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_gifs(tmp_path, monkeypatch, {})
    assert demo.check(argparse.Namespace()) == 0


def test_a_keyed_capped_budget_with_room_has_no_live_problems() -> None:
    assert demo.live_problems(True, 1.5, 0.0) == []
    assert demo.live_problems(True, 1.0, 0.9) == []


def test_a_live_take_needs_the_key() -> None:
    (problem,) = demo.live_problems(False, 1.5, 0.0)
    assert "ANTHROPIC_API_KEY" in problem


def test_a_live_take_refuses_a_budget_over_the_cap_or_off() -> None:
    (raised,) = demo.live_problems(True, 5.0, 0.0)
    assert "$5.00" in raised
    (off,) = demo.live_problems(True, None, 0.0)
    assert "off" in off


def test_a_live_take_refuses_once_the_budget_is_spent() -> None:
    (problem,) = demo.live_problems(True, 1.5, 1.5)
    assert "429" in problem


def test_only_the_model_scenarios_are_live() -> None:
    assert demo.LIVE_SLUGS == {"ask-agent", "document-chat"}
    assert demo.LIVE_SLUGS <= set(demo.SCENARIOS)
