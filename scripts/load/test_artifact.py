"""Tests for the load_v1 artifact: quantiles, statistics, the scenario digest,
capture, and validation (run: python -m pytest ../scripts -q). No Docker."""

from __future__ import annotations

import json
from pathlib import Path

import artifact
import pytest

# --- quantiles ---------------------------------------------------------------------


def test_quantiles_are_nearest_rank_against_a_hand_computed_list() -> None:
    # Ten values: rank = ceil(permille * 10 / 1000).
    values = [15.0, 20.0, 35.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 95.0]
    assert artifact.quantile(values, 10) == 15.0  # p1: rank ceil(0.1) = 1
    assert artifact.quantile(values, 250) == 35.0  # p25: rank ceil(2.5) = 3
    assert artifact.quantile(values, 500) == 50.0  # p50: rank 5
    assert artifact.quantile(values, 750) == 70.0  # p75: rank ceil(7.5) = 8
    assert artifact.quantile(values, 900) == 80.0  # p90: rank 9
    assert artifact.quantile(values, 950) == 95.0  # p95: rank ceil(9.5) = 10
    assert artifact.quantile(values, 999) == 95.0  # p99.9: rank 10


def test_p99_9_of_a_thousand_is_the_999th_not_the_maximum() -> None:
    """99.9 / 100 * 1000 is 999.0000000000001 in floating point, which a
    float ceil turns into rank 1000; the integer rank does not."""
    values = [float(n) for n in range(1, 1001)]
    assert artifact.quantile(values, 999) == 999.0
    assert artifact.quantile(values, 990) == 990.0
    assert artifact.quantile(values, 1000) == 1000.0


def test_a_single_value_is_every_quantile_and_none_is_an_error() -> None:
    assert artifact.quantile([7.0], 10) == 7.0
    with pytest.raises(ValueError, match="no samples"):
        artifact.quantile([], 500)


def test_summarize_reports_count_range_mean_and_every_quantile() -> None:
    summary = artifact.summarize([4.0, 1.0, 3.0, 2.0])
    assert summary["count"] == 4 and summary["min"] == 1.0 and summary["max"] == 4.0
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.0 and summary["p95"] == 4.0 and summary["p99.9"] == 4.0
    assert [label for label, _ in artifact.QUANTILES] == [
        "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99.9",
    ]
    assert artifact.summarize([]) is None


def test_median_and_spread_of_repetitions() -> None:
    assert artifact.median([3.0, 1.0, 2.0]) == 2.0
    assert artifact.median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert artifact.spread([9.0, 11.0, 10.0]) == {"n": 3, "median": 10.0, "min": 9.0, "max": 11.0}
    with pytest.raises(ValueError):
        artifact.spread([])


# --- samples ----------------------------------------------------------------------


def sample(name: str, start: float, ms: float, error: str | None = None, **extra: object) -> dict:
    return {"name": name, "start": start, "ms": ms, "ok": error is None, "error": error, **extra}


def test_a_killed_run_leaves_a_last_line_that_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(sample("a", 1, 2)) + "\n" + json.dumps(sample("b", 2, 3))[:20])
    assert [s["name"] for s in artifact.read_samples(path)] == ["a"]


def test_the_window_selects_by_start_so_slow_requests_stay() -> None:
    samples = [sample("a", 9.9, 5), sample("a", 10.0, 5000), sample("a", 20.0, 5), sample("a", 20.1, 5)]
    kept = artifact.in_window(samples, 10.0, 20.0)
    assert [s["start"] for s in kept] == [10.0, 20.0]  # the 5 s request that ends at 15 s counts


def test_endpoint_stats_time_only_the_successes_and_count_the_errors() -> None:
    samples = [sample("search", t, ms) for t, ms in enumerate([10.0, 20.0, 30.0, 40.0])]
    samples.append(sample("search", 5, 1.0, error="http_500"))
    samples.append(sample("search", 6, 1.0, error="http_500"))
    samples.append(sample("ask", 7, 100.0, error="no_citation"))
    stats = artifact.endpoint_stats(samples, seconds=10)
    assert stats["search"]["count"] == 6 and stats["search"]["errors"] == 2
    assert stats["search"]["error_rate"] == pytest.approx(2 / 6)
    assert stats["search"]["errors_by_code"] == {"http_500": 2}
    assert stats["search"]["rps"] == pytest.approx(0.6)
    assert stats["search"]["latency_ms"]["count"] == 4
    assert stats["search"]["latency_ms"]["p50"] == 20.0  # the fast errors do not flatter it
    assert stats["ask"]["latency_ms"] is None and stats["ask"]["errors_by_code"] == {"no_citation": 1}


def test_a_stream_also_gets_its_first_event_and_answer_quantiles() -> None:
    samples = [
        sample("ask_stream", 1, 90.0, first_event_ms=10.0, answer_ms=80.0),
        sample("ask_stream", 2, 110.0, first_event_ms=20.0, answer_ms=100.0),
        sample("ask_stream", 3, 5.0, error="stream_error", first_event_ms=1.0),
    ]
    entry = artifact.endpoint_stats(samples, 3)["ask_stream"]
    assert entry["first_event_ms"]["count"] == 2 and entry["first_event_ms"]["p50"] == 10.0
    assert entry["answer_ms"]["p95"] == 100.0


# --- the scenario digest -----------------------------------------------------------


def test_the_digest_ignores_key_order_and_follows_every_value() -> None:
    one = {"mix": {"search": 70, "ask": 15}, "queries": ["a", "b"]}
    reordered = {"queries": ["a", "b"], "mix": {"ask": 15, "search": 70}}
    assert artifact.scenario_digest(one) == artifact.scenario_digest(reordered)
    assert len(artifact.scenario_digest(one)) == 64
    assert artifact.scenario_digest({**one, "queries": ["a", "c"]}) != artifact.scenario_digest(one)
    assert artifact.scenario_digest({**one, "mix": {"search": 71, "ask": 15}}) != artifact.scenario_digest(one)


def test_a_json_round_trip_keeps_the_digest() -> None:
    description = {"plan": {"workers": (1, 2, 4), "scale": 0.1}, "n": 1}
    assert artifact.scenario_digest(json.loads(json.dumps(description))) == artifact.scenario_digest(description)


# --- capture -----------------------------------------------------------------------


def fake_runner(answers: dict[str, str]):
    def run(cmd: list[str]) -> str:
        key = " ".join(cmd)
        for fragment, answer in answers.items():
            if fragment in key:
                return answer
        raise RuntimeError(f"no answer for {key}")

    return run


def test_a_mac_is_read_from_sysctl_with_its_two_core_kinds() -> None:
    run = fake_runner(
        {
            "machdep.cpu.brand_string": "Apple M1\n",
            "hw.ncpu": "8\n",
            "hw.memsize": "17179869184\n",
            "perflevel0": "4\n",
            "perflevel1": "4\n",
            "sw_vers": "26.0\n",
        }
    )
    machine = artifact.capture_machine(run, system="Darwin")
    assert machine["cpu"] == "Apple M1" and machine["cores"] == 8
    assert machine["cores_performance"] == 4 and machine["cores_efficiency"] == 4
    assert machine["memory_bytes"] == 17179869184 and machine["os"] == "macOS 26.0"


def test_a_linux_host_is_read_from_proc_and_nproc(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "/proc/cpuinfo": "processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core Processor\n\nprocessor\t: 1\nmodel name\t: AMD EPYC 7763 64-Core Processor\n",
        "/proc/meminfo": "MemTotal:       16379432 kB\nMemFree:         1000 kB\n",
        "/etc/os-release": 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.2 LTS"\n',
    }
    monkeypatch.setattr(artifact.Path, "read_text", lambda self, **kwargs: files[str(self)])
    machine = artifact.capture_machine(fake_runner({"nproc": "4\n"}), system="Linux")
    assert machine["cpu"] == "AMD EPYC 7763 64-Core Processor" and machine["cores"] == 4
    assert machine["memory_bytes"] == 16379432 * 1024 and machine["os"] == "Ubuntu 24.04.2 LTS"
    document = good_artifact()
    document["machine"] = machine
    assert artifact.validate(document) == []


def test_an_unknown_system_records_no_cores_which_validation_refuses() -> None:
    machine = artifact.capture_machine(fake_runner({}), system="Plan9")
    assert machine["cores"] == 0
    document = good_artifact()
    document["machine"] = machine
    assert any("machine" in problem for problem in artifact.validate(document))


def test_docker_is_its_version_cpus_and_memory() -> None:
    info = {"ServerVersion": "29.4.0", "NCPU": 8, "MemTotal": 8321994752, "OperatingSystem": "Docker Desktop", "Architecture": "aarch64", "Secret": "x"}
    docker = artifact.capture_docker(fake_runner({"docker info": json.dumps(info)}))
    assert docker == {
        "version": "29.4.0",
        "ncpu": 8,
        "memory_bytes": 8321994752,
        "operating_system": "Docker Desktop",
        "architecture": "aarch64",
    }


def test_git_is_the_commit_and_whether_tracked_files_differ() -> None:
    seen: list[str] = []

    def run(cmd: list[str]) -> str:
        seen.append(" ".join(cmd))
        return "abc123\n" if "rev-parse" in cmd else " M scripts/load/load.py\n"

    assert artifact.capture_git(run) == {"commit": "abc123", "dirty": True}
    assert any("--untracked-files=no" in call for call in seen)
    assert artifact.capture_git(lambda cmd: "abc\n" if "rev-parse" in cmd else "") == {
        "commit": "abc",
        "dirty": False,
    }


def test_without_git_the_tree_is_unknown_and_dirty() -> None:
    def run(cmd: list[str]) -> str:
        raise RuntimeError("no git")

    assert artifact.capture_git(run) == {"commit": "unknown", "dirty": True}


def test_postgres_is_the_version_pgvector_and_the_settings_that_matter() -> None:
    answers = {
        "pg_settings": "max_connections|100|\nshared_buffers|16384|8kB\nshared_preload_libraries|pg_stat_statements|\n",
        "version()": "PostgreSQL 16.15 on aarch64\n",
        "extversion": "0.8.7\n",
    }

    def psql(sql: str) -> str:
        return next(answer for fragment, answer in answers.items() if fragment in sql)

    postgres = artifact.capture_postgres(psql)
    assert postgres["version"] == "PostgreSQL 16.15 on aarch64" and postgres["pgvector"] == "0.8.7"
    assert postgres["max_connections"] == 100
    assert postgres["settings"]["shared_buffers"] == "16384" + "8kB"
    assert postgres["settings"]["shared_preload_libraries"] == "pg_stat_statements"


def config(**api: str) -> dict:
    env = {
        "ANTHROPIC_API_KEY": "",
        "MODEL_PROVIDER": "scripted",
        "SCRIPTED_MODEL_PROFILE": "replay",
        "SCRIPTED_MODEL_LATENCY_SCALE": "0.1",
        "SCRIPTED_AGENT_STEP_MS": "3000",
        "SCRIPTED_JUDGE_MS": "4000",
        "EMBEDDING_BACKEND": "fastembed",
        "ASK_RATE_LIMIT_PER_MINUTE": "0",
        "DB_POOL_SIZE": "7",
        "EXTRACTION_MODEL": "scripted",
        "TRANSCRIPTION_MODEL": "scripted",
        "AGENT_MODEL": "scripted",
        "JUDGE_MODEL": "scripted",
        "DATABASE_URL": "postgresql+asyncpg://docpilot_runtime:secret-password@db/x",
        **api,
    }
    return {"services": {"api": {"environment": env}, "worker": {"environment": {"ANTHROPIC_API_KEY": ""}}}}


def test_the_stack_record_names_pools_workers_and_overrides_but_no_secret() -> None:
    stack = artifact.capture_stack(config(), ["uvicorn", "app.main:app", "--workers", "3"])
    assert stack["api_pool"] == {"size": 7, "max_overflow": 10, "timeout_seconds": 30.0}
    assert stack["uvicorn_workers"] == 3 and stack["embedder"] == "fastembed"
    assert stack["anthropic_key_blank"] is True
    assert stack["overrides"]["api"]["ASK_RATE_LIMIT_PER_MINUTE"] == "0"
    assert "secret-password" not in json.dumps(stack) and "DATABASE_URL" not in json.dumps(stack)
    assert artifact.capture_stack(config(), ["uvicorn"])["uvicorn_workers"] == 1


def test_a_key_in_the_environment_is_recorded_only_as_not_blank() -> None:
    stack = artifact.capture_stack(config(ANTHROPIC_API_KEY="sk-ant-real"), ["uvicorn"])
    assert stack["anthropic_key_blank"] is False
    assert "sk-ant-real" not in json.dumps(stack)
    assert any("blank Anthropic key" in problem for problem in artifact.validate({**good_artifact(), "stack": stack}))


def test_the_model_record_is_the_profile_and_scale_the_api_runs() -> None:
    model = artifact.capture_model(config())
    assert model["provider"] == "scripted" and model["profile"] == "replay"
    assert model["latency_scale"] == 0.1 and model["agent_step_ms"] == 3000.0
    assert artifact.capture_model({"services": {"api": {"environment": {}}}})["latency_scale"] is None


# --- validation ---------------------------------------------------------------------


def good_artifact(**changes: object) -> dict:
    scenario = {"workload": {"mix": {"search": 70}}, "plan": {"preset": "full"}}
    document = {
        "schema": "load_v1",
        "run_id": "20261025T120000Z",
        "label": "baseline",
        "preset": "full",
        "started_at": "2026-10-25T12:00:00.000000Z",
        "ended_at": "2026-10-25T13:00:00.000000Z",
        "command": ["python", "scripts/load/load.py", "run", "--label", "baseline"],
        "git": {"commit": "abc123", "dirty": False},
        "machine": {"cpu": "Apple M1", "cores": 8, "memory_bytes": 17179869184, "os": "macOS 26.0"},
        "docker": {"version": "29.4.0", "ncpu": 8, "memory_bytes": 8321994752},
        "postgres": {"version": "PostgreSQL 16.15", "max_connections": 100},
        "stack": artifact.capture_stack(config(), ["uvicorn"]),
        "model": {"provider": "scripted", "profile": "zero", "latency_scale": 1.0},
        "scenario": {"digest": artifact.scenario_digest(scenario), "description": scenario},
        "tenants": {},
        "generator": {"loop": "closed"},
        "methodology": artifact.METHODOLOGY,
        "clock_offset_s": 0.02,
        "phases": [{"name": "scaling", "repetition": 1, "valid": True, "invalid_reasons": []}],
        "summary": {},
        "spend_usd": 0.0,
        "models_seen": ["scripted"],
    }
    return {**document, **changes}


def test_a_good_artifact_has_no_problems_even_publishable() -> None:
    assert artifact.validate(good_artifact()) == []
    assert artifact.validate(good_artifact(), publishable=True) == []


def test_nonzero_spend_is_refused() -> None:
    assert any("spend" in p for p in artifact.validate(good_artifact(spend_usd=0.000001)))
    assert any("spend" in p for p in artifact.validate(good_artifact(spend_usd="0")))
    assert any("spend" in p for p in artifact.validate(good_artifact(spend_usd=None)))
    assert any("spend" in p for p in artifact.validate({k: v for k, v in good_artifact().items() if k != "spend_usd"}))


def test_a_model_other_than_the_scripted_one_is_refused_but_none_is_fine() -> None:
    assert any("models seen" in p for p in artifact.validate(good_artifact(models_seen=["scripted", "claude-haiku-4-5"])))
    assert artifact.validate(good_artifact(models_seen=[])) == []  # an ingest with no worker records no model
    document = good_artifact(model={"provider": "anthropic", "profile": "zero", "latency_scale": 1.0})
    assert any("provider" in p for p in artifact.validate(document))


def test_a_missing_machine_record_is_refused() -> None:
    assert any("machine" in p for p in artifact.validate({k: v for k, v in good_artifact().items() if k != "machine"}))
    assert any("machine" in p for p in artifact.validate(good_artifact(machine={"cpu": "Apple M1", "os": "macOS"})))
    assert any("machine" in p for p in artifact.validate(good_artifact(machine={"cores": 8, "memory_bytes": 1})))
    assert any("Docker" in p for p in artifact.validate(good_artifact(docker={"version": "29"})))
    assert any("Docker" in p for p in artifact.validate(good_artifact(docker={})))


def test_a_dirty_tree_is_a_problem_only_for_a_publishable_artifact() -> None:
    dirty = good_artifact(git={"commit": "abc123", "dirty": True})
    assert artifact.validate(dirty) == []
    assert any("dirty" in p for p in artifact.validate(dirty, publishable=True))
    assert any("git" in p for p in artifact.validate(good_artifact(git={"commit": "abc"})))
    unknown = good_artifact(git={"commit": "unknown", "dirty": False})
    assert any("unknown" in p for p in artifact.validate(unknown, publishable=True))


def test_only_a_full_plan_is_publishable() -> None:
    for preset in ("quick", "smoke", "custom"):
        document = good_artifact(preset=preset)
        assert artifact.validate(document) == []
        assert any("not publishable" in p for p in artifact.validate(document, publishable=True))
    assert any("preset" in p for p in artifact.validate(good_artifact(preset="huge")))


def test_an_invalid_phase_is_named_and_blocks_publishing() -> None:
    phase = {"name": "ramp", "repetition": 2, "valid": False, "invalid_reasons": ["the generator used 91% of a core"]}
    document = good_artifact(phases=[phase])
    assert artifact.validate(document) == []
    [problem] = artifact.validate(document, publishable=True)  # no knee found: not the allowed kind of invalid
    assert "ramp repetition 2" in problem and "91%" in problem


def ramp_step(repetition: int, users: int, *, valid: bool = True, knee: int | None = 4) -> dict:
    return {
        "name": "ramp", "repetition": repetition, "params": {"users": users}, "valid": valid,
        "invalid_reasons": [] if valid else ["the generator's p95 CPU was 85% of a core"],
        "results": {"knee_users": knee},
    }


def test_a_ramp_may_end_on_an_invalid_step_once_it_found_its_knee() -> None:
    """A ramp stops at the step that fails or saturates the generator, so that last step
    may be invalid and the artifact still publishable."""
    steps = [ramp_step(1, 2), ramp_step(1, 4), ramp_step(1, 8, valid=False)]
    assert artifact.validate(good_artifact(phases=steps), publishable=True) == []


def test_an_invalid_step_in_the_middle_of_a_ramp_blocks_publishing() -> None:
    steps = [ramp_step(1, 2), ramp_step(1, 4, valid=False), ramp_step(1, 8)]
    [problem] = artifact.validate(good_artifact(phases=steps), publishable=True)
    assert "ramp repetition 1 is invalid" in problem


def test_the_exception_is_per_repetition_and_needs_a_knee() -> None:
    first = [ramp_step(1, 2), ramp_step(1, 4, valid=False)]  # not the last of repetition 1
    second = [ramp_step(2, 2), ramp_step(2, 4, valid=False)]
    problems = artifact.validate(good_artifact(phases=[*first, ramp_step(1, 8), *second]), publishable=True)
    assert len(problems) == 1 and "repetition 1" in problems[0]
    no_knee = [ramp_step(1, 2, knee=None, valid=False)]
    assert len(artifact.validate(good_artifact(phases=no_knee), publishable=True)) == 1


def test_invalid_ingest_and_scaling_phases_always_block_publishing() -> None:
    for name in ("ingest", "scaling"):
        phase = {"name": name, "repetition": 1, "valid": False, "invalid_reasons": ["x"], "results": {"knee_users": 4}}
        assert len(artifact.validate(good_artifact(phases=[phase]), publishable=True)) == 1


def test_the_artifact_carries_its_methodology_and_loop_kind() -> None:
    document = good_artifact()
    assert "Closed-loop" in document["methodology"] and "no think time" in document["methodology"]
    assert "nearest-rank" in document["methodology"] and "not an open-loop" in document["methodology"]
    assert document["generator"]["loop"] == "closed"


def test_a_clock_more_than_a_second_off_is_refused() -> None:
    assert artifact.validate(good_artifact(clock_offset_s=-0.9)) == []
    for bad in (1.5, -1.5, None, "0"):
        assert any("clock_offset_s" in p for p in artifact.validate(good_artifact(clock_offset_s=bad)))


def test_a_digest_that_does_not_match_its_description_is_refused() -> None:
    document = good_artifact()
    document["scenario"]["description"]["workload"]["mix"]["search"] = 71
    assert any("digest" in p for p in artifact.validate(document))


def test_phases_are_required() -> None:
    assert any("no phases" in p for p in artifact.validate(good_artifact(phases=[])))
    assert any("name or validity" in p for p in artifact.validate(good_artifact(phases=[{"name": "x"}])))


def test_a_wrong_schema_is_refused() -> None:
    assert any("schema" in p for p in artifact.validate(good_artifact(schema="load_v0")))


def test_the_builder_attaches_the_validation_and_the_digest() -> None:
    base = good_artifact()
    built = artifact.build_artifact(
        run_id=base["run_id"],
        label=base["label"],
        preset="quick",
        command=base["command"],
        git=base["git"],
        machine=base["machine"],
        docker=base["docker"],
        postgres=base["postgres"],
        stack=base["stack"],
        model=base["model"],
        scenario=base["scenario"]["description"],
        tenants={},
        generator={},
        phases=base["phases"],
        summary={},
        clock_offset_s=0.02,
        spend_usd=0.0,
        models_seen=["scripted"],
        started_at=base["started_at"],
        ended_at=base["ended_at"],
    )
    assert built["scenario"]["digest"] == base["scenario"]["digest"]
    assert built["validation"]["problems"] == []
    assert built["validation"]["publishable_problems"] == ["preset 'quick' is not publishable"]
    assert built["schema"] == "load_v1"


# --- writing -----------------------------------------------------------------------


def test_the_file_name_is_the_run_and_a_safe_label() -> None:
    assert artifact.artifact_filename("20261025T120000Z", "Dev Check!") == "20261025T120000Z_dev-check.json"
    with pytest.raises(ValueError):
        artifact.artifact_filename("20261025T120000Z", "!!!")
    # A path in a label is reduced to a slug, never followed.
    assert artifact.artifact_filename("20261025T120000Z", "../../etc") == "20261025T120000Z_etc.json"


def test_the_written_file_round_trips_and_keeps_one_phase_per_line(tmp_path: Path) -> None:
    document = good_artifact(phases=[{"name": "a", "valid": True, "series": [[1, {"x": 1}]] * 5}, {"name": "b", "valid": True}])
    path = artifact.write_artifact(document, tmp_path / "nested")
    assert path.name == "20261025T120000Z_baseline.json"
    assert json.loads(path.read_text()) == json.loads(json.dumps(document))
    lines = path.read_text().splitlines()
    assert sum(1 for line in lines if line.startswith('  {"name":')) == 2
