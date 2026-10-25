"""Tests for the README Performance report: the tables from a fixture artifact,
the charts, the refusals, and the sync of the README and the committed charts
with the committed artifacts (run: python -m pytest ../scripts -q). No Docker."""

from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import artifact
import load
import pytest
import report

REPO_ROOT = Path(__file__).resolve().parents[2]
SVG = "{http://www.w3.org/2000/svg}"

# --- fixture artifacts -------------------------------------------------------------


def _latency(p95: float) -> dict:
    """Quantiles of a request class whose p95 is `p95` milliseconds."""
    return {"count": 100, "min": p95 / 10, "mean": p95 / 3, "max": p95 * 2, "p50": p95 / 3, "p95": p95, "p99": p95 * 1.5}


def ingest_phase(repetition: int, users: int, rate: float) -> dict:
    return {
        "name": "ingest",
        "repetition": repetition,
        "params": {"users": users, "documents": 100, "step": 1},
        "valid": True,
        "invalid_reasons": [],
        "results": {"users": users, "uploads_per_s": rate},
        "endpoints": {"upload": {"count": 100, "errors": 0, "error_rate": 0.0, "rps": rate, "latency_ms": _latency(40.0 / rate * 10)}},
    }


def scaling_phase(repetition: int, workers: int, documents_per_s: float, *, valid: bool = True) -> dict:
    return {
        "name": "scaling",
        "repetition": repetition,
        "params": {"workers": workers, "backlog_documents": 300},
        "valid": valid,
        "invalid_reasons": [] if valid else ["a job failed"],
        "results": {
            "workers": workers,
            "jobs_per_s": documents_per_s * 2,
            "documents_per_s": documents_per_s,
            "jobs_counted": "extract+index",
        },
        "endpoints": {},
    }


def ramp_steps(repetition: int, search_p95: dict[int, float], *, slo: float = 1000.0) -> list[dict]:
    """The steps of a read ramp: `search_p95` by users. The ramp stops at the
    first step over the SLO, and its knee is the step before."""
    knee = None
    records = []
    for users, p95 in search_p95.items():
        ok = p95 <= slo
        endpoints = {
            "search": {"count": 1000, "errors": 0, "error_rate": 0.0, "rps": users * 5.0, "latency_ms": _latency(p95)},
            "ask": {"count": 100, "errors": 0, "error_rate": 0.0, "rps": users * 1.0, "latency_ms": _latency(p95 * 2)},
            "ask_stream": {
                "count": 50,
                "errors": 0,
                "error_rate": 0.0,
                "rps": users * 0.5,
                "latency_ms": _latency(p95 * 2),
                "first_event_ms": _latency(p95 / 2),
                "answer_ms": _latency(p95 * 2),
            },
            "documents": {"count": 50, "errors": 0, "error_rate": 0.0, "rps": users * 0.5, "latency_ms": _latency(p95 / 4)},
        }
        records.append(
            {
                "name": "ramp",
                "repetition": repetition,
                "params": {"users": users, "slo_search_p95_ms": slo, "max_error_rate": 0.01},
                "valid": True,
                "invalid_reasons": [],
                "results": {
                    "verdict": {"ok": ok},
                    "tenants": {"load-large": {"documents": 2000, "chunks": 6000}, "load-small": {"documents": 30, "chunks": 90}},
                },
                "endpoints": endpoints,
            }
        )
        if not ok:
            break
        knee = users
    for record in records:
        record["results"]["knee_users"] = knee
    return records


def make_artifact(
    *,
    label: str = "baseline",
    run_id: str = "20261101T100000Z",
    profile: str = "zero",
    scale: float = 1.0,
    scaling: dict[int, list[float]] | None = None,
    ingest: dict[int, list[float]] | None = None,
    ramp: list[dict[int, float]] | None = None,
    dirty: bool = False,
    mutate=None,
) -> dict:
    """A small publishable artifact: `scaling` maps workers to the documents/s
    of each repetition, `ingest` users to uploads/s, `ramp` is one search-p95
    mapping per repetition."""
    scaling = {1: [6.0, 6.2], 2: [11.0, 12.0], 4: [20.0, 21.0]} if scaling is None else scaling
    ingest = {2: [50.0, 54.0]} if ingest is None else ingest
    ramp = [{2: 40.0, 4: 80.0, 8: 300.0, 16: 1500.0}, {2: 42.0, 4: 90.0, 8: 500.0, 16: 1400.0}] if ramp is None else ramp
    records = []
    for users, rates in ingest.items():
        records += [ingest_phase(rep, users, rate) for rep, rate in enumerate(rates, start=1)]
    for workers, rates in scaling.items():
        records += [scaling_phase(rep, workers, rate) for rep, rate in enumerate(rates, start=1)]
    for rep, steps in enumerate(ramp, start=1):
        records += ramp_steps(rep, steps)
    scenario = {"workload": {"version": 1}, "plan": {"preset": "full"}}
    document = artifact.build_artifact(
        run_id=run_id,
        label=label,
        preset="full",
        command=["python", "scripts/load/load.py", "run", "--profile", profile, "--scale", str(scale), "--label", label],
        git={"commit": "0123456789abcdef0123456789abcdef01234567", "dirty": dirty},
        machine={
            "system": "Darwin", "arch": "arm64", "cpu": "Apple M1", "cores": 8, "memory_bytes": 17179869184,
            "os": "macOS 27.0.1", "cores_performance": 4, "cores_efficiency": 4,
        },
        docker={
            "version": "29.4.0", "ncpu": 8, "memory_bytes": 8321994752,
            "operating_system": "Docker Desktop", "architecture": "aarch64",
        },
        postgres={"version": "PostgreSQL 16.4 (Debian 16.4-1) on aarch64-unknown-linux-gnu", "pgvector": "0.8.0", "max_connections": 100, "settings": {}},
        stack={
            "api_pool": {"size": 5, "max_overflow": 10, "timeout_seconds": 30.0},
            "worker_pool": {"size": 5, "max_overflow": 10, "timeout_seconds": 30.0},
            "uvicorn_workers": 1,
            "embedder": "fastembed",
            "anthropic_key_blank": True,
            "overrides": {"api": {"ASK_JUDGE_SAMPLE_RATE": "0.25"}, "worker": {"WORKER_RECLAIM_AFTER_SECONDS": "600"}},
        },
        model={
            "provider": "scripted", "profile": profile, "latency_scale": scale, "agent_step_ms": 3000.0,
            "judge_ms": 4000.0, "models": {"EXTRACTION_MODEL": "scripted"},
        },
        scenario=scenario,
        tenants={},
        generator={"tool": "locust"},
        phases=records,
        summary={},
        clock_offset_s=0.05,
        spend_usd=0.0,
        models_seen=["scripted"],
        started_at="2026-11-01T10:00:00.000000Z",
        ended_at="2026-11-01T11:00:00.000000Z",
    )
    if mutate:
        mutate(document)
        document["validation"] = {
            "problems": artifact.validate(document),
            "publishable_problems": artifact.validate(document, publishable=True),
        }
    return document


def write_runs(directory: Path, *documents: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for document in documents:
        (directory / artifact.artifact_filename(document["run_id"], document["label"])).write_text(
            artifact.dumps(document), encoding="utf-8"
        )


def with_scenario(document: dict, description: dict) -> None:
    document["scenario"] = {"digest": artifact.scenario_digest(description), "description": description}


def after_of(**changes) -> dict:
    """An after run that matches `make_artifact()` except for `changes`."""
    return make_artifact(label="after-pool-fix", run_id="20261102T100000Z", **changes)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    write_runs(tmp_path / "results", make_artifact(), after_of(scaling={1: [6.1, 6.3], 2: [20.0, 21.0], 4: [38.0, 40.0]}))
    return tmp_path / "results"


# --- the tables --------------------------------------------------------------------

INGEST_TABLE = """| Concurrent users | Uploads/s (median) | Range | Upload p50 (ms) | Upload p95 (ms) | Runs |
| --- | --- | --- | --- | --- | --- |
| 2 | 52.0 | 50.0 to 54.0 | 2.57 | 7.70 | 2 |"""

SCALING_TABLE = """| Workers | Documents/s (median) | Range | Jobs (extract+index)/s (median) | Range | Speedup vs 1 worker | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.10 | 6.00 to 6.20 | 12.2 | 12.0 to 12.4 | 1.00x | 2 |
| 2 | 11.5 | 11.0 to 12.0 | 23.0 | 22.0 to 24.0 | 1.89x | 2 |
| 4 | 20.5 | 20.0 to 21.0 | 41.0 | 40.0 to 42.0 | 3.36x | 2 |"""

LATENCY_TABLE = """| Step | Users | Endpoint | p50 (ms) | p95 (ms, range) | p99 (ms) | Requests/s | Errors | Runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| At the knee | 8 | GET /search | 133 | 400 (300 to 500) | 600 | 40.0 | 0.0% | 2 |
| At the knee | 8 | POST /ask | 267 | 800 (600 to 1,000) | 1,200 | 8.00 | 0.0% | 2 |
| At the knee | 8 | POST /ask/stream, first event | 66.7 | 200 (150 to 250) | 300 | 4.00 | 0.0% | 2 |
| At the knee | 8 | POST /ask/stream, answer | 267 | 800 (600 to 1,000) | 1,200 | 4.00 | 0.0% | 2 |
| At the knee | 8 | GET /documents | 33.3 | 100 (75.0 to 125) | 150 | 4.00 | 0.0% | 2 |
| One step past | 16 | GET /search | 483 | 1,450 (1,400 to 1,500) | 2,175 | 80.0 | 0.0% | 2 |
| One step past | 16 | POST /ask | 967 | 2,900 (2,800 to 3,000) | 4,350 | 16.0 | 0.0% | 2 |
| One step past | 16 | POST /ask/stream, first event | 242 | 725 (700 to 750) | 1,088 | 8.00 | 0.0% | 2 |
| One step past | 16 | POST /ask/stream, answer | 967 | 2,900 (2,800 to 3,000) | 4,350 | 8.00 | 0.0% | 2 |
| One step past | 16 | GET /documents | 121 | 362 (350 to 375) | 544 | 8.00 | 0.0% | 2 |"""

COMPARISON_TABLE = """| Measure | Before (median, range) | After (median, range) | Change | After outside the before range |
| --- | --- | --- | --- | --- |
| Documents/s at 1 worker | 6.10 (6.00 to 6.20) | 6.20 (6.10 to 6.30) | +1.6% | no |
| Documents/s at 2 workers | 11.5 (11.0 to 12.0) | 20.5 (20.0 to 21.0) | +78.3% | yes |
| Documents/s at 4 workers | 20.5 (20.0 to 21.0) | 39.0 (38.0 to 40.0) | +90.2% | yes |
| Uploads/s at 2 users | 52.0 (50.0 to 54.0) | 52.0 (50.0 to 54.0) | +0.0% | no |
| Knee (users) | 8 | 8 | +0.0% | no |
| GET /search p95 (ms) at 8 users | 400 (300 to 500) | 400 (300 to 500) | +0.0% | no |
| POST /ask p95 (ms) at 8 users | 800 (600 to 1,000) | 800 (600 to 1,000) | +0.0% | no |
| POST /ask/stream, answer p95 (ms) at 8 users | 800 (600 to 1,000) | 800 (600 to 1,000) | +0.0% | no |
| GET /documents p95 (ms) at 8 users | 100 (75.0 to 125) | 100 (75.0 to 125) | +0.0% | no |
| GET /search p95 (ms) at 16 users | 1,450 (1,400 to 1,500) | 1,450 (1,400 to 1,500) | +0.0% | no |
| POST /ask p95 (ms) at 16 users | 2,900 (2,800 to 3,000) | 2,900 (2,800 to 3,000) | +0.0% | no |
| POST /ask/stream, answer p95 (ms) at 16 users | 2,900 (2,800 to 3,000) | 2,900 (2,800 to 3,000) | +0.0% | no |
| GET /documents p95 (ms) at 16 users | 362 (350 to 375) | 362 (350 to 375) | +0.0% | no |"""


def render_fixture(directory: Path) -> report.Rendered:
    return report.render(directory, publishable=True)


def test_the_tables_come_from_the_fixture_byte_for_byte(results: Path) -> None:
    markdown = render_fixture(results).markdown
    for table in (INGEST_TABLE, SCALING_TABLE, LATENCY_TABLE, COMPARISON_TABLE):
        assert table in markdown


def test_the_region_names_the_machine_the_docker_share_and_the_exact_command(results: Path) -> None:
    markdown = render_fixture(results).markdown
    assert (
        "**Machine.** Apple M1, 8 cores (4 performance, 4 efficiency), 16 GiB, macOS 27.0.1. "
        "**Docker.** Docker Desktop 29.4.0 with 8 CPUs and 7.75 GiB for every container, the load generator included. "
        "**Stack.** PostgreSQL 16.4 with pgvector 0.8.0 (max_connections 100), API pool 5 plus 10 overflow, "
        "1 uvicorn worker, embedder fastembed."
    ) in markdown
    command = "python scripts/load/load.py run --profile zero --scale 1.0 --label baseline"
    assert f"```\n{command}\n```" in markdown
    assert "Baseline (commit `0123456`, run `20261101T100000Z`)" in markdown
    assert "After (commit `0123456`, run `20261102T100000Z`)" in markdown
    assert "### Scripted model `zero`, latency scale 1" in markdown
    assert "#### Before and after: `pool-fix`" in markdown


def test_the_method_is_the_artifacts_own_methodology_string(results: Path) -> None:
    assert f"**Method.** {artifact.METHODOLOGY}" in render_fixture(results).markdown


def test_the_knee_sentence_states_the_limits_and_the_runs(results: Path) -> None:
    markdown = render_fixture(results).markdown
    assert (
        "A step fails when more than 1% of its reads fail or the `/search` p95 is over 1,000 ms. "
        "The knee, the last step that met both limits, was 8 users over 2 runs."
    ) in markdown
    assert "Tenants: `load-large` 2,000 documents and 6,000 chunks, `load-small` 30 documents and 90 chunks." in markdown


def test_an_invalid_scaling_repetition_makes_the_artifact_unpublishable(tmp_path: Path) -> None:
    """Only the last step of a ramp may be invalid; a scaling repetition that
    failed would be a median over fewer runs than the table claims."""
    document = make_artifact()
    document["phases"].append(scaling_phase(3, 1, 1.0, valid=False))
    write_runs(tmp_path, document)
    with pytest.raises(report.ReportError, match="phase scaling repetition 3 is invalid"):
        report.render(tmp_path, publishable=True)


def test_a_ramp_that_never_failed_has_no_step_past_the_knee(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(ramp=[{2: 40.0, 4: 80.0}]))
    markdown = render_fixture(tmp_path).markdown
    assert "| At the knee | 4 | GET /search |" in markdown
    assert "One step past" not in markdown


def test_a_ramp_whose_first_step_failed_shows_only_that_step_past(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(ramp=[{2: 1500.0}]))
    markdown = render_fixture(tmp_path).markdown
    assert "At the knee" not in markdown
    assert "| One step past | 2 | GET /search |" in markdown
    assert "No step met both limits" in markdown


def test_fewer_phases_give_fewer_tables(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(ingest={}, ramp=[]))
    markdown = render_fixture(tmp_path).markdown
    assert "#### Worker scaling" in markdown
    assert "#### Ingest" not in markdown and "#### Read latency" not in markdown


def test_two_profiles_are_two_groups_in_a_fixed_order(tmp_path: Path) -> None:
    write_runs(
        tmp_path,
        make_artifact(profile="replay", scale=0.1, run_id="20261101T090000Z"),
        make_artifact(profile="zero", run_id="20261101T100000Z"),
    )
    markdown = render_fixture(tmp_path).markdown
    assert markdown.index("### Scripted model `zero`") < markdown.index("### Scripted model `replay`, latency scale 0.1")
    assert "Agent step 3000 ms and judge call 4000 ms are assumptions." in markdown


def test_the_region_has_no_em_dash(results: Path) -> None:
    rendered = render_fixture(results)
    assert "\N{EM DASH}" not in rendered.markdown
    assert all("\N{EM DASH}" not in svg for svg in rendered.charts.values())


# --- no published run ----------------------------------------------------------------


@pytest.mark.parametrize("make_dir", [lambda base: base / "missing", lambda base: (base / "empty").mkdir() or base / "empty"])
def test_with_no_artifact_the_region_says_there_is_no_published_run(tmp_path: Path, make_dir) -> None:
    rendered = report.render(make_dir(tmp_path), publishable=True)
    assert "**No published run yet.**" in rendered.markdown
    assert "####" not in rendered.markdown
    for svg in rendered.charts.values():
        root = ET.fromstring(svg)
        assert root.find(f"{SVG}desc").text == "No published run yet."
        assert "No published run yet" in svg


# --- refusals --------------------------------------------------------------------------


def test_a_dirty_artifact_is_refused_with_the_validate_rule(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(dirty=True))
    with pytest.raises(report.ReportError, match="the git tree was dirty"):
        report.render(tmp_path, publishable=True)


def test_one_bad_file_stops_the_report_and_every_bad_file_is_named(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(), make_artifact(label="baseline-b", run_id="20261103T100000Z", dirty=True))
    (tmp_path / "notes.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "other.json").write_text(json.dumps({"schema": "eval_v1"}), encoding="utf-8")
    (tmp_path / "README.txt").write_text("ignored: not a .json file", encoding="utf-8")
    with pytest.raises(report.ReportError) as raised:
        report.render(tmp_path, publishable=True)
    message = str(raised.value)
    assert "notes.json: not readable JSON" in message
    assert "other.json: not a load_v1 artifact" in message
    assert "20261103T100000Z_baseline-b.json: the git tree was dirty" in message


def test_a_quick_preset_is_refused(tmp_path: Path) -> None:
    quick = make_artifact(mutate=lambda doc: doc.update(preset="quick"))
    write_runs(tmp_path, quick)
    with pytest.raises(report.ReportError, match="preset 'quick' is not publishable"):
        report.render(tmp_path, publishable=True)


def test_two_baselines_of_one_model_profile_are_ambiguous(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(), make_artifact(label="baseline-again", run_id="20261103T100000Z"))
    with pytest.raises(report.ReportError, match="2 baselines for model profile 'zero' at scale 1"):
        report.render(tmp_path, publishable=True)


MISMATCHES = {
    "machine": lambda doc: doc["machine"].update(cpu="Apple M2"),
    "machine memory": lambda doc: doc["machine"].update(memory_bytes=34359738368),
    "docker cpus": lambda doc: doc["docker"].update(ncpu=4),
    "docker memory": lambda doc: doc["docker"].update(memory_bytes=4294967296),
    "scenario": lambda doc: with_scenario(doc, {"workload": {"version": 2}}),
    "model durations": lambda doc: doc["model"].update(agent_step_ms=1000.0),
    "override": lambda doc: doc["stack"]["overrides"]["api"].update(ASK_JUDGE_SAMPLE_RATE="0.5"),
    "embedder": lambda doc: doc["stack"].update(embedder="hashing"),
    "methodology": lambda doc: doc.update(methodology="something else"),
}


@pytest.mark.parametrize("name", sorted(MISMATCHES))
def test_a_pair_that_cannot_be_compared_is_refused(tmp_path: Path, name: str) -> None:
    changed = after_of(mutate=MISMATCHES[name])
    before = make_artifact()
    assert report.comparability_problems(before, changed)
    write_runs(tmp_path, before, changed)
    # methodology is checked for the whole set first; the others, pair by pair
    with pytest.raises(report.ReportError, match="cannot be compared|different methodologies"):
        report.render(tmp_path, publishable=True)


def test_the_refusal_names_what_differs(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(), after_of(mutate=lambda doc: doc["machine"].update(cpu="Apple M2", cores=10)))
    with pytest.raises(report.ReportError) as raised:
        report.render(tmp_path, publishable=True)
    message = str(raised.value)
    assert "machine cpu differs: baseline 'Apple M1', after 'Apple M2'" in message
    assert "machine cores differs: baseline 8, after 10" in message


def test_an_after_of_another_model_profile_has_no_baseline(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(), after_of(profile="replay", scale=0.1))
    with pytest.raises(report.ReportError, match="has no baseline of model profile 'replay' at scale 0.1"):
        report.render(tmp_path, publishable=True)
    write_runs(tmp_path / "scale", make_artifact(), after_of(scale=0.5))
    with pytest.raises(report.ReportError, match="no baseline of model profile 'zero' at scale 0.5"):
        report.render(tmp_path / "scale", publishable=True)


def test_matching_pairs_have_no_problems_and_commit_and_label_are_not_compared() -> None:
    assert report.comparability_problems(make_artifact(), after_of()) == []


def test_an_after_run_alone_is_refused_not_rendered_as_a_baseline(tmp_path: Path) -> None:
    write_runs(tmp_path, after_of())
    with pytest.raises(report.ReportError, match="has no baseline"):
        report.render(tmp_path, publishable=True)


# --- the charts ----------------------------------------------------------------------------


def series_colors_in_globals_css() -> tuple[dict[str, str], dict[str, str]]:
    css = (REPO_ROOT / "frontend" / "src" / "app" / "globals.css").read_text(encoding="utf-8")
    block = css[css.index("Chart series colors") :]
    dark_at = block.index("prefers-color-scheme: dark")

    def pick(text: str) -> dict[str, str]:
        return dict(re.findall(r"--(series-[12]):\s*(#[0-9a-f]{6})", text))

    return pick(block[:dark_at]), pick(block[dark_at:])


def test_the_chart_colors_are_the_dashboards_series_slots() -> None:
    light, dark = series_colors_in_globals_css()
    assert (report.LIGHT.baseline, report.LIGHT.after) == (light["series-1"], light["series-2"])
    assert (report.DARK.baseline, report.DARK.after) == (dark["series-1"], dark["series-2"])


def test_each_theme_draws_its_own_colors(results: Path) -> None:
    charts = render_fixture(results).charts
    light, dark = charts[report.LIGHT_CHART], charts[report.DARK_CHART]
    for theme, svg, other in ((report.LIGHT, light, report.DARK), (report.DARK, dark, report.LIGHT)):
        for color in (theme.surface, theme.ink, theme.muted, theme.baseline, theme.after):
            assert f'"{color}"' in svg
        for color in (other.surface, other.baseline, other.after):
            assert f'"{color}"' not in svg


def test_the_charts_parse_as_xml_with_a_title_and_a_description(results: Path) -> None:
    for svg in render_fixture(results).charts.values():
        root = ET.fromstring(svg)
        assert root.tag == f"{SVG}svg" and root.get("role") == "img"
        assert root.find(f"{SVG}title").text == "Load test results"
        description = root.find(f"{SVG}desc").text
        assert "Baseline documents per second: 1 workers 6.10" in description
        assert "After (pool-fix) /search p95" in description
        assert root.get("aria-labelledby") == "title desc"


def test_the_charts_label_axes_with_units_and_name_every_series(results: Path) -> None:
    svg = render_fixture(results).charts[report.LIGHT_CHART]
    texts = [element.text for element in ET.fromstring(svg).iter(f"{SVG}text")]
    for expected in (
        "Documents per second",
        "Worker containers",
        "p95 latency (ms, log scale)",
        "Concurrent users (closed loop)",
        "Throughput vs worker containers",
        "/search p95 latency vs concurrent users",
        "Baseline",
        "After (pool-fix)",
        "Ideal linear scaling",
        "SLO 1,000 ms",
    ):
        assert expected in texts


def test_everything_the_chart_draws_is_inside_the_image(results: Path) -> None:
    root = ET.fromstring(render_fixture(results).charts[report.LIGHT_CHART])
    width, height = float(root.get("width")), float(root.get("height"))
    for element in root.iter():
        for x_name, y_name in (("x", "y"), ("cx", "cy"), ("x1", "y1"), ("x2", "y2")):
            if element.get(x_name) is not None:
                assert 0 <= float(element.get(x_name)) <= width, element.attrib
                assert 0 <= float(element.get(y_name)) <= height, element.attrib
        for pair in (element.get("points") or "").split():
            x, y = (float(part) for part in pair.split(","))
            assert 0 <= x <= width and 0 <= y <= height


def test_the_charts_are_byte_identical_across_runs_and_file_order(results: Path, tmp_path: Path) -> None:
    first = render_fixture(results)
    again = render_fixture(results)
    assert first.charts == again.charts and first.markdown == again.markdown
    copy_dir = tmp_path / "copy"
    write_runs(copy_dir, *reversed([json.loads(path.read_text(encoding="utf-8")) for path in sorted(results.glob("*.json"))]))
    assert report.render(copy_dir, publishable=True).charts == first.charts


def test_a_baseline_alone_draws_one_series_and_no_after_label(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact())
    svg = report.render(tmp_path, publishable=True).charts[report.LIGHT_CHART]
    assert "After" not in svg and "pool-fix" not in svg
    assert f'"{report.LIGHT.after}"' not in svg
    assert "Before and after" not in report.render(tmp_path, publishable=True).markdown


def test_a_panel_without_data_says_so(tmp_path: Path) -> None:
    write_runs(tmp_path, make_artifact(ramp=[]))
    root = ET.fromstring(report.render(tmp_path, publishable=True).charts[report.LIGHT_CHART])
    texts = [element.text for element in root.iter(f"{SVG}text")]
    assert "Not measured in the published runs." in texts
    assert "Documents per second" in texts


def test_end_labels_that_would_collide_are_dropped_and_the_legend_names_them() -> None:
    svg = report._Svg(report.LIGHT)
    report._draw_end_labels(svg, [(100.0, 50.0, "Top"), (100.0, 55.0, "Close"), (100.0, 80.0, "Far")])
    drawn = "".join(svg.parts)
    assert "Top" in drawn and "Far" in drawn and "Close" not in drawn


def test_the_linear_axis_rounds_up_to_clean_ticks() -> None:
    assert report._linear_axis(41.0) == (50.0, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    assert report._linear_axis(0.9)[0] == pytest.approx(1.0)
    assert report._linear_axis(8.0) == (8.0, [0.0, 2.0, 4.0, 6.0, 8.0])


def test_the_headline_group_prefers_replay_and_falls_back_per_panel(tmp_path: Path) -> None:
    write_runs(
        tmp_path,
        make_artifact(profile="replay", scale=0.1, ramp=[], run_id="20261101T090000Z"),
        make_artifact(profile="zero", run_id="20261101T100000Z"),
    )
    root = ET.fromstring(report.render(tmp_path, publishable=True).charts[report.LIGHT_CHART])
    texts = [element.text for element in root.iter(f"{SVG}text")]
    assert texts.count("replay profile, latency scale 0.1") == 1  # the scaling panel
    assert texts.count("zero profile, latency scale 1") == 1  # the latency panel, which replay did not measure


# --- the README -------------------------------------------------------------------------------


README_TEXT = f"intro\n\n## Performance\n\n{report.START_MARKER}\n\nold\n\n{report.END_MARKER}\n\n## Next\n"


def test_update_published_writes_the_region_and_both_charts_and_is_idempotent(results: Path, tmp_path: Path) -> None:
    readme, charts = tmp_path / "README.md", tmp_path / "charts"
    readme.write_text(README_TEXT, encoding="utf-8")
    changed = report.update_published(readme, results, charts)
    assert {path.name for path in changed} == {"README.md", report.LIGHT_CHART, report.DARK_CHART}
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n\n## Performance\n\n") and text.endswith("\n\n## Next\n")
    assert report.marked_region(text) == report.region_for(render_fixture(results).markdown)
    snapshot = {path: path.read_bytes() for path in (readme, charts / report.LIGHT_CHART, charts / report.DARK_CHART)}
    assert report.update_published(readme, results, charts) == []
    assert snapshot == {path: path.read_bytes() for path in snapshot}


def test_a_refusal_leaves_the_readme_and_the_charts_untouched(tmp_path: Path) -> None:
    readme, charts, bad = tmp_path / "README.md", tmp_path / "charts", tmp_path / "results"
    readme.write_text(README_TEXT, encoding="utf-8")
    write_runs(bad, make_artifact(dirty=True))
    with pytest.raises(report.ReportError):
        report.update_published(readme, bad, charts)
    assert readme.read_text(encoding="utf-8") == README_TEXT
    assert not charts.exists()


def test_a_readme_without_markers_is_an_error_and_writes_nothing(results: Path, tmp_path: Path) -> None:
    readme, charts = tmp_path / "README.md", tmp_path / "charts"
    readme.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(report.ReportError, match="expected"):
        report.update_published(readme, results, charts)
    assert not charts.exists()


def test_the_published_charts_and_readme_are_regenerated_from_the_committed_artifacts() -> None:
    """The sync test: the README region and both committed charts equal what
    report.py renders from scripts/load/results (the no-run state until the
    first baseline is committed)."""
    rendered = report.render(report.RESULTS_DIR, publishable=True)
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert report.marked_region(readme) == report.region_for(rendered.markdown), (
        "README Performance region is stale: run `python scripts/load/load.py report --update-readme`"
    )
    for name, svg in rendered.charts.items():
        committed = (report.CHART_DIR / name).read_text(encoding="utf-8")
        assert committed == svg, f"{name} is stale: run `python scripts/load/load.py report --update-readme`"


def performance_section(readme: str) -> str:
    start = readme.index("\n## Performance\n")
    return readme[start : readme.index("\n## ", start + 1)]


def test_the_readme_section_has_the_markers_once_and_the_picture() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(report.START_MARKER) == 1 and readme.count(report.END_MARKER) == 1
    section = performance_section(readme)
    assert report.START_MARKER in section and report.END_MARKER in section
    assert '<source media="(prefers-color-scheme: dark)" srcset="screenshots/performance/load-dark.svg">' in section
    assert 'src="screenshots/performance/load-light.svg"' in section
    assert "<picture>" in section and "</picture>" in section
    assert re.search(r'<img [^>]*alt="[^"]{20,}"', section)


def test_no_number_is_typed_in_the_section_outside_the_generated_region() -> None:
    """Every number in the section comes from an artifact: the prose around the
    markers carries none."""
    section = performance_section((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    outside = section[: section.index(report.START_MARKER)] + section[section.index(report.END_MARKER) :]
    outside = re.sub(r"(?:srcset|src)=\"[^\"]*\"", "", outside)  # file names are not claims
    assert not re.search(r"\d", outside), re.findall(r".*\d.*", outside)
    assert "\N{EM DASH}" not in section


# --- the command ----------------------------------------------------------------------------------


def test_from_writes_a_scratch_render_and_never_the_readme(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source, out = tmp_path / "dev", tmp_path / "scratch"
    quick = make_artifact(label="l5-dev", mutate=lambda doc: doc.update(preset="quick"))
    write_runs(source, quick)
    published = [REPO_ROOT / "README.md", *report.CHART_DIR.glob("*.svg")]
    before = {path: path.read_bytes() for path in published}
    listing = sorted(path.name for path in report.CHART_DIR.iterdir()) if report.CHART_DIR.exists() else []

    assert load.main(["report", "--from", str(source), "--out", str(out)]) == 0

    assert before == {path: path.read_bytes() for path in published}
    assert (sorted(path.name for path in report.CHART_DIR.iterdir()) if report.CHART_DIR.exists() else []) == listing
    assert sorted(path.name for path in out.iterdir()) == [report.DARK_CHART, report.LIGHT_CHART, "report.md"]
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert markdown.startswith(report.NOTICE) and "**Development render, not publishable.**" in markdown
    assert "l5-dev" in markdown and "preset 'quick' is not publishable" in markdown
    assert "development render, not publishable" in (out / report.LIGHT_CHART).read_text(encoding="utf-8")
    assert "note: " in capsys.readouterr().err


def test_from_skips_what_fails_validation_and_keeps_the_newest_of_two_baselines(tmp_path: Path) -> None:
    source = tmp_path / "dev"
    old = make_artifact(label="first", run_id="20261101T100000Z", scaling={1: [1.0]})
    new = make_artifact(label="second", run_id="20261102T100000Z", scaling={1: [2.0]})
    broken = make_artifact(label="broken", run_id="20261103T100000Z", mutate=lambda doc: doc.update(spend_usd=1.0))
    write_runs(source, old, new, broken)
    (source / "tenants.json").write_text(json.dumps({"load-large": {"api_key": "dpk_secret"}}), encoding="utf-8")
    rendered = report.render(source, publishable=False)
    assert "| 1 | 2.00 |" in rendered.markdown and "| 1 | 1.00 |" not in rendered.markdown
    assert "dpk_secret" not in rendered.markdown + "".join(rendered.charts.values())
    assert any("broken.json: spend_usd" in note for note in rendered.notes)
    assert any("tenants.json: not a load_v1 artifact" in note for note in rendered.notes)
    assert any("using the newest of" in note for note in rendered.notes)


@pytest.mark.parametrize("inside", [report.RESULTS_DIR / "scratch", report.CHART_DIR, report.CHART_DIR / "scratch"])
def test_from_refuses_to_write_into_the_committed_results_or_charts(inside: Path, tmp_path: Path) -> None:
    existed = inside.exists()
    with pytest.raises(report.ReportError, match="never writes under"):
        report.render_development(tmp_path, inside)
    assert inside.exists() == existed


def test_from_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(report.ReportError, match="is not a directory"):
        report.render_development(tmp_path / "nope", tmp_path / "out")


def test_the_command_needs_exactly_one_mode(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    for argv in (["report"], ["report", "--update-readme", "--from", str(tmp_path)]):
        with pytest.raises(SystemExit) as raised:
            load.main(argv)
        assert raised.value.code == 2
    assert load.main(["report", "--update-readme", "--out", str(tmp_path)]) == 1
    assert "--out only applies with --from" in capsys.readouterr().err


def test_update_readme_through_the_cli_refuses_a_bad_committed_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_dir = tmp_path / "results"
    write_runs(results_dir, make_artifact(dirty=True))
    readme = tmp_path / "README.md"
    readme.write_text(README_TEXT, encoding="utf-8")
    monkeypatch.setattr(report, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(report, "README_PATH", readme)
    monkeypatch.setattr(report, "CHART_DIR", tmp_path / "charts")
    assert load.main(["report", "--update-readme"]) == 1
    assert "the git tree was dirty" in capsys.readouterr().err
    assert readme.read_text(encoding="utf-8") == README_TEXT


def test_the_fixture_helper_builds_a_publishable_artifact() -> None:
    document = make_artifact()
    assert artifact.validate(document, publishable=True) == []
    assert copy.deepcopy(document) == document
