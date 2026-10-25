"""Tests for the eval CLIs: which prompt and documents evals/run.py runs, and that
a --mock run (or any run under MODEL_PROVIDER=scripted, in evals/run.py,
evals/run_agent.py and evals/run_judge_calibration.py) can never reach
evals/results, where the README tables would read it as a measurement.

Pure/offline like the other eval-suite tests: the mock extract_fn has no
network, no key, and no database. Every run writes into pytest's tmp_path.
"""

import importlib.util
import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import SCRIPTED_MODEL, get_settings
from app.evals.agent import AgentRunResult
from app.evals.dataset import EVALS_DIR
from app.evals.report import load_results

RESULTS_DIR = EVALS_DIR / "results"


def _load_cli(script: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EVALS_DIR / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_cli = _load_cli("run.py", "evals_run_cli")
agent_cli = _load_cli("run_agent.py", "evals_run_agent_cli")
calibration_cli = _load_cli("run_judge_calibration.py", "evals_run_judge_calibration_cli")


def _results_listing() -> list[tuple[str, int]]:
    """Every file under evals/results, with its size, so a write anywhere below
    it shows."""
    return sorted(
        (str(p.relative_to(RESULTS_DIR)), p.stat().st_size)
        for p in RESULTS_DIR.rglob("*")
        if p.is_file()
    )


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run_cli.main()


def _written(out_dir: Path) -> dict:
    (artifact,) = out_dir.glob("*.json")
    return json.loads(artifact.read_text(encoding="utf-8"))


class TestMockNeverWritesToTheResults:
    def test_by_default_it_writes_to_a_temporary_directory(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
        before = _results_listing()

        _run(monkeypatch, "--mock", "--limit", "3")

        assert _written(tmp_path)["model"] == "mock-model"
        assert _results_listing() == before
        assert str(tmp_path) in capsys.readouterr().out
        assert all(run["model"] != "mock-model" for run in load_results())

    def test_it_refuses_the_results_directory(self, monkeypatch, capsys):
        before = _results_listing()

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--out", str(RESULTS_DIR))

        assert exit_info.value.code == 2
        assert "never writes" in capsys.readouterr().err
        assert _results_listing() == before

    def test_it_refuses_the_results_directory_spelled_another_way(self, monkeypatch):
        before = _results_listing()

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--out", str(RESULTS_DIR / ".." / "results"))

        assert exit_info.value.code == 2
        assert _results_listing() == before

    def test_an_explicit_other_directory_is_used(self, monkeypatch, tmp_path):
        _run(monkeypatch, "--mock", "--limit", "2", "--out", str(tmp_path / "scratch"))

        assert _written(tmp_path / "scratch")["n_scored"] == 2

    @pytest.mark.parametrize("sub", ["smoke", "judge", "retrieval"])
    def test_it_refuses_anything_under_the_results_directory(self, monkeypatch, sub):
        before = _results_listing()
        existed = (RESULTS_DIR / sub).exists()

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--limit", "1", "--out", str(RESULTS_DIR / sub))

        assert exit_info.value.code == 2
        assert _results_listing() == before
        assert (RESULTS_DIR / sub).exists() == existed

    def test_it_says_where_the_temporary_artifact_went(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))

        _run(monkeypatch, "--mock", "--limit", "1")

        out = capsys.readouterr().out
        assert f"wrote {tmp_path}" in out
        assert "temporary directory" in out

    def test_it_cannot_be_combined_with_update_readme(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--update-readme")

        assert exit_info.value.code == 2
        assert "--mock" in capsys.readouterr().err


@pytest.fixture
def scripted_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """MODEL_PROVIDER=scripted as the load stack sets it, read by the one cached
    get_settings(), which is rebuilt on the way in and on the way out."""
    for key, value in {
        "MODEL_PROVIDER": SCRIPTED_MODEL,
        "EXTRACTION_MODEL": SCRIPTED_MODEL,
        "TRANSCRIPTION_MODEL": SCRIPTED_MODEL,
        "AGENT_MODEL": SCRIPTED_MODEL,
        "JUDGE_MODEL": SCRIPTED_MODEL,
        "ANTHROPIC_API_KEY": "",
        "EMBEDDING_BACKEND": "hashing",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    monkeypatch.undo()
    get_settings.cache_clear()


@pytest.mark.usefixtures("scripted_environment")
class TestScriptedNeverWritesToTheResults:
    """The scripted twin of TestMockNeverWritesToTheResults: a run with no model
    is not a measurement, and `--mock` is not the only way to make one."""

    def test_by_default_it_writes_to_a_temporary_directory(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
        before = _results_listing()

        _run(monkeypatch, "--limit", "2")

        artifact = _written(tmp_path)
        assert artifact["model"] == SCRIPTED_MODEL
        assert artifact["n_scored"] == 2
        assert artifact["total_cost_usd"] == 0
        assert _results_listing() == before
        out = capsys.readouterr().out
        assert f"wrote {tmp_path}" in out
        assert "temporary directory" in out
        assert all(run["model"] != SCRIPTED_MODEL for run in load_results())

    @pytest.mark.parametrize("sub", ["", "smoke", "judge", "retrieval", "agent"])
    def test_it_refuses_the_results_directory_and_anything_under_it(
        self, monkeypatch, sub, capsys
    ):
        before = _results_listing()
        existed = (RESULTS_DIR / sub).exists()

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--limit", "1", "--out", str(RESULTS_DIR / sub))

        assert exit_info.value.code == 2
        assert "never writes" in capsys.readouterr().err
        assert _results_listing() == before
        assert (RESULTS_DIR / sub).exists() == existed

    def test_an_explicit_other_directory_is_used(self, monkeypatch, tmp_path):
        _run(monkeypatch, "--limit", "2", "--out", str(tmp_path / "scratch"))

        assert _written(tmp_path / "scratch")["model"] == SCRIPTED_MODEL

    def test_it_cannot_be_combined_with_update_readme(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--update-readme")

        assert exit_info.value.code == 2
        assert "MODEL_PROVIDER=scripted" in capsys.readouterr().err

    def test_it_cannot_update_the_readme_from_a_report_only_run_either(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--report-only", "--update-readme")

        assert exit_info.value.code == 2
        assert "MODEL_PROVIDER=scripted" in capsys.readouterr().err


def _run_other(cli, script: str, monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", [script, *argv])
    return cli.main()


@pytest.fixture
def agent_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_agent.py without a database, a tracer or a model: the eval itself
    returns an empty scripted run, so what is under test is where the CLI writes
    it. SCRIPTED_MODEL_ALLOW_API_KEY with a placeholder key gets past the CLI's
    own key check, the way a host process reading backend/.env would."""
    import app.db
    import app.evals.agent
    import app.telemetry

    async def nothing() -> None:
        return None

    async def run_agent_eval(*args, **kwargs) -> AgentRunResult:
        return AgentRunResult(
            model=SCRIPTED_MODEL, effort="medium", prompt_version="agent_v1",
            question_set_version="v1", started_at_utc="20261025T000000Z", n_questions=0,
            n_run=0, n_skipped_cost_cap=0, max_cost_usd=2.0, summary={}, by_type={},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("SCRIPTED_MODEL_ALLOW_API_KEY", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(app.evals.agent, "run_agent_eval", run_agent_eval)
    monkeypatch.setattr(app.db, "engine", SimpleNamespace(dispose=nothing))
    monkeypatch.setattr(app.telemetry, "shutdown_tracing", lambda: None)
    monkeypatch.setattr(app.evals.agent, "render_agent_table", lambda results: "table")


@pytest.mark.usefixtures("scripted_environment")
class TestScriptedAgentEvalNeverWritesToTheResults:
    def test_by_default_it_writes_to_a_temporary_directory(
        self, monkeypatch, tmp_path, capsys, agent_run
    ):
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
        before = _results_listing()

        assert _run_other(agent_cli, "run_agent.py", monkeypatch) == 0

        (artifact,) = tmp_path.glob("*.json")
        assert json.loads(artifact.read_text(encoding="utf-8"))["model"] == SCRIPTED_MODEL
        assert _results_listing() == before
        assert f"wrote {artifact}" in capsys.readouterr().out

    @pytest.mark.parametrize("sub", ["", "agent"])
    def test_it_refuses_the_results_directory_and_anything_under_it(
        self, monkeypatch, sub, capsys, agent_run
    ):
        before = _results_listing()

        code = _run_other(agent_cli, "run_agent.py", monkeypatch, "--out", str(RESULTS_DIR / sub))

        assert code == 2
        assert "never writes" in capsys.readouterr().err
        assert _results_listing() == before

    def test_an_explicit_other_directory_is_used(self, monkeypatch, tmp_path, agent_run):
        assert _run_other(agent_cli, "run_agent.py", monkeypatch, "--out", str(tmp_path)) == 0

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_it_cannot_be_combined_with_update_readme(self, monkeypatch, capsys, agent_run):
        readme = (EVALS_DIR.parent / "README.md").read_bytes()

        code = _run_other(agent_cli, "run_agent.py", monkeypatch, "--update-readme")

        assert code == 2
        assert "MODEL_PROVIDER=scripted" in capsys.readouterr().err
        assert (EVALS_DIR.parent / "README.md").read_bytes() == readme


@pytest.mark.usefixtures("scripted_environment")
class TestScriptedJudgeCalibrationNeverWritesToTheResults:
    def test_by_default_it_writes_to_a_temporary_directory(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
        before = _results_listing()

        assert _run_other(calibration_cli, "run_judge_calibration.py", monkeypatch) == 0

        assert len(list(tmp_path.glob("*.json"))) == 1
        assert _results_listing() == before
        assert f"wrote {tmp_path}" in capsys.readouterr().out

    @pytest.mark.parametrize("sub", ["", "judge"])
    def test_it_refuses_the_results_directory_and_anything_under_it(
        self, monkeypatch, sub, capsys
    ):
        before = _results_listing()

        code = _run_other(
            calibration_cli, "run_judge_calibration.py", monkeypatch,
            "--out", str(RESULTS_DIR / sub),
        )

        assert code == 2
        assert "never writes" in capsys.readouterr().err
        assert _results_listing() == before

    def test_an_explicit_other_directory_is_used(self, monkeypatch, tmp_path):
        code = _run_other(
            calibration_cli, "run_judge_calibration.py", monkeypatch, "--out", str(tmp_path)
        )

        assert code == 0
        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_it_cannot_be_combined_with_update_readme(self, monkeypatch, capsys):
        readme = (EVALS_DIR.parent / "README.md").read_bytes()

        code = _run_other(
            calibration_cli, "run_judge_calibration.py", monkeypatch, "--update-readme"
        )

        assert code == 2
        assert "MODEL_PROVIDER=scripted" in capsys.readouterr().err
        assert (EVALS_DIR.parent / "README.md").read_bytes() == readme


class TestPromptAndDocuments:
    def test_the_default_prompt_is_v1(self, monkeypatch, tmp_path):
        _run(monkeypatch, "--mock", "--limit", "3", "--out", str(tmp_path))

        artifact = _written(tmp_path)

        assert artifact["prompt_version"] == "extract_v1"
        assert "evidence_coverage" not in artifact["summary"]
        assert len(artifact["per_doc"][0]["fields"]) == 7

    def test_prompt_v2_runs_and_scores_the_v2_fields_with_evidence(self, monkeypatch, tmp_path, capsys):
        _run(monkeypatch, "--mock", "--prompt", "extract_v2", "--out", str(tmp_path))

        artifact = _written(tmp_path)
        out = capsys.readouterr().out

        assert artifact["prompt_version"] == "extract_v2"
        assert artifact["n_scored"] == 32
        assert artifact["summary"]["overall_accuracy"] == 638 / 640
        assert len(artifact["per_doc"][0]["fields"]) == 20
        assert "predicted mock accuracy: 0.996875" in out
        assert "caught_by_quotes: 0.5" in out
        assert "evidence: 813/813 values quoted (100.0%)" in out

    def test_prompt_v3_runs_and_scores_like_v2(self, monkeypatch, tmp_path, capsys):
        _run(monkeypatch, "--mock", "--prompt", "extract_v3", "--out", str(tmp_path))

        artifact = _written(tmp_path)

        assert artifact["prompt_version"] == "extract_v3"
        assert artifact["summary"]["overall_accuracy"] == 638 / 640
        assert len(artifact["per_doc"][0]["fields"]) == 20
        assert "evidence: 813/813 values quoted (100.0%)" in capsys.readouterr().out

    def test_doc_is_repeatable_and_keeps_dataset_order(self, monkeypatch, tmp_path):
        _run(
            monkeypatch,
            "--mock",
            "--prompt",
            "extract_v2",
            "--doc",
            "032-multipage-invoice-totals-page-two",
            "--doc",
            "026-invoice-discount-shipping-two-taxes",
            "--out",
            str(tmp_path),
        )

        artifact = _written(tmp_path)

        assert [doc["doc_id"][:3] for doc in artifact["per_doc"]] == ["026", "032"]

    def test_an_unknown_document_is_an_error_before_anything_runs(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--doc", "026", "--out", str(tmp_path))

        assert exit_info.value.code == 2
        assert "no label with doc_id '026'" in capsys.readouterr().err
        assert not list(tmp_path.glob("*.json"))

    def test_an_unknown_prompt_is_an_error_naming_the_known_ones(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--prompt", "extract_v9", "--out", str(tmp_path))

        assert exit_info.value.code == 2
        err = capsys.readouterr().err
        assert "extract_v1" in err
        assert "extract_v2" in err
        assert "extract_v3" in err


class TestReportOnly:
    def test_update_readme_from_another_directory_is_refused_and_changes_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        readme = (EVALS_DIR.parent / "README.md").read_bytes()
        latest = (RESULTS_DIR / "latest.md").read_bytes()

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--report-only", "--out", str(tmp_path), "--update-readme")

        assert exit_info.value.code == 2
        assert "from evals/results only" in capsys.readouterr().err
        assert (EVALS_DIR.parent / "README.md").read_bytes() == readme
        assert (RESULTS_DIR / "latest.md").read_bytes() == latest

    def test_a_live_run_cannot_update_the_readme_from_another_directory_either(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--out", str(tmp_path / "smoke"), "--update-readme")

        assert exit_info.value.code == 2

    def test_reporting_from_another_directory_without_update_is_fine(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--report-only", "--out", str(tmp_path))

        assert exit_info.value.code == 0
        assert "No eval runs recorded yet" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "flags",
        [["--mock"], ["--prompt", "extract_v2"], ["--doc", "026-invoice-discount-shipping-two-taxes"], ["--limit", "3"]],
    )
    def test_it_refuses_the_flags_it_would_ignore(self, monkeypatch, capsys, flags):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--report-only", *flags)

        assert exit_info.value.code == 2
        assert flags[0] in capsys.readouterr().err
