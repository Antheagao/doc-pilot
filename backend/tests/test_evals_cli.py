"""Tests for the extraction eval CLI (evals/run.py): which prompt and documents
it runs, and that a --mock run can never reach evals/results, where the README
tables would read it as a measurement.

Pure/offline like the other eval-suite tests: the mock extract_fn has no
network, no key, and no database. Every run writes into pytest's tmp_path.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.evals.dataset import EVALS_DIR
from app.evals.report import load_results

RESULTS_DIR = EVALS_DIR / "results"


def _load_run_cli():
    spec = importlib.util.spec_from_file_location("evals_run_cli", EVALS_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_cli = _load_run_cli()


def _results_listing() -> list[tuple[str, int]]:
    return sorted((p.name, p.stat().st_size) for p in RESULTS_DIR.iterdir() if p.is_file())


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run_cli.main()


def _written(out_dir: Path) -> dict:
    (artifact,) = out_dir.glob("*.json")
    return json.loads(artifact.read_text(encoding="utf-8"))


class TestMockNeverWritesToTheResults:
    def test_by_default_it_writes_to_a_temporary_directory(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(run_cli.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
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

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--limit", "1", "--out", str(RESULTS_DIR / sub))

        assert exit_info.value.code == 2
        assert _results_listing() == before
        assert not (RESULTS_DIR / "smoke").exists()

    def test_it_says_where_the_temporary_artifact_went(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(run_cli.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

        _run(monkeypatch, "--mock", "--limit", "1")

        out = capsys.readouterr().out
        assert f"wrote {tmp_path}" in out
        assert "temporary directory" in out

    def test_it_cannot_be_combined_with_update_readme(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, "--mock", "--update-readme")

        assert exit_info.value.code == 2
        assert "--mock" in capsys.readouterr().err


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
