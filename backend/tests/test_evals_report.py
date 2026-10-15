"""Tests for app.evals.report.

Pure/offline like its sibling eval-suite tests: no DB, no network, no
extraction. Artifacts are hand-built dicts matching
app.evals.runner.RunResult.to_dict()'s shape rather than real eval runs.
"""

import json
from copy import deepcopy
from typing import Any

import pytest

from app.evals.dataset import EVALS_DIR
from app.evals.report import (
    _OPS_HEADER,
    END_MARKER,
    LATEST_MD_PATH,
    NO_RUNS_PLACEHOLDER,
    START_MARKER,
    load_results,
    render_latest_md,
    render_tables,
    update_readme,
)
from app.extraction_fields import SCHEMA_V1, SCHEMAS


@pytest.fixture
def synthetic_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """These artifacts name a made-up extract_v2 to tell two runs apart; register
    it with the v1 shape so a replay can look the schema up."""
    monkeypatch.setitem(SCHEMAS, "extract_v2", SCHEMA_V1)


def _make_summary(**overrides: Any) -> dict[str, Any]:
    base = {
        "n_docs": 2,
        "per_field_accuracy": {
            "vendor": 1.0,
            "document_date": 1.0,
            "line_items": 1.0,
            "subtotal": 1.0,
            "tax": 0.5,
            "total": 1.0,
            "currency": 1.0,
        },
        "overall_accuracy": 13 / 14,
        "mean_confidence": 0.9,
        "mean_confidence_correct": 0.95,
        "mean_confidence_incorrect": 0.4,
        "caught_by_review": 1.0,
        "total_hallucinations": 0,
        "total_schema_violations": 0,
        "mean_item_accuracy": 1.0,
        "mean_item_accuracy_docs_with_items": 1.0,
    }
    base.update(overrides)
    return base


def _make_artifact(**overrides: Any) -> dict[str, Any]:
    base = {
        "model": "claude-sonnet-5",
        "prompt_version": "extract_v1",
        "dataset_version": "v1",
        "review_threshold": 0.8,
        "started_at_utc": "2026-07-01T00:00:00+00:00",
        "concurrency": 3,
        "max_cost_usd": 1.0,
        "summary": _make_summary(),
        "total_cost_usd": 0.02,
        "total_error_cost_usd": 0.0,
        "mean_cost_per_doc": 0.01,
        "total_input_tokens": 2000,
        "total_output_tokens": 400,
        "latency_p50_ms": 3000.0,
        "latency_p95_ms": 4000.0,
        "n_scored": 2,
        "n_errors": 0,
        "n_skipped_cost_cap": 0,
        "errors": [],
        "skipped_cost_cap": [],
        "per_doc": [],
    }
    base.update(overrides)
    return base


def _field(expected, actual, confidence=0.9, correct=True) -> dict[str, Any]:
    return {"correct": correct, "expected": expected, "actual": actual, "confidence": confidence}


def _leaf(value, confidence=0.9) -> dict[str, Any]:
    return {"value": value, "confidence": confidence}


def _doc_with_wrong_tax(tax_confidence: float = 0.9) -> dict[str, Any]:
    """A scored document whose tax was read as 1.5 instead of 1.0: the amounts no
    longer add up, so subtotal + tax = total fails."""
    row = {"description": _leaf("Widget"), "quantity": _leaf(2), "unit_price": _leaf(5.0), "total": _leaf(10.0)}
    expected_row = {"description": "Widget", "quantity": 2, "unit_price": 5.0, "total": 10.0}
    return {
        "doc_id": "042-wrong-tax",
        "status": "scored",
        "difficulty": "easy",
        "fields": {
            "vendor": _field("Acme", "Acme"),
            "document_date": _field("2026-01-01", "2026-01-01"),
            "currency": _field("USD", "USD"),
            "subtotal": _field(10.0, 10.0),
            "tax": _field(1.0, 1.5, tax_confidence, correct=False),
            "total": _field(11.0, 11.0),
            "line_items": _field([expected_row], [row]),
        },
        "schema_violations": 0,
        "hallucinations": 0,
        "item_accuracy": 1.0,
    }


def _ops_cell(row: str, column: str) -> str:
    """One cell of an Ops table row, found by its header label."""
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return cells[_OPS_HEADER.index(column)]


class TestRenderTables:
    def test_empty_results_is_placeholder(self):
        assert render_tables([]) == NO_RUNS_PLACEHOLDER

    def test_two_rows_percentages_err_annotation_and_na(self, synthetic_v2):
        artifact_a = _make_artifact(
            model="claude-sonnet-5",
            prompt_version="extract_v1",
            started_at_utc="2026-07-01T00:00:00+00:00",
            n_scored=2,
            n_errors=0,
            summary=_make_summary(),
        )
        artifact_b = _make_artifact(
            model="claude-haiku-4-5",
            prompt_version="extract_v2",
            started_at_utc="2026-07-15T00:00:00+00:00",
            n_scored=3,
            n_errors=2,
            summary=_make_summary(
                caught_by_review=None,
                overall_accuracy=1.0,
                per_field_accuracy=dict.fromkeys(
                    _make_summary()["per_field_accuracy"], 1.0
                ),
            ),
        )

        markdown = render_tables([artifact_a, artifact_b])

        # both rows present
        assert "claude-sonnet-5" in markdown
        assert "extract_v1" in markdown
        assert "claude-haiku-4-5" in markdown
        assert "extract_v2" in markdown

        # percentage formatting, 1 decimal
        assert "92.9%" in markdown  # 13/14 overall_accuracy for artifact_a
        assert "50.0%" in markdown  # tax per-field accuracy for artifact_a
        assert "100.0%" in markdown  # overall/per-field for artifact_b

        # Docs = n_scored, "+N err" appended when errors > 0
        assert "3 +2 err" in markdown
        assert "2 +0 err" not in markdown
        assert "| 2 |" in markdown  # artifact_a has no errors: bare n_scored

        # n/a handling: artifact_b's caught_by_review is None
        assert "n/a" in markdown

        # a one-line caption under each table
        assert markdown.count("*") >= 4  # two *...* captions, each wrapped

    def test_ops_table_has_a_caught_by_checks_column(self):
        markdown = render_tables([_make_artifact()])

        ops_header = next(line for line in markdown.splitlines() if "Caught by review" in line)
        assert "| Caught by review | Caught by checks |" in ops_header
        assert "Caught by checks is the share" in markdown

    def test_caught_by_checks_is_replayed_from_the_stored_answers(self):
        artifact = _make_artifact(n_scored=1, per_doc=[_doc_with_wrong_tax()])

        ops_row = render_tables([artifact]).split("### Ops")[1].splitlines()[4]

        # The 1.5 tax breaks total_math, and no unique fix exists, so tax is
        # flagged by the arithmetic even though the model was confident.
        assert _ops_cell(ops_row, "Caught by checks") == "100.0%"

    def test_caught_by_checks_moves_apart_from_caught_by_review(self):
        # A wrong vendor has no arithmetic: the same summary (caught by review
        # 100%) now reads 0.0% caught by checks.
        doc = _doc_with_wrong_tax()
        doc["fields"]["tax"] = _field(1.0, 1.0)
        doc["fields"]["vendor"] = _field("Acme", "Acme Corp", correct=False)
        artifact = _make_artifact(n_scored=1, per_doc=[doc])

        ops_row = render_tables([artifact]).split("### Ops")[1].splitlines()[4]

        assert _ops_cell(ops_row, "Caught by review") == "100.0%"
        assert _ops_cell(ops_row, "Caught by checks") == "0.0%"

    def test_caught_by_checks_is_na_without_scored_answers(self):
        ops_row = render_tables([_make_artifact(per_doc=[])]).split("### Ops")[1].splitlines()[4]

        assert ops_row.count("n/a") == 1
        assert "| 100.0% | n/a |" in ops_row

    def test_summary_none_renders_na_row(self):
        artifact = _make_artifact(n_scored=0, n_errors=5, summary=None)

        markdown = render_tables([artifact])

        assert "0 +5 err" in markdown
        assert "n/a / n/a" in markdown


class TestUpdateReadme:
    def test_replaces_only_marker_region(self, tmp_path):
        readme_path = tmp_path / "README.md"
        before = "# Title\n\nsome text before\n\n"
        after = "\n\nsome text after\n"
        original = f"{before}<!-- EVAL_TABLE:START -->\nold content\n<!-- EVAL_TABLE:END -->{after}"
        readme_path.write_text(original, encoding="utf-8")

        update_readme("NEW TABLE CONTENT", readme_path)
        updated = readme_path.read_text(encoding="utf-8")

        assert updated.startswith(before + "<!-- EVAL_TABLE:START -->")
        assert updated.endswith("<!-- EVAL_TABLE:END -->" + after)
        assert "NEW TABLE CONTENT" in updated
        assert "old content" not in updated

    def test_idempotent(self, tmp_path):
        readme_path = tmp_path / "README.md"
        readme_path.write_text(
            "# Title\n\n<!-- EVAL_TABLE:START -->\nold\n<!-- EVAL_TABLE:END -->\n\ntail\n",
            encoding="utf-8",
        )

        update_readme("SAME CONTENT", readme_path)
        first_pass = readme_path.read_text(encoding="utf-8")

        update_readme("SAME CONTENT", readme_path)
        second_pass = readme_path.read_text(encoding="utf-8")

        assert first_pass == second_pass

    def test_missing_markers_raises_clear_error(self, tmp_path):
        readme_path = tmp_path / "README.md"
        readme_path.write_text("# Title\n\nno markers here\n", encoding="utf-8")

        with pytest.raises(ValueError, match="EVAL_TABLE"):
            update_readme("content", readme_path)

    def test_end_before_start_raises(self, tmp_path):
        readme_path = tmp_path / "README.md"
        readme_path.write_text(
            "<!-- EVAL_TABLE:END -->\n<!-- EVAL_TABLE:START -->\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="EVAL_TABLE"):
            update_readme("content", readme_path)


class TestLoadResults:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_results(tmp_path / "does-not-exist") == []

    def test_sorted_by_started_at_utc_and_skips_corrupt(self, tmp_path, capsys):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        newer = _make_artifact(started_at_utc="2026-07-15T00:00:00+00:00")
        older = _make_artifact(started_at_utc="2026-07-01T00:00:00+00:00")

        (results_dir / "newer.json").write_text(json.dumps(newer), encoding="utf-8")
        (results_dir / "older.json").write_text(json.dumps(older), encoding="utf-8")
        (results_dir / "corrupt.json").write_text("{not valid json", encoding="utf-8")
        (results_dir / "incomplete.json").write_text(
            json.dumps({"model": "x"}), encoding="utf-8"
        )
        (results_dir / "not-an-object.json").write_text(json.dumps([1, 2]), encoding="utf-8")
        # Caught by checks replays at this threshold: without it, skip, not crash.
        no_threshold = _make_artifact()
        del no_threshold["review_threshold"]
        (results_dir / "no-threshold.json").write_text(json.dumps(no_threshold), encoding="utf-8")

        loaded = load_results(results_dir)

        assert [r["started_at_utc"] for r in loaded] == [
            "2026-07-01T00:00:00+00:00",
            "2026-07-15T00:00:00+00:00",
        ]

        captured = capsys.readouterr()
        assert "corrupt.json" in captured.err
        assert "incomplete.json" in captured.err
        assert "not-an-object.json" in captured.err
        assert "no-threshold.json" in captured.err


class TestUnknownPromptVersion:
    """An artifact this code has no schema for must not stop a report."""

    def test_an_all_errors_run_renders_with_no_incorrect_fields(self):
        artifact = _make_artifact(
            model=None,
            prompt_version=None,
            summary=None,
            n_scored=0,
            n_errors=1,
            per_doc=[{"doc_id": "001-boom", "status": "error", "difficulty": "easy"}],
        )

        markdown = render_latest_md([artifact])

        assert "*No incorrect fields in the most recent run.*" in markdown

    def test_an_unregistered_version_shows_na_in_caught_by_checks(self, capsys):
        artifact = _make_artifact(
            prompt_version="extract_v9", n_scored=1, per_doc=[_doc_with_wrong_tax()]
        )

        ops_row = render_tables([artifact]).split("### Ops")[1].splitlines()[4]

        assert _ops_cell(ops_row, "Caught by checks") == "n/a"
        assert _ops_cell(ops_row, "Caught by review") == "100.0%"
        assert "extract_v9" in capsys.readouterr().err

    def test_an_unregistered_version_shows_na_as_the_flagged_cell(self):
        artifact = _make_artifact(
            prompt_version="extract_v9", n_scored=1, per_doc=[_doc_with_wrong_tax()]
        )

        detail = render_latest_md([artifact]).split("### Incorrect fields")[1]
        row = next(line for line in detail.splitlines() if "042-wrong-tax" in line)

        assert [cell.strip() for cell in row.strip().strip("|").split("|")][-1] == "n/a"


class TestRenderLatestMd:
    def test_empty_results_is_placeholder(self):
        assert render_latest_md([]) == NO_RUNS_PLACEHOLDER

    def test_no_failures_in_latest_run(self):
        artifact = _make_artifact(per_doc=[])

        markdown = render_latest_md([artifact])

        assert "No incorrect fields" in markdown

    def test_failure_detail_for_most_recent_run_only(self, synthetic_v2):
        older = _make_artifact(
            model="claude-sonnet-5",
            prompt_version="extract_v1",
            started_at_utc="2026-07-01T00:00:00+00:00",
            per_doc=[
                {
                    "doc_id": "999-old-run-doc",
                    "status": "scored",
                    "difficulty": "easy",
                    "fields": {
                        "vendor": {
                            "correct": False,
                            "expected": "Old Corp",
                            "actual": "Wrong Corp",
                            "confidence": 0.3,
                        },
                    },
                    "schema_violations": 0,
                    "hallucinations": 0,
                    "item_accuracy": 1.0,
                    "cost_usd": 0.01,
                    "latency_ms": 3000,
                    "input_tokens": 1000,
                    "output_tokens": 200,
                },
            ],
        )
        newest = _make_artifact(
            model="claude-sonnet-5",
            prompt_version="extract_v2",
            started_at_utc="2026-07-15T00:00:00+00:00",
            n_scored=1,
            per_doc=[
                {
                    "doc_id": "007-diner",
                    "status": "scored",
                    "difficulty": "medium",
                    "fields": {
                        "vendor": {
                            "correct": True,
                            "expected": "Diner",
                            "actual": "Diner",
                            "confidence": 0.95,
                        },
                        "tax": {
                            "correct": False,
                            "expected": 1.0,
                            "actual": 1.5,
                            "confidence": 0.4,
                        },
                    },
                    "schema_violations": 0,
                    "hallucinations": 0,
                    "item_accuracy": 1.0,
                    "cost_usd": 0.01,
                    "latency_ms": 3000,
                    "input_tokens": 1000,
                    "output_tokens": 200,
                },
                {
                    "doc_id": "008-skipped",
                    "status": "error",
                    "difficulty": "hard",
                },
            ],
        )

        markdown = render_latest_md([older, newest])

        assert "extract_v2" in markdown

        detail_section = markdown.split("### Incorrect fields", 1)[1]
        assert "007-diner" in detail_section
        assert "tax" in detail_section
        assert "1.0" in detail_section  # expected
        assert "1.5" in detail_section  # actual
        assert "0.40" in detail_section  # confidence, 2 decimals

        # only the most recent run's failures appear -- not the older run's
        assert "999-old-run-doc" not in detail_section
        assert "Old Corp" not in detail_section
        # vendor was correct in the newest run -- must not show up as a failure
        assert "vendor" not in detail_section

    def test_flagged_column_says_why_the_product_would_review_a_field(self):
        confident = _make_artifact(n_scored=1, per_doc=[_doc_with_wrong_tax(0.9)])
        doubtful = _make_artifact(n_scored=1, per_doc=[_doc_with_wrong_tax(0.3)])

        confident_md = render_latest_md([confident]).split("### Incorrect fields")[1]
        doubtful_md = render_latest_md([doubtful]).split("### Incorrect fields")[1]

        assert "| Doc | Field | Expected | Actual | Confidence | Flagged |" in confident_md
        # The arithmetic names the field whatever the model's confidence was.
        assert "| 0.90 | reconciliation:total_math |" in confident_md
        assert "| 0.30 | reconciliation:total_math |" in doubtful_md

    def test_flagged_column_for_a_field_nothing_flags(self):
        doc = _doc_with_wrong_tax()
        doc["fields"]["vendor"] = _field("Acme", "Acme Corp Ltd", 0.95, correct=False)
        doc["fields"]["tax"] = _field(1.0, 1.0)
        artifact = _make_artifact(n_scored=1, per_doc=[doc])

        detail = render_latest_md([artifact]).split("### Incorrect fields")[1]

        assert "| 0.95 | not flagged |" in detail

    def test_does_not_mutate_input(self):
        artifact = _make_artifact(per_doc=[])
        snapshot = deepcopy(artifact)

        render_latest_md([artifact])

        assert artifact == snapshot


def test_readme_eval_table_and_latest_md_are_current():
    # Regenerate with `python evals/run.py --report-only --update-readme`. The
    # tables replay the committed runs through the rules, so a rules change
    # makes this fail until the README catches up.
    results = load_results()
    readme = (EVALS_DIR.parent / "README.md").read_text(encoding="utf-8")
    start = readme.index(START_MARKER) + len(START_MARKER)
    end = readme.index(END_MARKER)

    assert readme[start:end] == f"\n\n{render_tables(results)}\n\n"
    assert LATEST_MD_PATH.read_text(encoding="utf-8") == render_latest_md(results)
