"""Tests for app.evals.report.

Pure/offline like its sibling eval-suite tests: no DB, no network, no
extraction. Artifacts are hand-built dicts matching
app.evals.runner.RunResult.to_dict()'s shape rather than real eval runs.
"""

import json
from copy import deepcopy
from typing import Any

import pytest

from app.evals.dataset import CORE_DOC_IDS, EVALS_DIR, load_cases
from app.evals.report import (
    _ACCURACY_HEADER,
    _OPS_HEADER,
    END_MARKER,
    LATEST_MD_PATH,
    NO_RUNS_PLACEHOLDER,
    START_MARKER,
    core_counts,
    load_results,
    render_latest_md,
    render_tables,
    update_readme,
)
from app.evals.runner import build_mock_extract_fn, run_eval
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2, SCHEMAS


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


def _accuracy_cell(row: str, column: str) -> str:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return cells[_ACCURACY_HEADER.index(column)]


def _table_rows(markdown: str, heading: str) -> list[str]:
    """The data rows of the first table under a heading."""
    table: list[str] = []
    for line in markdown.split(heading, 1)[1].splitlines():
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    return table[2:]


async def _mock_v2_artifact(limit: int | None = None) -> dict[str, Any]:
    cases = load_cases(limit=limit)
    result = await run_eval(cases, extract_fn=build_mock_extract_fn(cases, SCHEMA_V2), schema=SCHEMA_V2)
    return result.to_dict()


def _core_doc(doc_id: str, correct: bool) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "status": "scored",
        "fields": {
            "vendor": _field("A", "A", correct=correct),
            "tax": _field(1.0, 1.0, correct=True),
            # A v2-only field: the core column never reads it.
            "payment_terms": _field("Net 30", "Net 60", correct=False),
        },
    }


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

        # Caught by checks and Caught by quotes both have nothing to replay.
        assert ops_row.count("n/a") == 2
        assert "| 100.0% | n/a | n/a |" in ops_row

    def test_ops_table_has_a_caught_by_quotes_column_after_the_checks_one(self):
        markdown = render_tables([_make_artifact()])

        ops_header = next(line for line in markdown.splitlines() if "Caught by review" in line)
        assert "| Caught by checks | Caught by quotes | Halluc. |" in ops_header
        assert "Caught by quotes is the part of that share" in markdown

    def test_caught_by_quotes_is_na_for_a_v1_run(self):
        artifact = _make_artifact(n_scored=1, per_doc=[_doc_with_wrong_tax()])

        ops_row = render_tables([artifact]).split("### Ops")[1].splitlines()[4]

        assert _ops_cell(ops_row, "Caught by checks") == "100.0%"
        assert _ops_cell(ops_row, "Caught by quotes") == "n/a"

    async def test_caught_by_quotes_is_the_quotes_share_for_a_v2_run(self):
        run = await _mock_v2_artifact()

        ops_row = render_tables([run]).split("### Ops")[1].splitlines()[4]

        assert _ops_cell(ops_row, "Caught by checks") == "50.0%"
        assert _ops_cell(ops_row, "Caught by quotes") == "50.0%"

    def test_summary_none_renders_na_row(self):
        artifact = _make_artifact(n_scored=0, n_errors=5, summary=None)

        markdown = render_tables([artifact])

        assert "0 +5 err" in markdown
        assert "n/a / n/a" in markdown


class TestCoreColumn:
    def test_it_follows_overall_in_the_accuracy_table(self):
        assert _ACCURACY_HEADER[3:5] == ("Overall", "Core (v1 fields, docs 001-025)")

    def test_it_is_the_share_of_core_v1_fields_the_run_got_right(self):
        docs = [_core_doc(CORE_DOC_IDS[0], True), _core_doc(CORE_DOC_IDS[1], False)]
        artifact = _make_artifact(prompt_version="extract_v2", n_scored=2, per_doc=docs)

        row = _table_rows(render_tables([artifact]), "### Accuracy")[0]

        # 2 docs x the 2 v1 fields they carry = 4 fields, 3 right; payment_terms
        # is not a v1 field and is wrong in both.
        assert core_counts(artifact) == (3, 4)
        assert _accuracy_cell(row, "Core (v1 fields, docs 001-025)") == "3/4 (75.0%)"

    def test_documents_outside_the_core_do_not_count(self):
        docs = [_core_doc(CORE_DOC_IDS[0], True), _core_doc("026-invoice-discount-shipping-two-taxes", False)]
        artifact = _make_artifact(n_scored=2, per_doc=docs)

        assert core_counts(artifact) == (2, 2)

    def test_errored_and_skipped_documents_do_not_count(self):
        docs = [
            _core_doc(CORE_DOC_IDS[0], True),
            {"doc_id": CORE_DOC_IDS[1], "status": "error"},
            {"doc_id": CORE_DOC_IDS[2], "status": "skipped_cost_cap"},
        ]

        assert core_counts(_make_artifact(per_doc=docs)) == (2, 2)

    def test_a_run_with_no_core_document_shows_na(self):
        artifact = _make_artifact(
            prompt_version="extract_v2", n_scored=1, per_doc=[_core_doc("029-receipt-card-number-pii", True)]
        )

        row = _table_rows(render_tables([artifact]), "### Accuracy")[0]

        assert core_counts(artifact) == (0, 0)
        assert _accuracy_cell(row, "Core (v1 fields, docs 001-025)") == "n/a"

    def test_the_committed_runs_core_is_computed_from_their_documents(self):
        # Recomputed here from the artifacts' own fields, not read from a table.
        runs = load_results()
        markdown = render_tables(runs)

        for run, row in zip(runs, _table_rows(markdown, "### Accuracy"), strict=True):
            correct = scored = 0
            for doc in run["per_doc"]:
                if doc["doc_id"] in CORE_DOC_IDS:
                    for field in doc["fields"].values():
                        scored += 1
                        correct += field["correct"]
            assert (correct, scored) == core_counts(run)
            assert (
                _accuracy_cell(row, "Core (v1 fields, docs 001-025)")
                == f"{correct}/{scored} ({correct / scored * 100:.1f}%)"
            )

    def test_the_committed_runs_core_values(self):
        runs = load_results()

        values = [(run["model"], run["n_scored"], core_counts(run)) for run in runs]

        assert values == [
            ("claude-sonnet-5", 3, (21, 21)),
            ("claude-sonnet-5", 25, (174, 175)),
            ("claude-haiku-4-5", 25, (167, 175)),
        ]
        # On the 25 documents the core is the run's own accuracy: nothing moved.
        for run in runs:
            assert core_counts(run)[0] / core_counts(run)[1] == run["summary"]["overall_accuracy"]

    async def test_a_v2_run_compares_with_a_v1_run_on_the_same_documents(self):
        cases = load_cases()
        v1 = (await run_eval(cases, extract_fn=build_mock_extract_fn(cases))).to_dict()
        v2 = await _mock_v2_artifact()

        # The same two mistakes in both runs, both on core documents (003, 007),
        # although the runs score 7 and 20 fields on 32 documents.
        assert core_counts(v1) == core_counts(v2) == (173, 175)
        assert v1["summary"]["overall_accuracy"] != v2["summary"]["overall_accuracy"]


class TestSchemaV2Table:
    def test_v1_only_results_render_no_such_section(self):
        markdown = render_tables(load_results())

        assert "Schema v2 fields" not in markdown
        assert markdown.count("### ") == 2

    async def test_a_v2_run_gets_a_table_with_a_row_per_field_and_a_total(self):
        run = await _mock_v2_artifact()

        markdown = render_tables([run])
        rows = _table_rows(markdown, "### Schema v2 fields")

        assert "### Schema v2 fields" in markdown
        assert "#### mock-model / extract_v2 (32 docs)" in markdown
        assert [row.split("|")[1].strip() for row in rows] == [*SCHEMA_V2.field_names, "All"]
        by_field = {row.split("|")[1].strip(): [c.strip() for c in row.strip().strip("|").split("|")] for row in rows}
        assert by_field["tax"][1:] == ["96.9%", "30", "100.0%", "100.0%", "100.0%"]
        assert by_field["po_number"][1:] == ["100.0%", "3", "100.0%", "100.0%", "100.0%"]
        assert by_field["line_items"][2] == "508"
        assert by_field["All"][1:] == ["99.7%", "813", "100.0%", "100.0%", "100.0%"]

    async def test_it_shows_a_coverage_miss_and_a_page_miss_apart(self):
        run = await _mock_v2_artifact(limit=1)
        # total: quoted, page unknown. subtotal: no quote at all.
        coverage = run["summary"]["evidence_coverage"]
        coverage["per_field"]["total"].update(page_agree=0, page_agreement=0.0)
        coverage["per_field"]["subtotal"].update(with_quote=0, coverage=0.0, page_agree=0, page_agreement=0.0)

        rows = _table_rows(render_tables([run]), "### Schema v2 fields")
        by_field = {row.split("|")[1].strip(): [c.strip() for c in row.strip().strip("|").split("|")] for row in rows}

        assert by_field["total"][3:5] == ["100.0%", "0.0%"]
        assert by_field["subtotal"][3:5] == ["0.0%", "0.0%"]

    async def test_the_table_is_part_of_latest_md(self):
        run = await _mock_v2_artifact(limit=2)

        assert "### Schema v2 fields" in render_latest_md([run])

    def test_a_run_without_a_summary_has_no_table(self):
        assert "Schema v2 fields" not in render_tables([_make_artifact(summary=None, n_scored=0)])


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
