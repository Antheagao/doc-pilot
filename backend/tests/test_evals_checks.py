"""Tests for app.evals.checks: caught_by_checks over the committed runs, and
the reconciliation (mutation) eval over the labels.

Pure/offline like the other eval-suite tests: no db_session, no network, no
model. The committed artifacts under evals/results/ are read and must come
back unchanged.
"""

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from app.evals import checks
from app.evals.checks import (
    CHECKS_END_MARKER,
    CHECKS_START_MARKER,
    MUTATION_KINDS,
    CheckMetrics,
    check_metrics,
    doc_routing,
    failing_labels,
    flagged_by_checks,
    label_tool_input,
    reconcile_label,
    render_checks_markdown,
    replay,
    run_mutation_eval,
)
from app.evals.dataset import EVALS_DIR, load_cases
from app.reconcile import RULES_VERSION
from app.review_routing import Routing

RESULTS_DIR = EVALS_DIR / "results"
README = EVALS_DIR.parent / "README.md"


def _artifact_path(pattern: str) -> Path:
    (path,) = sorted(RESULTS_DIR.glob(pattern))
    return path


def _artifact(pattern: str) -> dict:
    return json.loads(_artifact_path(pattern).read_text(encoding="utf-8"))


def _haiku() -> dict:
    return _artifact("*_claude-haiku-4-5_extract_v1.json")


def _sonnet_25() -> dict:
    # Two Sonnet runs are committed: the 3-document smoke and the full 25.
    for path in sorted(RESULTS_DIR.glob("*_claude-sonnet-5_extract_v1.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["n_scored"] == 25:
            return run
    raise AssertionError("no 25-document Sonnet artifact")


def _sonnet_3() -> dict:
    for path in sorted(RESULTS_DIR.glob("*_claude-sonnet-5_extract_v1.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["n_scored"] == 3:
            return run
    raise AssertionError("no 3-document Sonnet artifact")


def _caught_fields(run: dict) -> set[tuple[str, str]]:
    """The (doc_id, field) pairs of incorrect fields a failing check flagged."""
    caught = set()
    for doc in run["per_doc"]:
        routing = doc_routing(doc, run["review_threshold"])
        for name, field in doc["fields"].items():
            if not field["correct"] and flagged_by_checks(routing[name]):
                caught.add((doc["doc_id"], name))
    return caught


# --- caught by checks ---------------------------------------------------------


class TestCheckMetrics:
    def test_haiku_misses_are_caught_by_arithmetic_not_by_confidence(self):
        run = _haiku()

        metrics = check_metrics(run)

        assert metrics == CheckMetrics(
            rules_version=RULES_VERSION,
            n_docs=25,
            n_docs_failing=3,
            n_incorrect=8,
            n_caught_by_checks=4,
            n_correct_flagged=2,
        )
        assert metrics.caught_by_checks == 0.5
        # Confidence caught none of them (the Ops table's 0.0%).
        assert run["summary"]["caught_by_review"] == 0.0
        assert _caught_fields(run) == {
            ("018-dense-office-outfitters", "line_items"),
            ("019-dense-hardware-warehouse", "line_items"),
            ("019-dense-hardware-warehouse", "subtotal"),
            ("020-dense-electronics-order", "line_items"),
        }

    def test_a_failing_document_alone_does_not_count_as_caught(self):
        # 019 tax, total, and document_date are wrong in a document that fails
        # its checks, but no check names them.
        run = _haiku()
        routing = doc_routing(
            next(d for d in run["per_doc"] if d["doc_id"].startswith("019-")),
            run["review_threshold"],
        )

        assert routing["tax"].review_reason is None
        assert routing["total"].review_reason is None

    def test_sonnet_full_run_has_one_miss_and_no_check_to_catch_it(self):
        metrics = check_metrics(_sonnet_25())

        assert (metrics.n_incorrect, metrics.n_docs_failing) == (1, 0)
        assert metrics.caught_by_checks == 0.0

    def test_a_run_with_no_incorrect_field_has_nothing_to_catch(self):
        metrics = check_metrics(_sonnet_3())

        assert metrics.n_incorrect == 0
        assert metrics.caught_by_checks is None

    def test_a_run_without_a_summary_has_no_metrics(self):
        run = _haiku()
        run["summary"] = None

        assert check_metrics(run) is None

    def test_a_run_without_scored_documents_has_no_metrics(self):
        run = _haiku()
        run["per_doc"] = [{"doc_id": "x", "status": "error", "difficulty": "easy"}]

        assert check_metrics(run) is None

    def test_a_partial_field_set_does_not_crash(self):
        run = _haiku()
        run["per_doc"] = [
            {
                "doc_id": "partial",
                "status": "scored",
                "fields": {"tax": {"correct": False, "expected": 1.0, "actual": 1.5, "confidence": 0.9}},
            }
        ]

        metrics = check_metrics(run)

        assert metrics.n_docs == 1
        assert metrics.n_incorrect == 1

    def test_reading_a_run_leaves_it_and_its_file_unchanged(self):
        path = _artifact_path("*_claude-haiku-4-5_extract_v1.json")
        before_bytes = path.read_bytes()
        run = json.loads(before_bytes)
        snapshot = deepcopy(run)

        check_metrics(run)
        for doc in run["per_doc"]:
            doc_routing(doc, run["review_threshold"])

        assert run == snapshot
        assert path.read_bytes() == before_bytes


class TestReplay:
    def test_label_tool_input_turns_line_cells_into_leaves(self):
        fields = {
            "vendor": "Acme",
            "currency": "USD",
            "subtotal": 10.0,
            "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 5.0, "total": 10.0}],
        }

        tool_input = label_tool_input(fields, 0.7)

        assert tool_input["subtotal"] == {"value": 10.0, "confidence": 0.7}
        assert tool_input["line_items"]["value"][0]["quantity"] == {"value": 2, "confidence": 0.7}

    def test_a_low_confidence_field_with_a_passing_document_is_not_flagged_by_checks(self):
        report, routing = replay(label_tool_input(load_cases()[0].fields, 0.1), 0.8)

        assert report.status != "fail"
        assert routing["vendor"] == Routing(True, "confidence")
        assert not flagged_by_checks(routing["vendor"])

    def test_every_label_is_consistent(self):
        assert failing_labels(load_cases()) == []

    def test_a_broken_label_is_reported_with_its_failing_check(self):
        case = load_cases()[0]
        fields = {**case.fields, "total": float(Decimal(str(case.fields["total"])) + 1)}

        report = reconcile_label(fields)

        assert [c.id for c in report.failing_checks] == ["total_math"]


# --- reconciliation eval -------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated():
    return run_mutation_eval(load_cases())


class TestMutationEval:
    def test_it_is_deterministic(self, evaluated):
        assert run_mutation_eval(load_cases()) == evaluated

    def test_pinned_tallies(self, evaluated):
        got = {
            t.kind: (
                t.mutations,
                t.detected,
                t.flagged,
                t.fixed,
                t.wrong_fixes,
                t.other_fields_flagged,
            )
            for t in evaluated.by_kind
        }

        assert tuple(got) == MUTATION_KINDS
        assert got == {
            "one_minor_unit": (264, 264, 264, 120, 0, 98),
            "transposed_digits": (262, 262, 262, 119, 0, 98),
            "misread_digit": (359, 359, 359, 146, 0, 46),
            "dropped_point": (264, 264, 264, 169, 0, 0),
            "extra_leading_digit": (359, 359, 359, 154, 0, 30),
            "wrong_row": (120, 96, 96, 96, 0, 0),
        }
        overall = evaluated.overall
        assert (overall.mutations, overall.detected, overall.fixed) == (1628, 1604, 804)
        assert evaluated.n_labels == 25
        assert evaluated.rules_version == RULES_VERSION

    def test_the_engine_never_suggests_a_wrong_value(self, evaluated):
        assert all(t.wrong_fixes == 0 for t in evaluated.by_kind)

    def test_every_digit_misread_is_detected(self, evaluated):
        for tally in evaluated.by_kind:
            if tally.kind != "wrong_row":
                assert tally.detected == tally.mutations, tally.kind

    def test_the_only_misses_are_a_total_read_as_the_subtotal(self, evaluated):
        by_doc = {case.doc_id: case.fields for case in load_cases()}
        taxed = {
            doc_id
            for doc_id, fields in by_doc.items()
            if fields["tax"] and fields["subtotal"] is not None and fields["total"] is not None
        }

        assert {m.doc_id for m in evaluated.misses} <= taxed
        for miss in evaluated.misses:
            assert (miss.kind, miss.path) == ("wrong_row", "total")
            assert miss.mutated == Decimal(str(by_doc[miss.doc_id]["subtotal"]))
        assert len(evaluated.misses) == evaluated.overall.mutations - evaluated.overall.detected == 24

    def test_a_label_that_fails_unmutated_stops_the_eval(self):
        case = load_cases()[0]
        broken = type(case)(
            doc_id=case.doc_id,
            image_path=case.image_path,
            mime_type=case.mime_type,
            source=case.source,
            dataset_version=case.dataset_version,
            difficulty=case.difficulty,
            fields={**case.fields, "total": 999.0},
        )

        with pytest.raises(ValueError, match=case.doc_id):
            run_mutation_eval([broken])


class TestMutationKinds:
    """Each kind on a printed amount, through the same functions the eval runs."""

    mutators = checks._STRING_MUTATORS

    def test_examples_from_46_25(self):
        assert self.mutators["one_minor_unit"]("46.25", 2) == "46.26"
        assert self.mutators["transposed_digits"]("46.25", 2) == "46.52"
        assert self.mutators["misread_digit"]("46.25", 2) == "96.25"
        assert self.mutators["dropped_point"]("46.25", 2) == "4625"
        assert self.mutators["extra_leading_digit"]("46.25", 2) == "146.25"

    def test_digits_that_cannot_be_swapped_are_skipped(self):
        assert self.mutators["transposed_digits"]("11.11", 2) is None
        assert self.mutators["transposed_digits"]("2", 0) is None

    def test_a_whole_number_has_no_point_to_drop(self):
        assert self.mutators["dropped_point"]("2", 0) is None

    def test_a_zero_minor_unit_currency_gets_plus_one(self):
        assert self.mutators["one_minor_unit"]("1500", 0) == "1501"

    def test_a_sign_stays_in_front_of_the_extra_digit(self):
        assert self.mutators["extra_leading_digit"]("-4.00", 2) == "-14.00"

    def test_a_swap_reads_across_the_point(self):
        assert self.mutators["transposed_digits"]("10.05", 2) == "10.50"

    def test_quantities_are_never_bumped_by_a_minor_unit(self):
        assert all(
            m.kind != "one_minor_unit" or not m.path.endswith(".quantity")
            for m in checks._mutations(
                "x",
                {
                    "currency": "USD",
                    "subtotal": None,
                    "tax": None,
                    "total": None,
                    "line_items": [{"description": "a", "quantity": 2, "unit_price": 5.0, "total": 10.0}],
                },
            )
        )


# --- README --------------------------------------------------------------------


def test_readme_checks_table_is_current(evaluated):
    text = README.read_text(encoding="utf-8")
    start = text.index(CHECKS_START_MARKER) + len(CHECKS_START_MARKER)
    end = text.index(CHECKS_END_MARKER)

    assert text[start:end] == f"\n\n{render_checks_markdown(evaluated)}\n\n"
