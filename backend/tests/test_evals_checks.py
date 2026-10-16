"""Tests for app.evals.checks: caught_by_checks over the committed runs, and
the reconciliation (mutation) eval over the labels.

Pure/offline like the other eval-suite tests: no db_session, no network, no
model. The committed artifacts under evals/results/ are read and must come
back unchanged.
"""

import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from app import reconcile as reconcile_module
from app.evals import checks
from app.evals.checks import (
    CHECKS_END_MARKER,
    CHECKS_START_MARKER,
    MUTATION_KINDS,
    CheckMetrics,
    _broken_tool_input,
    _coupled_paths,
    _mutations,
    check_metrics,
    doc_routing,
    failing_labels,
    flagged_by_checks,
    flagged_by_quotes,
    label_tool_input,
    reconcile_label,
    render_checks_markdown,
    replay,
    run_mutation_eval,
)
from app.evals.dataset import EVALS_DIR, load_cases
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2
from app.money import currency_precision, to_api_string
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
        routing = doc_routing(doc, run["review_threshold"], SCHEMA_V1)
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
        # The committed answers carry no evidence, so no quote check ran.
        assert (metrics.n_caught_by_quotes, metrics.n_correct_flagged_by_quotes) == (0, 0)
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
            SCHEMA_V1,
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

    def test_an_incorrect_line_cell_whose_quote_disagrees_is_caught_by_the_quote(self):
        # Line item cells are stored as leaves, so an artifact that kept the
        # model's evidence replays it; one misread line total here.
        def cell(value, quote):
            return {"value": value, "confidence": 1.0, "evidence": {"page": 1, "quote": quote}}

        lines = [
            {
                "description": {"value": "a", "confidence": 1.0},
                "quantity": cell(1, "1"),
                "unit_price": cell(9.65, "9.65"),
                "total": cell(9.65, "9.56"),
            }
        ]
        run = {
            "summary": {},
            "prompt_version": "extract_v1",
            "review_threshold": 0.8,
            "per_doc": [
                {
                    "status": "scored",
                    "fields": {
                        "currency": {"actual": "USD", "confidence": 1.0, "correct": True},
                        "line_items": {"actual": lines, "confidence": 1.0, "correct": False},
                    },
                }
            ],
        }

        metrics = check_metrics(run)

        assert (metrics.n_incorrect, metrics.n_caught_by_checks, metrics.n_caught_by_quotes) == (1, 1, 1)
        assert (metrics.n_correct_flagged, metrics.n_correct_flagged_by_quotes) == (0, 0)
        assert metrics.caught_by_checks == 1.0

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
            doc_routing(doc, run["review_threshold"], SCHEMA_V1)

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

    def test_label_tool_input_wraps_tax_line_cells_too(self):
        fields = {
            "tax": 0.8,
            "tax_lines": [{"label": "Tax", "rate": 8.0, "base": 10.0, "amount": 0.8}],
        }

        tool_input = label_tool_input(fields, 0.7)

        assert tool_input["tax_lines"]["value"][0]["rate"] == {"value": 8.0, "confidence": 0.7}
        assert tool_input["tax_lines"]["value"][0]["label"] == {"value": "Tax", "confidence": 0.7}
        assert "evidence" not in tool_input["tax"]

    def test_gold_evidence_becomes_the_evidence_of_each_leaf_and_null_where_there_is_none(self):
        fields = {
            "subtotal": 10.0,
            "tax_lines": [{"label": "Tax", "rate": None, "base": None, "amount": 0.8}],
        }
        gold = {
            "subtotal": {"page": 1, "quote": "Subtotal $10.00"},
            "tax_lines[0].amount": {"page": 2, "quote": "Tax $0.80"},
        }

        tool_input = label_tool_input(fields, evidence=gold)

        assert tool_input["subtotal"]["evidence"] == gold["subtotal"]
        cells = tool_input["tax_lines"]["value"][0]
        assert cells["amount"]["evidence"] == gold["tax_lines[0].amount"]
        assert cells["label"]["evidence"] is None
        assert "evidence" not in tool_input["tax_lines"]

    def test_a_gold_quote_that_shows_another_number_fails_the_label(self):
        case = next(case for case in load_cases() if case.doc_id.startswith("026"))
        wrong = {**case.evidence, "discount": {"page": 1, "quote": "Volume discount (5%): -$23.29"}}

        report = reconcile_label(case.fields, wrong)

        assert [check.id for check in report.failing_checks] == ["quote:discount"]
        assert reconcile_label(case.fields, case.evidence).status == "pass"
        assert reconcile_label(case.fields).status == "pass"

    def test_a_low_confidence_field_with_a_passing_document_is_not_flagged_by_checks(self):
        report, routing = replay(label_tool_input(load_cases()[0].fields, 0.1), 0.8, SCHEMA_V1)

        assert report.status != "fail"
        assert routing["vendor"] == Routing(True, "confidence")
        assert not flagged_by_checks(routing["vendor"])

    @pytest.mark.parametrize(
        ("reason", "by_checks", "by_quotes"),
        [
            ("reconciliation:total_math", True, False),
            ("quote:subtotal", True, True),
            ("quote:line_items[2].total", True, True),
            ("confidence", False, False),
            (None, False, False),
        ],
    )
    def test_both_deterministic_reasons_count_as_checks_and_only_quotes_as_quotes(
        self, reason, by_checks, by_quotes
    ):
        routing = Routing(reason is not None, reason)

        assert flagged_by_checks(routing) is by_checks
        assert flagged_by_quotes(routing) is by_quotes

    def test_every_label_is_consistent(self):
        assert failing_labels(load_cases()) == []

    def test_a_broken_label_is_reported_with_its_failing_check(self):
        case = load_cases()[0]
        fields = {**case.fields, "total": float(Decimal(str(case.fields["total"])) + 1)}

        report = reconcile_label(fields)

        assert [c.id for c in report.failing_checks] == ["total_math"]


# --- reconciliation eval -------------------------------------------------------


def _put_evidence(tool_input, path, evidence):
    """Set the evidence of the leaf at a path (a header field or one row cell)."""
    if "[" not in path:
        if isinstance(tool_input.get(path), dict):
            tool_input[path]["evidence"] = evidence
        return
    head, key = path.rsplit(".", 1)
    name, index = head.removesuffix("]").split("[")
    tool_input[name]["value"][int(index)][key]["evidence"] = evidence


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
            "one_minor_unit": (348, 348, 348, 165, 0, 118),
            "transposed_digits": (355, 355, 355, 165, 0, 115),
            "misread_digit": (475, 475, 475, 195, 0, 55),
            "dropped_point": (348, 348, 348, 219, 0, 7),
            "extra_leading_digit": (475, 475, 475, 202, 0, 42),
            "wrong_row": (164, 138, 138, 135, 0, 9),
        }
        overall = evaluated.overall
        assert (overall.mutations, overall.detected, overall.flagged, overall.fixed) == (
            2165,
            2139,
            2139,
            1081,
        )
        assert overall.wrong_fixes == 0
        assert evaluated.n_labels == 32
        assert evaluated.rules_version == RULES_VERSION

    def test_a_misread_that_reaches_two_fields_is_never_a_wrong_fix(self, evaluated, monkeypatch):
        # Broken in one cell only, as if the two readings of a once-printed tax
        # were independent, the engine never suggests a wrong value.
        uncoupled = [replace(case, evidence={}) for case in load_cases()]

        assert all(t.wrong_fixes == 0 for t in run_mutation_eval(uncoupled).by_kind)
        # Broken in both (the tax and its only tax line), the two still agree. Taken
        # one at a time, the total is the one number that disagrees and was blamed
        # (87 wrong fixes); as one candidate they are as ambiguous as the tax and
        # the total of a document with no tax lines.
        assert all(t.wrong_fixes == 0 for t in evaluated.by_kind)
        monkeypatch.setattr(reconcile_module, "_tax_is_one_reading", lambda inp, report: False)
        assert run_mutation_eval(load_cases()).overall.wrong_fixes == 87

    @pytest.mark.parametrize(
        "perturb",
        [
            None,
            lambda quote: quote.split(":", 1)[-1].strip(),
            lambda quote: quote.replace(" ", "  ", 1),
            lambda quote: quote.replace(":", "", 1),
            lambda quote: quote + " ",
        ],
        ids=["as-gold", "label-stripped", "doubled-space", "colon-dropped", "trailing-space"],
    )
    def test_a_misread_with_quotes_in_front_of_the_engine_is_never_a_wrong_fix(self, perturb):
        # The eval above replays no quotes, so it cannot see how the engine
        # decides that a tax and its only tax line are one reading when both cite
        # evidence. Here every broken document carries the gold evidence, the quote
        # of each broken cell is rewritten the way its value was (a same-digit
        # misread), and the tax line's quote is then written as another reader
        # might: the pairing must not depend on the words. The gold quotes are
        # replayed against every mutation; the rewordings against the tax's, the
        # only cell with a twin.
        wrong = fixed = 0
        for case in load_cases():
            minor_units = currency_precision(case.fields.get("currency")).minor_units
            for mutation in _mutations(case.doc_id, case.fields):
                if perturb is not None and mutation.path != "tax":
                    continue
                tool = _broken_tool_input(case.fields, mutation, case.evidence)
                broken = {mutation.path, *_coupled_paths(case.fields, case.evidence, mutation)}
                before = to_api_string(mutation.original, minor_units)
                after = to_api_string(mutation.mutated, minor_units)
                for path, cited in case.evidence.items():
                    quote = cited["quote"]
                    if path in broken:
                        if before not in quote:
                            continue  # printed in another style: no quote rather than a false one
                        quote = quote.replace(before, after)
                    if path == "tax_lines[0].amount" and perturb is not None:
                        quote = perturb(quote)
                    _put_evidence(tool, path, {"page": cited["page"], "quote": quote})
                report = replay(tool, 1.0, SCHEMA_V2)[0]
                suggestion = report.suggestion
                if suggestion is None:
                    continue
                if (suggestion.path, suggestion.value) == (mutation.path, mutation.original):
                    fixed += 1
                else:
                    wrong += 1

        assert wrong == 0
        assert fixed > 0

    def test_a_number_printed_once_is_broken_everywhere_it_was_read(self):
        case = next(case for case in load_cases() if case.doc_id.startswith("001"))
        tax = next(m for m in _mutations(case.doc_id, case.fields) if m.path == "tax")

        broken = _broken_tool_input(case.fields, tax, case.evidence)

        mutated = {"value": tax.mutated, "confidence": 1.0}
        assert broken["tax"] == mutated
        assert broken["tax_lines"]["value"][0]["amount"] == mutated
        # A line's unit price and total are two printed numbers, even when equal.
        line = next(m for m in _mutations(case.doc_id, case.fields) if m.path == "line_items[0].unit_price")
        assert _coupled_paths(case.fields, case.evidence, line) == []

    def test_a_tax_printed_twice_is_broken_in_one_cell(self):
        # 026 prints each tax line and a total-tax line: two printed numbers.
        case = next(case for case in load_cases() if case.doc_id.startswith("026"))
        tax = next(m for m in _mutations(case.doc_id, case.fields) if m.path == "tax")

        assert _coupled_paths(case.fields, case.evidence, tax) == []

    def test_every_digit_misread_is_detected(self, evaluated):
        for tally in evaluated.by_kind:
            if tally.kind != "wrong_row":
                assert tally.detected == tally.mutations, tally.kind

    def test_the_only_misses_are_a_total_read_as_the_subtotal(self, evaluated):
        by_doc = {case.doc_id: case.fields for case in load_cases()}
        taxed = {
            doc_id
            for doc_id, fields in by_doc.items()
            if (fields["tax"] or fields["tax_lines"])
            and fields["subtotal"] is not None
            and fields["total"] is not None
        }

        assert {m.doc_id for m in evaluated.misses} <= taxed
        for miss in evaluated.misses:
            assert (miss.kind, miss.path) == ("wrong_row", "total")
            assert miss.mutated == Decimal(str(by_doc[miss.doc_id]["subtotal"]))
        assert len(evaluated.misses) == evaluated.overall.mutations - evaluated.overall.detected == 26

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
