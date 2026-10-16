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
    QUOTE_VARIANTS,
    QUOTES_END_MARKER,
    QUOTES_START_MARKER,
    CheckMetrics,
    Mutation,
    _broken_tool_input,
    _coupled_paths,
    _misread_quote,
    _mutations,
    _quoted_tool_input,
    _replace_number,
    check_metrics,
    doc_routing,
    failing_labels,
    flagged_by_checks,
    flagged_by_quotes,
    label_tool_input,
    reconcile_label,
    render_checks_markdown,
    render_quotes_markdown,
    replay,
    run_mutation_eval,
    run_quote_eval,
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
            "one_minor_unit": (371, 363, 363, 174, 0, 136),
            "transposed_digits": (383, 381, 381, 174, 0, 134),
            "misread_digit": (506, 505, 505, 205, 0, 72),
            "dropped_point": (375, 374, 374, 230, 0, 21),
            "extra_leading_digit": (506, 505, 505, 212, 0, 59),
            "wrong_row": (173, 147, 147, 143, 0, 11),
        }
        overall = evaluated.overall
        assert (overall.mutations, overall.detected, overall.flagged, overall.fixed) == (
            2314,
            2275,
            2275,
            1138,
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
            for mutation in _mutations(case.doc_id, case.fields, case.evidence):
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

    def test_every_digit_misread_is_detected_except_where_nothing_can_check_it(self, evaluated):
        # The blind spots: an amount paid on a document that prints no amount due
        # (029), and a tax line's base that moves by too little to change the tax
        # its rate gives.
        misses = [m for m in evaluated.misses if m.kind != "wrong_row"]
        missed = {
            (m.doc_id[:3], m.path.split("[")[0] + ("." + m.path.rsplit(".", 1)[1] if "[" in m.path else ""))
            for m in misses
        }
        assert len(misses) == 13
        assert missed == {
            ("026", "tax_lines.base"),
            ("029", "amount_paid"),
            ("030", "tax_lines.base"),
            ("031", "tax_lines.base"),
        }

    def test_the_wrong_row_misses_are_a_total_read_as_the_subtotal(self, evaluated):
        by_doc = {case.doc_id: case.fields for case in load_cases()}
        taxed = {
            doc_id
            for doc_id, fields in by_doc.items()
            if (fields["tax"] or fields["tax_lines"])
            and fields["subtotal"] is not None
            and fields["total"] is not None
        }
        wrong_row = [m for m in evaluated.misses if m.kind == "wrong_row"]

        assert {m.doc_id for m in wrong_row} <= taxed
        for miss in wrong_row:
            assert miss.path == "total"
            assert miss.mutated == Decimal(str(by_doc[miss.doc_id]["subtotal"]))
        assert len(wrong_row) == 26
        assert len(evaluated.misses) == evaluated.overall.mutations - evaluated.overall.detected == 39

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


# --- header evidence in a replay ------------------------------------------------


def _v2_cell(value, quote=None, confidence=1.0):
    return {"value": value, "confidence": confidence, "evidence": {"page": 1, "quote": quote} if quote else None}


def _doc_with_wrong_subtotal(stored_evidence: bool) -> dict:
    """A scored v2 document whose subtotal was read as 11.30 while the page
    prints 10.30: the lines and the total say so too, and so does the quote."""
    subtotal = {"correct": False, "expected": 10.3, "actual": 11.3, "confidence": 1.0}
    if stored_evidence:
        subtotal["evidence"] = {"page": 1, "quote": "Subtotal: $10.30"}
    row = {
        "description": _v2_cell("Widget"),
        "quantity": _v2_cell(1),
        "unit_price": _v2_cell(10.3),
        "total": _v2_cell(10.3),
    }
    return {
        "status": "scored",
        "fields": {
            "currency": {"correct": True, "actual": "USD", "confidence": 1.0},
            "line_items": {"correct": True, "actual": [row], "confidence": 1.0},
            "subtotal": subtotal,
            "total": {"correct": True, "actual": 10.3, "confidence": 1.0},
        },
    }


class TestReplayedHeaderEvidence:
    def test_a_header_field_brings_its_stored_quote_to_the_replay(self):
        routing = doc_routing(_doc_with_wrong_subtotal(True), 0.8, SCHEMA_V2)

        assert routing["subtotal"].review_reason == "quote:subtotal"
        assert flagged_by_quotes(routing["subtotal"])

    def test_without_the_stored_quote_only_the_arithmetic_can_flag_it(self):
        routing = doc_routing(_doc_with_wrong_subtotal(False), 0.8, SCHEMA_V2)

        assert routing["subtotal"].review_reason is not None
        assert routing["subtotal"].review_reason.startswith("reconciliation:")
        assert not flagged_by_quotes(routing["subtotal"])

    def test_check_metrics_counts_the_quote_catch_apart_from_the_arithmetic(self):
        def run(stored_evidence):
            return {
                "summary": {},
                "prompt_version": "extract_v2",
                "review_threshold": 0.8,
                "per_doc": [_doc_with_wrong_subtotal(stored_evidence)],
            }

        with_quote = check_metrics(run(True))
        without = check_metrics(run(False))

        assert (with_quote.n_incorrect, with_quote.n_caught_by_checks, with_quote.n_caught_by_quotes) == (1, 1, 1)
        assert with_quote.caught_by_quotes == 1.0
        assert (without.n_caught_by_checks, without.n_caught_by_quotes) == (1, 0)
        assert without.caught_by_quotes == 0.0

    def test_a_v1_run_has_no_quotes_to_catch_with(self):
        metrics = check_metrics(_haiku())

        assert metrics.quotes_checkable is False
        assert metrics.caught_by_quotes is None


# --- the mutation eval's cells ---------------------------------------------------


def _case(prefix):
    return next(case for case in load_cases() if case.doc_id.startswith(prefix))


class TestMutationCells:
    def test_the_new_money_fields_and_tax_line_cells_are_broken(self):
        case = _case("026")

        paths = {m.path for m in _mutations(case.doc_id, case.fields, case.evidence)}

        assert {"discount", "shipping"} <= paths
        for cell in ("rate", "base", "amount"):
            assert {f"tax_lines[0].{cell}", f"tax_lines[1].{cell}"} <= paths
        assert {m.path for m in _mutations(_case("027").doc_id, _case("027").fields)} >= {"tip"}
        paths_028 = {m.path for m in _mutations(_case("028").doc_id, _case("028").fields)}
        assert {"amount_paid", "amount_due"} <= paths_028

    def test_a_tax_printed_once_is_one_cell_not_two(self):
        # 001 reads its tax into the tax and into its only tax line; breaking the
        # line's amount too would replay the document the tax's mutation made.
        case = _case("001")
        with_twins = {m.path for m in _mutations(case.doc_id, case.fields)}
        without_twins = {m.path for m in _mutations(case.doc_id, case.fields, case.evidence)}

        assert "tax_lines[0].amount" in with_twins
        assert "tax_lines[0].amount" not in without_twins
        assert without_twins == with_twins - {"tax_lines[0].amount"}

    def test_a_tax_printed_twice_keeps_its_tax_line_amounts(self):
        case = _case("026")

        paths = {m.path for m in _mutations(case.doc_id, case.fields, case.evidence)}

        assert {"tax", "tax_lines[0].amount", "tax_lines[1].amount"} <= paths

    def test_wrong_row_covers_the_balance_and_the_tax_base(self):
        by_path = {}
        for prefix in ("028", "026"):
            case = _case(prefix)
            for m in _mutations(case.doc_id, case.fields, case.evidence):
                if m.kind == "wrong_row":
                    by_path.setdefault((prefix, m.path), set()).add(m.mutated)

        fields_028 = _case("028").fields
        total, due = Decimal(str(fields_028["total"])), Decimal(str(fields_028["amount_due"]))
        # The total read as the balance due, and the balance due as the total.
        assert due in by_path[("028", "total")]
        assert total in by_path[("028", "amount_due")]
        # A tax amount read as the base it is charged on.
        assert Decimal("454.48") in by_path[("026", "tax_lines[0].amount")]

    def test_a_wrong_row_that_reads_the_same_number_is_no_mutation(self):
        # 032's balance due equals its total: reading one as the other changes nothing.
        case = _case("032")

        mutations = _mutations(case.doc_id, case.fields, case.evidence)

        assert not [m for m in mutations if m.kind == "wrong_row" and m.mutated == m.original]
        assert case.fields["amount_due"] == case.fields["total"]
        assert {m.mutated for m in mutations if m.kind == "wrong_row" and m.path == "total"} == {
            Decimal(str(case.fields["subtotal"]))
        }


# --- quote eval ------------------------------------------------------------------


@pytest.fixture(scope="module")
def quoted():
    return run_quote_eval(load_cases())


def _tallies(quoted):
    return {t.variant: t for t in quoted.by_variant}


class TestQuoteEval:
    def test_it_replays_the_mutation_evals_documents_in_four_ways(self, quoted, evaluated):
        assert tuple(t.variant for t in quoted.by_variant) == QUOTE_VARIANTS
        assert quoted.n_labels == 32
        assert quoted.rules_version == RULES_VERSION
        assert {t.mutations for t in quoted.by_variant} == {evaluated.overall.mutations}

    def test_pinned_tallies(self, quoted):
        got = {
            t.variant: (t.mutations, t.detected, t.quote_flagged, t.flagged, t.fixed, t.wrong_fixes)
            for t in quoted.by_variant
        }

        assert got == {
            "none": (2314, 2275, 0, 2275, 1138, 0),
            "faithful": (2314, 2314, 2196, 2314, 1352, 0),
            "same_misread": (2314, 2275, 0, 2275, 1138, 0),
            "missing": (2314, 2275, 0, 2275, 1138, 0),
        }

    def test_a_mutation_eval_run_beforehand_is_reused_not_replayed_and_gives_the_same_table(self, quoted, evaluated, monkeypatch):
        calls = []
        monkeypatch.setattr(checks, "run_mutation_eval", lambda cases: calls.append(cases))

        again = run_quote_eval(load_cases(), evaluated)

        assert again == quoted
        assert calls == []

    def test_with_no_quotes_it_is_the_mutation_eval(self, quoted, evaluated):
        none = _tallies(quoted)["none"]
        overall = evaluated.overall

        assert (none.detected, none.flagged, none.fixed, none.wrong_fixes) == (
            overall.detected,
            overall.flagged,
            overall.fixed,
            overall.wrong_fixes,
        )

    def test_a_misread_whose_quote_shows_what_is_printed_is_always_detected(self, quoted):
        faithful = _tallies(quoted)["faithful"]

        assert faithful.detected == faithful.flagged == faithful.mutations
        # The rest are caught by the arithmetic, not a quote: the misread value is
        # another number the same quoted line prints, or the cell has no quote.
        assert 0 < faithful.mutations - faithful.quote_flagged < 200
        assert faithful.fixed > _tallies(quoted)["none"].fixed

    def test_a_value_and_quote_misread_alike_fall_back_to_the_arithmetic_exactly(self, quoted):
        tallies = _tallies(quoted)

        for variant in ("same_misread", "missing"):
            assert tallies[variant].quote_flagged == 0
            assert tallies[variant].detected == tallies["none"].detected
            assert tallies[variant].fixed == tallies["none"].fixed

    def test_no_way_of_writing_quotes_causes_a_wrong_fix(self, quoted):
        assert [t.wrong_fixes for t in quoted.by_variant] == [0, 0, 0, 0]

    def test_a_label_that_fails_unmutated_stops_the_quote_eval(self):
        case = load_cases()[0]
        broken = replace(case, fields={**case.fields, "total": 999.0})

        with pytest.raises(ValueError, match=case.doc_id):
            run_quote_eval([broken])


def _total_mutation(prefix="001"):
    case = _case(prefix)
    mutation = next(
        m for m in _mutations(case.doc_id, case.fields, case.evidence) if m.path == "total" and m.kind == "misread_digit"
    )
    return case, mutation


class TestQuoteVariants:
    """The two cases the value-quote check was specified with."""

    def test_change_a_value_and_keep_its_quote_and_the_quote_check_fails(self):
        case, mutation = _total_mutation()

        tool = _quoted_tool_input(case.fields, mutation, case.evidence, "faithful")
        report, routing = replay(tool, 1.0, SCHEMA_V2)

        assert tool["total"]["value"] == mutation.mutated
        assert tool["total"]["evidence"] == case.evidence["total"]
        assert "quote:total" in [check.id for check in report.failing_checks]
        assert routing["total"].review_reason == "quote:total"

    def test_change_both_together_and_the_quote_check_is_silent(self):
        case, mutation = _total_mutation()

        tool = _quoted_tool_input(case.fields, mutation, case.evidence, "same_misread")
        report, routing = replay(tool, 1.0, SCHEMA_V2)

        assert tool["total"]["evidence"]["quote"] != case.evidence["total"]["quote"]
        assert tool["total"]["evidence"]["quote"].startswith("Total: $")
        assert [c.id for c in report.failing_checks if c.kind == "quote"] == []
        # Only the arithmetic notices.
        assert [c.id for c in report.failing_checks] == ["total_math"]
        assert routing["total"].review_reason == "reconciliation:total_math"

    def test_a_missing_quote_leaves_the_broken_cell_unchecked_and_the_rest_quoted(self):
        case, mutation = _total_mutation()

        tool = _quoted_tool_input(case.fields, mutation, case.evidence, "missing")

        assert tool["total"]["evidence"] is None
        assert tool["subtotal"]["evidence"] == case.evidence["subtotal"]
        assert replay(tool, 1.0, SCHEMA_V2)[1]["total"].review_reason == "reconciliation:total_math"

    def test_a_number_printed_once_has_its_quote_broken_in_both_places(self):
        case = _case("001")
        tax = next(m for m in _mutations(case.doc_id, case.fields, case.evidence) if m.path == "tax")

        tool = _quoted_tool_input(case.fields, tax, case.evidence, "same_misread")

        shown = tool["tax"]["evidence"]["quote"]
        assert shown == tool["tax_lines"]["value"][0]["amount"]["evidence"]["quote"]
        assert to_api_string(tax.mutated, 2) in shown
        assert to_api_string(tax.original, 2) not in shown

    def test_every_misread_can_be_written_into_its_quote(self):
        # A quote that cannot be rewritten would silently turn "misread alike"
        # into "missing", so every cell the eval breaks must be one it can write.
        for case in load_cases():
            minor_units = currency_precision(case.fields.get("currency")).minor_units
            for mutation in _mutations(case.doc_id, case.fields, case.evidence):
                cited = case.evidence.get(mutation.path)
                if cited is None:
                    continue
                money = not mutation.path.endswith((".quantity", ".rate"))
                rewritten = _misread_quote(cited, mutation, minor_units, money)
                assert rewritten is not None, (case.doc_id, mutation)
                assert rewritten["quote"] != cited["quote"]
                assert rewritten["page"] == cited["page"]

    def test_the_gold_evidence_is_never_changed(self):
        case, mutation = _total_mutation("032")
        before = deepcopy(case.evidence)

        for variant in QUOTE_VARIANTS:
            _quoted_tool_input(case.fields, mutation, case.evidence, variant)

        assert case.evidence == before


class TestWritingAMisreadQuote:
    def test_a_number_is_replaced_in_the_style_it_is_printed_in(self):
        cited = {"page": 1, "quote": "Total: $2,226.78"}
        mutation = Mutation("032", "total", "misread_digit", Decimal("2226.78"), Decimal("7226.78"))

        assert _misread_quote(cited, mutation, 2, True) == {"page": 1, "quote": "Total: $7,226.78"}

    def test_a_decimal_comma_is_kept(self):
        cited = {"page": 1, "quote": "VAT 19% on 125,19 EUR"}
        mutation = Mutation("030", "tax_lines[0].base", "one_minor_unit", Decimal("125.19"), Decimal("125.20"))

        assert _misread_quote(cited, mutation, 2, True)["quote"] == "VAT 19% on 125,20 EUR"

    def test_european_grouping_is_kept(self):
        cited = {"page": 1, "quote": "Summe 2.226,78 EUR"}
        mutation = Mutation("x", "total", "misread_digit", Decimal("2226.78"), Decimal("7226.78"))

        assert _misread_quote(cited, mutation, 2, True)["quote"] == "Summe 7.226,78 EUR"

    def test_only_a_whole_number_is_replaced(self):
        # The quantity 2 is a token of its own; the 2 in 12.25 and 20.00 are not.
        assert _replace_number("Widget   2   $12.25   $20.00", "2", "7") == "Widget   7   $12.25   $20.00"

    def test_a_number_not_printed_in_the_quote_gives_no_quote(self):
        mutation = Mutation("x", "total", "misread_digit", Decimal("46.25"), Decimal("96.25"))

        assert _misread_quote({"page": 1, "quote": "Total due on receipt"}, mutation, 2, True) is None
        assert _misread_quote(None, mutation, 2, True) is None


# --- README --------------------------------------------------------------------


def _marked(text, start, end):
    first = text.index(start) + len(start)
    return text[first : text.index(end)]


def test_readme_checks_table_is_current(evaluated):
    text = README.read_text(encoding="utf-8")

    assert _marked(text, CHECKS_START_MARKER, CHECKS_END_MARKER) == f"\n\n{render_checks_markdown(evaluated)}\n\n"


def test_readme_quotes_table_is_current(quoted):
    text = README.read_text(encoding="utf-8")

    assert _marked(text, QUOTES_START_MARKER, QUOTES_END_MARKER) == f"\n\n{render_quotes_markdown(quoted)}\n\n"
