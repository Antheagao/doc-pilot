"""Tests for app.evals.dataset and app.evals.scoring.

Deliberately has no db_session fixture anywhere -- these modules are pure
functions over dicts/dataclasses and must stay importable/runnable with
Postgres stopped (see the "no DB" verification step for E2).
"""

import json
from pathlib import Path

import pytest

import app.evals.dataset as dataset_module
from app.evals.checks import label_tool_input
from app.evals.dataset import (
    CORE_DOC_IDS,
    FIELD_KEYS,
    EvalCase,
    _validate_document_date,
    _validate_line_items,
    _validate_numeric,
    _validate_string_or_null,
    _validate_vendor,
    load_cases,
)
from app.evals.report import load_results
from app.evals.scoring import (
    CaseScore,
    EvidenceCounts,
    FieldScore,
    aggregate,
    match_amount,
    match_card_last4,
    match_currency,
    match_date,
    match_identifier,
    match_line_items,
    match_rate,
    match_tax_lines,
    match_terms,
    match_vendor,
    score_case,
)
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2, schema_for


def _leaf(value, confidence: float = 0.9) -> dict:
    return {"value": value, "confidence": confidence}


def _case(**field_overrides) -> EvalCase:
    fields = {
        "vendor": "Acme Corp",
        "document_date": "2026-01-01",
        "currency": "USD",
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
        "line_items": [
            {"description": "Widget", "quantity": 2, "unit_price": 5.0, "total": 10.0}
        ],
    }
    fields.update(field_overrides)
    return EvalCase(
        doc_id="test-case",
        image_path=Path("unused.png"),
        mime_type="image/png",
        source="synthetic",
        dataset_version="v1",
        difficulty="clean",
        fields=fields,
    )


def _base_tool_input() -> dict:
    return {
        "vendor": _leaf("Acme Corp"),
        "document_date": _leaf("2026-01-01"),
        "currency": _leaf("USD"),
        "subtotal": _leaf(10.0),
        "tax": _leaf(1.0),
        "total": _leaf(11.0),
        "line_items": _leaf(
            [
                {
                    "description": _leaf("Widget"),
                    "quantity": _leaf(2),
                    "unit_price": _leaf(5.0),
                    "total": _leaf(10.0),
                }
            ]
        ),
    }


class TestMatchDate:
    def test_iso_exact(self):
        assert match_date("2026-03-05", "2026-03-05") is True

    def test_iso_mismatch(self):
        assert match_date("2026-03-05", "2026-03-06") is False

    def test_mmddyyyy(self):
        assert match_date("2026-01-31", "01/31/2026") is True

    def test_ddmmyyyy_swap_when_month_component_gt_12(self):
        # 25 can't be a month, so the MM/DD-first guess is rejected and
        # the components are swapped -> DD/MM/YYYY -> Jan 25, 2026.
        assert match_date("2026-01-25", "25/01/2026") is True

    def test_yyyymmdd_slash(self):
        assert match_date("2026-05-09", "2026/05/09") is True

    def test_genuinely_ambiguous_date_reads_as_mmddyyyy(self):
        # 03/05/2026: both components are <= 12, so this is genuinely
        # ambiguous between March 5 and May 3. The documented rule (US
        # convention tried first) reads it as March 5.
        assert match_date("2026-03-05", "03/05/2026") is True
        assert match_date("2026-05-03", "03/05/2026") is False

    def test_unparseable_string(self):
        assert match_date("2026-03-05", "not a date") is False

    def test_non_string_input_does_not_raise(self):
        assert match_date("2026-03-05", None) is False
        assert match_date("2026-03-05", 20260305) is False


class TestMatchAmount:
    def test_one_cent_apart_pairs_pass_despite_float_imprecision(self):
        # `abs(expected - actual) <= 0.01` is float-broken exactly at
        # this boundary: 19.99 - 20.00 == -0.010000000000001563 in
        # binary float64, which fails a naive `<= 0.01` even though the
        # two amounts are one cent apart and should pass. These pairs
        # (plus 10.00/10.01, which happened to land on the passing side
        # of the old bug and so pinned nothing) previously scored WRONG.
        for expected, actual in [
            (19.99, 20.00),
            (1.13, 1.14),
            (0.03, 0.04),
            (99.99, 100.00),
            (10.00, 10.01),
        ]:
            matched, coerced = match_amount(expected, actual)
            assert matched is True, f"{expected} vs {actual} should be within tolerance"
            assert coerced is False

    def test_two_cents_apart_fails(self):
        matched, _coerced = match_amount(19.99, 20.01)
        assert matched is False

    def test_within_tolerance_lower_boundary_passes(self):
        matched, _coerced = match_amount(10.00, 9.99)
        assert matched is True

    def test_dollar_sign_string_is_not_an_amount(self):
        # The product's parser (money.to_decimal) refuses "$17.44", so the
        # eval scores it wrong rather than guessing what the model meant.
        matched, coerced = match_amount(17.44, "$17.44")
        assert matched is False
        assert coerced is True

    def test_thousands_comma_string_is_not_an_amount(self):
        matched, coerced = match_amount(1234.50, "1,234.50")
        assert matched is False
        assert coerced is True

    def test_plain_decimal_string_matches_and_is_coerced(self):
        matched, coerced = match_amount(17.44, "17.44")
        assert matched is True
        assert coerced is True

    def test_float_sum_noise_does_not_matter(self):
        # 0.1 + 0.2 is 0.30000000000000004 in binary float; both sides are
        # read as the shortest decimal that round-trips.
        matched, _coerced = match_amount(0.3, 0.1 + 0.2)
        assert matched is True

    def test_tolerance_is_exactly_one_cent(self):
        matched, _coerced = match_amount(10.00, 10.014)
        assert matched is False

    def test_unparseable_string_is_coerced_and_fails(self):
        matched, coerced = match_amount(10.0, "not a number")
        assert matched is False
        assert coerced is True

    def test_plain_number_is_not_coerced(self):
        matched, coerced = match_amount(10.0, 10.0)
        assert matched is True
        assert coerced is False


class TestMatchVendor:
    def test_exact(self):
        assert match_vendor("Acme Corp", "Acme Corp") is True

    def test_case_and_punctuation_insensitive(self):
        assert match_vendor("Acme Corp.", "ACME CORP") is True

    def test_near_match_passes(self):
        assert match_vendor("Cascade Coffee Roasters", "Cascade Coffee Roaster") is True

    def test_different_vendor_fails(self):
        assert match_vendor("Cascade Coffee Roasters", "Green Valley Grocery") is False

    def test_non_string_input_does_not_raise(self):
        assert match_vendor("Acme Corp", None) is False

    def test_ratio_exactly_at_threshold_passes(self):
        # difflib ratio == 0.85 exactly -- pins the `>=` in _fuzzy_match:
        # a `>` mutant would flip this case to False.
        expected = "abcdefghijklmnopqrstuvwxyz0123456789abcd"
        actual = "abcdefghijklmnopqrstuvwxyz01234567ZYXWVU"
        assert match_vendor(expected, actual) is True

    def test_ratio_just_below_threshold_fails(self):
        expected = "abcdefghijklmnopqrstuvwxyz0123456789abcd"
        actual = "abcdefghijklmnopqrstuvwxyz0123456ZYXWVUT"  # ratio == 0.825
        assert match_vendor(expected, actual) is False

    def test_ratio_just_above_threshold_passes(self):
        expected = "abcdefghijklmnopqrstuvwxyz0123456789abcd"
        actual = "abcdefghijklmnopqrstuvwxyz012345678ZYXWV"  # ratio == 0.875
        assert match_vendor(expected, actual) is True

    def test_punctuation_only_gold_does_not_vacuously_match(self):
        # Previously, two punctuation-only strings both normalized to ""
        # and difflib's convention (ratio() == 1.0 for two empty
        # sequences) let this vacuously "match". A non-null gold that
        # normalizes to nothing is now always a non-match.
        assert match_vendor("***", "!!!") is False
        assert match_vendor("***", "Acme Corp") is False


class TestMatchCurrency:
    def test_exact(self):
        assert match_currency("USD", "USD") is True

    def test_case_and_whitespace_insensitive(self):
        assert match_currency("usd", " USD ") is True

    def test_mismatch(self):
        assert match_currency("USD", "EUR") is False


class TestMatchLineItems:
    def test_both_null_or_empty_is_full_credit(self):
        assert match_line_items(None, []) == (True, 1.0, 0)
        assert match_line_items([], None) == (True, 1.0, 0)

    def test_wrapped_leaves_scored_correctly(self):
        gold = [
            {"description": "Widget", "quantity": 2, "unit_price": 5.0, "total": 10.0}
        ]
        pred = [
            {
                "description": _leaf("Widget"),
                "quantity": _leaf(2),
                "unit_price": _leaf(5.0),
                "total": _leaf(10.0),
            }
        ]
        correct, item_accuracy, schema_violations = match_line_items(gold, pred)
        assert correct is True
        assert item_accuracy == 1.0
        assert schema_violations == 0

    def test_order_mismatch_fails_full_but_partial_credit_reflects_it(self):
        gold = [
            {"description": "Widget", "quantity": 1, "unit_price": 5.0, "total": 5.0},
            {"description": "Gadget", "quantity": 1, "unit_price": 7.0, "total": 7.0},
        ]
        pred = [
            {
                "description": _leaf("Gadget"),
                "quantity": _leaf(1),
                "unit_price": _leaf(7.0),
                "total": _leaf(7.0),
            },
            {
                "description": _leaf("Widget"),
                "quantity": _leaf(1),
                "unit_price": _leaf(5.0),
                "total": _leaf(5.0),
            },
        ]
        correct, item_accuracy, _sv = match_line_items(gold, pred)
        assert correct is False
        assert item_accuracy == 0.0  # neither position lines up

    def test_null_entry_in_predicted_array_is_tolerated(self):
        gold = [
            {"description": "Widget", "quantity": 1, "unit_price": 5.0, "total": 5.0}
        ]
        pred = [None]
        correct, item_accuracy, _sv = match_line_items(gold, pred)
        assert correct is False
        assert item_accuracy == 0.0

    def test_count_mismatch_gives_partial_credit(self):
        gold = [
            {"description": "Widget", "quantity": 1, "unit_price": 5.0, "total": 5.0},
            {"description": "Gadget", "quantity": 1, "unit_price": 7.0, "total": 7.0},
        ]
        pred = [
            {
                "description": _leaf("Widget"),
                "quantity": _leaf(1),
                "unit_price": _leaf(5.0),
                "total": _leaf(5.0),
            }
        ]
        correct, item_accuracy, _sv = match_line_items(gold, pred)
        assert correct is False
        assert item_accuracy == 0.5

    def test_empty_vs_nonempty_is_zero(self):
        gold = [
            {"description": "Widget", "quantity": 1, "unit_price": 5.0, "total": 5.0}
        ]
        assert match_line_items(gold, []) == (False, 0.0, 0)
        pred = [
            {
                "description": _leaf("Widget"),
                "quantity": _leaf(1),
                "unit_price": _leaf(5.0),
                "total": _leaf(5.0),
            }
        ]
        correct, item_accuracy, _sv = match_line_items([], pred)
        assert correct is False
        assert item_accuracy == 0.0

    def test_amount_coercion_inside_a_matched_item_counts_as_schema_violation(self):
        gold = [
            {"description": "Widget", "quantity": 2, "unit_price": 5.0, "total": 10.0}
        ]
        pred = [
            {
                "description": _leaf("Widget"),
                "quantity": _leaf(2),
                "unit_price": _leaf("5.00"),
                "total": _leaf(10.0),
            }
        ]
        correct, _item_accuracy, schema_violations = match_line_items(gold, pred)
        assert correct is True
        assert schema_violations == 1

    def test_formatted_amount_inside_an_item_is_not_matched_but_still_counted(self):
        gold = [
            {"description": "Widget", "quantity": 2, "unit_price": 5.0, "total": 10.0}
        ]
        pred = [
            {
                "description": _leaf("Widget"),
                "quantity": _leaf(2),
                "unit_price": _leaf("$5.00"),
                "total": _leaf(10.0),
            }
        ]
        correct, _item_accuracy, schema_violations = match_line_items(gold, pred)
        assert correct is False
        assert schema_violations == 1


class TestMatchLineItemsDescriptionThreshold:
    """Description matching (ITEM_DESC_THRESHOLD = 0.8) lives inside
    match_line_items -- there's no standalone exported function for it --
    so these pin the >= boundary the same way TestMatchVendor does for
    match_vendor, via single-item lists where every other sub-field
    matches exactly.
    """

    @staticmethod
    def _item(description):
        return {
            "description": _leaf(description),
            "quantity": _leaf(1),
            "unit_price": _leaf(5.0),
            "total": _leaf(5.0),
        }

    @staticmethod
    def _gold(description):
        return [
            {"description": description, "quantity": 1, "unit_price": 5.0, "total": 5.0}
        ]

    def test_ratio_exactly_at_threshold_matches(self):
        gold = self._gold("abcdefghijklmnopqrst")
        pred = [self._item("abcdefghijklmnopZYXW")]  # ratio == 0.8 exactly
        correct, item_accuracy, _sv = match_line_items(gold, pred)
        assert correct is True
        assert item_accuracy == 1.0

    def test_ratio_just_below_threshold_fails(self):
        gold = self._gold("abcdefghijklmnopqrst")
        pred = [self._item("abcdefghijklmnoZYXWV")]  # ratio == 0.75
        correct, item_accuracy, _sv = match_line_items(gold, pred)
        assert correct is False
        assert item_accuracy == 0.0

    def test_ratio_just_above_threshold_matches(self):
        gold = self._gold("abcdefghijklmnopqrst")
        pred = [self._item("abcdefghijklmnopqZYX")]  # ratio == 0.85
        correct, item_accuracy, _sv = match_line_items(gold, pred)
        assert correct is True
        assert item_accuracy == 1.0


class TestScoreCase:
    def test_all_correct(self):
        case = _case()
        score = score_case(case, _base_tool_input(), review_threshold=0.8)
        assert all(fs.correct for fs in score.fields.values())
        assert score.hallucinations == 0
        assert score.schema_violations == 0
        assert score.item_accuracy == 1.0

    def test_null_gold_null_pred_is_correct(self):
        case = _case(tax=None)
        tool_input = _base_tool_input()
        tool_input["tax"] = _leaf(None)
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["tax"].correct is True
        assert score.hallucinations == 0

    def test_null_gold_nonnull_pred_is_hallucination(self):
        case = _case(tax=None)
        tool_input = _base_tool_input()  # tax leaf value is 1.0, non-null
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["tax"].correct is False
        assert score.hallucinations == 1

    def test_nonnull_gold_null_pred_is_a_miss_not_a_hallucination(self):
        case = _case()  # gold tax = 1.0
        tool_input = _base_tool_input()
        tool_input["tax"] = _leaf(None)
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["tax"].correct is False
        assert score.hallucinations == 0

    def test_line_items_hallucination_when_gold_has_none(self):
        case = _case(line_items=None)
        tool_input = _base_tool_input()  # predicted line_items is non-empty
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["line_items"].correct is False
        assert score.hallucinations == 1

    def test_line_items_miss_when_pred_empty(self):
        case = _case()  # gold has one line item
        tool_input = _base_tool_input()
        tool_input["line_items"] = _leaf([])
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["line_items"].correct is False
        assert score.hallucinations == 0

    def test_amount_string_coercion_counts_as_schema_violation(self):
        case = _case()
        tool_input = _base_tool_input()
        tool_input["total"] = _leaf("$11.00")
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["total"].correct is False
        assert score.schema_violations == 1

    def test_plain_amount_string_matches_but_is_still_a_schema_violation(self):
        case = _case()
        tool_input = _base_tool_input()
        tool_input["total"] = _leaf("11.00")
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["total"].correct is True
        assert score.schema_violations == 1

    def test_line_item_amount_coercion_counts_as_schema_violation(self):
        # Previously match_line_items discarded the coerced flag from
        # its inner match_amount calls, so a string like "5.00" inside
        # a line item never reached score_case's schema_violations count.
        case = _case()
        tool_input = _base_tool_input()
        tool_input["line_items"]["value"][0]["unit_price"] = _leaf("5.00")
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["line_items"].correct is True
        assert score.schema_violations == 1

    def test_hallucinated_amount_with_string_coercion_counts_both(self):
        # gold tax is null, pred asserts "$1.00" -- a hallucination AND
        # a schema violation (the model returned a string, not a bare
        # number), previously only the former was counted.
        case = _case(tax=None)
        tool_input = _base_tool_input()
        tool_input["tax"] = _leaf("$1.00")
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["tax"].correct is False
        assert score.hallucinations == 1
        assert score.schema_violations == 1

    def test_malformed_leaf_degrades_to_none_via_coerce_leaf(self):
        case = _case()
        tool_input = _base_tool_input()
        # A bare scalar instead of a {value, confidence} object -- the
        # same adversarial shape app.extraction._coerce_leaf defends
        # against -- degrades the whole leaf to (None, 0.0).
        tool_input["vendor"] = "Acme Corp"
        score = score_case(case, tool_input, review_threshold=0.8)
        assert score.fields["vendor"].actual is None
        assert score.fields["vendor"].confidence == 0.0
        assert score.fields["vendor"].correct is False


class TestAggregate:
    def test_caught_by_review_math(self):
        case_a = CaseScore(
            doc_id="a",
            fields={
                "vendor": FieldScore(
                    correct=True, expected="X", actual="X", confidence=0.9
                ),
                "total": FieldScore(
                    correct=False, expected=10.0, actual=20.0, confidence=0.5
                ),
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=1.0,
            review_threshold=0.8,
        )
        case_b = CaseScore(
            doc_id="b",
            fields={
                "vendor": FieldScore(
                    correct=False, expected="Y", actual="Z", confidence=0.9
                ),
                "total": FieldScore(
                    correct=False, expected=5.0, actual=6.0, confidence=0.95
                ),
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=1.0,
            review_threshold=0.8,
        )
        summary = aggregate([case_a, case_b])

        # Incorrect fields: a.total (conf 0.5, below 0.8 -> caught),
        # b.vendor (conf 0.9, above 0.8 -> not caught),
        # b.total (conf 0.95, above 0.8 -> not caught).
        # caught_by_review = 1 / 3.
        assert summary.caught_by_review == pytest.approx(1 / 3)
        assert summary.overall_accuracy == pytest.approx(
            1 / 4
        )  # 1 correct of 4 scored fields
        assert summary.per_field_accuracy["vendor"] == pytest.approx(0.5)
        assert summary.per_field_accuracy["total"] == pytest.approx(0.0)
        assert summary.mean_confidence_correct == pytest.approx(0.9)
        assert summary.mean_confidence_incorrect == pytest.approx(
            (0.5 + 0.9 + 0.95) / 3
        )

    def test_caught_by_review_is_none_when_nothing_incorrect(self):
        case_a = CaseScore(
            doc_id="a",
            fields={
                "vendor": FieldScore(
                    correct=True, expected="X", actual="X", confidence=0.9
                )
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=1.0,
            review_threshold=0.8,
        )
        summary = aggregate([case_a])
        assert summary.caught_by_review is None
        assert summary.mean_confidence_incorrect is None

    def test_confidence_exactly_at_threshold_is_not_caught(self):
        # Strict `<`, matching app.extraction's own
        # `confidence < settings.review_threshold` -- a field scored
        # with confidence exactly equal to the threshold is NOT "below"
        # it, so it must not count as caught. A `<` -> `<=` mutant would
        # flip this to 1.0.
        case_a = CaseScore(
            doc_id="a",
            fields={
                "total": FieldScore(
                    correct=False, expected=10.0, actual=20.0, confidence=0.8
                ),
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=1.0,
            review_threshold=0.8,
        )
        summary = aggregate([case_a])
        assert summary.caught_by_review == pytest.approx(0.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            aggregate([])

    def test_mean_item_accuracy_docs_with_items_excludes_no_item_docs(self):
        case_with_items = CaseScore(
            doc_id="a",
            fields={
                "line_items": FieldScore(
                    correct=False,
                    expected=[
                        {
                            "description": "Widget",
                            "quantity": 1,
                            "unit_price": 5.0,
                            "total": 5.0,
                        }
                    ],
                    actual=[],
                    confidence=0.5,
                ),
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=0.0,
            review_threshold=0.8,
        )
        case_without_items = CaseScore(
            doc_id="b",
            fields={
                "line_items": FieldScore(
                    correct=True, expected=None, actual=[], confidence=0.9
                ),
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=1.0,  # flattering default for a doc with no gold items
            review_threshold=0.8,
        )
        summary = aggregate([case_with_items, case_without_items])
        assert summary.mean_item_accuracy == pytest.approx(0.5)
        assert summary.mean_item_accuracy_docs_with_items == pytest.approx(0.0)

    def test_mean_item_accuracy_docs_with_items_is_none_when_no_docs_have_items(self):
        case = CaseScore(
            doc_id="a",
            fields={
                "line_items": FieldScore(
                    correct=True, expected=None, actual=[], confidence=0.9
                )
            },
            schema_violations=0,
            hallucinations=0,
            item_accuracy=1.0,
            review_threshold=0.8,
        )
        summary = aggregate([case])
        assert summary.mean_item_accuracy_docs_with_items is None


class TestDatasetValidation:
    def test_numeric_rejects_bool(self):
        # bool is a subclass of int, so isinstance(True, int | float) is
        # True -- must be excluded explicitly or "subtotal": true would
        # pass as a number.
        with pytest.raises(ValueError, match="numeric"):
            _validate_numeric("doc", "subtotal", True)
        with pytest.raises(ValueError, match="numeric"):
            _validate_numeric("doc", "subtotal", False)

    def test_numeric_accepts_int_float_or_null(self):
        _validate_numeric("doc", "subtotal", 10)
        _validate_numeric("doc", "subtotal", 10.5)
        _validate_numeric("doc", "subtotal", None)

    def test_numeric_rejects_string(self):
        with pytest.raises(ValueError, match="numeric"):
            _validate_numeric("doc", "subtotal", "10.00")

    def test_line_items_rejects_string_quantity(self):
        # The verified defect: "quantity": "2" (string) previously
        # passed load-time validation (key presence only) and would
        # crash score_case with a TypeError deep inside match_amount's
        # cents arithmetic once scoring actually touched it.
        with pytest.raises(ValueError, match="quantity"):
            _validate_line_items(
                "doc",
                [
                    {
                        "description": "Widget",
                        "quantity": "2",
                        "unit_price": 5.0,
                        "total": 10.0,
                    }
                ],
            )

    def test_line_items_rejects_bool_numeric(self):
        with pytest.raises(ValueError, match="unit_price"):
            _validate_line_items(
                "doc",
                [
                    {
                        "description": "Widget",
                        "quantity": 2,
                        "unit_price": True,
                        "total": 10.0,
                    }
                ],
            )

    def test_line_items_rejects_non_string_description(self):
        with pytest.raises(ValueError, match="description"):
            _validate_line_items(
                "doc",
                [{"description": 123, "quantity": 2, "unit_price": 5.0, "total": 10.0}],
            )

    def test_line_items_accepts_null_scalars(self):
        _validate_line_items(
            "doc",
            [
                {
                    "description": None,
                    "quantity": None,
                    "unit_price": None,
                    "total": None,
                }
            ],
        )

    def test_document_date_rejects_basic_iso_form(self):
        # date.fromisoformat("20260101") is valid Python (3.11+) but not
        # the dashed YYYY-MM-DD form scoring's match_date requires --
        # accepting it here would produce a label that "loads fine" but
        # is a guaranteed permanent zero at score time.
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _validate_document_date("doc", "20260101")

    def test_document_date_rejects_iso_week_date(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _validate_document_date("doc", "2026-W01-1")

    def test_document_date_accepts_dashed_form(self):
        _validate_document_date("doc", "2026-01-01")
        _validate_document_date("doc", None)

    def test_document_date_rejects_invalid_calendar_date(self):
        with pytest.raises(ValueError):
            _validate_document_date("doc", "2026-02-30")

    def test_vendor_rejects_punctuation_only(self):
        with pytest.raises(ValueError, match="word character"):
            _validate_vendor("doc", "***")

    def test_vendor_rejects_whitespace_only(self):
        with pytest.raises(ValueError, match="word character"):
            _validate_vendor("doc", "   ")

    def test_vendor_rejects_non_string(self):
        with pytest.raises(ValueError, match="string"):
            _validate_vendor("doc", 123)

    def test_vendor_accepts_null_and_normal_string(self):
        _validate_vendor("doc", None)
        _validate_vendor("doc", "Acme Corp")

    def test_currency_rejects_non_string(self):
        with pytest.raises(ValueError, match="string"):
            _validate_string_or_null("doc", "currency", 840)

    def test_currency_accepts_null_and_string(self):
        _validate_string_or_null("doc", "currency", None)
        _validate_string_or_null("doc", "currency", "USD")

    def test_bad_label_rejected_at_load_time_not_at_score_time(
        self, tmp_path, monkeypatch
    ):
        """Integration check for the verified defect: a label with a
        string quantity used to pass load_cases() and only blow up with
        a TypeError once score_case actually scored it. It must now fail
        loudly during load_cases()/_load_case, before scoring ever sees
        it.
        """
        label = {
            "doc_id": "bad-label",
            "image": "bad-label.png",
            "mime_type": "image/png",
            "source": "synthetic",
            "dataset_version": "v2",
            "difficulty": "clean",
            "fields": {
                "vendor": "Acme Corp",
                "document_date": "2026-01-01",
                "currency": "USD",
                "subtotal": 10.0,
                "tax": 1.0,
                "total": 11.0,
                "line_items": [
                    {
                        "description": "Widget",
                        "quantity": "2",
                        "unit_price": 5.0,
                        "total": 10.0,
                    }
                ],
                # The v2 fields a label must also have, all absent here.
                **dict.fromkeys(
                    FIELD_KEYS
                    - {"vendor", "document_date", "currency", "subtotal", "tax", "total", "line_items"}
                ),
            },
        }
        label_path = tmp_path / "bad-label.json"
        label_path.write_text(json.dumps(label), encoding="utf-8")
        (tmp_path / "bad-label.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(dataset_module, "EVALS_DIR", tmp_path)
        with pytest.raises(ValueError, match="quantity"):
            dataset_module._load_case(label_path)


class TestLoadRealDataset:
    def test_shipped_labels_load_without_exceptions(self):
        cases = load_cases()
        assert len([c for c in cases if c.source == "synthetic"]) == 32
        doc_ids = [c.doc_id for c in cases]
        assert doc_ids == sorted(doc_ids)
        for case in cases:
            assert case.image_path.is_file()
            assert set(case.fields.keys()) == FIELD_KEYS

    def test_limit_slices_after_sorting(self):
        all_cases = load_cases()
        limited = load_cases(limit=3)
        assert [c.doc_id for c in limited] == [c.doc_id for c in all_cases[:3]]


def _rebuilt_tool_input(doc: dict) -> dict:
    """The tool_input a scored artifact document was stored from: every field a
    {value, confidence} leaf, plus the evidence its schema stored (an array's
    cells keep theirs inside `actual`)."""
    tool_input = {}
    for name, field in doc["fields"].items():
        leaf = {"value": field["actual"], "confidence": field["confidence"]}
        if "evidence" in field:
            leaf["evidence"] = field["evidence"]
        tool_input[name] = leaf
    return tool_input


def test_rescoring_the_committed_runs_reproduces_their_scores():
    # The scorer reads amounts through the product's parser; this pins that the
    # change moved no committed score. Each artifact's stored answers are
    # rebuilt into a tool_input and scored again with the schema its own prompt
    # version was extracted with (its stored evidence included), and the run's
    # evidence totals are rolled up again from the same scores.
    cases = {case.doc_id: case for case in load_cases()}
    results = load_results()
    assert results
    assert {run["prompt_version"] for run in results} >= {"extract_v1", "extract_v3"}
    for run in results:
        schema = schema_for(run["prompt_version"])
        case_scores = []
        for doc in run["per_doc"]:
            if doc.get("status") != "scored":
                continue
            score = score_case(
                cases[doc["doc_id"]],
                _rebuilt_tool_input(doc),
                run["review_threshold"],
                schema,
            )
            case_scores.append(score)
            assert set(score.fields) == set(doc["fields"]), doc["doc_id"]
            for name, field in doc["fields"].items():
                assert score.fields[name].correct == field["correct"], (doc["doc_id"], name)
                assert score.fields[name].evidence == field.get("evidence"), (doc["doc_id"], name)
            assert score.item_accuracy == doc["item_accuracy"], doc["doc_id"]
            assert score.schema_violations == doc["schema_violations"], doc["doc_id"]
            assert score.hallucinations == doc["hallucinations"], doc["doc_id"]
        assert len(case_scores) == run["n_scored"]
        summary = aggregate(case_scores).to_dict()
        assert summary.get("evidence_coverage") == run["summary"].get("evidence_coverage")
        assert summary["total_hallucinations"] == run["summary"]["total_hallucinations"]
        assert summary["total_schema_violations"] == run["summary"]["total_schema_violations"]


# --- schema v2 matchers --------------------------------------------------------


class TestMatchIdentifier:
    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            ("INV-2026-0417", "INV-2026-0417"),
            ("INV-2026-0417", "inv 2026 0417"),
            ("INV-2026-0417", "INV20260417"),
            ("PO #88213", "po88213"),
            ("91-4827365", "91 4827365"),
            ("AB_12", "ab12"),
        ],
    )
    def test_case_and_punctuation_do_not_matter(self, expected, actual):
        assert match_identifier(expected, actual) is True

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            ("INV-2026-0417", "INV-2026-0471"),
            ("INV-2026-0417", "INV-2026-041"),
            ("INV-0417", "0417"),
            ("A0", "AO"),
        ],
    )
    def test_one_character_off_is_wrong(self, expected, actual):
        assert match_identifier(expected, actual) is False

    def test_an_expected_value_with_nothing_to_compare_never_matches(self):
        assert match_identifier("---", "---") is False
        assert match_identifier("", "") is False

    def test_non_strings_are_a_non_match(self):
        assert match_identifier("7745", 7745) is False
        assert match_identifier(None, "x") is False


class TestMatchTerms:
    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            ("Net 30", "Net 30"),
            ("Net 30", "net 30 days"),
            ("Net 30", "Net 30 days from invoice date"),
            ("Net 30", "Terms: Net 30"),
            ("Net 30", "NET-30"),
            ("Net 30 days", "Net 30"),
            ("Due on receipt", "Due upon receipt"),
            ("Due on receipt", "due on receipt."),
        ],
    )
    def test_the_same_terms_in_other_words_match(self, expected, actual):
        assert match_terms(expected, actual) is True

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            ("Net 30", "Net 300"),
            ("Net 300", "Net 30"),
            ("Net 30", "Net 3"),
            ("Net 30", "Net 60"),
            ("Net 30", "Net 31 days"),
            ("Net 30", "Due on receipt"),
            ("Due on receipt", "Net 30"),
            ("2/10 net 30", "Net 30"),
        ],
    )
    def test_different_terms_never_match(self, expected, actual):
        assert match_terms(expected, actual) is False

    def test_an_expected_value_of_only_filler_never_matches(self):
        assert match_terms("Terms", "Terms") is False

    def test_non_strings_are_a_non_match(self):
        assert match_terms("Net 30", 30) is False
        assert match_terms(None, "Net 30") is False


class TestMatchCardLast4:
    def test_exact_after_strip(self):
        assert match_card_last4("1111", "1111") is True
        assert match_card_last4("1111", " 1111 ") is True

    def test_a_different_or_longer_value_is_wrong(self):
        assert match_card_last4("1111", "1112") is False
        assert match_card_last4("1111", "****1111") is False
        assert match_card_last4("1111", 1111) is False


class TestMatchRate:
    def test_exact_decimal(self):
        assert match_rate(8.875, 8.875) == (True, False)
        assert match_rate(6.5, 6.50) == (True, False)
        assert match_rate(7, 7.0) == (True, False)

    def test_no_cent_of_tolerance(self):
        assert match_rate(8.875, 8.88) == (False, False)
        assert match_rate(6.5, 6.51) == (False, False)

    def test_a_string_is_coerced_and_counted(self):
        assert match_rate(6.5, "6.5") == (True, True)
        assert match_rate(6.5, "6.5%") == (False, True)

    def test_a_fraction_is_not_the_percent(self):
        assert match_rate(8.875, 0.08875) == (False, False)


def _tax_row(label="State sales tax", rate=6.5, base=454.48, amount=29.54):
    return {"label": label, "rate": rate, "base": base, "amount": amount}


def _tax_cells(row):
    return {key: _leaf(value) for key, value in row.items()}


class TestMatchTaxLines:
    def test_equal_rows_match(self):
        gold = [_tax_row(), _tax_row("County transit tax", 1.25, 454.48, 5.68)]
        pred = [_tax_cells(row) for row in gold]

        assert match_tax_lines(gold, pred) == (True, 1.0, 0)

    def test_both_empty_or_null_match(self):
        assert match_tax_lines(None, []) == (True, 1.0, 0)
        assert match_tax_lines([], None) == (True, 1.0, 0)

    def test_a_missing_or_extra_row_is_wrong_with_partial_credit(self):
        gold = [_tax_row(), _tax_row("County transit tax", 1.25, 454.48, 5.68)]

        correct, accuracy, _ = match_tax_lines(gold, [_tax_cells(gold[0])])
        assert (correct, accuracy) == (False, 0.5)
        assert match_tax_lines([], [_tax_cells(gold[0])]) == (False, 0.0, 0)

    def test_rows_are_compared_in_the_order_printed(self):
        gold = [_tax_row(), _tax_row("County transit tax", 1.25, 454.48, 5.68)]
        pred = [_tax_cells(gold[1]), _tax_cells(gold[0])]

        assert match_tax_lines(gold, pred) == (False, 0.0, 0)

    def test_the_label_is_fuzzy(self):
        gold = [_tax_row("State sales tax")]

        assert match_tax_lines(gold, [_tax_cells(_tax_row("State Sales Tax."))])[0] is True
        assert match_tax_lines(gold, [_tax_cells(_tax_row("Excise duty"))])[0] is False

    def test_the_rate_is_exact(self):
        gold = [_tax_row(rate=8.875)]

        assert match_tax_lines(gold, [_tax_cells(_tax_row(rate=8.875))])[0] is True
        assert match_tax_lines(gold, [_tax_cells(_tax_row(rate=8.88))])[0] is False

    def test_the_base_and_amount_tolerate_a_cent(self):
        gold = [_tax_row()]

        assert match_tax_lines(gold, [_tax_cells(_tax_row(base=454.49, amount=29.55))])[0] is True
        assert match_tax_lines(gold, [_tax_cells(_tax_row(amount=29.60))])[0] is False

    def test_a_null_cell_matches_only_a_null_cell(self):
        gold = [_tax_row(rate=None, base=None)]

        assert match_tax_lines(gold, [_tax_cells(_tax_row(rate=None, base=None))])[0] is True
        assert match_tax_lines(gold, [_tax_cells(_tax_row(rate=6.5, base=None))])[0] is False
        assert match_tax_lines([_tax_row()], [_tax_cells(_tax_row(base=None))])[0] is False

    def test_a_string_amount_is_a_schema_violation(self):
        gold = [_tax_row()]
        row = _tax_cells(_tax_row())
        row["amount"] = _leaf("29.54")
        row["rate"] = _leaf("6.5")

        assert match_tax_lines(gold, [row]) == (True, 1.0, 2)

    @pytest.mark.parametrize(
        ("gold", "predicted"),
        [
            ("VAT", "VAT 19%"),
            ("VAT", "VAT 19% on 125,19 EUR"),
            ("State sales tax", "State sales tax 4% on $1,018.00"),
            ("State sales tax", "State sales tax 6.5%"),
            ("County transit tax", "County transit tax 1.25 %"),
        ],
    )
    def test_a_rate_and_base_left_in_the_label_are_not_a_wrong_tax(self, gold, predicted):
        assert match_tax_lines([_tax_row(gold)], [_tax_cells(_tax_row(predicted))])[0] is True

    def test_stripping_the_rate_does_not_make_another_label_match(self):
        assert match_tax_lines([_tax_row("VAT")], [_tax_cells(_tax_row("Excise duty 19%"))])[0] is False
        assert match_tax_lines([_tax_row("VAT")], [_tax_cells(_tax_row("19%"))])[0] is False

    def test_a_bare_null_row_degrades_instead_of_raising(self):
        assert match_tax_lines([_tax_row()], [None]) == (False, 0.0, 0)


# --- scoring a v2 case -----------------------------------------------------------


def _label_case(prefix: str) -> EvalCase:
    return next(case for case in load_cases() if case.doc_id.startswith(prefix))


def _gold_tool_input(case: EvalCase) -> dict:
    """The answer a flawless reader gives for a label, quotes and all."""
    return label_tool_input(case.fields, 0.9, case.evidence)


def _score_v2(case: EvalCase, tool_input: dict) -> CaseScore:
    return score_case(case, tool_input, 0.8, SCHEMA_V2)


class TestScoreCaseV2:
    @pytest.mark.parametrize("prefix", ["001", "026", "028", "029", "030", "031", "032"])
    def test_the_gold_answer_scores_every_field_correct(self, prefix):
        case = _label_case(prefix)

        score = _score_v2(case, _gold_tool_input(case))

        assert list(score.fields) == list(SCHEMA_V2.field_names)
        assert all(field.correct for field in score.fields.values()), prefix
        assert (score.hallucinations, score.schema_violations) == (0, 0)

    def test_an_identifier_is_exact_on_its_letters_and_digits(self):
        case = _label_case("026")
        tool = _gold_tool_input(case)

        tool["invoice_number"]["value"] = "inv 2026 0417"
        tool["po_number"]["value"] = "PO-88231"
        score = _score_v2(case, tool)

        assert score.fields["invoice_number"].correct is True
        assert score.fields["po_number"].correct is False

    def test_payment_terms_match_by_their_words_and_numbers(self):
        case = _label_case("026")
        tool = _gold_tool_input(case)

        tool["payment_terms"]["value"] = "net 30 days"
        assert _score_v2(case, tool).fields["payment_terms"].correct is True
        tool["payment_terms"]["value"] = "Net 300"
        assert _score_v2(case, tool).fields["payment_terms"].correct is False
        tool["payment_terms"]["value"] = "Due on receipt"
        assert _score_v2(case, tool).fields["payment_terms"].correct is False

    def test_a_due_date_is_normalized_like_a_date(self):
        case = _label_case("026")
        tool = _gold_tool_input(case)

        tool["due_date"]["value"] = "09/02/2026"
        assert _score_v2(case, tool).fields["due_date"].correct is True
        tool["due_date"]["value"] = "2026-09-03"
        assert _score_v2(case, tool).fields["due_date"].correct is False

    def test_the_card_digits_are_exact(self):
        case = _label_case("029")
        tool = _gold_tool_input(case)

        tool["payment_card_last4"]["value"] = "1112"
        score = _score_v2(case, tool)

        assert score.fields["payment_card_last4"].correct is False
        assert score.fields["payment_method"].correct is True

    @pytest.mark.parametrize("name", ["discount", "shipping", "amount_paid", "amount_due", "tip"])
    def test_a_new_amount_is_within_a_cent(self, name):
        case = next(c for c in load_cases() if c.fields[name] is not None)
        tool = _gold_tool_input(case)

        tool[name]["value"] = round(case.fields[name] + 0.01, 2)
        assert _score_v2(case, tool).fields[name].correct is True
        tool[name]["value"] = round(case.fields[name] + 0.02, 2)
        assert _score_v2(case, tool).fields[name].correct is False

    def test_a_new_amount_as_a_string_is_a_schema_violation(self):
        case = _label_case("026")
        tool = _gold_tool_input(case)

        tool["shipping"]["value"] = "$18.50"
        score = _score_v2(case, tool)

        assert score.fields["shipping"].correct is False
        assert score.schema_violations == 1

    def test_a_tax_line_misread_makes_the_field_wrong_and_the_rest_right(self):
        case = _label_case("026")
        tool = _gold_tool_input(case)

        tool["tax_lines"]["value"][1]["amount"]["value"] = 5.86
        score = _score_v2(case, tool)

        assert score.fields["tax_lines"].correct is False
        assert score.fields["tax"].correct is True
        assert score.fields["line_items"].correct is True

    def test_a_hallucinated_new_field_is_counted(self):
        case = _label_case("001")
        tool = _gold_tool_input(case)

        tool["po_number"] = _leaf("PO-1")
        tool["tax_lines"]["value"].append(_tax_cells(_tax_row()))
        score = _score_v2(case, tool)

        assert score.fields["po_number"].correct is False
        # The extra tax line makes tax_lines wrong, but gold has tax lines, so it is no hallucination.
        assert score.hallucinations == 1
        assert score.fields["tax_lines"].correct is False

    def test_tax_lines_hallucinated_where_gold_has_none(self):
        case = _label_case("030")
        no_tax_lines = replace_fields(case, tax_lines=None)

        score = _score_v2(no_tax_lines, _gold_tool_input(case))

        assert score.fields["tax_lines"].correct is False
        assert score.hallucinations == 1

    def test_the_default_schema_is_v1_and_reads_only_its_fields(self):
        case = _label_case("026")

        score = score_case(case, _gold_tool_input(case), 0.8)

        assert tuple(score.fields) == SCHEMA_V1.field_names
        assert score.evidence is None

    def test_v1_scoring_of_a_v2_label_is_the_same_as_before(self):
        case = _label_case("026")

        v1 = score_case(case, _gold_tool_input(case), 0.8, SCHEMA_V1)
        v2 = _score_v2(case, _gold_tool_input(case))

        for name in SCHEMA_V1.field_names:
            assert v1.fields[name].correct == v2.fields[name].correct
        assert v1.item_accuracy == v2.item_accuracy

    def test_the_card_number_never_reaches_the_score(self):
        # extract_document masks card data, but a score is the artifact: whatever
        # extract_fn produced, an unmasked number must not be written into it.
        case = _label_case("029")
        tool = _gold_tool_input(case)
        tool["payment_method"]["value"] = "VISA 4111 1111 1111 1111"
        tool["payment_method"]["evidence"] = {"page": 1, "quote": "Paid by VISA 4111 1111 1111 1111 $88.57"}
        tool["amount_paid"]["evidence"] = {"page": 1, "quote": "Paid by VISA 4111 1111 1111 1111 $88.57"}

        score = _score_v2(case, tool)
        artifact = json.dumps(score.to_dict())

        assert "4111" not in artifact
        assert score.fields["payment_method"].actual == "VISA ****1111"
        assert score.fields["payment_method"].evidence["quote"] == "Paid by VISA ****1111 $88.57"


def replace_fields(case: EvalCase, **overrides) -> EvalCase:
    return EvalCase(
        doc_id=case.doc_id,
        image_path=case.image_path,
        mime_type=case.mime_type,
        source=case.source,
        dataset_version=case.dataset_version,
        difficulty=case.difficulty,
        fields={**case.fields, **overrides},
        evidence=case.evidence,
    )


# --- evidence in a score -----------------------------------------------------------


class TestFieldScoreEvidence:
    def test_a_v1_field_score_keeps_its_exact_shape(self):
        assert FieldScore(True, 1.0, 1.0, 0.9).to_dict() == {
            "correct": True,
            "expected": 1.0,
            "actual": 1.0,
            "confidence": 0.9,
        }

    def test_evidence_is_serialized_only_when_present(self):
        cited = {"page": 1, "quote": "Tax: $1.00"}

        assert FieldScore(True, 1.0, 1.0, 0.9, evidence=cited).to_dict()["evidence"] == cited
        assert "evidence" not in FieldScore(True, 1.0, 1.0, 0.9, evidence=None).to_dict()

    def test_a_v1_score_has_no_evidence_anywhere(self):
        score = score_case(_case(), _base_tool_input(), 0.8)

        assert score.evidence is None
        assert all("evidence" not in field for field in score.to_dict()["fields"].values())
        assert aggregate([score]).evidence_coverage is None
        assert "evidence_coverage" not in aggregate([score]).to_dict()

    def test_a_v2_header_field_records_the_evidence_it_cited(self):
        case = _label_case("001")
        score = _score_v2(case, _gold_tool_input(case))

        assert score.fields["total"].evidence == case.evidence["total"]
        assert score.to_dict()["fields"]["total"]["evidence"] == case.evidence["total"]
        # A null value has no evidence, and an array's cells keep theirs inside `actual`.
        assert "evidence" not in score.to_dict()["fields"]["po_number"]
        assert score.fields["line_items"].evidence is None
        assert score.fields["line_items"].actual[0]["total"]["evidence"] == case.evidence["line_items[0].total"]


def _counts(case: EvalCase, mutate=None) -> dict[str, EvidenceCounts]:
    tool = _gold_tool_input(case)
    if mutate is not None:
        mutate(tool)
    return _score_v2(case, tool).evidence


class TestEvidenceCounts:
    def test_the_gold_answer_is_fully_covered_and_agrees_everywhere(self):
        case = _label_case("026")

        counts = _counts(case)

        total = sum(counts.values(), EvidenceCounts())
        assert total.leaves == total.with_quote == total.gold_judged == total.page_agree == total.quote_agree
        assert counts["invoice_number"] == EvidenceCounts(1, 1, 1, 1, 1)
        assert counts["po_number"] == EvidenceCounts(1, 1, 1, 1, 1)
        # Every cell of every line item and tax line is a leaf.
        assert counts["line_items"].leaves == 4 * 4
        assert counts["tax_lines"].leaves == 2 * 4

    def test_a_null_value_is_no_leaf(self):
        case = _label_case("001")

        assert _counts(case)["po_number"] == EvidenceCounts()

    def test_a_value_with_no_quote_is_a_coverage_miss_and_a_page_miss(self):
        case = _label_case("001")

        counts = _counts(case, lambda tool: tool["total"].update(evidence=None))

        assert counts["total"] == EvidenceCounts(1, 0, 1, 0, 0)

    def test_a_null_page_is_a_page_miss_not_a_coverage_miss(self):
        case = _label_case("001")

        def drop_page(tool):
            tool["total"]["evidence"] = {"page": None, "quote": case.evidence["total"]["quote"]}

        counts = _counts(case, drop_page)

        assert counts["total"] == EvidenceCounts(1, 1, 1, 0, 1)

    def test_a_wrong_page_is_a_page_miss(self):
        case = _label_case("032")

        def wrong_page(tool):
            tool["total"]["evidence"] = {"page": 1, "quote": case.evidence["total"]["quote"]}

        assert case.evidence["total"]["page"] == 2
        assert _counts(case, wrong_page)["total"] == EvidenceCounts(1, 1, 1, 0, 1)

    def test_a_quote_is_compared_with_its_whitespace_collapsed(self):
        case = _label_case("001")
        gold_quote = case.evidence["line_items[0].total"]["quote"]
        assert "  " in gold_quote

        def tidy(tool):
            tool["line_items"]["value"][0]["total"]["evidence"] = {
                "page": 1,
                "quote": " ".join(gold_quote.split()),
            }

        counts = _counts(case, tidy)["line_items"]

        assert counts.quote_agree == counts.gold_judged == counts.leaves

    def test_a_cell_may_quote_part_of_the_gold_row(self):
        case = _label_case("001")
        gold_quote = case.evidence["line_items[0].description"]["quote"]
        description = case.fields["line_items"][0]["description"]
        assert description in gold_quote and description != " ".join(gold_quote.split())

        def quote_the_cell(tool):
            tool["line_items"]["value"][0]["description"]["evidence"] = {
                "page": 1,
                "quote": description,
            }

        counts = _counts(case, quote_the_cell)["line_items"]

        assert counts.quote_agree == counts.gold_judged == counts.leaves

    def test_a_cell_quoting_something_outside_the_gold_row_is_a_quote_miss(self):
        case = _label_case("001")

        def quote_another_row(tool):
            tool["line_items"]["value"][0]["description"]["evidence"] = {
                "page": 1,
                "quote": "Amount due $1.00",
            }

        counts = _counts(case, quote_another_row)["line_items"]

        assert counts.quote_agree == counts.gold_judged - 1

    @pytest.mark.parametrize(
        ("quote", "agrees"),
        [
            ("Cat Litter 10lb", True),
            ("1", True),  # a whole word of the row "... 1 $11.30 $11.30"
            ("$11.30", True),
            ("$11.30 $11.30", True),
            ("$", False),  # part of a word, not a word
            ("11.30", False),  # "$11.30" with its sign cut off
            ("Cat Lit", False),  # the front of a word
            ("Litter 10", False),  # the front of a word
            ("$11.30 Cat", False),  # words out of order
        ],
    )
    def test_a_cell_quote_agrees_only_as_whole_words_of_the_gold_row(self, quote, agrees):
        case = _label_case("001")
        assert case.evidence["line_items[0].total"]["quote"].split() == ["Cat", "Litter", "10lb", "1", "$11.30", "$11.30"]

        def quote_total(tool):
            tool["line_items"]["value"][0]["total"]["evidence"] = {"page": 1, "quote": quote}

        counts = _counts(case, quote_total)["line_items"]

        assert counts.quote_agree == counts.gold_judged - (0 if agrees else 1)

    def test_a_header_field_must_quote_the_gold_text_exactly(self):
        case = _label_case("001")
        gold_quote = case.evidence["total"]["quote"]
        part = gold_quote.split()[-1]
        assert part != " ".join(gold_quote.split())

        counts = _counts(
            case, lambda tool: tool["total"].update(evidence={"page": 1, "quote": part})
        )

        assert counts["total"] == EvidenceCounts(1, 1, 1, 1, 0)

    def test_another_quote_is_a_quote_miss_but_still_covered(self):
        case = _label_case("001")

        counts = _counts(
            case, lambda tool: tool["total"].update(evidence={"page": 1, "quote": "Amount due $1.00"})
        )

        assert counts["total"] == EvidenceCounts(1, 1, 1, 1, 0)

    def test_a_quote_that_is_not_usable_does_not_cover(self):
        case = _label_case("001")

        counts = _counts(
            case, lambda tool: tool["total"].update(evidence={"page": 1, "quote": "x" * 201})
        )

        assert counts["total"].with_quote == 0

    def test_a_value_the_gold_label_does_not_cite_is_covered_but_not_judged(self):
        case = _label_case("001")

        def invent(tool):
            tool["po_number"] = {"value": "PO-1", "confidence": 0.9, "evidence": {"page": 1, "quote": "PO-1"}}

        assert _counts(case, invent)["po_number"] == EvidenceCounts(1, 1, 0, 0, 0)

    def test_the_masked_payment_quote_agrees_with_the_gold_one(self):
        case = _label_case("029")

        counts = _counts(case)

        assert case.evidence["payment_method"]["quote"] == "Paid by VISA ****1111 $88.57"
        assert counts["payment_method"] == EvidenceCounts(1, 1, 1, 1, 1)
        assert counts["payment_card_last4"] == EvidenceCounts(1, 1, 1, 1, 1)

    def test_a_row_past_the_gold_rows_is_covered_but_not_judged(self):
        case = _label_case("001")

        def extra_row(tool):
            row = {key: dict(cell) for key, cell in tool["line_items"]["value"][0].items()}
            tool["line_items"]["value"].append(row)

        counts = _counts(case, extra_row)["line_items"]
        base = _counts(case)["line_items"]

        assert counts.leaves == base.leaves + 4
        assert counts.gold_judged == base.gold_judged

    def test_rates_are_none_where_nothing_was_counted(self):
        assert EvidenceCounts().to_dict()["coverage"] is None
        assert EvidenceCounts(2, 1, 0, 0, 0).to_dict()["page_agreement"] is None
        assert EvidenceCounts(2, 1, 2, 1, 0).to_dict()["coverage"] == 0.5


class TestAggregateEvidence:
    def test_the_run_totals_are_the_sum_of_its_cases_and_fields(self):
        cases = [_label_case("001"), _label_case("026")]
        scores = [_score_v2(case, _gold_tool_input(case)) for case in cases]

        coverage = aggregate(scores).evidence_coverage

        per_field = coverage["per_field"]
        assert list(per_field) == list(SCHEMA_V2.field_names)
        assert coverage["leaves"] == sum(field["leaves"] for field in per_field.values())
        assert coverage["leaves"] == sum(
            counts.leaves for score in scores for counts in score.evidence.values()
        )
        assert coverage["coverage"] == coverage["page_agreement"] == coverage["quote_agreement"] == 1.0
        assert per_field["vendor"]["leaves"] == 2
        assert per_field["po_number"]["leaves"] == 1

    def test_a_missing_quote_lowers_coverage_and_both_agreements(self):
        case = _label_case("001")
        tool = _gold_tool_input(case)
        tool["total"]["evidence"] = None

        coverage = aggregate([_score_v2(case, tool)]).evidence_coverage

        assert coverage["per_field"]["total"]["coverage"] == 0.0
        assert coverage["per_field"]["subtotal"]["coverage"] == 1.0
        assert coverage["coverage"] < 1.0
        assert coverage["page_agreement"] < 1.0

    def test_it_is_part_of_the_summary_dict_for_v2_only(self):
        case = _label_case("001")

        summary = aggregate([_score_v2(case, _gold_tool_input(case))]).to_dict()

        assert summary["evidence_coverage"]["leaves"] > 0
        assert json.loads(json.dumps(summary)) == summary


# --- the core documents ------------------------------------------------------------


class TestCoreDocuments:
    def test_the_core_is_the_first_25_labels(self):
        assert [case.doc_id for case in load_cases(limit=25)] == list(CORE_DOC_IDS)

    def test_documents_can_be_selected_by_doc_id(self):
        cases = load_cases(doc_ids=["032-multipage-invoice-totals-page-two", "026-invoice-discount-shipping-two-taxes"])

        assert [case.doc_id[:3] for case in cases] == ["026", "032"]
        assert len(load_cases(doc_ids=["026-invoice-discount-shipping-two-taxes"], limit=5)) == 1

    def test_an_unknown_doc_id_is_an_error(self):
        with pytest.raises(ValueError, match="no-such-doc"):
            load_cases(doc_ids=["026-invoice-discount-shipping-two-taxes", "no-such-doc"])
