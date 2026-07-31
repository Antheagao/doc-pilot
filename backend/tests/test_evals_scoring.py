"""Tests for app.evals.dataset and app.evals.scoring.

Deliberately has no db_session fixture anywhere -- these modules are pure
functions over dicts/dataclasses and must stay importable/runnable with
Postgres stopped (see the "no DB" verification step for E2).
"""

import json
from pathlib import Path

import pytest

import app.evals.dataset as dataset_module
from app.evals.dataset import (
    EvalCase,
    _validate_document_date,
    _validate_line_items,
    _validate_numeric,
    _validate_string_or_null,
    _validate_vendor,
    load_cases,
)
from app.evals.scoring import (
    CaseScore,
    FieldScore,
    aggregate,
    match_amount,
    match_currency,
    match_date,
    match_line_items,
    match_vendor,
    score_case,
)


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

    def test_dollar_sign_string_is_coerced(self):
        matched, coerced = match_amount(17.44, "$17.44")
        assert matched is True
        assert coerced is True

    def test_thousands_comma_string_is_coerced(self):
        matched, coerced = match_amount(1234.50, "1,234.50")
        assert matched is True
        assert coerced is True

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
                "unit_price": _leaf("$5.00"),
                "total": _leaf(10.0),
            }
        ]
        correct, _item_accuracy, schema_violations = match_line_items(gold, pred)
        assert correct is True
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
        assert score.fields["total"].correct is True
        assert score.schema_violations == 1

    def test_line_item_amount_coercion_counts_as_schema_violation(self):
        # Previously match_line_items discarded the coerced flag from
        # its inner match_amount calls, so a string like "$5.00" inside
        # a line item never reached score_case's schema_violations count.
        case = _case()
        tool_input = _base_tool_input()
        tool_input["line_items"]["value"][0]["unit_price"] = _leaf("$5.00")
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
            "dataset_version": "v1",
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
        assert len(cases) == 25
        doc_ids = [c.doc_id for c in cases]
        assert doc_ids == sorted(doc_ids)
        for case in cases:
            assert case.image_path.is_file()
            assert set(case.fields.keys()) == {
                "vendor",
                "document_date",
                "currency",
                "subtotal",
                "tax",
                "total",
                "line_items",
            }

    def test_limit_slices_after_sorting(self):
        all_cases = load_cases()
        limited = load_cases(limit=3)
        assert [c.doc_id for c in limited] == [c.doc_id for c in all_cases[:3]]
