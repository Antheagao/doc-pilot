"""Field-level scoring of a VLM extraction result against a gold label.

Each rule (match_vendor, match_date, match_currency, match_amount,
match_line_items) is a small standalone function so tests can exercise
the normalization/tolerance logic directly, independent of a full
score_case() run.

score_case() reads tool_input leaves through app.extraction._coerce_leaf
-- the same defensive unwrap app.extraction.process_document_job uses
before persisting an ExtractedField -- so this scores what actually
lands in the DB, not Claude's raw (possibly schema-violating) tool_input.
Importing app.extraction (and transitively app.models/app.db) is
tolerated here per the module's own docstring: engine creation in app.db
is lazy, so this stays importable with Postgres down. app.worker is
never imported and no DB session is ever opened.

Null semantics, applied uniformly to every top-level field including
line_items as a whole:
- gold None, pred None -> correct
- gold None, pred non-None -> incorrect, and `hallucinations` +1 (the
  model asserted a value the document doesn't have)
- gold non-None, pred None -> incorrect (a miss -- the model failed to
  extract a value that exists -- NOT a hallucination)
"""

import difflib
import re
from dataclasses import dataclass
from datetime import date
from statistics import mean
from typing import Any

from app.evals.dataset import EvalCase
from app.extraction import TOP_LEVEL_FIELDS, _coerce_leaf

VENDOR_MATCH_THRESHOLD = 0.85
ITEM_DESC_THRESHOLD = 0.8
AMOUNT_TOLERANCE = 0.01

_LINE_ITEM_KEYS = ("description", "quantity", "unit_price", "total")
_AMOUNT_FIELD_NAMES = ("subtotal", "tax", "total")

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_CURRENCY_JUNK_RE = re.compile(r"[^\d.\-]")


# --- text normalization -----------------------------------------------


def _normalize_text(value: str) -> str:
    """casefold, strip punctuation, collapse whitespace -- used by both
    match_vendor and the line-item description comparison so "Acme Corp."
    and "ACME CORP" normalize to the same string before fuzzy matching.
    """
    value = value.casefold()
    value = _PUNCTUATION_RE.sub("", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def _fuzzy_match(expected: str, actual: str, threshold: float) -> bool:
    a = _normalize_text(expected)
    b = _normalize_text(actual)
    if not a:
        # `expected` was a non-null string that normalizes to nothing --
        # empty, or punctuation/whitespace-only (e.g. "***") -- so there
        # is nothing meaningful to compare against. difflib.SequenceMatcher
        # treats two empty sequences as a ratio of 1.0, which used to let
        # this fall through to a vacuous match (gold "***" vs pred "!!"
        # scored correct). Guarding here makes that always a non-match,
        # regardless of what `actual` normalizes to. Dataset loading now
        # rejects punctuation-only vendor labels outright (see
        # app.evals.dataset._validate_vendor); this is the general
        # backstop for any other caller (e.g. line-item descriptions).
        return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def match_vendor(expected: str, actual: str) -> bool:
    """Fuzzy string match: casefold, strip punctuation, collapse
    whitespace, then a difflib.SequenceMatcher ratio >= VENDOR_MATCH_THRESHOLD.
    Callers are expected to have already excluded the null cases (see
    module docstring) -- non-string input is treated as a non-match
    rather than raising.
    """
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    return _fuzzy_match(expected, actual, VENDOR_MATCH_THRESHOLD)


# --- dates ---------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLASH_DATE_RE = re.compile(r"^(\d{1,4})/(\d{1,2})/(\d{1,4})$")


def _normalize_date(value: str) -> str | None:
    """Normalize a date string to ISO YYYY-MM-DD, or None if it can't be
    parsed as one of the formats a VLM is known to emit.

    Accepts ISO YYYY-MM-DD as-is (the format the prompt asks the model
    for). Also tolerates slash-separated MM/DD/YYYY, DD/MM/YYYY, and
    YYYY/MM/DD, since the model is asked to normalize a document's
    printed date and doesn't always comply. A 4-digit first component
    is unambiguous (YYYY/MM/DD). Otherwise the two remaining components
    are ambiguous when both are <= 12 (e.g. 03/05/2026): the US
    convention MM/DD/YYYY is tried first; if the first component can't
    be a valid month (> 12), the components are swapped and retried as
    DD/MM/YYYY. A genuinely ambiguous date like 03/05/2026 is therefore
    always read as MM/DD/YYYY (March 5) by this rule, not DD/MM/YYYY.
    """
    value = value.strip()
    if _ISO_DATE_RE.match(value):
        try:
            date.fromisoformat(value)
        except ValueError:
            return None
        return value

    match = _SLASH_DATE_RE.match(value)
    if not match:
        return None
    a, b, c = match.groups()
    if len(a) == 4:
        year, month, day = a, b, c
    else:
        month, day, year = a, b, c
        if int(month) > 12:
            month, day = day, month
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def match_date(expected: str, actual: str) -> bool:
    """Normalize both sides to ISO YYYY-MM-DD (see _normalize_date) and
    compare exactly. Non-string input, or input that doesn't parse as
    any recognized date format, is a non-match rather than a raise.
    """
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    exp = _normalize_date(expected)
    act = _normalize_date(actual)
    return exp is not None and exp == act


def match_currency(expected: str, actual: str) -> bool:
    """Strip whitespace + uppercase, then exact match."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    return expected.strip().upper() == actual.strip().upper()


# --- amounts ---------------------------------------------------------------


def _coerce_amount(value: Any) -> tuple[float | None, bool]:
    """Coerce a numeric-ish value to float. Returns (value, coerced) where
    coerced is True iff the value was a string that had to be parsed
    (strip currency symbols, strip thousands-separator commas) rather
    than arriving as a plain int/float -- every such coercion is counted
    as a schema violation by the caller, since the model was asked for a
    bare number, not a formatted string. Kept intentionally simple:
    commas are always treated as thousands separators (no European
    "17,44"-as-decimal support).
    """
    if isinstance(value, bool):
        return None, False
    if isinstance(value, int | float):
        return float(value), False
    if isinstance(value, str):
        cleaned = _CURRENCY_JUNK_RE.sub("", value.replace(",", ""))
        try:
            return float(cleaned), True
        except ValueError:
            return None, True
    return None, False


def _amounts_match(expected: float, actual: float) -> bool:
    """Compare two dollar amounts in integer cents rather than raw
    floats. `abs(expected - actual) <= AMOUNT_TOLERANCE` is float-broken
    exactly at the boundary it's supposed to police: 19.99 - 20.00 is
    -0.010000000000001563 in binary float, which fails a naive
    `<= 0.01` even though the two amounts are one cent apart and should
    pass. Rounding both sides to whole cents first makes the comparison
    exact regardless of binary-float representation error.
    """
    tolerance_cents = round(AMOUNT_TOLERANCE * 100)
    return abs(round(expected * 100) - round(actual * 100)) <= tolerance_cents


def match_amount(expected: float, actual: Any) -> tuple[bool, bool]:
    """Returns (matched, coerced). `actual` is coerced to float first
    (see _coerce_amount); `coerced` is True whenever that coercion had to
    parse a string at all (e.g. "$17.44"), regardless of whether the
    parsed value ends up matching. `matched` is True iff the coerced
    actual is within AMOUNT_TOLERANCE of expected (see _amounts_match).
    """
    actual_value, coerced = _coerce_amount(actual)
    if actual_value is None:
        return False, coerced
    return _amounts_match(expected, actual_value), coerced


# --- line items --------------------------------------------------------


def _decode_predicted_item(item: Any) -> dict[str, Any]:
    """Unwrap a predicted line item's four sub-leaves via _coerce_leaf.
    A non-dict item (e.g. a bare `null` the model emitted in the array
    instead of a proper item object) degrades to all-None sub-fields
    rather than raising -- the array is untrusted the same way every
    other leaf is (see app.extraction._coerce_leaf).
    """
    if not isinstance(item, dict):
        return dict.fromkeys(_LINE_ITEM_KEYS)
    return {key: _coerce_leaf(item.get(key))[0] for key in _LINE_ITEM_KEYS}


def _item_field_matches(key: str, expected: Any, actual: Any) -> tuple[bool, bool]:
    """Returns (matches, coerced). `coerced` is only ever True for the
    three numeric sub-fields (see match_amount) -- description matching
    never coerces anything, so it always reports False.
    """
    if expected is None and actual is None:
        return True, False
    if expected is None or actual is None:
        return False, False
    if key == "description":
        return _fuzzy_match(str(expected), str(actual), ITEM_DESC_THRESHOLD), False
    return match_amount(expected, actual)


def match_line_items(
    expected: list | None, actual: list | None
) -> tuple[bool, float, int]:
    """Compare a gold line_items list to a predicted line_items list.

    `actual` must already be the persisted array -- i.e. a non-list
    tool_input value already coerced down to [] by the caller, mirroring
    app.extraction.process_document_job's own `value if isinstance(value,
    list) else []` handling, since this scores what gets persisted, not
    raw output.

    Returns (correct, item_accuracy, schema_violations):
    - correct is True iff the item counts match AND every item matches,
      in order -- description fuzzy >= ITEM_DESC_THRESHOLD, quantity/
      unit_price/total within AMOUNT_TOLERANCE (see _item_field_matches).
    - item_accuracy = matched_items / max(len(expected), len(actual)):
      1.0 when both sides are null/empty, 0.0 when one side is empty and
      the other isn't, and partial credit in between otherwise.
    - schema_violations counts every numeric sub-field (quantity/
      unit_price/total), across every gold/predicted item pair actually
      compared, that required string coercion (see match_amount) --
      mirrors the top-level schema_violations counter in score_case so a
      model that emits "$5.00" inside a line item isn't invisible to it.
    """
    gold = expected or []
    pred = actual or []

    if not gold and not pred:
        return True, 1.0, 0

    decoded_pred = [_decode_predicted_item(item) for item in pred]

    matched_items = 0
    schema_violations = 0
    for gold_item, pred_item in zip(gold, decoded_pred, strict=False):
        field_results = [
            _item_field_matches(key, gold_item.get(key), pred_item.get(key))
            for key in _LINE_ITEM_KEYS
        ]
        if all(matched for matched, _coerced in field_results):
            matched_items += 1
        schema_violations += sum(1 for _matched, coerced in field_results if coerced)

    correct = len(gold) == len(pred) and matched_items == len(gold)
    item_accuracy = matched_items / max(len(gold), len(pred))
    return correct, item_accuracy, schema_violations


# --- case / run scoring --------------------------------------------------


@dataclass
class FieldScore:
    correct: bool
    expected: Any
    actual: Any
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "expected": self.expected,
            "actual": self.actual,
            "confidence": self.confidence,
        }


@dataclass
class CaseScore:
    doc_id: str
    fields: dict[str, FieldScore]
    schema_violations: int
    hallucinations: int
    item_accuracy: float
    review_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "fields": {name: fs.to_dict() for name, fs in self.fields.items()},
            "schema_violations": self.schema_violations,
            "hallucinations": self.hallucinations,
            "item_accuracy": self.item_accuracy,
            "review_threshold": self.review_threshold,
        }


def score_case(
    case: EvalCase, tool_input: dict[str, Any], review_threshold: float
) -> CaseScore:
    """Score one extraction result (a raw tool_input dict, e.g. from
    ExtractionResult.tool_input) against `case`'s gold label.

    See the module docstring for the null/hallucination/miss rule
    applied to every field, including line_items as a whole.
    """
    gold = case.fields
    field_scores: dict[str, FieldScore] = {}
    schema_violations = 0
    hallucinations = 0
    item_accuracy = 1.0

    for field_name in TOP_LEVEL_FIELDS:
        expected = gold[field_name]
        value, confidence = _coerce_leaf(tool_input.get(field_name))

        if field_name == "line_items":
            actual = value if isinstance(value, list) else []
            correct, item_accuracy, line_schema_violations = match_line_items(
                expected, actual
            )
            schema_violations += line_schema_violations
            if not expected and actual:
                hallucinations += 1
        else:
            actual = value
            if expected is None and actual is None:
                correct = True
            elif expected is None and actual is not None:
                correct = False
                hallucinations += 1
                # A hallucinated amount can *also* be a schema violation
                # (e.g. gold tax is null, the model asserts "$1.00") --
                # that string-coercion shouldn't go uncounted just
                # because this branch is about the null mismatch, not a
                # value comparison.
                if field_name in _AMOUNT_FIELD_NAMES:
                    _, coerced = _coerce_amount(actual)
                    if coerced:
                        schema_violations += 1
            elif expected is not None and actual is None:
                correct = False
            elif field_name == "vendor":
                correct = match_vendor(expected, actual)
            elif field_name == "document_date":
                correct = match_date(expected, actual)
            elif field_name == "currency":
                correct = match_currency(expected, actual)
            else:  # subtotal, tax, total
                correct, coerced = match_amount(expected, actual)
                if coerced:
                    schema_violations += 1

        field_scores[field_name] = FieldScore(
            correct=correct, expected=expected, actual=actual, confidence=confidence
        )

    return CaseScore(
        doc_id=case.doc_id,
        fields=field_scores,
        schema_violations=schema_violations,
        hallucinations=hallucinations,
        item_accuracy=item_accuracy,
        review_threshold=review_threshold,
    )


@dataclass
class RunSummary:
    n_docs: int
    per_field_accuracy: dict[str, float]
    overall_accuracy: float
    mean_confidence: float
    mean_confidence_correct: float | None
    mean_confidence_incorrect: float | None
    caught_by_review: float | None
    total_hallucinations: int
    total_schema_violations: int
    mean_item_accuracy: float
    mean_item_accuracy_docs_with_items: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_docs": self.n_docs,
            "per_field_accuracy": self.per_field_accuracy,
            "overall_accuracy": self.overall_accuracy,
            "mean_confidence": self.mean_confidence,
            "mean_confidence_correct": self.mean_confidence_correct,
            "mean_confidence_incorrect": self.mean_confidence_incorrect,
            "caught_by_review": self.caught_by_review,
            "total_hallucinations": self.total_hallucinations,
            "total_schema_violations": self.total_schema_violations,
            "mean_item_accuracy": self.mean_item_accuracy,
            "mean_item_accuracy_docs_with_items": self.mean_item_accuracy_docs_with_items,
        }


def aggregate(case_scores: list[CaseScore]) -> RunSummary:
    """Roll a list of per-case CaseScores up into a RunSummary.

    overall_accuracy is the micro-average across every scored field
    (correct fields / total fields scored) -- equal to the "7 fields x
    n_docs" denominator described for the real dataset, but computed
    from whatever fields each CaseScore actually carries so a
    hand-built CaseScore with a reduced field set aggregates sanely too.

    caught_by_review reads the review_threshold each CaseScore was
    scored with -- (incorrect fields with confidence < that field's
    case's review_threshold) / (incorrect fields); None when there are
    no incorrect fields at all (nothing for review to "catch"). The
    comparison is strict `<`, matching app.extraction's own
    `confidence < settings.review_threshold` -- a field scored with
    confidence exactly equal to the threshold is NOT counted as caught.

    mean_item_accuracy averages item_accuracy over every case, including
    docs whose gold has no line items at all (which score a flattering
    1.0 by definition -- see match_line_items). mean_item_accuracy_docs_with_items
    is the same average restricted to cases whose gold line_items is
    non-empty, so a dataset skewed toward no-item docs can't hide poor
    line-item extraction behind the unrestricted metric.
    """
    if not case_scores:
        raise ValueError("aggregate() requires at least one CaseScore")

    per_field_correct: dict[str, int] = {}
    per_field_total: dict[str, int] = {}
    all_confidences: list[float] = []
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []
    incorrect_below_threshold = 0
    total_correct = 0
    total_fields = 0
    total_hallucinations = 0
    total_schema_violations = 0
    item_accuracies: list[float] = []
    item_accuracies_with_items: list[float] = []

    for case_score in case_scores:
        total_hallucinations += case_score.hallucinations
        total_schema_violations += case_score.schema_violations
        item_accuracies.append(case_score.item_accuracy)

        line_items_field = case_score.fields.get("line_items")
        if line_items_field is not None and line_items_field.expected:
            item_accuracies_with_items.append(case_score.item_accuracy)

        for field_name, field_score in case_score.fields.items():
            per_field_total[field_name] = per_field_total.get(field_name, 0) + 1
            all_confidences.append(field_score.confidence)
            total_fields += 1

            if field_score.correct:
                per_field_correct[field_name] = per_field_correct.get(field_name, 0) + 1
                total_correct += 1
                correct_confidences.append(field_score.confidence)
            else:
                incorrect_confidences.append(field_score.confidence)
                if field_score.confidence < case_score.review_threshold:
                    incorrect_below_threshold += 1

    per_field_accuracy = {
        name: per_field_correct.get(name, 0) / total
        for name, total in per_field_total.items()
    }
    incorrect_total = len(incorrect_confidences)

    return RunSummary(
        n_docs=len(case_scores),
        per_field_accuracy=per_field_accuracy,
        overall_accuracy=total_correct / total_fields if total_fields else 0.0,
        mean_confidence=mean(all_confidences) if all_confidences else 0.0,
        mean_confidence_correct=mean(correct_confidences)
        if correct_confidences
        else None,
        mean_confidence_incorrect=mean(incorrect_confidences)
        if incorrect_confidences
        else None,
        caught_by_review=(incorrect_below_threshold / incorrect_total)
        if incorrect_total
        else None,
        total_hallucinations=total_hallucinations,
        total_schema_violations=total_schema_violations,
        mean_item_accuracy=mean(item_accuracies) if item_accuracies else 0.0,
        mean_item_accuracy_docs_with_items=(
            mean(item_accuracies_with_items) if item_accuracies_with_items else None
        ),
    )
