"""Field-level scoring of a VLM extraction result against a gold label.

Each rule (match_vendor, match_identifier, match_date, match_currency,
match_amount, match_line_items, match_tax_lines, ...) is a small standalone
function so tests can exercise the normalization/tolerance logic directly,
independent of a full score_case() run. A case is scored over the fields of
the extraction schema it is given (SCHEMA_V1 by default, SCHEMA_V2 for the
richer fields), each by the rule its name selects.

score_case() reads tool_input leaves through app.extraction._coerce_leaf
-- the same defensive unwrap app.extraction.process_document_job uses
before persisting an ExtractedField -- so this scores what actually
lands in the DB, not Claude's raw (possibly schema-violating) tool_input.
Importing app.extraction (and transitively app.models/app.db) is
tolerated here per the module's own docstring: engine creation in app.db
is lazy, so this stays importable with Postgres down. app.worker is
never imported and no DB session is ever opened.

Amounts are read with the product's own parser (app.money.to_decimal) and
compared as exact decimals, so the eval and the reconciliation checks agree
on what a string such as "17.44" or "$17.44" means.

Null semantics, applied uniformly to every top-level field including
line_items as a whole:
- gold None, pred None -> correct
- gold None, pred non-None -> incorrect, and `hallucinations` +1 (the
  model asserted a value the document doesn't have)
- gold non-None, pred None -> incorrect (a miss -- the model failed to
  extract a value that exists -- NOT a hallucination)

Evidence. For a schema whose leaves carry evidence (v2), score_case also
counts how well the answer says where each value was read (EvidenceCounts):
coverage is the share of non-null extracted values that carry a usable quote,
and, for values the gold label cites, page agreement and quote agreement are
counted apart from it. A leaf with a quote but no page counts as covered and as
a page miss; a leaf with no quote at all is both. The gold label's quotes keep
the document's column padding, so quotes are compared with their whitespace
collapsed. A header field's quote must equal the gold quote; the cell of a line
item or tax line may also quote just the cell, so it agrees when it is whole
words of the gold quote, which cites the whole printed row.
"""

import difflib
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from statistics import mean
from typing import Any

from app.evals.dataset import EvalCase
from app.extraction import _coerce_leaf, _stored_leaf, mask_card_data
from app.extraction_fields import ARRAY_CELLS, SCHEMA_V1, ExtractionSchema
from app.money import MoneyError, to_decimal

VENDOR_MATCH_THRESHOLD = 0.85
ITEM_DESC_THRESHOLD = 0.8
AMOUNT_TOLERANCE = Decimal("0.01")

_LINE_ITEM_KEYS = tuple(ARRAY_CELLS["line_items"])
_TAX_LINE_KEYS = tuple(ARRAY_CELLS["tax_lines"])

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
# Everything that is not a letter or a digit, the underscore included.
_NON_ALNUM_RE = re.compile(r"[\W_]+")
# A rate and the base it is charged on, as a reader may leave them in a tax
# line's label: `19%`, `4% on $1,018.00`, `19% on 125,19 EUR`.
_LABEL_RATE_RE = re.compile(r"\s*\d+(?:[.,]\d+)?\s*%(\s+on\s+\S+(\s+\S+)?)?")
# The words a payment-terms phrase may add around the terms themselves.
_TERMS_FILLER = frozenset({"terms", "days", "day"})


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


def _normalize_identifier(value: str) -> str:
    return _NON_ALNUM_RE.sub("", value.casefold())


def match_identifier(expected: str, actual: str) -> bool:
    """Casefold and keep only letters and digits, then compare exactly, so
    `INV-2026-0417` and `inv 2026 0417` agree but `INV-2026-0471` does not: an
    identifier is right or wrong, never nearly right. An expected value with no
    letter or digit has nothing to compare, and non-string input is a
    non-match rather than a raise."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    wanted = _normalize_identifier(expected)
    return bool(wanted) and wanted == _normalize_identifier(actual)


def _terms_tokens(value: str) -> list[str]:
    text = _PUNCTUATION_RE.sub(" ", value.casefold())
    return [token for token in text.split() if token not in _TERMS_FILLER]


def _digit_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if any(char.isdigit() for char in token)]


def match_terms(expected: str, actual: str) -> bool:
    """Payment terms: `Net 30` is right for `net 30 days` and `Terms: Net 30`,
    and a different number never is (`Net 300`, `Net 3`). Both sides are
    casefolded, stripped of punctuation, and split into words, dropping the
    filler words `terms` and `days`. They match when the expected words appear
    in the actual ones as an unbroken run, or, failing that, when the whole
    phrases are fuzzy-equal (VENDOR_MATCH_THRESHOLD) and carry the same numbers
    (`Due on receipt` for `Due upon receipt`). Non-string input is a non-match."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    wanted, got = _terms_tokens(expected), _terms_tokens(actual)
    if not wanted:
        return False
    if any(got[i : i + len(wanted)] == wanted for i in range(len(got) - len(wanted) + 1)):
        return True
    if _digit_tokens(wanted) != _digit_tokens(got):
        return False
    return _fuzzy_match(" ".join(wanted), " ".join(got), VENDOR_MATCH_THRESHOLD)


def match_card_last4(expected: str, actual: str) -> bool:
    """Exact match of the last four digits, ignoring surrounding whitespace.
    Non-string input is a non-match."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    return expected.strip() == actual.strip()


# --- amounts ---------------------------------------------------------------


def _coerce_amount(value: Any) -> tuple[Decimal | None, bool]:
    """Read a numeric-ish value as a Decimal through app.money.to_decimal.
    Returns (value, coerced) where coerced is True iff the value was a
    string -- every string is counted as a schema violation by the caller,
    since the model was asked for a bare number, not text. A string that is
    not a plain number ("$17.44", "1,234.50") is refused by the parser, so
    it reads as None and scores wrong, the way the product would treat it.
    """
    if isinstance(value, bool):
        return None, False
    try:
        return to_decimal(value), isinstance(value, str)
    except MoneyError:
        return None, isinstance(value, str)


def _amounts_match(expected: Decimal, actual: Decimal) -> bool:
    """True when two amounts are within AMOUNT_TOLERANCE of each other.
    The float boundary bug this once guarded against (19.99 - 20.00 is
    -0.010000000000001563 in binary float, which failed a naive `<= 0.01`
    though the amounts are one cent apart) cannot happen in Decimal, so the
    plain comparison is exact.
    """
    return abs(expected - actual) <= AMOUNT_TOLERANCE


def match_amount(expected: float, actual: Any) -> tuple[bool, bool]:
    """Returns (matched, coerced). `actual` is read as a Decimal first
    (see _coerce_amount); `coerced` is True whenever it was a string,
    regardless of whether it parsed or matched. `matched` is True iff the
    parsed actual is within AMOUNT_TOLERANCE of expected (see
    _amounts_match). `expected` is a validated label value, so a malformed
    one raises MoneyError rather than scoring.
    """
    actual_value, coerced = _coerce_amount(actual)
    if actual_value is None:
        return False, coerced
    return _amounts_match(to_decimal(expected), actual_value), coerced


def match_rate(expected: float, actual: Any) -> tuple[bool, bool]:
    """Returns (matched, coerced) like match_amount, but a tax rate is a printed
    percent (8.875), so it matches only when it equals the label's exactly,
    with no cent of tolerance: 8.88 is not 8.875."""
    actual_value, coerced = _coerce_amount(actual)
    if actual_value is None:
        return False, coerced
    return to_decimal(expected) == actual_value, coerced


# --- rows: line items and tax lines -----------------------------------------


def _decode_predicted_row(row: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """Unwrap a predicted row's sub-leaves via _coerce_leaf.
    A non-dict row (e.g. a bare `null` the model emitted in the array
    instead of a proper row object) degrades to all-None sub-fields
    rather than raising -- the array is untrusted the same way every
    other leaf is (see app.extraction._coerce_leaf).
    """
    if not isinstance(row, dict):
        return dict.fromkeys(keys)
    return {key: _coerce_leaf(row.get(key))[0] for key in keys}


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


def _tax_line_cell_matches(key: str, expected: Any, actual: Any) -> tuple[bool, bool]:
    """Returns (matches, coerced) for one tax line cell: the label fuzzy like a
    line item's description, the rate exactly (see match_rate), the base and the
    amount within AMOUNT_TOLERANCE (see match_amount)."""
    if expected is None and actual is None:
        return True, False
    if expected is None or actual is None:
        return False, False
    if key == "label":
        # The prompt asks for the label without its rate, but a rate a reader
        # leaves in is a wording difference, not a wrong tax.
        label = _LABEL_RATE_RE.sub("", str(actual))
        return _fuzzy_match(str(expected), label, ITEM_DESC_THRESHOLD), False
    if key == "rate":
        return match_rate(expected, actual)
    return match_amount(expected, actual)


def _match_rows(
    keys: tuple[str, ...],
    cell_matches: Any,
    expected: list | None,
    actual: list | None,
) -> tuple[bool, float, int]:
    """The row comparison match_line_items and match_tax_lines share: rows are
    paired in the order printed, a row matches when every cell does, and the
    result is (correct, row_accuracy, schema_violations) as match_line_items
    documents."""
    gold = expected or []
    pred = actual or []

    if not gold and not pred:
        return True, 1.0, 0

    decoded_pred = [_decode_predicted_row(row, keys) for row in pred]

    matched_rows = 0
    schema_violations = 0
    for gold_row, pred_row in zip(gold, decoded_pred, strict=False):
        field_results = [
            cell_matches(key, gold_row.get(key), pred_row.get(key)) for key in keys
        ]
        if all(matched for matched, _coerced in field_results):
            matched_rows += 1
        schema_violations += sum(1 for _matched, coerced in field_results if coerced)

    correct = len(gold) == len(pred) and matched_rows == len(gold)
    row_accuracy = matched_rows / max(len(gold), len(pred))
    return correct, row_accuracy, schema_violations


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
    return _match_rows(_LINE_ITEM_KEYS, _item_field_matches, expected, actual)


def match_tax_lines(
    expected: list | None, actual: list | None
) -> tuple[bool, float, int]:
    """Compare a gold tax_lines list to a predicted one, the way
    match_line_items compares line items: the counts must match and every row
    match, in the order printed. A row matches when its label is fuzzy-equal
    (ITEM_DESC_THRESHOLD), its rate is exactly equal (match_rate), and its base
    and amount are within AMOUNT_TOLERANCE (match_amount). Returns (correct,
    row_accuracy, schema_violations) with the same meaning, the violations
    counting the rate, base, and amount cells that arrived as strings."""
    return _match_rows(_TAX_LINE_KEYS, _tax_line_cell_matches, expected, actual)


# --- evidence ------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceCounts:
    """How well one field's (or one run's) values say where they were read.

    `leaves` are the non-null extracted values (every cell of a line item or tax
    line counts as a leaf), `with_quote` those carrying a usable quote.
    `gold_judged` are the leaves whose path the gold label cites, and
    `page_agree` / `quote_agree` how many of those name the gold page and quote
    the gold text (whitespace collapsed; a cell may quote whole words of the gold
    row, see _leaf_evidence_counts). A leaf with no quote is a page miss and
    a quote miss too; one with a quote and no page is covered but a page miss.
    """

    leaves: int = 0
    with_quote: int = 0
    gold_judged: int = 0
    page_agree: int = 0
    quote_agree: int = 0

    def __add__(self, other: "EvidenceCounts") -> "EvidenceCounts":
        return EvidenceCounts(
            self.leaves + other.leaves,
            self.with_quote + other.with_quote,
            self.gold_judged + other.gold_judged,
            self.page_agree + other.page_agree,
            self.quote_agree + other.quote_agree,
        )

    def to_dict(self) -> dict[str, Any]:
        """The counts and their three rates (None where nothing was counted)."""

        def rate(count: int, total: int) -> float | None:
            return count / total if total else None

        return {
            "leaves": self.leaves,
            "with_quote": self.with_quote,
            "coverage": rate(self.with_quote, self.leaves),
            "gold_judged": self.gold_judged,
            "page_agree": self.page_agree,
            "page_agreement": rate(self.page_agree, self.gold_judged),
            "quote_agree": self.quote_agree,
            "quote_agreement": rate(self.quote_agree, self.gold_judged),
        }


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _leaf_evidence_counts(
    value: Any,
    evidence: dict[str, Any] | None,
    gold: dict[str, Any] | None,
    *,
    cell: bool = False,
) -> EvidenceCounts:
    """The counts for one leaf: nothing for a null value, else whether it has a
    quote and, when the gold label cites the path, whether page and quote agree.

    A header field's quote agrees only when it equals the gold quote. The gold
    quote of a line-item or tax-line cell is the whole printed row, and a model
    that quotes just the cell ("Hydraulic Hose Assembly 3/4in") has read the
    same line, so a `cell` agrees when its quote is the gold row or whole
    whitespace-separated words of it, in order (exact characters, whitespace
    collapsed on both sides). Words, not characters: a bare "$", or "154" against
    a row that prints "$154.00", is part of a token and is not that cell's quote."""
    if value is None:
        return EvidenceCounts()
    has_quote = evidence is not None
    judged = gold is not None
    page_agree = judged and has_quote and evidence["page"] == gold["page"]
    quote_agree = False
    if judged and has_quote:
        quoted = _collapse_whitespace(evidence["quote"])
        printed = _collapse_whitespace(gold["quote"])
        if cell:
            quote_agree = (
                bool(quoted)
                and re.search(rf"(?<!\S){re.escape(quoted)}(?!\S)", printed) is not None
            )
        else:
            quote_agree = quoted == printed
    return EvidenceCounts(1, int(has_quote), int(judged), int(page_agree), int(quote_agree))


def _array_evidence_counts(
    name: str, rows: list, schema: ExtractionSchema, gold_evidence: dict[str, dict[str, Any]]
) -> EvidenceCounts:
    """The counts of every cell of every row of an array field, each row paired
    with the gold evidence of the same index."""
    total = EvidenceCounts()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for key in ARRAY_CELLS[name]:
            raw = row.get(key)
            total += _leaf_evidence_counts(
                _coerce_leaf(raw)[0],
                _stored_leaf(raw, schema)["evidence"],
                gold_evidence.get(f"{name}[{index}].{key}"),
                cell=True,
            )
    return total


# --- case / run scoring --------------------------------------------------


@dataclass
class FieldScore:
    correct: bool
    expected: Any
    actual: Any
    confidence: float
    # The {page, quote} a v2 header field cited, when it cited a usable one; the
    # cells of an array field keep theirs inside `actual`. Left out of the
    # artifact when absent, so a v1 artifact keeps its exact shape.
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "correct": self.correct,
            "expected": self.expected,
            "actual": self.actual,
            "confidence": self.confidence,
        }
        if self.evidence is not None:
            data["evidence"] = self.evidence
        return data


@dataclass
class CaseScore:
    doc_id: str
    fields: dict[str, FieldScore]
    schema_violations: int
    hallucinations: int
    item_accuracy: float
    review_threshold: float
    # Evidence counts per field; None for a schema without evidence (v1). Kept
    # out of to_dict: the run summary carries the totals.
    evidence: dict[str, EvidenceCounts] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "fields": {name: fs.to_dict() for name, fs in self.fields.items()},
            "schema_violations": self.schema_violations,
            "hallucinations": self.hallucinations,
            "item_accuracy": self.item_accuracy,
            "review_threshold": self.review_threshold,
        }


# The rule each scalar field is scored by; the money fields go through
# match_amount instead (their kind says so) and the array fields through the
# row matchers.
_SCALAR_MATCHERS = {
    "vendor": match_vendor,
    "invoice_number": match_identifier,
    "po_number": match_identifier,
    "vendor_tax_id": match_identifier,
    "document_date": match_date,
    "due_date": match_date,
    # Free text a reader may word differently: terms by their words and numbers,
    # the method fuzzy at the vendor's threshold.
    "payment_terms": match_terms,
    "payment_method": match_vendor,
    "payment_card_last4": match_card_last4,
    "currency": match_currency,
}

_ROW_MATCHERS = {"line_items": match_line_items, "tax_lines": match_tax_lines}


def score_case(
    case: EvalCase,
    tool_input: dict[str, Any],
    review_threshold: float,
    schema: ExtractionSchema = SCHEMA_V1,
) -> CaseScore:
    """Score one extraction result (a raw tool_input dict, e.g. from
    ExtractionResult.tool_input) against `case`'s gold label, over the fields of
    `schema`.

    See the module docstring for the null/hallucination/miss rule
    applied to every field, including line_items and tax_lines as a whole, and
    for the evidence counts a schema with evidence adds. A schema with evidence
    also has the card-number rule applied to the answer first, the one the
    product applies before storing anything, so no artifact holds an unmasked
    number whatever produced the answer.
    """
    if schema.evidence:
        tool_input = mask_card_data(tool_input)
    gold = case.fields
    money_fields = schema.money_field_names
    field_scores: dict[str, FieldScore] = {}
    evidence_counts: dict[str, EvidenceCounts] | None = {} if schema.evidence else None
    schema_violations = 0
    hallucinations = 0
    item_accuracy = 1.0

    for spec in schema.fields:
        field_name = spec.name
        expected = gold[field_name]
        raw = tool_input.get(field_name)
        value, confidence = _coerce_leaf(raw)
        evidence = None

        if spec.kind == "array":
            actual = value if isinstance(value, list) else []
            correct, row_accuracy, row_schema_violations = _ROW_MATCHERS[field_name](
                expected, actual
            )
            if field_name == "line_items":
                item_accuracy = row_accuracy
            schema_violations += row_schema_violations
            if not expected and actual:
                hallucinations += 1
            if evidence_counts is not None:
                evidence_counts[field_name] = _array_evidence_counts(
                    field_name, actual, schema, case.evidence
                )
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
                if field_name in money_fields:
                    _, coerced = _coerce_amount(actual)
                    if coerced:
                        schema_violations += 1
            elif expected is not None and actual is None:
                correct = False
            elif field_name in money_fields:
                correct, coerced = match_amount(expected, actual)
                if coerced:
                    schema_violations += 1
            else:
                correct = _SCALAR_MATCHERS[field_name](expected, actual)
            if evidence_counts is not None:
                evidence = _stored_leaf(raw, schema)["evidence"]
                evidence_counts[field_name] = _leaf_evidence_counts(
                    actual, evidence, case.evidence.get(field_name)
                )

        field_scores[field_name] = FieldScore(
            correct=correct,
            expected=expected,
            actual=actual,
            confidence=confidence,
            evidence=evidence,
        )

    return CaseScore(
        doc_id=case.doc_id,
        fields=field_scores,
        schema_violations=schema_violations,
        hallucinations=hallucinations,
        item_accuracy=item_accuracy,
        review_threshold=review_threshold,
        evidence=evidence_counts,
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
    # The evidence counts and rates of the whole run and of each field (see
    # EvidenceCounts.to_dict); None for a schema without evidence, and then left
    # out of the artifact.
    evidence_coverage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        if self.evidence_coverage is not None:
            data["evidence_coverage"] = self.evidence_coverage
        return data


def _evidence_coverage(case_scores: list[CaseScore]) -> dict[str, Any] | None:
    """The run's evidence counts, overall and per field, or None when no case
    was scored with evidence."""
    per_field: dict[str, EvidenceCounts] = {}
    scored = False
    for case_score in case_scores:
        if case_score.evidence is None:
            continue
        scored = True
        for name, counts in case_score.evidence.items():
            per_field[name] = per_field.get(name, EvidenceCounts()) + counts
    if not scored:
        return None
    overall = sum(per_field.values(), EvidenceCounts())
    return {
        **overall.to_dict(),
        "per_field": {name: counts.to_dict() for name, counts in per_field.items()},
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
        evidence_coverage=_evidence_coverage(case_scores),
    )
