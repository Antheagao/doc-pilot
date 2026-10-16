"""What the arithmetic checks add to the evals, measured offline.

Two measurements share this module, and both replay answers through the
product's own path (`build_extracted_fields`, `reconcile`, `route`) with no
database, no network, and no model call.

Caught by checks. `caught_by_review` asks how many wrong fields the model
itself doubted; `caught_by_checks` asks how many the product would route to a
person because a check in app/reconcile.py names them. It is the share of
incorrect fields (the same denominator as `caught_by_review`) whose replayed
`review_reason` starts with `reconciliation:` or `quote:`, the two reasons a
deterministic check gives (the quote check of app/quotes.py needs a run whose
answers carry evidence quotes, so it adds nothing to a run without them;
`flagged_by_quotes` and the `*_by_quotes` counts say how much of the total is
the quotes'). A document that merely fails a check does not count: the check
has to point at the wrong field. It is
computed whenever a table is rendered, from the answers each artifact already
stores, and never written back: the artifacts are what a model said and how it
scored, and the rules version is part of the measurement, so a new rule moves
the number without rewriting history.

Reconciliation eval. The labels are consistent documents (every label passes
the checks unmutated, with the quotes of its gold evidence, and `failing_labels`
is how a test or the CLI says so). `run_mutation_eval` breaks one printed number
at a time in the ways a reader misreads digits, replays each broken document
with full confidence (so only arithmetic can notice: the replay carries no
quotes), and counts what the engine says. A number printed once is broken
everywhere it was read (the gold evidence says which cells cite the same printed
number), and the rest of the label, its tax lines, discount, shipping, tip, and
payments, is replayed as it is, so a broken amount is also checked against them. It answers "which misreads would the checks catch
if the model were confidently wrong" without paying for a model run.

Quote eval. `run_quote_eval` replays the same broken documents with evidence
quotes in front of the engine, in three ways a reader can go wrong: the value is
misread and the quote still shows what is printed, the value and the quote are
misread the same way, and the quote of the broken cell is missing. It answers how
much the value-quote check (app/quotes.py) adds to the arithmetic, and where it
adds nothing.

Imports `report.py` nowhere: the report module imports this one.
"""

import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.evals.dataset import EvalCase, label_values
from app.extraction import build_extracted_fields
from app.extraction_fields import (
    ARRAY_CELLS,
    SCHEMA_V2,
    SCHEMAS,
    ExtractionSchema,
)
from app.models import REVIEW_REASON_QUOTE_PREFIX, REVIEW_REASON_RECONCILIATION_PREFIX
from app.money import currency_precision, to_api_string, to_decimal, to_plain_string
from app.reconcile import (
    RULES_VERSION,
    STATUS_FAIL,
    ReconcileInput,
    Report,
    line_path,
    reconcile,
    tax_line_path,
)
from app.review_routing import Routing, field_name_of, route

# Confidence every replayed label cell carries: high enough that no field is
# flagged for doubt, so a flag in the mutation eval is the arithmetic's doing.
_FULL_CONFIDENCE = 1.0

# The labels carry every v2 field, so the label replays and the mutation eval
# run on SCHEMA_V2; a scored run replays under its own prompt version. The
# mutation eval breaks every amount a label carries: the header amounts, and
# the cells of each line item and tax line.
_LABEL_SCHEMA = SCHEMA_V2
_HEADER_MONEY = SCHEMA_V2.money_field_names


# --- replaying answers through the product's path ----------------------------


def _leaf(value: Any, confidence: float) -> dict[str, Any]:
    return {"value": value, "confidence": confidence}


def label_tool_input(
    fields: Mapping[str, Any],
    confidence: float = _FULL_CONFIDENCE,
    evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """A label's `fields` as the tool_input a model would have produced for
    it: every field a `{value, confidence}` leaf, and every cell of a line item
    or tax line a leaf too.

    `evidence` is the label's gold evidence, `{path: {page, quote}}`. Given, it
    adds an `evidence` key to each leaf other than an array's own: the gold
    object for its path, else null. Left out (the default), leaves carry no
    quote, so only the arithmetic can notice a broken number.
    """

    def leaf(value: Any, path: str) -> dict[str, Any]:
        cell = _leaf(value, confidence)
        if evidence is not None:
            cell["evidence"] = evidence.get(path)
        return cell

    tool_input: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in ARRAY_CELLS or not isinstance(value, list):
            tool_input[name] = leaf(value, name)
            continue
        tool_input[name] = _leaf(
            [
                {key: leaf(row.get(key), f"{name}[{index}].{key}") for key in ARRAY_CELLS[name]}
                if isinstance(row, dict)
                else row
                for index, row in enumerate(value)
            ],
            confidence,
        )
    return tool_input


def replay(
    tool_input: dict[str, Any], review_threshold: float, schema: ExtractionSchema
) -> tuple[Report, dict[str, Routing]]:
    """The reconciliation report and per-field routing the worker would store
    for this tool_input. The fields are transient ORM rows, so nothing needs a
    session."""
    fields = build_extracted_fields(uuid.UUID(int=0), tool_input, review_threshold, schema)
    report = reconcile(ReconcileInput.from_fields(fields))
    return report, route(fields, report, review_threshold)


def _replay_doc(
    doc: Mapping[str, Any], review_threshold: float, schema: ExtractionSchema
) -> tuple[Report, dict[str, Routing]]:
    """`replay` for one scored `per_doc` entry, from its stored answers. Only
    the fields the entry carries are replayed and returned. A header field
    brings back the evidence the entry stored for it; the cells of an array
    field keep theirs inside the stored array."""
    tool_input = {}
    for name, field in doc["fields"].items():
        leaf = _leaf(field["actual"], field["confidence"])
        if field.get("evidence") is not None:
            leaf["evidence"] = field["evidence"]
        tool_input[name] = leaf
    report, routing = replay(tool_input, review_threshold, schema)
    return report, {name: routing[name] for name in doc["fields"]}


def doc_routing(
    doc: Mapping[str, Any], review_threshold: float, schema: ExtractionSchema
) -> dict[str, Routing]:
    """The routing of each field of one scored `per_doc` entry."""
    return _replay_doc(doc, review_threshold, schema)[1]


def flagged_by_quotes(routing: Routing) -> bool:
    """True when a quote that disagrees with its value flagged the field."""
    return routing.review_reason is not None and routing.review_reason.startswith(
        REVIEW_REASON_QUOTE_PREFIX
    )


def flagged_by_checks(routing: Routing) -> bool:
    """True when a failing deterministic check, arithmetic or quote, not the
    model's doubt, flagged the field."""
    return flagged_by_quotes(routing) or (
        routing.review_reason is not None
        and routing.review_reason.startswith(REVIEW_REASON_RECONCILIATION_PREFIX)
    )


def reconcile_label(
    fields: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]] | None = None
) -> Report:
    """The report for a label's fields, replayed as a fully confident answer.
    With the label's gold evidence its quotes are checked against its values
    too."""
    return replay(label_tool_input(fields, evidence=evidence), _FULL_CONFIDENCE, _LABEL_SCHEMA)[0]


def failing_labels(cases: Sequence[EvalCase]) -> list[tuple[str, Report]]:
    """Every label whose own numbers, or whose gold quotes, fail a check, with
    its report. Empty for a consistent dataset, which the mutation eval
    requires."""
    failing = []
    for case in cases:
        report = reconcile_label(case.fields, case.evidence)
        if report.status == STATUS_FAIL:
            failing.append((case.doc_id, report))
    return failing


# --- caught by checks, per run ----------------------------------------------


@dataclass(frozen=True)
class CheckMetrics:
    """What the checks do with one run's answers: documents that fail,
    incorrect fields a failing check flagged, and correct fields it flagged
    as well (a check cannot always tell which operand is wrong). The checks are
    the arithmetic and the quote agreement; `*_by_quotes` count the part of
    each number that is the quotes' (they are included in it)."""

    rules_version: str
    n_docs: int
    n_docs_failing: int
    n_incorrect: int
    n_caught_by_checks: int
    n_correct_flagged: int
    n_caught_by_quotes: int = 0
    n_correct_flagged_by_quotes: int = 0
    # Whether the run's schema carries evidence, so the quote check could run.
    quotes_checkable: bool = False

    @property
    def caught_by_checks(self) -> float | None:
        """None when no field was incorrect, so there was nothing to catch."""
        return self.n_caught_by_checks / self.n_incorrect if self.n_incorrect else None

    @property
    def caught_by_quotes(self) -> float | None:
        """The part of `caught_by_checks` that is the quotes'. None when no field
        was incorrect, and for a run whose schema has no evidence, where there
        is no quote to check."""
        if not self.quotes_checkable or not self.n_incorrect:
            return None
        return self.n_caught_by_quotes / self.n_incorrect


def check_metrics(run: Mapping[str, Any]) -> CheckMetrics | None:
    """The check metrics of one eval artifact (or `RunResult.to_dict()`), or
    None when the run has no summary, no scored document with answers, or a
    prompt version this code has no schema for (warned about on stderr, so one
    old artifact never stops a report rendering). The run is only read."""
    if run["summary"] is None:
        return None
    schema = SCHEMAS.get(run["prompt_version"])
    if schema is None:
        print(
            f"warning: no extraction schema for prompt version {run['prompt_version']!r} "
            f"(known: {', '.join(sorted(SCHEMAS))}); caught_by_checks not computed",
            file=sys.stderr,
        )
        return None
    n_docs = n_docs_failing = n_incorrect = n_caught = n_correct_flagged = 0
    n_caught_by_quotes = n_correct_flagged_by_quotes = 0
    for doc in run["per_doc"]:
        if doc.get("status") != "scored" or "fields" not in doc:
            continue
        report, routing = _replay_doc(doc, run["review_threshold"], schema)
        n_docs += 1
        n_docs_failing += report.status == STATUS_FAIL
        for name, field in doc["fields"].items():
            flagged = flagged_by_checks(routing[name])
            by_quotes = flagged_by_quotes(routing[name])
            if field["correct"]:
                n_correct_flagged += flagged
                n_correct_flagged_by_quotes += by_quotes
            else:
                n_incorrect += 1
                n_caught += flagged
                n_caught_by_quotes += by_quotes
    if n_docs == 0:
        return None
    return CheckMetrics(
        RULES_VERSION,
        n_docs,
        n_docs_failing,
        n_incorrect,
        n_caught,
        n_correct_flagged,
        n_caught_by_quotes,
        n_correct_flagged_by_quotes,
        quotes_checkable=schema.evidence,
    )


# --- reconciliation eval: break one printed number at a time -----------------

# README markers around the reconciliation eval's table, maintained by
# evals/run_checks.py --update-readme the way report.py maintains EVAL_TABLE.
CHECKS_START_MARKER = "<!-- CHECKS_TABLE:START -->"
CHECKS_END_MARKER = "<!-- CHECKS_TABLE:END -->"

MUTATION_KINDS = (
    "one_minor_unit",
    "transposed_digits",
    "misread_digit",
    "dropped_point",
    "extra_leading_digit",
    "wrong_row",
)

# What the leftmost digit is misread as: the pairs a reader confuses (0/8,
# 1/7, 2/7, 3/8, 4/9, 5/6, 6/5, 7/1, 8/3, 9/4). Fixed, so the eval is exact.
_MISREAD = {"0": "8", "1": "7", "2": "7", "3": "8", "4": "9", "5": "6", "6": "5", "7": "1", "8": "3", "9": "4"}

# The amount every Example cell is derived from.
_EXAMPLE = "46.25"


def _one_minor_unit(printed: str, minor_units: int) -> str | None:
    bumped = to_decimal(printed) + Decimal(1).scaleb(-minor_units)
    return to_api_string(bumped, minor_units)


def _transposed_digits(printed: str, minor_units: int) -> str | None:
    """Swap the rightmost pair of neighbouring digits that differ, reading
    across the decimal point; None when every neighbouring pair is equal."""
    positions = [i for i, char in enumerate(printed) if char.isdigit()]
    for left, right in zip(reversed(positions[:-1]), reversed(positions[1:]), strict=True):
        if printed[left] != printed[right]:
            chars = list(printed)
            chars[left], chars[right] = chars[right], chars[left]
            return "".join(chars)
    return None


def _misread_digit(printed: str, minor_units: int) -> str | None:
    index = next(i for i, char in enumerate(printed) if char.isdigit())
    return printed[:index] + _MISREAD[printed[index]] + printed[index + 1 :]


def _dropped_point(printed: str, minor_units: int) -> str | None:
    return printed.replace(".", "") if "." in printed else None


def _extra_leading_digit(printed: str, minor_units: int) -> str | None:
    sign = "-" if printed.startswith("-") else ""
    return f"{sign}1{printed.removeprefix('-')}"


# The kinds that rewrite a printed string (wrong_row copies another cell).
_STRING_MUTATORS: dict[str, Callable[[str, int], str | None]] = {
    "one_minor_unit": _one_minor_unit,
    "transposed_digits": _transposed_digits,
    "misread_digit": _misread_digit,
    "dropped_point": _dropped_point,
    "extra_leading_digit": _extra_leading_digit,
}

# "line_items[3].total" -> "line total", "tax_lines[0].base" -> "tax line base",
# for naming a cell kind in prose.
_ROW_CELL = re.compile(r"(line_items|tax_lines)\[\d+\]\.")
_ROW_CELL_NAMES = {"line_items": "line ", "tax_lines": "tax line "}

_KIND_LABELS = {
    "one_minor_unit": "One minor unit off",
    "transposed_digits": "Two digits swapped",
    "misread_digit": "Leading digit misread",
    "dropped_point": "Decimal point dropped",
    "extra_leading_digit": "Extra leading digit",
    "wrong_row": "Wrong row read",
}
_WRONG_ROW_EXAMPLE = "total read as the subtotal"


@dataclass(frozen=True)
class Mutation:
    """One broken cell: where, how, and the value it had and now has."""

    doc_id: str
    path: str
    kind: str
    original: Decimal
    mutated: Decimal


@dataclass(frozen=True)
class MutationTally:
    """What the engine did with the mutations of one kind. `other_fields_flagged`
    is the number of flags the mutations put on fields other than the one they
    broke, summed; `other_fields_per_mutation` is the mean."""

    kind: str
    mutations: int
    detected: int
    flagged: int
    fixed: int
    wrong_fixes: int
    other_fields_flagged: int

    @property
    def other_fields_per_mutation(self) -> float:
        return self.other_fields_flagged / self.mutations if self.mutations else 0.0


@dataclass(frozen=True)
class MutationEval:
    rules_version: str
    n_labels: int
    by_kind: tuple[MutationTally, ...]
    misses: tuple[Mutation, ...]

    @property
    def overall(self) -> MutationTally:
        return MutationTally(
            "all",
            *(sum(getattr(t, name) for t in self.by_kind) for name in _TALLY_COUNTS),
        )


_TALLY_COUNTS = (
    "mutations",
    "detected",
    "flagged",
    "fixed",
    "wrong_fixes",
    "other_fields_flagged",
)


@dataclass(frozen=True)
class _Cell:
    path: str
    original: Decimal
    printed: str
    money: bool


def _cells(fields: Mapping[str, Any], minor_units: int) -> list[_Cell]:
    """Every non-null amount a label carries: the header amounts (subtotal,
    discount, shipping, tax, tip, total, amount paid, amount due), each line's
    quantity, unit price, and total, and each tax line's rate, base, and amount."""
    cells = []

    def add(path: str, raw: Any, money: bool) -> None:
        if raw is None:
            return
        value = to_decimal(raw)
        printed = to_api_string(value, minor_units) if money else to_plain_string(value)
        cells.append(_Cell(path, value, printed, money))

    for name in _HEADER_MONEY:
        add(name, fields.get(name), True)
    for index, row in enumerate(fields.get("line_items") or []):
        add(line_path(index, "quantity"), row.get("quantity"), False)
        add(line_path(index, "unit_price"), row.get("unit_price"), True)
        add(line_path(index, "total"), row.get("total"), True)
    for index, row in enumerate(fields.get("tax_lines") or []):
        add(tax_line_path(index, "rate"), row.get("rate"), False)
        add(tax_line_path(index, "base"), row.get("base"), True)
        add(tax_line_path(index, "amount"), row.get("amount"), True)
    return cells


def _mutations(
    doc_id: str,
    fields: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Mutation]:
    """Every way of breaking one printed number of a label. A number printed once
    is one number: given the label's gold `evidence`, a cell that is only the
    twin of an earlier cell (the tax line amount that is the header tax read a
    second time, see `_coupled_paths`) is not broken again, since breaking it
    would replay the same document."""
    minor_units = currency_precision(fields.get("currency")).minor_units
    cells = _cells(fields, minor_units)
    by_path = {cell.path: cell for cell in cells}
    twins: set[str] = set()
    mutations = []
    for cell in cells:
        if cell.path in twins:
            continue
        if evidence is not None:
            twins.update(_twin_paths(fields, evidence, cell.path, cell.original))
        for kind in MUTATION_KINDS:
            if kind == "wrong_row":
                candidates = [
                    by_path[source].original
                    for source in _wrong_row_sources(cell.path)
                    if source in by_path
                ]
            else:
                if kind == "one_minor_unit" and not cell.money:
                    continue
                text = _STRING_MUTATORS[kind](cell.printed, minor_units)
                candidates = [to_decimal(text)] if text is not None else []
            for mutated in dict.fromkeys(candidates):
                if mutated != cell.original:
                    mutations.append(Mutation(doc_id, cell.path, kind, cell.original, mutated))
    return mutations


def _wrong_row_sources(path: str) -> tuple[str, ...]:
    """The cells a reader could take this one's value from by looking at the
    wrong row: a total from the subtotal or the balance due, a subtotal or a
    balance due from the total, a line total from its own unit price, a tax
    line's amount from the base it is charged on."""
    if path == "total":
        return ("subtotal", "amount_due")
    if path in ("subtotal", "amount_due"):
        return ("total",)
    if path.startswith("line_items[") and path.endswith(".total"):
        return (path.removesuffix(".total") + ".unit_price",)
    if path.startswith("tax_lines[") and path.endswith(".amount"):
        return (path.removesuffix(".amount") + ".base",)
    return ()


def _twin_paths(
    fields: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    path: str,
    original: Decimal,
) -> list[str]:
    """The other paths a misread of the cell at `path` is also a misread of: those
    whose gold evidence is the same page and quote, and whose value is the same
    number. A tax printed once is read into `tax` and into the amount of its tax
    line, so a reader who misreads the printed number misreads both; a document
    that prints it twice (a tax line and a total-tax line) has two printed numbers
    and couples nothing. The cells of a line item are never coupled: a unit price
    and a line total share a printed line, and are two printed numbers even when
    they are equal."""
    cited = evidence.get(path)
    if cited is None or path.startswith("line_items["):
        return []
    twins = []
    for other, value in label_values(fields).items():
        if other == path or other.startswith("line_items[") or evidence.get(other) != cited:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if to_decimal(value) == original:
            twins.append(other)
    return twins


def _coupled_paths(
    fields: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]], mutation: Mutation
) -> list[str]:
    """The other paths a mutation also breaks: its twins (see `_twin_paths`)."""
    return _twin_paths(fields, evidence, mutation.path, mutation.original)


def _set_cell(tool_input: dict[str, Any], path: str, leaf: dict[str, Any]) -> None:
    """Put a broken leaf at a path: a header amount, or one cell of a row."""
    if "[" not in path:
        tool_input[path] = leaf
        return
    head, key = path.rsplit(".", 1)
    name, index = head.removesuffix("]").split("[")
    tool_input[name]["value"][int(index)][key] = leaf


def _broken_tool_input(
    fields: Mapping[str, Any], mutation: Mutation, evidence: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    tool_input = label_tool_input(fields, _FULL_CONFIDENCE)
    for path in (mutation.path, *_coupled_paths(fields, evidence, mutation)):
        _set_cell(tool_input, path, _leaf(mutation.mutated, _FULL_CONFIDENCE))
    return tool_input


def _outcome(
    mutation: Mutation, report: Report, routing: Mapping[str, Routing]
) -> dict[str, int]:
    """What the engine did with one broken document, as the counts a tally adds
    up: whether a check failed, whether the broken field was routed to review
    because of one, whether the engine's suggestion restored exactly the
    original value (`fixed`) or changed something else (`wrong_fixes`), and how
    many other fields were flagged."""
    broken_field = field_name_of(mutation.path)
    suggestion = report.suggestion
    fixed = suggestion is not None and (
        suggestion.path == mutation.path and suggestion.value == mutation.original
    )
    return {
        "detected": int(report.status == STATUS_FAIL),
        "flagged": int(flagged_by_checks(routing[broken_field])),
        "quote_flagged": int(flagged_by_quotes(routing[broken_field])),
        "fixed": int(fixed),
        "wrong_fixes": int(suggestion is not None and not fixed),
        "other_fields_flagged": sum(
            flagged_by_checks(r) for name, r in routing.items() if name != broken_field
        ),
    }


def _require_consistent(cases: Sequence[EvalCase]) -> None:
    failing = failing_labels(cases)
    if failing:
        raise ValueError(
            "labels that fail the checks unmutated: " + ", ".join(doc_id for doc_id, _ in failing)
        )


def run_mutation_eval(cases: Sequence[EvalCase]) -> MutationEval:
    """Break every cell of every label in each way in `MUTATION_KINDS`, replay,
    and tally per kind. Raises ValueError naming any label that fails the
    checks unmutated, since a miss would then mean nothing."""
    _require_consistent(cases)

    counts = {kind: dict.fromkeys(_TALLY_COUNTS, 0) for kind in MUTATION_KINDS}
    misses = []
    for case in cases:
        for mutation in _mutations(case.doc_id, case.fields, case.evidence):
            report, routing = replay(
                _broken_tool_input(case.fields, mutation, case.evidence), _FULL_CONFIDENCE, _LABEL_SCHEMA
            )
            outcome = _outcome(mutation, report, routing)
            tally = counts[mutation.kind]
            tally["mutations"] += 1
            if not outcome["detected"]:
                misses.append(mutation)
            for name in _TALLY_COUNTS[1:]:
                tally[name] += outcome[name]
    return MutationEval(
        RULES_VERSION,
        len(cases),
        tuple(MutationTally(kind, **counts[kind]) for kind in MUTATION_KINDS),
        tuple(misses),
    )


# --- quote eval: the same broken documents, with quotes in front of the engine ---

# README markers around the quote eval's table, maintained like CHECKS_TABLE.
QUOTES_START_MARKER = "<!-- QUOTES_TABLE:START -->"
QUOTES_END_MARKER = "<!-- QUOTES_TABLE:END -->"

# How the quotes of a broken document are written: none at all (the arithmetic
# alone), faithful quotes on a misread value, the same misread in the value and
# its quote, and no quote on the broken cell.
QUOTE_VARIANTS = ("none", "faithful", "same_misread", "missing")

_VARIANT_LABELS = {
    "none": "No quotes",
    "faithful": "Value misread, quote shows what is printed",
    "same_misread": "Value and quote misread the same way",
    "missing": "No quote on the broken cell",
}

_QUOTE_TALLY_COUNTS = ("mutations", "detected", "quote_flagged", "flagged", "fixed", "wrong_fixes")


@dataclass(frozen=True)
class QuoteTally:
    """What the engine did with every mutation under one way of writing quotes.
    `quote_flagged` counts the broken fields routed to review because their
    quote disagrees with their value, `flagged` those routed because of any
    failing check, quote or arithmetic."""

    variant: str
    mutations: int
    detected: int
    quote_flagged: int
    flagged: int
    fixed: int
    wrong_fixes: int


@dataclass(frozen=True)
class QuoteEval:
    rules_version: str
    n_labels: int
    by_variant: tuple[QuoteTally, ...]


def _grouped(text: str) -> str:
    """`2226.78` as `2,226.78`."""
    return format(Decimal(text), ",")


def _number_styles(text: str) -> list[str]:
    """The ways a document may print the number `text` (plain, dot-decimal): as
    is, with US thousands separators, with a decimal comma, with European
    separators."""
    grouped = _grouped(text)
    return [
        text,
        grouped,
        text.replace(".", ","),
        grouped.replace(",", "\x00").replace(".", ",").replace("\x00", "."),
    ]


def _replace_number(quote: str, before: str, after: str) -> str:
    """`quote` with the number `before` written as `after`, where `before` is a
    whole number (not the middle of a longer one)."""
    pattern = r"(?<![\d.,])" + re.escape(before) + r"(?![\d]|[.,]\d)"
    return re.sub(pattern, lambda _match: after, quote)


def _misread_quote(
    cited: Mapping[str, Any] | None, mutation: Mutation, minor_units: int, money: bool
) -> dict[str, Any] | None:
    """The evidence of a cell whose reader misread the number the same way in the
    value and in the quote: the gold quote with the original number written as
    the broken one, in whatever style the quote prints it. None when there is no
    gold quote or the number is not printed in it in a style tried."""
    if cited is None:
        return None
    render = (lambda value: to_api_string(value, minor_units)) if money else to_plain_string
    for before, after in zip(
        _number_styles(render(mutation.original)), _number_styles(render(mutation.mutated)), strict=True
    ):
        quote = _replace_number(cited["quote"], before, after)
        if quote != cited["quote"]:
            return {**cited, "quote": quote}
    return None


def _quoted_tool_input(
    fields: Mapping[str, Any],
    mutation: Mutation,
    evidence: Mapping[str, Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    """The broken document for a quote variant: every leaf cites its gold
    evidence, except that the broken cell (and the twins of the same printed
    number) cite what the variant says. A misread quote that cannot be written
    (see `_misread_quote`) leaves its cell without one."""
    if variant == "none":
        return _broken_tool_input(fields, mutation, evidence)
    minor_units = currency_precision(fields.get("currency")).minor_units
    cell_money = {cell.path: cell.money for cell in _cells(fields, minor_units)}
    tool_input = label_tool_input(fields, _FULL_CONFIDENCE, evidence)
    for path in (mutation.path, *_coupled_paths(fields, evidence, mutation)):
        cited = evidence.get(path)
        if variant == "same_misread":
            cited = _misread_quote(cited, mutation, minor_units, cell_money[path])
        elif variant == "missing":
            cited = None
        leaf = _leaf(mutation.mutated, _FULL_CONFIDENCE)
        leaf["evidence"] = dict(cited) if cited is not None else None
        _set_cell(tool_input, path, leaf)
    return tool_input


def run_quote_eval(cases: Sequence[EvalCase], baseline: MutationEval | None = None) -> QuoteEval:
    """Replay every mutation of every label under each way of writing quotes
    (`QUOTE_VARIANTS`) and tally per variant. A label's gold evidence is what an
    ideal reader would quote for each of its values, so every leaf that has gold
    evidence cites it; only the broken cell differs between variants. Raises
    ValueError naming any label that fails the checks unmutated, as the mutation
    eval does.

    The "none" variant is the mutation eval on the same cases, so a caller that
    has already run it passes its result as `baseline` and nothing is replayed
    twice; without one, it is run here."""
    _require_consistent(cases)
    if baseline is None:
        baseline = run_mutation_eval(cases)
    overall = baseline.overall

    counts = {variant: dict.fromkeys(_QUOTE_TALLY_COUNTS, 0) for variant in QUOTE_VARIANTS}
    counts["none"] = {
        "mutations": overall.mutations,
        "detected": overall.detected,
        "quote_flagged": 0,
        "flagged": overall.flagged,
        "fixed": overall.fixed,
        "wrong_fixes": overall.wrong_fixes,
    }
    replayed = [variant for variant in QUOTE_VARIANTS if variant != "none"]
    for case in cases:
        for mutation in _mutations(case.doc_id, case.fields, case.evidence):
            for variant in replayed:
                tool_input = _quoted_tool_input(case.fields, mutation, case.evidence, variant)
                report, routing = replay(tool_input, _FULL_CONFIDENCE, _LABEL_SCHEMA)
                outcome = _outcome(mutation, report, routing)
                tally = counts[variant]
                tally["mutations"] += 1
                for name in _QUOTE_TALLY_COUNTS[1:]:
                    tally[name] += outcome[name]
    return QuoteEval(
        RULES_VERSION,
        len(cases),
        tuple(QuoteTally(variant, **counts[variant]) for variant in QUOTE_VARIANTS),
    )


# --- rendering ---------------------------------------------------------------

_CHECKS_HEADER = (
    "Misread",
    "Example",
    "Mutations",
    "Detected",
    "Field flagged",
    "Fix suggested",
    "Wrong fixes",
    "Correct fields flagged (mean)",
)


def _example(kind: str) -> str:
    if kind == "wrong_row":
        return _WRONG_ROW_EXAMPLE
    mutated = _STRING_MUTATORS[kind](_EXAMPLE, 2)
    return f"{_EXAMPLE} -> {mutated}"


def _share(count: int, total: int) -> str:
    return f"{count} ({count / total * 100:.1f}%)" if total else "n/a"


def _tally_row(label: str, example: str, tally: MutationTally) -> str:
    cells = [
        label,
        example,
        str(tally.mutations),
        _share(tally.detected, tally.mutations),
        _share(tally.flagged, tally.mutations),
        _share(tally.fixed, tally.mutations),
        str(tally.wrong_fixes),
        f"{tally.other_fields_per_mutation:.2f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _cell_name(path: str) -> str:
    """`line_items[3].total` as `line total`, `tax_lines[0].base` as `tax line
    base`, and a header path as its words."""
    named = _ROW_CELL.sub(lambda match: _ROW_CELL_NAMES[match.group(1)], path)
    return named.replace("_", " ")


def _misses_sentence(misses: Sequence[Mutation]) -> str:
    """The undetected mutations, grouped by kind and cell, from the data
    rather than written down, so the caption cannot drift from the table."""
    if not misses:
        return "Every mutation was detected."
    groups = Counter((mutation.kind, _cell_name(mutation.path)) for mutation in misses)
    parts = [
        f"{count} {_KIND_LABELS[kind].lower()} on {cell}"
        for (kind, cell), count in sorted(groups.items())
    ]
    return f"Undetected: {'; '.join(parts)}."


def render_checks_markdown(result: MutationEval) -> str:
    """The reconciliation eval as a GitHub-markdown table with a caption, for
    README between the CHECKS_TABLE markers."""
    rows = [
        _tally_row(_KIND_LABELS[t.kind], _example(t.kind), t) for t in result.by_kind
    ]
    rows.append(_tally_row("All", "", result.overall))
    header = "| " + " | ".join(_CHECKS_HEADER) + " |"
    separator = "| " + " | ".join(["---"] * len(_CHECKS_HEADER)) + " |"
    table = "\n".join([header, separator, *rows])
    caption = (
        f"*All {result.n_labels} labels pass {result.rules_version} unmutated; each row then breaks "
        "one printed amount (every header amount, every line's quantity, unit price, and total, and "
        "every tax line's rate, base, and amount) "
        "in one way and replays the document with full confidence, so only the arithmetic can notice; "
        "a number printed once is broken everywhere it was read (a tax that is read as the tax and "
        "as its only tax line is broken in both, once), and the document keeps its other amounts and tax "
        "lines as the label has them, so a broken amount is checked against those too. "
        "Detected means a check fails. Field flagged means the broken field is routed to review "
        "because of a failing check. Fix suggested means the engine proposes exactly the original "
        "value for that field; Wrong fixes counts any other suggestion, a change to something that "
        "was read correctly (a tax and its only tax line that are one reading are one candidate, so a "
        "misread that reaches both is not blamed on the total). "
        "Quantities, unit prices, tax rates, and tax bases are never suggested, so those misreads are "
        "flagged but not fixed. Correct fields flagged is "
        f"the mean number of other fields a mutation sends to review. {_misses_sentence(result.misses)}*"
    )
    return f"{table}\n\n{caption}"


_QUOTES_HEADER = (
    "Quotes",
    "Mutations",
    "Detected",
    "Field flagged by a quote",
    "Field flagged",
    "Fix suggested",
    "Wrong fixes",
)


def _quote_row(tally: QuoteTally) -> str:
    cells = [
        _VARIANT_LABELS[tally.variant],
        str(tally.mutations),
        _share(tally.detected, tally.mutations),
        _share(tally.quote_flagged, tally.mutations),
        _share(tally.flagged, tally.mutations),
        _share(tally.fixed, tally.mutations),
        str(tally.wrong_fixes),
    ]
    return "| " + " | ".join(cells) + " |"


def render_quotes_markdown(result: QuoteEval) -> str:
    """The quote eval as a GitHub-markdown table with a caption, for README
    between the QUOTES_TABLE markers."""
    header = "| " + " | ".join(_QUOTES_HEADER) + " |"
    separator = "| " + " | ".join(["---"] * len(_QUOTES_HEADER)) + " |"
    table = "\n".join([header, separator, *(_quote_row(t) for t in result.by_variant)])
    caption = (
        f"*The same {result.by_variant[0].mutations} broken documents as the table above, replayed with evidence "
        "quotes. Every leaf the gold label cites carries its gold quote, the line as the document prints it; "
        "the rows differ only in the broken cell, and in the cell of any twin of the same printed number. "
        "A misread value keeps its faithful quote: the quote check sees the value is not the number printed "
        "beside it. A value and quote misread the same way agree with each other, so the quote check is "
        "silent and the arithmetic does all the work, as it does with no quote or a missing one. "
        "Detected, Field flagged, Fix suggested and Wrong fixes mean what they mean in the table above; "
        "Field flagged by a quote counts the broken fields whose review reason is `quote:<path>`. "
        "What no row can show is a fabricated quote that agrees with a fabricated value.*"
    )
    return f"{table}\n\n{caption}"
