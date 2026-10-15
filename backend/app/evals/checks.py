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
    SCHEMA_V1,
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
)
from app.review_routing import Routing, field_name_of, route

# Confidence every replayed label cell carries: high enough that no field is
# flagged for doubt, so a flag in the mutation eval is the arithmetic's doing.
_FULL_CONFIDENCE = 1.0

# The labels carry every v2 field, so the label replays and the mutation eval
# run on SCHEMA_V2; a scored run replays under its own prompt version. The
# mutation eval breaks only the amounts v1 also reads (subtotal, tax, total, and
# the line cells): the v2 amounts stay as the label has them, and are what the
# broken ones are checked against.
_LABEL_SCHEMA = SCHEMA_V2
_HEADER_MONEY = SCHEMA_V1.money_field_names
_LINE_KEYS = tuple(ARRAY_CELLS["line_items"])


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
    the fields the entry carries are replayed and returned."""
    tool_input = {
        name: _leaf(field["actual"], field["confidence"]) for name, field in doc["fields"].items()
    }
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

    @property
    def caught_by_checks(self) -> float | None:
        """None when no field was incorrect, so there was nothing to catch."""
        return self.n_caught_by_checks / self.n_incorrect if self.n_incorrect else None


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

# "line_items[3].total" -> "line total", for naming a cell kind in prose.
_LINE_CELL = re.compile(r"line_items\[\d+\]\.")

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
    """Every non-null amount a label carries that the checks read: subtotal,
    tax, total, and each line's quantity, unit price, and total."""
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
    return cells


def _mutations(doc_id: str, fields: Mapping[str, Any]) -> list[Mutation]:
    minor_units = currency_precision(fields.get("currency")).minor_units
    cells = _cells(fields, minor_units)
    by_path = {cell.path: cell for cell in cells}
    mutations = []
    for cell in cells:
        for kind in MUTATION_KINDS:
            if kind == "wrong_row":
                source = _wrong_row_source(cell.path)
                mutated = by_path[source].original if source in by_path else None
            else:
                if kind == "one_minor_unit" and not cell.money:
                    continue
                text = _STRING_MUTATORS[kind](cell.printed, minor_units)
                mutated = to_decimal(text) if text is not None else None
            if mutated is not None and mutated != cell.original:
                mutations.append(Mutation(doc_id, cell.path, kind, cell.original, mutated))
    return mutations


def _wrong_row_source(path: str) -> str | None:
    """The cell a reader could take this one's value from by looking at the
    wrong row: a total from the subtotal, a subtotal from the total, a line
    total from its own unit price."""
    if path == "total":
        return "subtotal"
    if path == "subtotal":
        return "total"
    if path.startswith("line_items[") and path.endswith(".total"):
        return path.removesuffix(".total") + ".unit_price"
    return None


def _coupled_paths(
    fields: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]], mutation: Mutation
) -> list[str]:
    """The other paths a misread of this cell is also a misread of: those whose
    gold evidence is the same page and quote, and whose value is the same number.
    A tax printed once is read into `tax` and into the amount of its tax line, so
    a reader who misreads the printed number misreads both; a document that
    prints it twice (a tax line and a total-tax line) has two printed numbers and
    couples nothing. The cells of a line item are never coupled: a unit price and
    a line total share a printed line, and are two printed numbers even when they
    are equal."""
    cited = evidence.get(mutation.path)
    if cited is None or mutation.path.startswith("line_items["):
        return []
    coupled = []
    for path, value in label_values(fields).items():
        if path == mutation.path or path.startswith("line_items[") or evidence.get(path) != cited:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if to_decimal(value) == mutation.original:
            coupled.append(path)
    return coupled


def _set_cell(tool_input: dict[str, Any], path: str, value: Decimal) -> None:
    """Put a broken value at a path: a header amount, or one cell of a row."""
    if "[" not in path:
        tool_input[path] = _leaf(value, _FULL_CONFIDENCE)
        return
    head, key = path.rsplit(".", 1)
    name, index = head.removesuffix("]").split("[")
    tool_input[name]["value"][int(index)][key] = _leaf(value, _FULL_CONFIDENCE)


def _broken_tool_input(
    fields: Mapping[str, Any], mutation: Mutation, evidence: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    tool_input = label_tool_input(fields, _FULL_CONFIDENCE)
    for path in (mutation.path, *_coupled_paths(fields, evidence, mutation)):
        _set_cell(tool_input, path, mutation.mutated)
    return tool_input


def run_mutation_eval(cases: Sequence[EvalCase]) -> MutationEval:
    """Break every cell of every label in each way in `MUTATION_KINDS`, replay,
    and tally per kind. Raises ValueError naming any label that fails the
    checks unmutated, since a miss would then mean nothing."""
    failing = failing_labels(cases)
    if failing:
        raise ValueError(
            "labels that fail the checks unmutated: " + ", ".join(doc_id for doc_id, _ in failing)
        )

    counts = {kind: dict.fromkeys(_TALLY_COUNTS, 0) for kind in MUTATION_KINDS}
    misses = []
    for case in cases:
        for mutation in _mutations(case.doc_id, case.fields):
            report, routing = replay(
                _broken_tool_input(case.fields, mutation, case.evidence), _FULL_CONFIDENCE, _LABEL_SCHEMA
            )
            tally = counts[mutation.kind]
            tally["mutations"] += 1
            if report.status == STATUS_FAIL:
                tally["detected"] += 1
            else:
                misses.append(mutation)
            broken_field = field_name_of(mutation.path)
            tally["flagged"] += flagged_by_checks(routing[broken_field])
            tally["other_fields_flagged"] += sum(
                flagged_by_checks(r) for name, r in routing.items() if name != broken_field
            )
            suggestion = report.suggestion
            if suggestion is not None:
                if suggestion.path == mutation.path and suggestion.value == mutation.original:
                    tally["fixed"] += 1
                else:
                    tally["wrong_fixes"] += 1
    return MutationEval(
        RULES_VERSION,
        len(cases),
        tuple(MutationTally(kind, **counts[kind]) for kind in MUTATION_KINDS),
        tuple(misses),
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


def _misses_sentence(misses: Sequence[Mutation]) -> str:
    """The undetected mutations, grouped by kind and cell, from the data
    rather than written down, so the caption cannot drift from the table."""
    if not misses:
        return "Every mutation was detected."
    groups = Counter(
        (mutation.kind, _LINE_CELL.sub("line ", mutation.path)) for mutation in misses
    )
    parts = [
        f"{count} {_KIND_LABELS[kind].lower()} on {cell.replace('_', ' ')}"
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
        "one printed amount (subtotal, tax, total, and every line's quantity, unit price, and total) "
        "in one way and replays the document with full confidence, so only the arithmetic can notice; "
        "a number printed once is broken everywhere it was read (a tax that is read as the tax and "
        "as its only tax line is broken in both), and the document keeps its other amounts and tax "
        "lines as the label has them, so a broken amount is checked against those too. "
        "Detected means a check fails. Field flagged means the broken field is routed to review "
        "because of a failing check. Fix suggested means the engine proposes exactly the original "
        "value for that field; Wrong fixes counts any other suggestion (it happens when a misread "
        "that reaches two fields leaves the arithmetic agreeing on both, so the total is blamed). "
        "Quantities and unit prices "
        "are never suggested, so those misreads are flagged but not fixed. Correct fields flagged is "
        f"the mean number of other fields a mutation sends to review. {_misses_sentence(result.misses)}*"
    )
    return f"{table}\n\n{caption}"
