"""Markdown reporting over eval-run artifacts: render the accuracy/ops
history tables consumed by README.md and evals/results/latest.md, and
inject them into README between the EVAL_TABLE marker comments.

Offline: reads only the JSON artifacts app.evals.runner.write_result
already wrote, plus the README.md text -- no DB, no network, no
re-running extraction. Two columns are computed, never written into an
artifact: Core, the accuracy over the seven v1 fields of the 25 original
documents, read from each run's per_doc entries so that runs of different
prompt versions compare on the same ground; and Caught by checks (with its
Caught by quotes part), which replays each run's stored answers through the
reconciliation rules every time the tables are rendered (app.evals.checks),
so it follows the rules version. Deliberately tolerant of a corrupt or
half-written artifact file (warns to stderr, skips it) so one bad run
can't take down reporting for every other run in evals/results/.
"""

import json
import sys
from pathlib import Path
from typing import Any

from app.evals.checks import check_metrics, doc_routing
from app.evals.dataset import CORE_DOC_IDS, EVALS_DIR
from app.extraction_fields import SCHEMA_V1, SCHEMAS
from app.reconcile import RULES_VERSION
from app.review_routing import Routing

RESULTS_DIR = EVALS_DIR / "results"
LATEST_MD_PATH = RESULTS_DIR / "latest.md"

START_MARKER = "<!-- EVAL_TABLE:START -->"
END_MARKER = "<!-- EVAL_TABLE:END -->"

# Must match the placeholder committed in README.md between the markers
# before the first eval run -- kept as one constant so the two never
# drift out of sync with each other.
NO_RUNS_PLACEHOLDER = (
    "*No eval runs recorded yet - table appears after the first "
    "`evals/run.py` run with `--update-readme`.*"
)

# Keys render_tables/render_latest_md dereference directly (some nested,
# inside `summary`) -- an artifact missing any of these can't be
# rendered without a KeyError, so load_results treats that the same as
# an unparseable file: warn to stderr and skip, rather than letting one
# bad artifact take down the whole report.
_REQUIRED_KEYS = (
    "model",
    "prompt_version",
    "dataset_version",
    "started_at_utc",
    "summary",
    "n_scored",
    "n_errors",
    "total_cost_usd",
    "total_error_cost_usd",
    "mean_cost_per_doc",
    "latency_p50_ms",
    "latency_p95_ms",
    "per_doc",
    "review_threshold",
)

# (column label, per_field_accuracy key) -- in the table's display
# order, which is not the schema's field order (line_items moves to the
# end, after the amount fields, to match the spec's column layout).
_ACCURACY_FIELD_COLUMNS = (
    ("Vendor", "vendor"),
    ("Date", "document_date"),
    ("Currency", "currency"),
    ("Subtotal", "subtotal"),
    ("Tax", "tax"),
    ("Total", "total"),
    ("Line items", "line_items"),
)

_CORE_HEADER = "Core (v1 fields, docs 001-025)"

_ACCURACY_HEADER = (
    "Model",
    "Prompt",
    "Docs",
    "Overall",
    _CORE_HEADER,
    *(label for label, _key in _ACCURACY_FIELD_COLUMNS),
)
_OPS_HEADER = (
    "Model",
    "Prompt",
    "Conf ✓/✗",
    "Caught by review",
    "Caught by checks",
    "Caught by quotes",
    "Halluc.",
    "$/doc",
    "Total $",
    "p50/p95 latency",
)

_ACCURACY_CAPTION = (
    "*Percentages are field-level accuracy (see this README's Evals section "
    "for the per-field match rules); Docs is n_scored for that run, with "
    "`+N err` appended when the run had N extraction errors. Each row is one "
    "eval run, tied to its own model / prompt_version / dataset_version, "
    "ordered oldest to newest by started_at_utc. Core is the accuracy over "
    "the seven v1 fields of documents 001-025 only, the ground every prompt "
    "version has been run on, so a v1 run and a v2 run compare there whatever "
    "else each was run on.*"
)
_OPS_CAPTION = (
    "*Conf ✓/✗ is mean model confidence on correct vs. incorrect fields; "
    "Caught by review is the share of incorrect fields whose confidence fell "
    "below that run's review_threshold -- the empirical case for the "
    "human-review queue; Caught by checks is the share of incorrect fields "
    "the deterministic checks route to review: the arithmetic (app/reconcile.py, "
    f"{RULES_VERSION}) and, for a run whose answers carry evidence quotes, the "
    "agreement of each value with the number in its quote (app/quotes.py), "
    "replayed from each run's stored answers whenever this table is "
    "rendered (a field can count in both); Caught by quotes is the part of that share "
    "that the quote check made, n/a for a prompt version whose answers carry no quotes; "
    "$/doc and Total $ are extraction spend in USD; "
    "latency is wall-clock p50/p95 per document.*"
)
_SCHEMA_V2_CAPTION = (
    "*Accuracy is the field-level accuracy of that field over the run's documents. "
    "Values counts the non-null values the model extracted for the field (every cell of "
    "a line item or tax line counts). Evidence coverage is the share of them that cite a "
    "usable quote. Page agreement and Quote agreement are the shares of the values the gold "
    "label cites that name the gold page and copy the gold quote (whitespace collapsed); a "
    "value with no quote, or with a quote and no page, is a miss on both, and the first is "
    "a coverage miss too.*"
)


def load_results(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load every *.json artifact in `results_dir` (default
    EVALS_DIR/"results"), sorted by started_at_utc.

    A file that isn't valid JSON, isn't a JSON object, or is missing one
    of _REQUIRED_KEYS is skipped with a warning printed to stderr -- one
    corrupt or half-written artifact must not prevent the rest of the
    run history from being reported. Returns [] if `results_dir` doesn't
    exist (e.g. no eval has ever been run yet).
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []

    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warning: skipping unreadable eval artifact {path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            print(
                f"warning: skipping eval artifact {path}: not a JSON object "
                f"(got {type(data).__name__})",
                file=sys.stderr,
            )
            continue

        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            print(
                f"warning: skipping eval artifact {path}: missing key(s) {missing}",
                file=sys.stderr,
            )
            continue

        results.append(data)

    results.sort(key=lambda r: r["started_at_utc"])
    return results


# --- cell formatting -------------------------------------------------------


def _esc(value: str) -> str:
    """Escape a literal pipe so it can't be mistaken for a GitHub-markdown
    table column separator -- only expected/actual leaf values (free-form
    strings from a label or a model's output) can plausibly contain one.
    """
    return value.replace("|", "\\|")


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _conf(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _ms(value: float | None) -> str:
    return f"{value:.0f}ms" if value is not None else "n/a"


def _docs_cell(result: dict[str, Any]) -> str:
    n_scored = result["n_scored"]
    n_errors = result["n_errors"]
    return f"{n_scored} +{n_errors} err" if n_errors else str(n_scored)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: tuple[str, ...], rows: list[str]) -> str:
    header_line = _row(list(header))
    sep_line = _row(["---"] * len(header))
    return "\n".join([header_line, sep_line, *rows])


def core_counts(result: dict[str, Any]) -> tuple[int, int]:
    """(correct, scored) over the seven v1 fields of the core documents
    (CORE_DOC_IDS) the run scored, read from its per_doc entries. (0, 0) when it
    scored none of them. Counts rather than a share so two runs can be compared
    field by field."""
    correct = scored = 0
    for doc in result["per_doc"]:
        if doc.get("status") != "scored" or doc["doc_id"] not in CORE_DOC_IDS:
            continue
        for name in SCHEMA_V1.field_names:
            field = doc["fields"].get(name)
            if field is not None:
                scored += 1
                correct += bool(field["correct"])
    return correct, scored


def _core_cell(result: dict[str, Any]) -> str:
    """The core accuracy with its counts (`174/175 (99.4%)`), so a difference of
    one field is readable and a shrunk denominator is visible."""
    correct, scored = core_counts(result)
    return f"{correct}/{scored} ({_pct(correct / scored)})" if scored else "n/a"


def _accuracy_row(result: dict[str, Any]) -> str:
    summary = result["summary"]
    if summary is None:
        overall = "n/a"
        per_field = {key: "n/a" for _label, key in _ACCURACY_FIELD_COLUMNS}
    else:
        overall = _pct(summary["overall_accuracy"])
        per_field_accuracy = summary["per_field_accuracy"]
        per_field = {
            key: _pct(per_field_accuracy.get(key)) for _label, key in _ACCURACY_FIELD_COLUMNS
        }

    return _row(
        [
            result["model"] or "n/a",
            result["prompt_version"] or "n/a",
            _docs_cell(result),
            overall,
            _core_cell(result),
            *(per_field[key] for _label, key in _ACCURACY_FIELD_COLUMNS),
        ]
    )


def _ops_row(result: dict[str, Any]) -> str:
    summary = result["summary"]
    if summary is None:
        conf = "n/a / n/a"
        caught = "n/a"
        halluc = "n/a"
    else:
        conf = f"{_conf(summary['mean_confidence_correct'])} / {_conf(summary['mean_confidence_incorrect'])}"
        caught = _pct(summary["caught_by_review"])
        halluc = str(summary["total_hallucinations"])
    metrics = check_metrics(result)
    caught_by_checks = _pct(metrics.caught_by_checks) if metrics else "n/a"
    caught_by_quotes = _pct(metrics.caught_by_quotes) if metrics else "n/a"

    return _row(
        [
            result["model"] or "n/a",
            result["prompt_version"] or "n/a",
            conf,
            caught,
            caught_by_checks,
            caught_by_quotes,
            halluc,
            f"${result['mean_cost_per_doc']:.4f}",
            f"${result['total_cost_usd']:.4f}",
            f"{_ms(result['latency_p50_ms'])} / {_ms(result['latency_p95_ms'])}",
        ]
    )


_SCHEMA_V2_HEADER = (
    "Field",
    "Accuracy",
    "Values",
    "Evidence coverage",
    "Page agreement",
    "Quote agreement",
)


def _schema_v2_rows(summary: dict[str, Any]) -> list[str]:
    """One row per field of a run with evidence, and a last row for all of them."""
    coverage = summary["evidence_coverage"]
    accuracy = summary["per_field_accuracy"]

    def row(label: str, acc: float | None, counts: dict[str, Any]) -> str:
        return _row(
            [
                label,
                _pct(acc),
                str(counts["leaves"]),
                _pct(counts["coverage"]),
                _pct(counts["page_agreement"]),
                _pct(counts["quote_agreement"]),
            ]
        )

    rows = [row(name, accuracy.get(name), counts) for name, counts in coverage["per_field"].items()]
    rows.append(row("All", summary["overall_accuracy"], coverage))
    return rows


def _schema_v2_section(results: list[dict[str, Any]]) -> str:
    """The Schema v2 fields tables, one per run whose answers carry evidence, or
    an empty string when no run does."""
    tables = []
    for result in results:
        summary = result["summary"]
        if summary is None or summary.get("evidence_coverage") is None:
            continue
        tables.append(
            f"#### {result['model']} / {result['prompt_version']} ({_docs_cell(result)} docs)\n\n"
            + _table(_SCHEMA_V2_HEADER, _schema_v2_rows(summary))
        )
    if not tables:
        return ""
    return "### Schema v2 fields\n\n" + "\n\n".join(tables) + f"\n\n{_SCHEMA_V2_CAPTION}"


def render_tables(results: list[dict[str, Any]]) -> str:
    """Render the GitHub-markdown eval-history tables (accuracy, ops), one row
    per artifact in `results`, plus a one-line caption under each explaining what
    its metrics mean, and, when a run carries evidence, a Schema v2 fields table
    per such run. `results` is expected already sorted oldest-first (see
    load_results) -- this function renders rows in the order given rather than
    re-sorting.
    """
    if not results:
        return NO_RUNS_PLACEHOLDER

    accuracy_table = _table(_ACCURACY_HEADER, [_accuracy_row(r) for r in results])
    ops_table = _table(_OPS_HEADER, [_ops_row(r) for r in results])

    sections = [
        f"### Accuracy\n\n{accuracy_table}\n\n{_ACCURACY_CAPTION}",
        f"### Ops\n\n{ops_table}\n\n{_OPS_CAPTION}",
    ]
    schema_v2 = _schema_v2_section(results)
    if schema_v2:
        sections.append(schema_v2)
    return "\n\n".join(sections)


def update_readme(
    markdown: str,
    readme_path: Path,
    *,
    start_marker: str = START_MARKER,
    end_marker: str = END_MARKER,
) -> None:
    """Replace the block between START_MARKER and END_MARKER in
    `readme_path` with `markdown`, keeping the markers themselves in
    place. Idempotent: calling this twice with the same `markdown`
    leaves the file byte-identical after the second call, since the
    whole marker-to-marker span (not just an insertion point) is
    replaced each time.

    Raises ValueError if either marker is missing, or END appears
    before START -- README's Evals section is added once, by hand; a
    missing marker means it was edited out from under this function and
    needs a human to look, not a silent no-op.

    The markers default to the extraction eval's EVAL_TABLE pair; the
    retrieval eval passes its own (app.evals.retrieval) to maintain a
    second table in the same README the same way.
    """
    text = readme_path.read_text(encoding="utf-8")
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        raise ValueError(
            f"{readme_path}: table markers not found -- expected both "
            f"{start_marker!r} and {end_marker!r}"
        )
    if start_idx > end_idx:
        raise ValueError(f"{readme_path}: {end_marker!r} appears before {start_marker!r}")

    before = text[: start_idx + len(start_marker)]
    after = text[end_idx:]
    readme_path.write_text(f"{before}\n\n{markdown}\n\n{after}", encoding="utf-8")


# --- latest.md (per-doc failure detail for the most recent run) -----------


def _render_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return _esc(json.dumps(value, separators=(",", ":")))
    return _esc(str(value))


def _flagged_cell(routing: Routing) -> str:
    """Why the product would send this field to a person: the model's doubt
    (`confidence`), a failing check (`reconciliation:<id>`), or neither."""
    return routing.review_reason or "not flagged"


def _failure_rows(latest: dict[str, Any]) -> list[str]:
    rows = []
    # A run whose prompt version has no schema here cannot be replayed.
    schema = SCHEMAS.get(latest["prompt_version"])
    for doc in latest["per_doc"]:
        if doc.get("status") != "scored":
            continue
        routing = (
            doc_routing(doc, latest["review_threshold"], schema) if schema is not None else None
        )
        for field_name, field_score in doc["fields"].items():
            if field_score["correct"]:
                continue
            rows.append(
                _row(
                    [
                        _esc(doc["doc_id"]),
                        field_name,
                        _render_value(field_score["expected"]),
                        _render_value(field_score["actual"]),
                        _conf(field_score["confidence"]),
                        _flagged_cell(routing[field_name]) if routing is not None else "n/a",
                    ]
                )
            )
    return rows


def render_latest_md(results: list[dict[str, Any]]) -> str:
    """Render the same two tables as render_tables, plus a per-doc
    failure-detail section for the MOST RECENT run in `results` (its
    last element -- `results` is expected already sorted oldest-first,
    see load_results): every incorrect field of every scored doc in
    that one run, for eyeballing exactly what a model/prompt version got
    wrong. This is the inspection artifact written to
    evals/results/latest.md; README only ever gets the summary tables.
    """
    tables = render_tables(results)
    if not results:
        return tables

    latest = results[-1]
    rows = _failure_rows(latest)
    detail = (
        _table(("Doc", "Field", "Expected", "Actual", "Confidence", "Flagged"), rows)
        if rows
        else "*No incorrect fields in the most recent run.*"
    )

    header = (
        f"## Latest run: {latest['model']} / {latest['prompt_version']} "
        f"({latest['started_at_utc']})"
    )

    return f"{tables}\n\n{header}\n\n### Incorrect fields\n\n{detail}\n"


def write_latest_md(results: list[dict[str, Any]], out_path: Path | None = None) -> Path:
    """Render and write evals/results/latest.md (default LATEST_MD_PATH),
    creating its parent directory if missing -- mirrors
    app.evals.runner.write_result's own mkdir-then-write pattern.
    """
    if out_path is None:
        out_path = LATEST_MD_PATH
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_latest_md(results), encoding="utf-8")
    return out_path
