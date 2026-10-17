"""The pre-registered bar for making extraction schema v2 the default, checked
against two eval artifacts: the model's v1 run and its run with evidence (v2, or
a later prompt version with the same fields).

The terms are fixed before any paid run (PRESENT.md, Batch B): per model, core
accuracy (the seven v1 fields on documents 001-025) no worse than its v1 run by
more than one field; no residual schema violations; Sonnet evidence coverage of
at least 95%; no `max_tokens` stop; no unmasked card number in the artifact; and
a complete run (32 documents scored, none errored). Each term is computed from
the artifact alone and reported with the numbers it was decided on, so a pass or
a failure can be read without opening the files. An artifact that does not
record what a term needs fails the term rather than passing it by default.

Offline and pure: it reads two dicts (as app.evals.report.load_results does) and
returns the verdicts. evals/check_bar.py is the CLI.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from app.evals.dataset import CORE_DOC_IDS
from app.evals.report import core_counts
from app.extraction_fields import SCHEMA_V1, SCHEMAS
from app.pii import mask_card_numbers

MAX_CORE_FIELDS_WORSE = 1
MIN_SONNET_EVIDENCE_COVERAGE = 0.95
FULL_RUN_DOCUMENTS = 32


@dataclass(frozen=True)
class BarTerm:
    """One term of the bar: whether it holds and the numbers behind it."""

    name: str
    passed: bool
    detail: str


def core_misses(run: dict[str, Any]) -> list[tuple[str, str]]:
    """The (doc_id, field) pairs of the core documents' v1 fields a run got
    wrong, in document order."""
    misses = []
    for doc in run["per_doc"]:
        if doc.get("status") != "scored" or doc["doc_id"] not in CORE_DOC_IDS:
            continue
        for name in SCHEMA_V1.field_names:
            field = doc["fields"].get(name)
            if field is not None and not field["correct"]:
                misses.append((doc["doc_id"], name))
    return misses


def _listed(misses: list[tuple[str, str]]) -> str:
    return ", ".join(f"{doc_id} {name}" for doc_id, name in misses) or "none"


def _core_term(v1: dict[str, Any], v2: dict[str, Any]) -> BarTerm:
    (c1, n1), (c2, n2) = core_counts(v1), core_counts(v2)
    detail = (
        f"v1 {c1}/{n1} (misses: {_listed(core_misses(v1))}); "
        f"v2 {c2}/{n2} (misses: {_listed(core_misses(v2))})"
    )
    if n1 != n2 or n1 == 0:
        return BarTerm("core accuracy", False, f"the runs scored different core fields; {detail}")
    worse = c1 - c2
    return BarTerm(
        "core accuracy",
        worse <= MAX_CORE_FIELDS_WORSE,
        f"v2 is {worse} field(s) worse than v1 (allowed {MAX_CORE_FIELDS_WORSE}); {detail}",
    )


def _residual_term(v2: dict[str, Any]) -> BarTerm:
    recorded = v2.get("total_residual_schema_violations")
    if recorded is None:
        return BarTerm("residual schema violations", False, "not recorded in the artifact")
    per_doc = [
        doc.get("residual_schema_violations")
        for doc in v2["per_doc"]
        if doc.get("status") == "scored"
    ]
    if any(count is None for count in per_doc):
        return BarTerm("residual schema violations", False, "a scored document does not record it")
    return BarTerm(
        "residual schema violations",
        recorded == 0 and sum(per_doc) == 0,
        f"{recorded} in total, {sum(per_doc)} summed over the documents (required 0)",
    )


def _coverage_term(v2: dict[str, Any]) -> BarTerm:
    coverage = (v2["summary"] or {}).get("evidence_coverage")
    required = "sonnet" in (v2["model"] or "").lower()
    if coverage is None or coverage["coverage"] is None:
        return BarTerm("evidence coverage", False, "the run has no evidence coverage")
    detail = (
        f"{coverage['with_quote']}/{coverage['leaves']} values quoted = {coverage['coverage']:.1%}; "
        f"page agreement {_share(coverage['page_agreement'])}, "
        f"quote agreement {_share(coverage['quote_agreement'])}"
    )
    if not required:
        return BarTerm("evidence coverage", True, f"{detail} (required for Sonnet only)")
    return BarTerm(
        "evidence coverage",
        coverage["coverage"] >= MIN_SONNET_EVIDENCE_COVERAGE,
        f"{detail} (required >= {MIN_SONNET_EVIDENCE_COVERAGE:.0%})",
    )


def _share(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _stops(run: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(documents with a max_tokens stop, documents that do not record stop
    reasons), over scored and errored documents."""
    stopped, unrecorded = [], []
    for doc in run["per_doc"]:
        if doc.get("status") not in ("scored", "error"):
            continue
        reasons = doc.get("stop_reasons")
        text = json.dumps(reasons) + str(doc.get("error_message", ""))
        if "max_tokens" in text:
            stopped.append(doc["doc_id"])
        if reasons is None and doc.get("status") == "scored":
            unrecorded.append(doc["doc_id"])
    return stopped, unrecorded


def _stop_term(v2: dict[str, Any]) -> BarTerm:
    stopped, unrecorded = _stops(v2)
    if unrecorded:
        return BarTerm(
            "no max_tokens stop",
            False,
            f"stop reasons not recorded for {len(unrecorded)} scored document(s): {', '.join(unrecorded)}",
        )
    return BarTerm(
        "no max_tokens stop",
        not stopped,
        f"{len(stopped)} document(s) stopped at max_tokens{': ' + ', '.join(stopped) if stopped else ''}",
    )


def _texts(node: Any) -> Iterator[str]:
    """Every string in a JSON value (keys included), and every whole number, as
    text: the places a card number can be. A float is a measurement (a mean
    confidence, a latency) whose sixteen digits pass the Luhn check by chance
    about one time in ten, so it is not text a card number is written in."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, bool):
        return
    elif isinstance(node, int):
        yield str(node)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _texts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _texts(value)


def _card_term(v2: dict[str, Any]) -> BarTerm:
    exposed = [text for text in _texts(v2) if mask_card_numbers(text) != text]
    return BarTerm(
        "no unmasked card number",
        not exposed,
        "no string or whole number in the artifact is changed by the card-number mask"
        if not exposed
        else f"{len(exposed)} value(s) the card-number mask changes: a card number is in the artifact",
    )


def _complete_term(v2: dict[str, Any]) -> BarTerm:
    scored, errors, skipped = v2["n_scored"], v2["n_errors"], v2.get("n_skipped_cost_cap", 0)
    return BarTerm(
        "complete run",
        scored == FULL_RUN_DOCUMENTS and errors == 0 and skipped == 0,
        f"{scored} scored, {errors} errored, {skipped} skipped (required {FULL_RUN_DOCUMENTS} scored, 0 errored)",
    )


def _carries_evidence(prompt_version: str) -> bool:
    """Whether a prompt version is registered with a schema that asks for evidence
    (extract_v2, extract_v3, ...): the bar judges any of them against v1."""
    schema = SCHEMAS.get(prompt_version)
    return schema is not None and schema.evidence


def _same_run_term(v1: dict[str, Any], v2: dict[str, Any]) -> BarTerm:
    ok = (
        v1["model"] == v2["model"]
        and v1["prompt_version"] == "extract_v1"
        and _carries_evidence(v2["prompt_version"])
    )
    return BarTerm(
        "runs to compare",
        ok,
        f"v1 artifact {v1['model']} / {v1['prompt_version']}, v2 artifact {v2['model']} / {v2['prompt_version']}"
        " (required: the same model, extract_v1 then a prompt version with evidence, extract_v2 or later)",
    )


def evaluate_bar(v1: dict[str, Any], v2: dict[str, Any]) -> list[BarTerm]:
    """Every term of the bar for a model's v1 and v2 artifacts."""
    return [
        _same_run_term(v1, v2),
        _complete_term(v2),
        _core_term(v1, v2),
        _residual_term(v2),
        _coverage_term(v2),
        _stop_term(v2),
        _card_term(v2),
    ]
