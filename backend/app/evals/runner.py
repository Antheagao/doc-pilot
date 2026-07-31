"""Async eval-run orchestration: fan a dataset of EvalCases out through an
extraction function (real or fake), score each result, aggregate, and
serialize the whole run to a JSON artifact.

Kept separate from app.evals.scoring (pure field-comparison math) and
app.evals.dataset (pure dataset loading) because this module owns the
concerns those two don't: concurrency, a hard cost cap, per-doc failure
isolation, and turning a list of CaseScores into a single reportable
RunResult. Like its siblings, this stays importable/runnable with
Postgres stopped -- extract_document (the default extract_fn) never
touches the DB, and neither does anything here.

extract_fn is the mock/CI seam: run_eval() takes it as a parameter
instead of hardcoding a call to app.extraction.extract_document, so
tests (and the --mock CLI flag) can inject a zero-cost fake with no
network access and no Anthropic API key. build_mock_extract_fn() below
builds that fake from the same EvalCases being run, so the CLI and the
test suite exercise identical mock behavior.
"""

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.config import get_settings
from app.evals.dataset import EVALS_DIR, EvalCase, load_cases
from app.evals.scoring import aggregate, score_case
from app.extraction import (
    PROMPT_VERSION,
    TOP_LEVEL_FIELDS,
    ExtractionResult,
    extract_document,
)

RESULTS_DIR = EVALS_DIR / "results"

# The two fixed perturbations (see build_mock_extract_fn) shift the
# expected accuracy by exactly one wrong field each: a tax value nudged
# outside AMOUNT_TOLERANCE, and a vendor forced to null (a "miss" against
# a non-null gold value, per app.evals.scoring's null-semantics rules --
# not a hallucination, since the perturbation runs the other direction).
_MOCK_MODEL = "mock-model"
_MOCK_CONFIDENCE = 0.9
_MOCK_TAX_PERTURBATION = 0.10


class ExtractFn(Protocol):
    async def __call__(self, path: str | Path, mime_type: str) -> Any: ...


@dataclass
class RunResult:
    model: str | None
    prompt_version: str | None
    dataset_version: str | None
    review_threshold: float
    started_at_utc: str
    concurrency: int
    max_cost_usd: float
    summary: dict[str, Any] | None
    total_cost_usd: float
    total_error_cost_usd: float
    mean_cost_per_doc: float
    total_input_tokens: int
    total_output_tokens: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    n_scored: int
    n_errors: int
    n_skipped_cost_cap: int
    errors: list[dict[str, Any]]
    skipped_cost_cap: list[str]
    per_doc: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "dataset_version": self.dataset_version,
            "review_threshold": self.review_threshold,
            "started_at_utc": self.started_at_utc,
            "concurrency": self.concurrency,
            "max_cost_usd": self.max_cost_usd,
            "summary": self.summary,
            "total_cost_usd": self.total_cost_usd,
            "total_error_cost_usd": self.total_error_cost_usd,
            "mean_cost_per_doc": self.mean_cost_per_doc,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "n_scored": self.n_scored,
            "n_errors": self.n_errors,
            "n_skipped_cost_cap": self.n_skipped_cost_cap,
            "errors": self.errors,
            "skipped_cost_cap": self.skipped_cost_cap,
            "per_doc": self.per_doc,
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile: index = ceil(pct/100 * n) - 1, clamped to
    [0, n-1]. Used instead of statistics.quantiles because that function
    needs at least 2 data points and returns n-1 *cut points* rather than
    a value actually observed at the requested percentile -- a worse fit
    here, where a run can legitimately have as few as 1 successfully
    scored doc and we still want a well-defined p50/p95.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("cannot compute a percentile of an empty sequence")
    idx = max(0, min(n - 1, math.ceil(pct / 100 * n) - 1))
    return sorted_values[idx]


async def run_eval(
    cases: list[EvalCase],
    extract_fn: ExtractFn | None = None,
    *,
    concurrency: int = 3,
    max_cost_usd: float = 1.00,
) -> RunResult:
    """Run `extract_fn` (default: app.extraction.extract_document) over
    every case in `cases`, score each result against its gold label, and
    return a single RunResult.

    Concurrency is bounded by an asyncio.Semaphore(concurrency): one task
    per case is created up front, but at most `concurrency` of them are
    ever inside an extract_fn call at once. `concurrency` must be >= 1
    (Semaphore(0) would deadlock every task forever) -- raises ValueError
    otherwise.

    Cost cap: a single running total is checked, under a lock,
    immediately after a task acquires its semaphore slot and before it
    calls extract_fn. That total accumulates cost from BOTH successfully
    scored docs and from errored docs whose exception carries a cost_usd
    (see "Billed-but-errored calls" below) -- either way, real money was
    spent, so either way it counts against the cap. Once the running
    total exceeds max_cost_usd, every task that checks in after that
    point skips its extract_fn call entirely and is recorded as
    "skipped_cost_cap" instead.

    This is what actually stops unbounded spend mid-run, but it has two
    real limits callers sizing --max-cost need to know:
    - Worst-case overshoot: a doc whose extract_fn call already started
      before the cap tripped is allowed to finish (there is no way to
      know its cost, let alone un-spend it, before the call returns).
      Up to `concurrency` calls can be in flight at once, so actual
      spend can exceed max_cost_usd by as much as
      concurrency * (largest single-doc cost seen in the run).
    - Minimum spend: the first `concurrency` tasks all pass the cap
      check (0.0 <= max_cost_usd, even for max_cost_usd == 0) before any
      of them has a chance to complete and report cost, so at least
      min(concurrency, len(cases)) docs are always attempted regardless
      of how low max_cost_usd is set.

    Per-doc failures: any exception raised by extract_fn for one doc
    (ExtractionError, NonRetryableExtractionError, or anything else) is
    caught, recorded as an `error` entry, and does not stop the run --
    that doc is simply excluded from scoring, so aggregate() never sees
    it and the accuracy denominator shrinks accordingly.

    Billed-but-errored calls: app.extraction.extract_document raises
    after some failures where the API call already succeeded and was
    billed (a refusal/non-tool_use stop reason, a missing tool_use
    block, or -- for process_document_job, not this path -- a
    persistence failure). Those exceptions carry optional
    input_tokens/output_tokens/cost_usd attributes (default None on a
    plain exception). When present, that cost_usd is added to the cap
    accumulator above AND to the returned total_error_cost_usd -- kept
    separate from total_cost_usd, which stays scored-docs-only, so a
    reader can always tell "cost from extractions we could score" apart
    from "cost we know we spent but got nothing scoreable for".

    model / prompt_version are read from the FIRST successful
    ExtractionResult in `cases` order (not completion order, so the
    result is stable regardless of scheduling) -- never hardcoded. If a
    later successful result reports a different model or prompt_version,
    that is a bug in the caller's extract_fn (a single run must be
    uniform) and raises AssertionError. dataset_version is read the same
    way from EvalCase.dataset_version and is checked for uniformity
    across ALL cases, scored or not, since it's dataset metadata rather
    than something extract_fn reports.
    """
    if not cases:
        raise ValueError("run_eval requires at least one case")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    if extract_fn is None:
        extract_fn = extract_document

    dataset_versions = {c.dataset_version for c in cases}
    if len(dataset_versions) > 1:
        raise AssertionError(
            f"cases span multiple dataset_versions: {sorted(dataset_versions)} "
            "-- a single eval run must be over one dataset version"
        )
    dataset_version = cases[0].dataset_version

    settings = get_settings()
    review_threshold = settings.review_threshold
    started_at_utc = datetime.now(UTC).isoformat()

    semaphore = asyncio.Semaphore(concurrency)
    cost_lock = asyncio.Lock()
    # cap_cost is the cap-tripping accumulator: scored-doc cost AND
    # billed-but-errored cost both count against it (see docstring). It
    # is NOT the same number as the returned total_cost_usd, which stays
    # scored-docs-only.
    state = {"cap_cost": 0.0, "cap_hit": False}

    n = len(cases)
    results_by_idx: list[ExtractionResult | None] = [None] * n
    case_scores_by_idx: list[Any] = [None] * n
    doc_details: list[dict[str, Any] | None] = [None] * n

    async def _run_one(idx: int, case: EvalCase) -> None:
        async with semaphore:
            async with cost_lock:
                cap_hit = state["cap_hit"]
            if cap_hit:
                doc_details[idx] = {
                    "doc_id": case.doc_id,
                    "status": "skipped_cost_cap",
                    "difficulty": case.difficulty,
                }
                return

            try:
                result = await extract_fn(case.image_path, case.mime_type)
            except Exception as exc:  # noqa: BLE001 -- any failure isolates this doc, see docstring
                # Some failures (see docstring's "Billed-but-errored
                # calls") carry cost from an API call that already
                # succeeded and was billed before the exception was
                # raised -- getattr defaults to None for exceptions that
                # don't set these (a plain Exception, or a network-level
                # ExtractionError raised before any billing happened).
                error_cost_usd = getattr(exc, "cost_usd", None)
                doc_details[idx] = {
                    "doc_id": case.doc_id,
                    "status": "error",
                    "difficulty": case.difficulty,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "input_tokens": getattr(exc, "input_tokens", None),
                    "output_tokens": getattr(exc, "output_tokens", None),
                    "cost_usd": error_cost_usd,
                }
                if error_cost_usd is not None:
                    async with cost_lock:
                        state["cap_cost"] += error_cost_usd
                        if state["cap_cost"] > max_cost_usd:
                            state["cap_hit"] = True
                return

            score = score_case(case, result.tool_input, review_threshold)
            results_by_idx[idx] = result
            case_scores_by_idx[idx] = score
            doc_details[idx] = {
                "doc_id": case.doc_id,
                "status": "scored",
                "difficulty": case.difficulty,
                "fields": {name: fs.to_dict() for name, fs in score.fields.items()},
                "schema_violations": score.schema_violations,
                "hallucinations": score.hallucinations,
                "item_accuracy": score.item_accuracy,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }

            async with cost_lock:
                state["cap_cost"] += result.cost_usd
                if state["cap_cost"] > max_cost_usd:
                    state["cap_hit"] = True

    await asyncio.gather(*(_run_one(idx, case) for idx, case in enumerate(cases)))

    results = [r for r in results_by_idx if r is not None]
    case_scores = [s for s in case_scores_by_idx if s is not None]
    errors = [d for d in doc_details if d is not None and d["status"] == "error"]
    skipped = [d for d in doc_details if d is not None and d["status"] == "skipped_cost_cap"]

    first_result = next((r for r in results_by_idx if r is not None), None)
    if first_result is not None:
        model = first_result.model
        prompt_version = first_result.prompt_version
        for r in results:
            if r.model != model:
                raise AssertionError(
                    f"inconsistent model across run: {r.model!r} != {model!r} "
                    "-- extract_fn must return a uniform model within a single run"
                )
            if r.prompt_version != prompt_version:
                raise AssertionError(
                    f"inconsistent prompt_version across run: {r.prompt_version!r} != "
                    f"{prompt_version!r} -- extract_fn must return a uniform "
                    "prompt_version within a single run"
                )
    else:
        model = None
        prompt_version = None

    summary = aggregate(case_scores).to_dict() if case_scores else None

    total_cost_usd = sum(r.cost_usd for r in results)
    total_error_cost_usd = sum(
        d["cost_usd"] for d in errors if d.get("cost_usd") is not None
    )
    mean_cost_per_doc = total_cost_usd / len(results) if results else 0.0
    total_input_tokens = sum(r.input_tokens for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)

    latencies = sorted(r.latency_ms for r in results)
    latency_p50_ms = _percentile(latencies, 50) if latencies else None
    latency_p95_ms = _percentile(latencies, 95) if latencies else None

    return RunResult(
        model=model,
        prompt_version=prompt_version,
        dataset_version=dataset_version,
        review_threshold=review_threshold,
        started_at_utc=started_at_utc,
        concurrency=concurrency,
        max_cost_usd=max_cost_usd,
        summary=summary,
        total_cost_usd=total_cost_usd,
        total_error_cost_usd=total_error_cost_usd,
        mean_cost_per_doc=mean_cost_per_doc,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        n_scored=len(results),
        n_errors=len(errors),
        n_skipped_cost_cap=len(skipped),
        errors=errors,
        skipped_cost_cap=[d["doc_id"] for d in skipped],
        per_doc=[d for d in doc_details if d is not None],
    )


def write_result(result: RunResult, out_dir: Path | None = None) -> Path:
    """Write `result` as JSON to
    `<out_dir>/<UTC-ts>_<model>_<prompt_version>.json` (out_dir defaults
    to EVALS_DIR/"results", created if missing).

    The timestamp is derived from result.started_at_utc (the run's own
    start time), not wall-clock time at write() -- an artifact's filename
    should reflect when the eval ran, not the (usually near-identical,
    but not always) instant it happened to be persisted. That timestamp
    is only second-granularity, so two runs starting in the same second
    (or two calls with the same started_at_utc, e.g. in tests) would
    otherwise collide on one filename and silently clobber each other via
    write_text -- a `-1`, `-2`, ... suffix is appended until a free path
    is found, so no run's artifact is ever overwritten by another's.
    """
    if out_dir is None:
        out_dir = RESULTS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.fromisoformat(result.started_at_utc).strftime("%Y%m%dT%H%M%SZ")
    model = result.model or "unknown-model"
    prompt_version = result.prompt_version or "unknown-prompt"
    base_name = f"{ts}_{model}_{prompt_version}"

    out_path = out_dir / f"{base_name}.json"
    suffix = 1
    while out_path.exists():
        out_path = out_dir / f"{base_name}-{suffix}.json"
        suffix += 1

    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return out_path


# --- mock extract_fn (shared by the CLI and the test suite) --------------


def _wrap_leaf(value: Any, confidence: float = _MOCK_CONFIDENCE) -> dict[str, Any]:
    return {"value": value, "confidence": confidence}


def _mock_perturbation_targets() -> tuple[str | None, str | None]:
    """Returns (tax_perturb_doc_id, vendor_none_doc_id): the doc_ids of
    the 3rd and 7th cases, by sorted doc_id, of the FULL shipped dataset
    (load_cases() with no limit) -- deliberately NOT of whatever subset
    is being run, so a --limit run perturbs the same two docs (dropping
    one, both, or neither out of the run) rather than shifting to
    whichever docs happen to land at positions 3 and 7 of the smaller
    subset. That's what keeps predicted_mock_accuracy() computable from
    first principles regardless of --limit.
    """
    full_ordered = load_cases()
    tax_id = full_ordered[2].doc_id if len(full_ordered) > 2 else None
    vendor_id = full_ordered[6].doc_id if len(full_ordered) > 6 else None
    return tax_id, vendor_id


def _fake_tool_input(
    fields: dict[str, Any],
    doc_id: str,
    tax_perturb_doc_id: str | None,
    vendor_none_doc_id: str | None,
) -> dict[str, Any]:
    tool_input: dict[str, Any] = {}
    for name in TOP_LEVEL_FIELDS:
        if name == "line_items":
            items = fields["line_items"] or []
            wrapped_items = [
                {key: _wrap_leaf(item[key]) for key in ("description", "quantity", "unit_price", "total")}
                for item in items
            ]
            tool_input["line_items"] = _wrap_leaf(wrapped_items)
            continue

        value = fields[name]
        if name == "tax" and doc_id == tax_perturb_doc_id and value is not None:
            value = round(value + _MOCK_TAX_PERTURBATION, 2)
        elif name == "vendor" and doc_id == vendor_none_doc_id:
            value = None
        tool_input[name] = _wrap_leaf(value)
    return tool_input


def build_mock_extract_fn(cases: list[EvalCase]) -> ExtractFn:
    """Build a deterministic, zero-cost fake extract_fn from `cases`'
    own gold labels: every field is wrapped as {value, confidence 0.9}
    (line_items: each item's four sub-fields wrapped the same way), so a
    default run scores ~perfectly -- except for two fixed perturbations
    (see _mock_perturbation_targets / _fake_tool_input) that make the
    expected accuracy an exact, analytically-known constant rather than
    a vacuous 1.0. See predicted_mock_accuracy() for that computation.

    This is the shared mock seam: both the --mock CLI flag and
    test_evals_runner.py call this (not a duplicated local version), so
    they can never drift out of sync with each other.
    """
    tax_perturb_doc_id, vendor_none_doc_id = _mock_perturbation_targets()
    fields_by_doc_id = {c.doc_id: c.fields for c in cases}

    async def mock_extract_fn(path: str | Path, mime_type: str) -> ExtractionResult:
        doc_id = Path(path).stem
        fields = fields_by_doc_id.get(doc_id)
        if fields is None:
            raise KeyError(
                f"mock extract_fn: no case with doc_id {doc_id!r} was passed to "
                f"build_mock_extract_fn (path={path})"
            )
        tool_input = _fake_tool_input(fields, doc_id, tax_perturb_doc_id, vendor_none_doc_id)
        return ExtractionResult(
            tool_input=tool_input,
            model=_MOCK_MODEL,
            prompt_version=PROMPT_VERSION,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=1,
        )

    return mock_extract_fn


def predicted_mock_accuracy(cases: list[EvalCase]) -> float:
    """Analytically compute the exact overall_accuracy build_mock_extract_fn's
    output must score, for eyeball comparison against the actual run.

    total_slots = len(cases) * len(TOP_LEVEL_FIELDS) (7). Each of the two
    fixed perturbations (see _mock_perturbation_targets) costs exactly
    one wrong field IF its target doc_id is present in `cases` -- for the
    full 25-doc shipped dataset that's 25 * 7 = 175 slots, 2 wrong ->
    173/175 = 0.9885714285714285.
    """
    tax_perturb_doc_id, vendor_none_doc_id = _mock_perturbation_targets()
    doc_ids = {c.doc_id for c in cases}
    total_slots = len(cases) * len(TOP_LEVEL_FIELDS)
    if total_slots == 0:
        return 0.0
    wrong = sum(
        1
        for target in (tax_perturb_doc_id, vendor_none_doc_id)
        if target is not None and target in doc_ids
    )
    return (total_slots - wrong) / total_slots
