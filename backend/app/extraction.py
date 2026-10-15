"""VLM-based structured extraction from uploaded documents.

Reads a document image/PDF, asks Claude to call the `record_extraction`
tool with structured field data, and persists the result as an Extraction
row plus one ExtractedField row per top-level field. This module is the
real `process_document_job` handler wired into app.worker (see
app/worker.py).

Prompt versioning: the extraction prompt lives in prompts/extract_v1.md as
a plain file, not inline in code, so future prompt iterations are versioned
by filename (extract_v2.md, ...) and eval results can be tied to a specific
prompt_version string. Each version has an ExtractionSchema
(app/extraction_fields.py) holding its prompt, repair prompt, field list,
and token budget; every function below that reads or writes extracted
fields takes the schema it is working with.

Schema conformance is NOT guaranteed by tool_choice: forcing tool_choice
to record_extraction guarantees the model returns *a* tool_use block for
that tool, but says nothing about whether its `input` actually matches
input_schema (Claude can and does emit a bare scalar instead of a
{value, confidence} object, a null confidence, or a leaf missing "value"
entirely). Every place this module reads a leaf out of tool_input is
written assuming that input is adversarial/untrusted, not schema-clean --
see _coerce_leaf.
"""

import base64
import logging
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.extraction_fields import (
    ARRAY_CELLS,
    ExtractionSchema,
    record_extraction_tool,
    schema_for,
)
from app.models import (
    JOB_KIND_INDEX,
    REVIEW_REASON_CONFIDENCE,
    Document,
    ExtractedField,
    Extraction,
    Job,
)
from app.pdf import count_pdf_pages, split_pdf_pages
from app.telemetry import current_traceparent, model_call_span, record_model_response

logger = logging.getLogger(__name__)

# Anthropic media types the vision API accepts for an `image` content
# block. GIF is included deliberately -- the Claude API's image source
# media_type enum documents jpeg/png/webp/gif as supported, so it is kept
# in ALLOWED_MIME_TYPES (app/routers/documents.py) rather than dropped.
IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
PDF_MIME_TYPE = "application/pdf"

# $ per million tokens (input, output). Cached from the claude-api skill,
# 2026-06-24. Sonnet 5 is running introductory pricing ($2/$10) through
# 2026-08-31, after which it reverts to standard $3/$15 -- update this
# table when that happens (or when EXTRACTION_MODEL is changed).
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Added with the /ask agent (app/agent/), from the claude-api skill's
    # model table (cached 2026-09-25). claude-opus-4-8 is here because it
    # is where server-side refusal fallbacks route some declines, and a
    # response served by the fallback is priced at the fallback's rates.
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
}

class ExtractionError(Exception):
    """Raised when the VLM call fails to produce a usable extraction.
    Callers (the worker) treat this like any other handler exception --
    fail_job requeues with backoff, up to MAX_ATTEMPTS. Use this base
    class only for failures that are plausibly transient (network
    errors, 5xx, rate limits, a flaky disk read) where a retry has a
    real chance of succeeding. See NonRetryableExtractionError below for
    deterministic failures.

    input_tokens/output_tokens/cost_usd are optional structured fields
    (default None) set at raise sites where the API call already
    succeeded and was billed before this exception was raised (a
    refusal/non-tool_use stop reason, a missing tool_use block, or a
    persistence failure after a successful call) -- callers that only
    have a raised exception, not an ExtractionResult, still need a way
    to know that money was spent. The existing message text (which
    already embeds the same token counts for last_error/logging) is
    unchanged; these are additive attributes for programmatic callers
    like app.evals.runner.run_eval's cost cap, not a replacement for it.

    retry_after_seconds (default None) is set at raise sites triggered
    by a retryable anthropic.APIStatusError (429 rate limit, 529
    overloaded, other 5xx/408) whose response carried a parseable
    `retry-after` header -- see the classification in extract_document's
    messages.create() call. app.worker.fail_job reads it via
    getattr(exc, "retry_after_seconds", None) and feeds it into
    _backoff_delay as a floor on the requeue delay, so a job never
    retries sooner than the server explicitly asked for.
    """

    def __init__(
        self,
        message: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.retry_after_seconds = retry_after_seconds


class NonRetryableExtractionError(ExtractionError):
    """A deterministic failure: re-running the identical job (same
    document, same prompt, same model) would produce the identical
    outcome every time, so retrying only spends money for no chance of
    a different result. Raised for: an unsupported mime type, an
    EXTRACTION_MODEL with no pricing entry, a model refusal or other
    non-tool_use stop reason, and output truncation (stop_reason ==
    "max_tokens"). worker.fail_job checks for this type and skips the
    requeue-with-backoff path, going straight to permanent failure.
    """


class ModelRefusalError(NonRetryableExtractionError):
    """The model explicitly refused to extract the document
    (response.stop_reason == "refusal"), as distinct from other
    non-tool_use stop reasons like max_tokens. Still non-retryable --
    re-running the identical job would produce the identical refusal --
    but worker.fail_job marks the document "refused" instead of "failed"
    so the UI can tell the two apart.
    """


@dataclass
class ExtractionResult:
    tool_input: dict[str, Any]
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    # Appended last, both defaulted, so existing keyword-argument
    # constructions (e.g. app.evals.runner.build_mock_extract_fn) keep
    # working unmodified. repaired is True only when a repair reprompt
    # (see extract_document) both ran and was accepted; schema_violations
    # is the violation count (see _schema_violations) of whichever
    # tool_input ended up in this result -- the repaired one if accepted,
    # the first response's otherwise.
    repaired: bool = False
    schema_violations: int = 0


def _build_client(settings: Settings) -> anthropic.AsyncAnthropic:
    """Factored out so tests can monkeypatch just the client construction
    and inject a fake with a mocked `.messages.create` -- no network, no
    need to stub the whole extract_document call.

    max_retries/timeout are two independent retry layers with different
    horizons, not a duplication of each other: the SDK retries transport
    blips (connection errors, 429/5xx) in-process, within a single
    extract_document call, honoring any `retry-after` header on the
    failing response. The Postgres job queue (app.worker.fail_job) is the
    outer layer -- it retries on a minutes-long horizon, across separate
    worker invocations, only after the SDK's own retries are exhausted
    and extract_document raises. Neither layer replaces the other.
    """
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        max_retries=settings.anthropic_max_retries,
        timeout=settings.anthropic_timeout_seconds,
    )


def _ensure_model_priced(model: str) -> None:
    """Validate EXTRACTION_MODEL has a pricing entry BEFORE spending any
    money on an API call. This is a pure config check -- catching it
    after the call (the previous behavior) meant a misconfigured model
    name paid for a real request and then blew up trying to price it.
    """
    if model not in PRICING_PER_MTOK:
        raise NonRetryableExtractionError(
            f"no pricing entry for model {model!r}; add it to PRICING_PER_MTOK "
            "before using it as EXTRACTION_MODEL"
        )


def _compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    # model is validated by _ensure_model_priced before extract_document
    # ever reaches this point, so this lookup cannot fail in practice --
    # the KeyError->NonRetryableExtractionError translation is a safety
    # net in case that invariant is ever broken by a future refactor.
    try:
        price_in, price_out = PRICING_PER_MTOK[model]
    except KeyError:
        raise NonRetryableExtractionError(
            f"no pricing entry for model {model!r}; add it to PRICING_PER_MTOK"
        ) from None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _leaf_is_malformed(raw: Any) -> bool:
    """True iff `raw` does NOT have the shape _coerce_leaf accepts: not a
    dict, missing the "value" key, or a "confidence" that doesn't convert
    to a finite float. This is the single rule both _coerce_leaf's reject
    branch and _schema_violations' violation detector delegate to, so the
    two can never drift out of sync with each other -- see both callers'
    docstrings.

    Note this is NOT the same question as "is this leaf's `value`
    correct" -- a leaf with a perfectly well-formed {"value": null,
    "confidence": 0.2} shape is not malformed, even though its value is
    null, because null is a legitimate value per the extraction prompt.
    """
    if not isinstance(raw, dict) or "value" not in raw:
        return True
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        # OverflowError: float(x) on an int too large to represent as a
        # float (e.g. a huge integer literal, valid JSON) -- same
        # "can't trust this leaf" degrade as a non-numeric confidence.
        return True
    return not math.isfinite(confidence)


def _coerce_leaf(raw: Any) -> tuple[Any, float]:
    """Defensively normalize a single {"value", "confidence"} leaf out of
    the model's (untrusted) tool_use.input. Returns (value, confidence).

    A leaf is only trusted if it is a dict, has a "value" key (even if
    that value is legitimately null -- the prompt tells the model to use
    null for absent/illegible fields, which is different from the key
    being missing entirely), and has a "confidence" that converts to a
    finite float (see _leaf_is_malformed, the shared rule). Any other
    shape -- a bare scalar instead of a dict (`"vendor": "Acme"`), a
    missing "value" key, a null/non-numeric confidence -- degrades the
    WHOLE leaf to (None, 0.0) rather than salvaging the parts that
    happened to look fine: if the model didn't follow the schema for a
    field, nothing in that field is trustworthy enough to keep.

    confidence is clamped to [0.0, 1.0] even when float-convertible,
    since a schema `minimum`/`maximum` on the tool definition is only a
    hint to the model, not an enforced constraint.
    """
    if _leaf_is_malformed(raw):
        return None, 0.0

    confidence = float(raw["confidence"])
    return raw["value"], max(0.0, min(1.0, confidence))


# --- H6 chunked-PDF merge (merge_page_tool_inputs) --------------------------
#
# Kept next to _coerce_leaf/_leaf_is_malformed (rather than in app.pdf,
# which must stay pypdf+stdlib only, or its own module) so it can reuse
# them directly without a circular import back into this module.

# When two pages disagree on a scalar field's value, the merged leaf's
# confidence is capped at this value regardless of how confident either
# page was individually -- a real cross-page disagreement must reach a
# human reviewer even if one page reported very high confidence.
MERGE_CONFLICT_CONFIDENCE = 0.5


def _values_disagree(a: Any, b: Any) -> bool:
    """True iff two non-null candidate leaf values for the same scalar
    field are meaningfully different. Strings are compared via
    strip().casefold() (whitespace/case-insensitive), numbers via
    round(v, 2) (tolerates float noise from independent per-page
    extractions), anything else by plain equality.

    Page tool_inputs are untrusted (see the module docstring) and can
    contain adversarial numerics -- e.g. an oversized int literal like
    10**400, valid JSON but too large for float() to represent, raising
    OverflowError. That's treated as a disagreement (the safe direction:
    it caps the merged confidence via MERGE_CONFLICT_CONFIDENCE rather
    than silently trusting an unrepresentable number), not re-raised.
    """
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() != b.strip().casefold()
    if isinstance(a, int | float) and isinstance(b, int | float):
        try:
            return round(float(a), 2) != round(float(b), 2)
        except OverflowError:
            return True
    return a != b


def _merge_scalar_leaf(candidates: list[tuple[int, Any, float]], *, tie_break: str) -> dict[str, Any]:
    """Merge one scalar field's per-page (page_index, value, confidence)
    candidates -- already read defensively via _coerce_leaf, so a
    malformed leaf on any page has already degraded to
    (page_index, None, 0.0) by the time it reaches here -- into a single
    {"value", "confidence"} leaf, shape-identical to a single-call leaf.

    All-null (every candidate's value is None): the merged leaf's value
    is None, confidence is the MINIMUM confidence across every
    candidate (including the null ones) -- an absent field should land
    low-confidence and route to review, not look artificially confident
    just because no page actually reported anything wrong.

    Otherwise: pick the highest-confidence NON-null candidate; ties
    (equal confidence) are broken by `tie_break` ("earliest" or
    "latest") using page index. If any non-null candidate's value
    disagrees with another's (see _values_disagree), the merged
    confidence is capped at MERGE_CONFLICT_CONFIDENCE -- a genuine
    cross-page conflict must reach a human even if the winning page was
    very confident.
    """
    non_null = [c for c in candidates if c[1] is not None]
    if not non_null:
        return {"value": None, "confidence": min((c[2] for c in candidates), default=0.0)}

    best_confidence = max(c[2] for c in non_null)
    tied = [c for c in non_null if c[2] == best_confidence]
    winner = (min if tie_break == "earliest" else max)(tied, key=lambda c: c[0])

    base_value = non_null[0][1]
    disagreement = any(_values_disagree(base_value, c[1]) for c in non_null[1:])

    confidence = min(winner[2], MERGE_CONFLICT_CONFIDENCE) if disagreement else winner[2]
    return {"value": winner[1], "confidence": confidence}


def _merge_line_items(page_line_items: list[tuple[Any, float]]) -> dict[str, Any]:
    """Merge per-page (value, confidence) candidates for the "line_items"
    leaf (already read via _coerce_leaf) into one leaf: every page's
    array that actually IS a list is concatenated in page order, and
    the merged confidence is the MINIMUM confidence across only the
    pages that contributed a list -- a page whose line_items leaf was
    null/malformed contributed nothing to concatenate, so it doesn't
    drag the confidence down either. If no page produced a list at all,
    returns {"value": [], "confidence": 0.0} so the field routes to
    review like any other all-null field.
    """
    concatenated: list[Any] = []
    contributing_confidences: list[float] = []
    for value, confidence in page_line_items:
        if isinstance(value, list):
            concatenated.extend(value)
            contributing_confidences.append(confidence)

    if not contributing_confidences:
        return {"value": [], "confidence": 0.0}
    return {"value": concatenated, "confidence": min(contributing_confidences)}


def merge_page_tool_inputs(
    page_inputs: list[dict[str, Any]], schema: ExtractionSchema
) -> dict[str, Any]:
    """Merge a list of per-page tool_input dicts (one per PDF page, in
    page order, each shaped like a single-call response to the schema's
    extraction tool) into ONE tool_input identical in SHAPE to a
    single-call one -- so downstream persistence (process_document_job),
    scoring (_coerce_leaf-based eval scoring), and this module's own
    _coerce_leaf are all untouched by chunking. See _merge_scalar_leaf
    and _merge_line_items for the field-by-field merge rules (each
    scalar field's FieldSpec.merge_tie_break says whether ties go to the
    earliest or the latest page, all-null uses min confidence,
    disagreement caps confidence at MERGE_CONFLICT_CONFIDENCE).

    Every page's tool_input is untrusted the same way a single
    response's is (see the module docstring / _coerce_leaf): a
    non-dict page_input, or one missing/mangling a field entirely,
    degrades that page's contribution to (None, 0.0) via _coerce_leaf
    rather than raising.
    """
    array_candidates: dict[str, list[tuple[Any, float]]] = {
        field.name: [] for field in schema.fields if field.kind == "array"
    }
    scalar_candidates: dict[str, list[tuple[int, Any, float]]] = {
        field.name: [] for field in schema.fields if field.kind != "array"
    }

    for page_idx, page_input in enumerate(page_inputs):
        for field_name, array_field_candidates in array_candidates.items():
            raw_array = page_input.get(field_name) if isinstance(page_input, dict) else None
            array_field_candidates.append(_coerce_leaf(raw_array))

        for field_name, candidates in scalar_candidates.items():
            raw_leaf = page_input.get(field_name) if isinstance(page_input, dict) else None
            value, confidence = _coerce_leaf(raw_leaf)
            candidates.append((page_idx, value, confidence))

    merged: dict[str, Any] = {}
    for field in schema.fields:
        if field.kind == "array":
            merged[field.name] = _merge_line_items(array_candidates[field.name])
        else:
            assert field.merge_tie_break is not None  # every scalar field has one
            merged[field.name] = _merge_scalar_leaf(
                scalar_candidates[field.name], tie_break=field.merge_tie_break
            )

    return merged


# _schema_violations truncates its result to this many dotted-path
# entries (plus one trailing summary entry) so a response that's
# malformed in nearly every leaf -- e.g. every line item missing every
# sub-leaf -- can't blow up the repair prompt built from the list.
MAX_SCHEMA_VIOLATIONS = 20

_MISSING = object()


def _schema_violations(tool_input: dict[str, Any], schema: ExtractionSchema) -> list[str]:
    """Enumerate dotted-path violations of the schema's extraction tool
    in `tool_input`, the model's untrusted tool_use.input.

    A violation path is bare `field_name` for a missing or malformed
    top-level leaf (missing and malformed collapse to the same path --
    a field can't be both), or `line_items.value[<idx>].<subfield>` for
    a malformed/missing sub-leaf of one line item. Uses
    _leaf_is_malformed as the single source of truth for "malformed" so
    this can never disagree with what _coerce_leaf actually rejects.

    line_items is special-cased: if its OWN leaf is malformed, that's
    the one violation for it and there's nothing to descend into (no
    {"value": [...]} shape exists yet). Only when its leaf is
    well-formed AND `value` is an actual list do we walk each item's
    four sub-leaves; a non-dict item degrades to "every sub-leaf
    missing" the same way _leaf_is_malformed(None) would.

    Returns the FULL, untruncated list -- callers use its true length for
    both the repair-acceptance comparison (extract_document: "strictly
    fewer violations than before") and the returned schema_violations
    count, so truncating here would corrupt both (a repair that cuts 50
    violations down to 25 must still compare as strictly better than 50,
    not tie at "21 == 21" after both get capped the same way). Truncation
    to MAX_SCHEMA_VIOLATIONS entries + a trailing "… (N more)" summary
    happens only where it belongs: at the one call site that renders
    this list into repair prompt text (extract_document).

    `tool_input` is untrusted the same way every leaf is: a non-dict
    tool_input (the model returning something other than an object at
    all) degrades to "every top-level field missing" rather than raising
    -- one violation per schema field -- so this is always safe
    to call on a raw tool_use.input without an isinstance guard at the
    call site.
    """
    if not isinstance(tool_input, dict):
        return list(schema.field_names)

    violations: list[str] = []

    for field in schema.fields:
        raw = tool_input.get(field.name, _MISSING)
        if raw is _MISSING or _leaf_is_malformed(raw):
            violations.append(field.name)
            continue

        if field.kind == "array":
            items = raw["value"]
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    item_dict = item if isinstance(item, dict) else {}
                    for sub_field in ARRAY_CELLS[field.name]:
                        if _leaf_is_malformed(item_dict.get(sub_field)):
                            violations.append(f"{field.name}.value[{idx}].{sub_field}")

    return violations


def _format_violations_for_prompt(violations: list[str]) -> str:
    """Render a (possibly long) violations list as the repair prompt's
    bullet list, truncated to MAX_SCHEMA_VIOLATIONS entries plus a
    trailing "… (N more)" summary line -- so a response that's malformed
    in nearly every leaf can't blow up the repair prompt. This is the
    ONLY place truncation happens; _schema_violations itself always
    returns the full list (see its docstring for why).
    """
    shown = violations
    suffix_lines: list[str] = []
    if len(violations) > MAX_SCHEMA_VIOLATIONS:
        remaining = len(violations) - MAX_SCHEMA_VIOLATIONS
        shown = violations[:MAX_SCHEMA_VIOLATIONS]
        suffix_lines = [f"… ({remaining} more)"]
    return "\n".join(f"- {v}" for v in shown + suffix_lines)


def _content_type_for_mime(mime_type: str) -> str:
    """Validate mime_type and return the Anthropic content block `type`
    ("image" or "document") for it, or raise NonRetryableExtractionError
    for anything unsupported. Factored out of _build_document_block so
    this check can run before any read/threadpool hop is scheduled --
    exactly the order it ran in inline before this was split out.
    """
    if mime_type == PDF_MIME_TYPE:
        return "document"
    if mime_type in IMAGE_MIME_TYPES:
        return "image"
    raise NonRetryableExtractionError(f"unsupported mime type for extraction: {mime_type}")


async def _read_document_bytes(path: Path) -> bytes:
    """Threaded raw-bytes read shared by every path that needs the
    document's bytes: the image/single-call path (via
    _build_document_block below) and both PDF code paths in
    extract_document (page counting and, for a chunked PDF,
    app.pdf.split_pdf_pages) -- factored out so a PDF's bytes are read
    from disk exactly once regardless of which path ends up handling it.
    run_in_threadpool keeps a large file's read off the event loop the
    worker shares with everything else in the process. OSError (missing
    file, permission error, ...) is wrapped as ExtractionError --
    retryable, since the underlying cause (e.g. a networked/mounted
    uploads volume being briefly unavailable) is plausibly transient,
    unlike the deterministic failures that use NonRetryableExtractionError.
    """
    try:
        return await run_in_threadpool(path.read_bytes)
    except OSError as exc:
        raise ExtractionError(f"failed to read document file {path}: {exc}") from exc


def _encode_document_block(raw_bytes: bytes, mime_type: str, content_type: str) -> dict[str, Any]:
    """Pure: base64-encode already-read bytes into the image/document
    content block shape the Anthropic API expects. Shared by the
    single-call path (_build_document_block, below) and both PDF
    paths in extract_document -- the whole-PDF single-call case reuses
    the bytes already read for page counting rather than reading the
    file twice, and the H6 chunked path calls this once per split
    single-page PDF.
    """
    data = base64.standard_b64encode(raw_bytes).decode("utf-8")
    return {
        "type": content_type,
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    }


async def _build_document_block(path: Path, mime_type: str) -> dict[str, Any]:
    """Build the image/document content block for the user message.

    Mime-type validation happens before the (threaded) file read so an
    unsupported type fails fast without touching disk -- see
    _content_type_for_mime. The read (_read_document_bytes) and the
    base64 encoding (_encode_document_block) are the same two steps
    every code path uses; this is just their composition for the
    whole-file, non-PDF-page-counted case (images, and a PDF at or
    below settings.pdf_max_pages_per_call handled outside this
    function reuses the already-read bytes directly).
    """
    content_type = _content_type_for_mime(mime_type)
    raw_bytes = await _read_document_bytes(path)
    return _encode_document_block(raw_bytes, mime_type, content_type)


# Status codes an anthropic.APIStatusError can carry that are deterministic
# given the same request -- retrying would spend money for the identical
# rejection every time. Everything else (429 rate limit, 529 overloaded,
# 408 timeout, other 5xx, and any status not explicitly deterministic,
# e.g. 409) is treated as retryable: see _classify_api_error below.
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 413, 422}


def _parse_retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a retryable response's `retry-after` header as an integer
    number of seconds, defensively: the header may be absent, or in the
    HTTP-date form RFC 7231 also allows (rather than delay-seconds) --
    either case returns None rather than raising or guessing, matching
    ExtractionError.retry_after_seconds' own "optional, best-effort" contract.
    """
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return float(int(header.strip()))
    except (TypeError, ValueError):
        return None


def _classify_api_error(exc: anthropic.AnthropicError) -> ExtractionError:
    """Map an exception raised by messages.create() onto the job queue's
    retry semantics: NonRetryableExtractionError for failures that are
    deterministic given the same request, a plain (retryable)
    ExtractionError for everything else. Shared by every call site that
    talks to the model (_extract_one_block here, app.transcription) so a
    429 or a 400 is treated identically no matter which job hit it.
    """
    if isinstance(exc, anthropic.APIStatusError):
        # RateLimitError (429) and OverloadedError (529) are both
        # APIStatusError subclasses with a fixed status_code (see the
        # claude-api skill's cached exception hierarchy), so classifying
        # by exc.status_code here covers them without a separate except
        # clause per class -- same treatment as any other 5xx/408.
        # NON_RETRYABLE_STATUS_CODES are deterministic given the same
        # request/document/model: retrying only spends money for the
        # identical rejection, so they skip straight to permanent failure.
        if exc.status_code in NON_RETRYABLE_STATUS_CODES:
            message = f"Anthropic API error ({exc.status_code}): {exc}"
            if exc.status_code == 413:
                # Deterministic given the same bytes -- retrying an
                # unchanged request would 413 again every time. This
                # fires on both the single-call path (the whole
                # document is too large) and the H6 chunked path (one
                # single-page PDF, from _extract_one_block, is still
                # too large on its own) -- page-chunking has already
                # split as far as it can by the time a single page hits
                # this, so a bigger fix (downsampling/re-encoding the
                # source) is the only way forward.
                message += (
                    " -- the document (or a single page of it) exceeds the "
                    "API's request size limit even after page chunking; "
                    "reduce the source resolution"
                )
            return NonRetryableExtractionError(message)
        # Retryable: 429, 529, 408, other 5xx, and any status not
        # explicitly deterministic above (e.g. 409). retry_after_seconds
        # is parsed from the response's `retry-after` header (when
        # present and in delay-seconds form) so app.worker.fail_job can
        # honor it as a floor on the requeue delay.
        return ExtractionError(
            f"Anthropic API error ({exc.status_code}): {exc}",
            retry_after_seconds=_parse_retry_after_seconds(exc.response),
        )
    if isinstance(exc, anthropic.APIConnectionError):
        # Covers network-shaped failures with no HTTP response to classify
        # by status code -- including anthropic.APITimeoutError, a
        # subclass of APIConnectionError. No response means no
        # `retry-after` to parse. Retryable.
        return ExtractionError(f"Anthropic API connection error: {exc}")
    # Outermost SDK branch: any anthropic exception that isn't an
    # APIStatusError or APIConnectionError (e.g. a bare
    # AnthropicError/RetryableError, or a future SDK exception type), so
    # it can't escape unclassified. Treated as transient/retryable rather
    # than silently propagating as an unhandled exception type fail_job
    # doesn't know how to bucket.
    return ExtractionError(f"Anthropic SDK error: {exc}")


@dataclass
class _PageResult:
    """The outcome of one _extract_one_block call -- either the whole
    document (single-call path) or one PDF page (H6 chunked path).

    Deliberately a subset of ExtractionResult: model/prompt_version are
    constant across every block within one extract_document call, so
    the caller (extract_document / _extract_chunked_pdf) attaches those
    once rather than repeating them per block. latency_ms is scoped to
    just THIS block's call(s) -- the first call plus an optional repair
    call -- not the whole extract_document invocation; the chunked path
    sums every page's latency_ms into its own total.
    """

    tool_input: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    repaired: bool
    schema_violations: int


async def _extract_one_block(
    client: anthropic.AsyncAnthropic,
    settings: Settings,
    document_block: dict[str, Any],
    schema: ExtractionSchema,
) -> _PageResult:
    """Send ONE document content block (a whole image/PDF for the
    single-call path, or a single PDF page for the H6 chunked path)
    through record_extraction and return the result.

    This is every step extract_document used to run inline, factored
    out unchanged so the single-call and per-page chunked paths share
    identical behavior: build the user message from `document_block` ->
    create() -> classify any anthropic.APIStatusError/APIConnectionError/
    AnthropicError -> check stop_reason -> find the tool_use block ->
    schema-repair reprompt -> sum this block's tokens. See
    extract_document's docstring for the full failure classification
    (what's retryable vs not) and the schema-repair reprompt semantics
    -- none of that changed, it just moved here.
    """
    user_message = {
        "role": "user",
        "content": [
            document_block,
            {"type": "text", "text": "Extract the document now."},
        ],
    }

    tool = record_extraction_tool(schema)

    start = time.perf_counter()
    with model_call_span(
        settings.extraction_model, max_tokens=schema.max_tokens, prompt_version=schema.prompt_version
    ) as span:
        try:
            response = await client.messages.create(
                model=settings.extraction_model,
                max_tokens=schema.max_tokens,
                system=schema.prompt_text,
                tools=[tool],
                tool_choice={"type": "tool", "name": "record_extraction"},
                messages=[user_message],
            )
        except anthropic.AnthropicError as exc:
            raise _classify_api_error(exc) from exc
        record_model_response(
            span,
            response,
            _compute_cost_usd(
                settings.extraction_model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
        )

    if response.stop_reason != "tool_use":
        error_cls = (
            ModelRefusalError
            if response.stop_reason == "refusal"
            else NonRetryableExtractionError
        )
        raise error_cls(
            f"model did not return a usable tool call (stop_reason={response.stop_reason!r}); "
            f"billed usage: input_tokens={response.usage.input_tokens}, "
            f"output_tokens={response.usage.output_tokens}",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=_compute_cost_usd(
                settings.extraction_model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
        )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"), None
    )
    if tool_use_block is None:
        raise ExtractionError(
            "stop_reason was tool_use but no tool_use block was found; "
            f"billed usage: input_tokens={response.usage.input_tokens}, "
            f"output_tokens={response.usage.output_tokens}",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=_compute_cost_usd(
                settings.extraction_model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
        )

    tool_input = tool_use_block.input
    violations = _schema_violations(tool_input, schema)
    repaired = False
    total_input_tokens = response.usage.input_tokens
    total_output_tokens = response.usage.output_tokens

    # Schema-repair reprompt: tool_choice guarantees *a* tool_use block,
    # not one that matches the schema's extraction tool (see module
    # docstring). If the first response violated it, make exactly one
    # additional call asking the model to fix only the structure. The
    # repair call's cost is billed regardless of whether it's accepted
    # (see docstring below) -- money was spent either way -- but a
    # failed/unusable repair call must never turn an otherwise-usable
    # extraction into a failed job, so any exception, refusal, or missing
    # tool_use block from the repair call just falls back to the first
    # response.
    if violations and settings.schema_repair:
        repair_text = schema.repair_prompt_text.replace(
            "{{violations}}", _format_violations_for_prompt(violations)
        )
        try:
            with model_call_span(
                settings.extraction_model,
                max_tokens=schema.max_tokens,
                prompt_version=schema.repair_prompt_version,
            ) as repair_span:
                repair_response = await client.messages.create(
                    model=settings.extraction_model,
                    max_tokens=schema.max_tokens,
                    system=schema.prompt_text,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "record_extraction"},
                    messages=[
                        user_message,
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": tool_use_block.id,
                                    "name": tool_use_block.name,
                                    "input": tool_input,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_block.id,
                                    "content": repair_text,
                                    "is_error": True,
                                }
                            ],
                        },
                    ],
                )
                record_model_response(
                    repair_span,
                    repair_response,
                    _compute_cost_usd(
                        settings.extraction_model,
                        repair_response.usage.input_tokens,
                        repair_response.usage.output_tokens,
                    ),
                )
        except Exception:  # a failed repair must never fail an otherwise-usable extraction
            logger.warning(
                "schema repair call failed for record_extraction; falling back to "
                "the first response's tool_input",
                exc_info=True,
            )
        else:
            # The repair call is billed the moment it succeeds, whether
            # or not its result is actually accepted below (see
            # docstring) -- both calls spent real money.
            total_input_tokens += repair_response.usage.input_tokens
            total_output_tokens += repair_response.usage.output_tokens

            repair_tool_use_block = None
            if repair_response.stop_reason == "tool_use":
                repair_tool_use_block = next(
                    (block for block in repair_response.content if block.type == "tool_use"),
                    None,
                )

            if repair_tool_use_block is None:
                logger.warning(
                    "schema repair call returned no usable tool_use block "
                    "(stop_reason=%r); falling back to the first response's tool_input",
                    repair_response.stop_reason,
                )
            else:
                repair_violations = _schema_violations(repair_tool_use_block.input, schema)
                if len(repair_violations) < len(violations):
                    tool_input = repair_tool_use_block.input
                    violations = repair_violations
                    repaired = True
                else:
                    logger.warning(
                        "schema repair did not reduce violations (%d -> %d); "
                        "falling back to the first response's tool_input",
                        len(violations),
                        len(repair_violations),
                    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    return _PageResult(
        tool_input=tool_input,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        latency_ms=latency_ms,
        repaired=repaired,
        schema_violations=len(violations),
    )


async def _extract_chunked_pdf(
    raw_bytes: bytes, settings: Settings, schema: ExtractionSchema
) -> ExtractionResult:
    """The H6 chunked-PDF path: settings.pdf_max_pages_per_call < page
    count <= settings.pdf_max_pages. Split the PDF into one single-page
    PDF per page (app.pdf.split_pdf_pages), extract each page
    SEQUENTIALLY via _extract_one_block -- a plain for loop, not
    asyncio.gather, so this never bursts multiple pages' worth of
    requests against the rate limiter at once -- then merge the
    per-page tool_inputs (merge_page_tool_inputs) into one tool_input
    shape-identical to a single-call extraction's, plus a `_chunks`
    provenance list keyed by page number (see process_document_job:
    raw_response is this tool_input verbatim, and its per-field
    iteration over the schema ignores the extra `_chunks` key, so persistence and eval
    scoring are untouched by it).

    If any page's _extract_one_block call raises, the tokens/cost
    already spent on EARLIER pages in this call are added onto the
    raised exception's own input_tokens/output_tokens/cost_usd (the
    existing billed-but-errored convention documented on
    ExtractionError) before it's re-raised, so a requeued job's
    last_error and any programmatic caller (e.g. the eval runner's cost
    cap) never lose track of spend from pages that succeeded before the
    failure.
    """
    try:
        page_pdfs = split_pdf_pages(raw_bytes)
    except ValueError as exc:
        raise NonRetryableExtractionError(f"unreadable or encrypted PDF: {exc}") from exc

    client = _build_client(settings)

    page_inputs: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0
    total_schema_violations = 0
    any_repaired = False

    for page_num, page_pdf_bytes in enumerate(page_pdfs, start=1):
        document_block = _encode_document_block(page_pdf_bytes, PDF_MIME_TYPE, "document")
        try:
            page_result = await _extract_one_block(client, settings, document_block, schema)
        except ExtractionError as exc:
            exc.input_tokens = total_input_tokens + (exc.input_tokens or 0)
            exc.output_tokens = total_output_tokens + (exc.output_tokens or 0)
            exc.cost_usd = _compute_cost_usd(
                settings.extraction_model, exc.input_tokens, exc.output_tokens
            )
            raise

        page_inputs.append(page_result.tool_input)
        chunks.append(
            {
                "page": page_num,
                "tool_input": page_result.tool_input,
                "input_tokens": page_result.input_tokens,
                "output_tokens": page_result.output_tokens,
                "cost_usd": _compute_cost_usd(
                    settings.extraction_model,
                    page_result.input_tokens,
                    page_result.output_tokens,
                ),
                "latency_ms": page_result.latency_ms,
                "repaired": page_result.repaired,
            }
        )
        total_input_tokens += page_result.input_tokens
        total_output_tokens += page_result.output_tokens
        total_latency_ms += page_result.latency_ms
        total_schema_violations += page_result.schema_violations
        any_repaired = any_repaired or page_result.repaired

    merged_tool_input = merge_page_tool_inputs(page_inputs, schema)
    merged_tool_input["_chunks"] = chunks

    return ExtractionResult(
        tool_input=merged_tool_input,
        model=settings.extraction_model,
        prompt_version=schema.prompt_version,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=_compute_cost_usd(
            settings.extraction_model, total_input_tokens, total_output_tokens
        ),
        latency_ms=total_latency_ms,
        repaired=any_repaired,
        schema_violations=total_schema_violations,
    )


async def extract_document(
    path: str | Path, mime_type: str, *, schema: ExtractionSchema | None = None
) -> ExtractionResult:
    """Call Claude with the extraction tool forced, and return the parsed
    result. `schema` is the extraction schema to ask for (default: the one
    named by settings.extraction_prompt_version), and the result records
    its prompt_version. Dispatches to one of three paths based on document type and
    (for PDFs) page count -- see the H6 task notes for the full design:

    - Images, and PDFs with page count <= settings.pdf_max_pages_per_call
      (default 5): unchanged single-call path, byte-identical to the
      pre-H6 behavior -- one _extract_one_block call on the whole
      document, wrapped straight into an ExtractionResult.
    - PDFs with settings.pdf_max_pages_per_call < page count <=
      settings.pdf_max_pages (default 20): the H6 chunked path -- see
      _extract_chunked_pdf.
    - PDFs with page count > settings.pdf_max_pages, or a PDF pypdf
      can't read/decrypt (app.pdf.count_pdf_pages raising ValueError):
      raise NonRetryableExtractionError BEFORE any API call -- refusing
      an oversized or unreadable document is free, and retrying an
      identical request would fail identically every time.

    Failure classification for the actual extraction calls (both the
    single-call and per-page chunked cases) lives in _extract_one_block
    -- see its docstring for what's retryable vs not. An unpriced model
    is still validated up front here, before any API call or even a
    PDF page count, and raises NonRetryableExtractionError for free.

    Schema-repair reprompt: unchanged from before H6, see
    _extract_one_block's docstring -- it runs per block (once for the
    whole document on the single-call path, once per page on the
    chunked path), never across the whole extract_document call.

    Page counting: only relevant to PDFs. The file's bytes are read
    exactly once (_read_document_bytes) regardless of which path ends
    up handling the document -- the single-call PDF path reuses those
    bytes for _encode_document_block rather than reading the file
    again, and the chunked path passes them straight to
    app.pdf.split_pdf_pages.
    """
    settings = get_settings()
    if schema is None:
        schema = schema_for(settings.extraction_prompt_version)
    _ensure_model_priced(settings.extraction_model)

    path = Path(path)

    if mime_type == PDF_MIME_TYPE:
        raw_bytes = await _read_document_bytes(path)
        try:
            page_count = count_pdf_pages(raw_bytes)
        except ValueError as exc:
            raise NonRetryableExtractionError(
                f"unreadable or encrypted PDF, cannot extract: {exc}"
            ) from exc

        if page_count == 0:
            # A guaranteed-rejection API call otherwise: there is
            # nothing for the model to extract from, and no page count
            # threshold above catches this (0 is <= every positive
            # pdf_max_pages_per_call/pdf_max_pages). Refuse locally,
            # free, alongside the other pre-flight checks.
            raise NonRetryableExtractionError("PDF has 0 pages; nothing to extract")

        if page_count > settings.pdf_max_pages:
            raise NonRetryableExtractionError(
                f"PDF has {page_count} pages, exceeding pdf_max_pages="
                f"{settings.pdf_max_pages}; refusing to extract before spending "
                "any API budget"
            )

        if page_count > settings.pdf_max_pages_per_call:
            return await _extract_chunked_pdf(raw_bytes, settings, schema)

        document_block = _encode_document_block(raw_bytes, mime_type, "document")
    else:
        document_block = await _build_document_block(path, mime_type)

    client = _build_client(settings)
    page_result = await _extract_one_block(client, settings, document_block, schema)

    return ExtractionResult(
        tool_input=page_result.tool_input,
        model=settings.extraction_model,
        prompt_version=schema.prompt_version,
        input_tokens=page_result.input_tokens,
        output_tokens=page_result.output_tokens,
        cost_usd=_compute_cost_usd(
            settings.extraction_model, page_result.input_tokens, page_result.output_tokens
        ),
        latency_ms=page_result.latency_ms,
        repaired=page_result.repaired,
        schema_violations=page_result.schema_violations,
    )


def build_extracted_fields(
    extraction_id: uuid.UUID,
    tool_input: dict,
    review_threshold: float,
    schema: ExtractionSchema,
) -> list[ExtractedField]:
    """One ExtractedField per schema field, read defensively out of
    the model's tool_input (see _coerce_leaf); fields under review_threshold
    get needs_review with the reason "confidence" (the reconciliation can flag
    more afterwards, see app/review_routing.py). Shared by the worker and the demo seed so a replayed
    extraction is persisted exactly as a live one would be.
    """
    fields = []
    for field in schema.fields:
        value, confidence = _coerce_leaf(tool_input.get(field.name))
        if field.kind == "array":
            # An array field (line_items): store the array itself as
            # the ExtractedField value (its own confidence already
            # lives in the confidence column below). Every other
            # field is a scalar leaf, so the reconstructed
            # {"value", "confidence"} object is stored.
            field_value: Any = value if isinstance(value, list) else []
        else:
            field_value = {"value": value, "confidence": confidence}
        low_confidence = confidence < review_threshold
        fields.append(
            ExtractedField(
                extraction_id=extraction_id,
                field_name=field.name,
                value=field_value,
                confidence=confidence,
                needs_review=low_confidence,
                review_reason=REVIEW_REASON_CONFIDENCE if low_confidence else None,
            )
        )
    return fields


async def process_document_job(session: AsyncSession, job: Job) -> None:
    """The real worker handler (wired in as the default in app/worker.py).

    Follows the worker's existing commit contract: this function only
    adds/mutates ORM objects on `session` and flushes when it needs a
    generated id (extraction.id, for the ExtractedField FK) -- it never
    calls session.commit() itself. The caller (worker.run_once) commits
    via complete_job on success, or rolls back and requeues via fail_job
    if this raises. That rollback is why a partial extraction never lands
    half-written: if anything below raises, everything added here is
    discarded along with it.

    The session is deliberately rolled back (releasing its DB connection
    back to the pool) before the multi-second VLM call, since nothing
    has been written yet at that point -- holding a connection idle in
    transaction for the duration of an external API call would tie it
    up for no reason and block other workers' SKIP LOCKED scans. See the
    rollback in app.worker.fail_job for the same "expire, then re-fetch
    by id" pattern this reuses.
    """
    # Imported here: review_routing reaches back into this module through
    # reconcile (which parses leaves with _coerce_leaf).
    from app.review_routing import refresh_reconciliation

    document = await session.get(Document, job.document_id)
    if document is None:
        raise ExtractionError(f"document {job.document_id} not found")

    document_id = document.id
    storage_path = document.storage_path
    mime_type = document.mime_type

    await session.rollback()

    settings = get_settings()
    result = await extract_document(storage_path, mime_type)

    # rollback() expires every attribute on every object in the session,
    # so `document` must be re-fetched rather than reused.
    document = await session.get(Document, document_id)
    if document is None:
        raise ExtractionError(f"document {document_id} vanished during extraction")

    try:
        extraction = Extraction(
            document_id=document_id,
            prompt_version=result.prompt_version,
            model=result.model,
            raw_response=result.tool_input,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )
        session.add(extraction)
        await session.flush()  # assigns extraction.id without committing

        extraction_id = extraction.id
        session.add_all(
            build_extracted_fields(
                extraction_id,
                result.tool_input,
                settings.review_threshold,
                schema_for(result.prompt_version),
            )
        )
        await session.flush()

        # Check the arithmetic and flag the fields it doubts. The extraction
        # was billed, so a failure here (a bug in the rules, say) must not cost
        # it: the savepoint undoes only the reconciliation, the fields keep
        # their confidence flags, the error is logged, and
        # scripts/reconcile_all.py fills in the missing row later. A retry
        # would re-bill the call for the same failure.
        try:
            async with session.begin_nested():
                await refresh_reconciliation(session, extraction_id, settings)
        except Exception:
            logger.exception(
                "reconciliation failed for extraction %s; kept the extraction without it",
                extraction_id,
            )
            # The savepoint rollback expired every field it touched: do not read
            # their attributes from here on (MissingGreenlet).

        document.status = "extracted"

        # Queue the retrieval index build (app/retrieval/index_job.py) in
        # this same transaction: it commits exactly when the extraction
        # does, so there's never an extracted document that was silently
        # skipped, nor an index job for an extraction that rolled back.
        if settings.index_after_extraction:
            session.add(
                Job(
                    document_id=document_id,
                    kind=JOB_KIND_INDEX,
                    traceparent=current_traceparent(),
                )
            )
    except ExtractionError:
        raise
    except Exception as exc:
        # The API call already happened and was billed (result.cost_usd)
        # -- if our own persistence code still fails (e.g. a DB error),
        # don't let that spend go invisible in the job's last_error.
        # Retryable: a persistence-layer failure (lock timeout, dropped
        # connection) is plausibly transient, unlike the deterministic
        # cases above.
        raise ExtractionError(
            "persistence failed after a successful, billed API call "
            f"(input_tokens={result.input_tokens}, output_tokens={result.output_tokens}, "
            f"cost_usd={result.cost_usd:.6f}): {exc}",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        ) from exc
