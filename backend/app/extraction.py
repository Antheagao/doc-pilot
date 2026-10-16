"""VLM-based structured extraction from uploaded documents.

Reads a document image/PDF, asks Claude to call the `record_extraction`
tool with structured field data, and persists the result as an Extraction
row plus one ExtractedField row per top-level field. This module is the
real `process_document_job` handler wired into app.worker (see
app/worker.py).

Prompt versioning: the extraction prompts live in prompts/ (extract_v1.md,
extract_v2.md) as plain files, not inline in code, so prompt iterations are
versioned by filename and eval results can be tied to a specific
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
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import anthropic
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.extraction_fields import (
    ARRAY_CELLS,
    EVIDENCE_QUOTE_MAX_CHARS,
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
from app.pii import digits_of, mask_card_numbers, mask_digit_runs
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
        stop_reasons: list[dict[str, str | None]] | None = None,
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.retry_after_seconds = retry_after_seconds
        # The stop reasons of the calls made before this failure, one entry per
        # block as on ExtractionResult.stop_reasons (None where no call finished).
        self.stop_reasons = stop_reasons


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
    # How each block's calls stopped, one {"first", "repair"} entry per block (a
    # whole document on the single-call path, each page on the chunked path):
    # "first" is the stop reason of the extraction call, "repair" that of the
    # repair call, None when none was made or it failed. A "max_tokens" here
    # means an answer was cut off, which tool_use alone would hide.
    stop_reasons: list[dict[str, str | None]] = dataclass_field(default_factory=list)


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


# No document has more pages than this; a page beyond it is junk, not a citation
# (the extraction limits are far lower, see Settings.pdf_max_pages).
MAX_EVIDENCE_PAGE = 10_000


def _evidence_quote(evidence: Any) -> str | None:
    """The quote of one raw evidence object, stripped, or None unless it is
    text that is non-empty once stripped and at most EVIDENCE_QUOTE_MAX_CHARS
    long."""
    if not isinstance(evidence, dict):
        return None
    quote = evidence.get("quote")
    if not isinstance(quote, str):
        return None
    quote = quote.strip()
    if not quote or len(quote) > EVIDENCE_QUOTE_MAX_CHARS:
        return None
    return quote


def _coerce_evidence(raw_leaf: Any) -> dict[str, Any] | None:
    """The `{"page", "quote"}` evidence of one raw leaf, or None.

    Evidence is untrusted like the rest of the leaf (see _coerce_leaf), and a
    half-right citation is worse than none, so it is all or nothing: a dict
    whose quote passes _evidence_quote and whose page is an int from 1 to
    MAX_EVIDENCE_PAGE (a bool is not a page) is kept, with the quote stripped.
    The page key must be there, and its value may be None, which is how
    extraction records a quote whose page the model got wrong (see
    _evidence_page): the quote is still checkable, only its place is unknown.
    The missing-key rule is reachable only from the seed, replay, and
    correction paths: on the model path _paged_leaf always adds the key.
    Anything else, including a missing key, becomes None. It reads only the
    "evidence" key: whether the leaf's value and confidence are usable is
    _coerce_leaf's question.
    """
    if not isinstance(raw_leaf, dict):
        return None
    evidence = raw_leaf.get("evidence")
    quote = _evidence_quote(evidence)
    if quote is None:
        return None
    if "page" not in evidence:
        return None
    page = evidence["page"]
    if page is not None and (
        isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= MAX_EVIDENCE_PAGE
    ):
        return None
    return {"page": page, "quote": quote}


def _evidence_page(page: Any, page_count: int | None) -> int | None:
    """The page an evidence object is stored with, given how many pages the
    document the model saw has. `page_count` None means an image, which is one
    page whatever the model wrote, so the page is 1. For a PDF the model's page
    is kept when it is an int from 1 to `page_count` and is None otherwise: a
    page the document does not have is a wrong citation, not a wrong quote, so
    the quote stays and the place is dropped."""
    if page_count is None:
        return 1
    if isinstance(page, int) and not isinstance(page, bool) and 1 <= page <= page_count:
        return page
    return None


def _paged_leaf(leaf: Any, page_count: int | None) -> Any:
    """One leaf, and the cells of the rows its value holds when it is an array,
    with the page of each evidence object that is an object set by
    _evidence_page. Anything that is not an evidence object (null, a bare
    scalar, a missing key) is left for _coerce_evidence to judge."""
    if not isinstance(leaf, dict):
        return leaf
    paged = leaf
    evidence = leaf.get("evidence")
    if isinstance(evidence, dict):
        paged = {**paged, "evidence": {**evidence, "page": _evidence_page(evidence.get("page"), page_count)}}
    value = leaf.get("value")
    if isinstance(value, list):
        rows = [
            {key: _paged_leaf(cell, page_count) for key, cell in row.items()}
            if isinstance(row, dict)
            else row
            for row in value
        ]
        paged = {**paged, "value": rows}
    return paged


def _with_evidence_pages(tool_input: Any, schema: ExtractionSchema, page_count: int | None) -> Any:
    """`tool_input` with the page of every evidence object set by
    _evidence_page; a schema without evidence (v1) is returned as it is."""
    if not schema.evidence or not isinstance(tool_input, dict):
        return tool_input
    return {name: _paged_leaf(leaf, page_count) for name, leaf in tool_input.items()}


# --- PII: no more than the last four digits of a card or account number ------
#
# The prompt asks the model to write only the last four digits, but a prompt
# is not a control, so the rule is enforced here on every tool_input before
# anything stores or logs it (_extract_one_block and build_extracted_fields).
# Review corrections get the same rule as a 422 (app/records.py).

PAYMENT_CARD_FIELD = "payment_card_last4"
PAYMENT_METHOD_FIELD = "payment_method"
_PAYMENT_FIELDS = (PAYMENT_METHOD_FIELD, PAYMENT_CARD_FIELD)


def _last_four_digits(value: Any) -> str | None:
    """The last four digits of a payment_card_last4 value, or None when it has
    fewer than four digits (or is not text or a number at all)."""
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    digits = digits_of(str(value))
    return digits[-4:] if len(digits) >= 4 else None


def _masked_payment_leaf(name: str, leaf: Any) -> dict[str, Any] | None:
    """One payment leaf rebuilt from the keys it is allowed to have.

    A payment leaf is the one place a card number is likely to be, so nothing
    the model sent is carried over by default: `value` is text with its long
    digit runs masked (payment_method) or the last four digits
    (payment_card_last4), and anything else, a list or an object holding the
    number, is null; `confidence` is kept only as a number; `evidence` is
    rebuilt as {page, quote} when its quote is usable (see _evidence_quote),
    with the quote masked, and is null otherwise; every other key is dropped. A leaf missing value or confidence stays missing it, so
    _leaf_is_malformed still reports it. A leaf that is not an object is
    dropped (None), which is the same violation as a bare scalar.
    """
    if not isinstance(leaf, dict):
        return None
    masked: dict[str, Any] = {}
    if "value" in leaf:
        value = leaf["value"]
        if name == PAYMENT_CARD_FIELD:
            masked["value"] = _last_four_digits(value)
        else:
            masked["value"] = mask_digit_runs(value) if isinstance(value, str) else None
    if "confidence" in leaf:
        try:
            masked["confidence"] = float(leaf["confidence"])
        except (TypeError, ValueError, OverflowError):
            pass  # left out: the leaf then fails _leaf_is_malformed as before
    if "evidence" in leaf:
        # Page-agnostic like _swept_evidence: the page is fixed on the model
        # path by _with_evidence_pages, and judged by _coerce_evidence when the
        # leaf is stored.
        quote = _evidence_quote(leaf["evidence"])
        masked["evidence"] = (
            None
            if quote is None
            else {"page": leaf["evidence"].get("page"), "quote": mask_digit_runs(quote)}
        )
    return masked


def mask_payment_fields(tool_input: Any) -> Any:
    """`tool_input` with the card number kept to its last four digits.

    payment_card_last4 is cut to its last four digits (null when it holds
    fewer), and any long digit run (see app/pii.py) in payment_method and in
    either payment field's evidence quote becomes "****" plus its last four
    ("VISA 4111 1111 1111 1111" becomes "VISA ****1111"). Both leaves are
    rebuilt from allowed keys only (_masked_payment_leaf), so a number cannot
    ride in an unexpected shape or key. Other fields are not touched here: an
    invoice number may be sixteen digits long. Returns a new dict when a payment
    field is present and the input itself when there is none (v1 has no payment
    fields), so a caller can always use the result.
    """
    if not isinstance(tool_input, dict):
        return tool_input
    present = [name for name in _PAYMENT_FIELDS if name in tool_input]
    if not present:
        return tool_input
    masked = dict(tool_input)
    for name in present:
        masked[name] = _masked_payment_leaf(name, tool_input[name])
    return masked


def _swept_evidence(evidence: Any) -> Any:
    """An evidence object with a card number masked out of its quote. Rebuilt
    as {page, quote}: a quote that is not text, or an extra key, could hold the
    number, so neither is kept. None and a non-object (nothing usable, see
    _coerce_evidence) become None."""
    if not isinstance(evidence, dict):
        return None
    quote = evidence.get("quote")
    return {
        "page": evidence.get("page"),
        "quote": mask_card_numbers(quote) if isinstance(quote, str) else None,
    }


def _swept_leaf(leaf: Any) -> Any:
    """One leaf, and the cells of the rows its value holds when it is an array,
    with card numbers masked out of their evidence quotes. A leaf with no
    evidence key (v1) is returned as it is."""
    if not isinstance(leaf, dict):
        return leaf
    swept = leaf
    if "evidence" in leaf:
        evidence = _swept_evidence(leaf["evidence"])
        if evidence != leaf["evidence"]:
            swept = {**swept, "evidence": evidence}
    value = leaf.get("value")
    if isinstance(value, list):
        rows = [
            {key: _swept_leaf(cell) for key, cell in row.items()} if isinstance(row, dict) else row
            for row in value
        ]
        if rows != value:
            swept = {**swept, "value": rows}
    return swept


def mask_evidence_quotes(tool_input: Any) -> Any:
    """`tool_input` with each card number masked out of every evidence quote,
    whatever the field.

    A quote is the printed line a value came from, and the line that shows the
    amount paid often shows the card as well ("VISA 4111 1111 1111 1111
    APPROVED $50.00"). The rule is the Luhn-checked one (app/pii.py,
    mask_card_numbers): a run of 13 to 19 digits that is a card number becomes
    "****" plus its last four, and an order reference that is not one is left
    alone. Quotes only, never values: a sixteen-digit invoice number stays
    whole. Returns the input itself when nothing changes.
    """
    if not isinstance(tool_input, dict):
        return tool_input
    swept = {name: _swept_leaf(leaf) for name, leaf in tool_input.items()}
    return swept if swept != tool_input else tool_input


def mask_card_data(tool_input: Any) -> Any:
    """Every card-number rule on one tool_input: the payment fields first
    (mask_payment_fields), then the quote sweep (mask_evidence_quotes). Safe to
    apply twice, and a no-op on a v1 tool_input."""
    return mask_evidence_quotes(mask_payment_fields(tool_input))


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


def _chunk_evidence(raw_leaf: Any, page_idx: int) -> dict[str, Any] | None:
    """The evidence of one page's raw scalar leaf, cited to the page it came
    from. Each chunk call saw a single page, so whatever page the model wrote
    is replaced by `page_idx + 1`; only the quote is taken from the model. A
    malformed leaf has no evidence, like its value (see _coerce_leaf)."""
    if _leaf_is_malformed(raw_leaf):
        return None
    quote = _evidence_quote(raw_leaf.get("evidence"))
    return None if quote is None else {"page": page_idx + 1, "quote": quote}


def _chunk_cell(cell: Any, page: int) -> Any:
    """One cell of an array row from the chunk that read `page`, with its
    evidence cited to that page (see _chunk_evidence). A cell with no
    evidence key is returned as it is."""
    if not isinstance(cell, dict) or "evidence" not in cell:
        return cell
    quote = _evidence_quote(cell["evidence"])
    return {**cell, "evidence": None if quote is None else {"page": page, "quote": quote}}


def _merge_scalar_leaf(
    candidates: list[tuple[int, Any, float, dict[str, Any] | None]],
    *,
    tie_break: str,
    evidence: bool,
) -> dict[str, Any]:
    """Merge one scalar field's per-page (page_index, value, confidence,
    evidence) candidates -- already read defensively via _coerce_leaf, so a
    malformed leaf on any page has already degraded to
    (page_index, None, 0.0, None) by the time it reaches here -- into a single
    {"value", "confidence"} leaf, shape-identical to a single-call leaf.
    With `evidence` the leaf also carries the winning candidate's evidence
    (already cited to its page, see _chunk_evidence); when the winner has none,
    that of the most confident candidate with the same value that does, and
    None when there is no such candidate; without `evidence` the leaf has no
    evidence key at all.

    All-null (every candidate's value is None): the merged leaf's value
    is None, confidence is the MINIMUM confidence across every
    candidate (including the null ones) -- an absent field should land
    low-confidence and route to review, not look artificially confident
    just because no page actually reported anything wrong. Its evidence is
    None: there is no value to cite.

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
        merged: dict[str, Any] = {
            "value": None,
            "confidence": min((c[2] for c in candidates), default=0.0),
        }
        if evidence:
            merged["evidence"] = None
        return merged

    def most_confident(pool: list[tuple[int, Any, float, dict[str, Any] | None]]):
        best_confidence = max(c[2] for c in pool)
        tied = [c for c in pool if c[2] == best_confidence]
        return (min if tie_break == "earliest" else max)(tied, key=lambda c: c[0])

    winner = most_confident(non_null)

    base_value = non_null[0][1]
    disagreement = any(_values_disagree(base_value, c[1]) for c in non_null[1:])

    confidence = min(winner[2], MERGE_CONFLICT_CONFIDENCE) if disagreement else winner[2]
    merged = {"value": winner[1], "confidence": confidence}
    if evidence:
        merged["evidence"] = winner[3]
        if winner[3] is None:
            # The winner has nothing to cite: borrow the quote of the most
            # confident candidate that read the same value and has one. The
            # value and confidence stay the winner's.
            cited = [
                c for c in non_null if c[3] is not None and not _values_disagree(winner[1], c[1])
            ]
            if cited:
                merged["evidence"] = most_confident(cited)[3]
    return merged


def _merge_line_items(
    page_line_items: list[tuple[int, Any, float]], *, evidence: bool
) -> dict[str, Any]:
    """Merge per-page (page_index, value, confidence) candidates for an array
    field (already read via _coerce_leaf) into one leaf: every page's
    array that actually IS a list is concatenated in page order, and
    the merged confidence is the MINIMUM confidence across only the
    pages that contributed a list -- a page whose line_items leaf was
    null/malformed contributed nothing to concatenate, so it doesn't
    drag the confidence down either. If no page produced a list at all,
    returns {"value": [], "confidence": 0.0} so the field routes to
    review like any other all-null field.

    With `evidence`, each cell of a page's rows cites that page (_chunk_cell):
    the cell keeps its quote and takes the page of the chunk it came from.
    """
    concatenated: list[Any] = []
    contributing_confidences: list[float] = []
    for page_idx, value, confidence in page_line_items:
        if isinstance(value, list):
            if evidence:
                value = [
                    {key: _chunk_cell(cell, page_idx + 1) for key, cell in row.items()}
                    if isinstance(row, dict)
                    else row
                    for row in value
                ]
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

    Under a schema with evidence, a scalar leaf carries its winning page's
    evidence (or an agreeing page's, when the winner has none) and an array's
    cells carry their own, every one cited to the page index + 1 of the chunk
    it came from, whatever page the model wrote: each chunk call saw one page,
    so the chunk is the only page that can be right.
    A v1 merge has no evidence and is unchanged.

    Every page's tool_input is untrusted the same way a single
    response's is (see the module docstring / _coerce_leaf): a
    non-dict page_input, or one missing/mangling a field entirely,
    degrades that page's contribution to (None, 0.0) via _coerce_leaf
    rather than raising.
    """
    array_candidates: dict[str, list[tuple[int, Any, float]]] = {
        field.name: [] for field in schema.fields if field.kind == "array"
    }
    scalar_candidates: dict[str, list[tuple[int, Any, float, dict[str, Any] | None]]] = {
        field.name: [] for field in schema.fields if field.kind != "array"
    }

    for page_idx, page_input in enumerate(page_inputs):
        for field_name, array_field_candidates in array_candidates.items():
            raw_array = page_input.get(field_name) if isinstance(page_input, dict) else None
            array_value, array_confidence = _coerce_leaf(raw_array)
            array_field_candidates.append((page_idx, array_value, array_confidence))

        for field_name, candidates in scalar_candidates.items():
            raw_leaf = page_input.get(field_name) if isinstance(page_input, dict) else None
            value, confidence = _coerce_leaf(raw_leaf)
            leaf_evidence = _chunk_evidence(raw_leaf, page_idx) if schema.evidence else None
            candidates.append((page_idx, value, confidence, leaf_evidence))

    merged: dict[str, Any] = {}
    for field in schema.fields:
        if field.kind == "array":
            merged[field.name] = _merge_line_items(
                array_candidates[field.name], evidence=schema.evidence
            )
        else:
            assert field.merge_tie_break is not None  # every scalar field has one
            merged[field.name] = _merge_scalar_leaf(
                scalar_candidates[field.name],
                tie_break=field.merge_tie_break,
                evidence=schema.evidence,
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
    a malformed/missing sub-leaf of one row of an array field (`tax_lines`
    has its own cells, see ARRAY_CELLS). A leaf's evidence is never a
    violation: a bad citation degrades to none (_coerce_evidence) and is not
    worth a repair call. Uses
    _leaf_is_malformed as the single source of truth for "malformed" so
    this can never disagree with what _coerce_leaf actually rejects.

    An array field (line_items, tax_lines) is special-cased: if its OWN
    leaf is malformed, that's the one violation for it and there's
    nothing to descend into (no {"value": [...]} shape exists yet). Only
    when its leaf is well-formed AND `value` is an actual list do we walk
    each item's sub-leaves (ARRAY_CELLS lists them); a non-dict item
    degrades to "every sub-leaf missing" the same way
    _leaf_is_malformed(None) would.

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
    stop_reason: str
    repair_stop_reason: str | None


async def _extract_one_block(
    client: anthropic.AsyncAnthropic,
    settings: Settings,
    document_block: dict[str, Any],
    schema: ExtractionSchema,
    page_count: int | None,
) -> _PageResult:
    """Send ONE document content block (a whole image/PDF for the
    single-call path, or a single PDF page for the H6 chunked path)
    through record_extraction and return the result.

    `page_count` is how many pages the block has, None for an image. Under a
    schema with evidence it fixes every evidence page the model wrote
    (_evidence_page): an image's evidence is on page 1, a PDF's is kept when
    it is a page the block has and is None otherwise, with the quote kept.

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
            stop_reasons=[{"first": response.stop_reason, "repair": None}],
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
            stop_reasons=[{"first": response.stop_reason, "repair": None}],
        )

    # Masked before anything else sees it: the repair prompt below, the
    # stored raw_response, the chunk provenance, and the eval artifacts are all
    # built from tool_input, so none of them can hold a full card number.
    model_input = mask_card_data(tool_use_block.input)
    tool_input = _with_evidence_pages(model_input, schema, page_count)
    violations = _schema_violations(tool_input, schema)
    repaired = False
    repair_stop_reason: str | None = None
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
                                    "input": model_input,
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
            repair_stop_reason = repair_response.stop_reason

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
                repair_tool_input = _with_evidence_pages(
                    mask_card_data(repair_tool_use_block.input), schema, page_count
                )
                repair_violations = _schema_violations(repair_tool_input, schema)
                if len(repair_violations) < len(violations):
                    tool_input = repair_tool_input
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
        stop_reason=response.stop_reason,
        repair_stop_reason=repair_stop_reason,
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
    stop_reasons: list[dict[str, str | None]] = []

    for page_num, page_pdf_bytes in enumerate(page_pdfs, start=1):
        document_block = _encode_document_block(page_pdf_bytes, PDF_MIME_TYPE, "document")
        try:
            page_result = await _extract_one_block(
                client, settings, document_block, schema, page_count=1
            )
        except ExtractionError as exc:
            exc.input_tokens = total_input_tokens + (exc.input_tokens or 0)
            exc.output_tokens = total_output_tokens + (exc.output_tokens or 0)
            exc.cost_usd = _compute_cost_usd(
                settings.extraction_model, exc.input_tokens, exc.output_tokens
            )
            exc.stop_reasons = [*stop_reasons, *(exc.stop_reasons or [])]
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
        stop_reasons.append(
            {"first": page_result.stop_reason, "repair": page_result.repair_stop_reason}
        )

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
        stop_reasons=stop_reasons,
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

    Evidence pages (schema v2): whatever page the model cites is fixed on
    the way out of _extract_one_block -- 1 for an image, None for a page the
    PDF does not have (the quote is kept) -- and the chunked path cites each
    scalar and cell to the chunk it came from (merge_page_tool_inputs), so a
    stored page is always one the document has.

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
        evidence_page_count: int | None = page_count
    else:
        document_block = await _build_document_block(path, mime_type)
        evidence_page_count = None

    client = _build_client(settings)
    page_result = await _extract_one_block(
        client, settings, document_block, schema, page_count=evidence_page_count
    )

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
        stop_reasons=[
            {"first": page_result.stop_reason, "repair": page_result.repair_stop_reason}
        ],
    )


def _stored_leaf(raw: Any, schema: ExtractionSchema) -> dict[str, Any]:
    """One scalar or cell leaf as stored: {value, confidence}, plus the
    evidence when the schema's leaves carry it. A malformed leaf degrades to
    nothing at all, evidence included (see _coerce_leaf)."""
    value, confidence = _coerce_leaf(raw)
    leaf: dict[str, Any] = {"value": value, "confidence": confidence}
    if schema.evidence:
        leaf["evidence"] = None if _leaf_is_malformed(raw) else _coerce_evidence(raw)
    return leaf


def _stored_array_rows(array_name: str, value: Any, schema: ExtractionSchema) -> list[dict[str, Any]]:
    """The rows of an array field as stored under schema v2: every row has
    every cell of ARRAY_CELLS, each a leaf of its own with its own evidence,
    and a row or cell that is not usable is a null leaf rather than missing."""
    rows = value if isinstance(value, list) else []
    return [
        {
            cell: _stored_leaf(row.get(cell) if isinstance(row, dict) else None, schema)
            for cell in ARRAY_CELLS[array_name]
        }
        for row in rows
    ]


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

    A schema without evidence (v1) stores the model's rows as they came; one
    with evidence (v2) stores {value, confidence, evidence} leaves, and an
    array's rows cell by cell (_stored_array_rows), the shape
    reconcile.ReconcileInput.from_fields reads. An array's own evidence is not
    kept: its cells carry the evidence that matters.

    The tool_input goes through the card-number rule first (mask_card_data), so
    every writer of fields (the worker, the demo seed, a replay) stores the same
    masked data whether or not it came through _extract_one_block.
    """
    tool_input = mask_card_data(tool_input)
    fields = []
    for field in schema.fields:
        value, confidence = _coerce_leaf(tool_input.get(field.name))
        if field.kind == "array":
            # An array field (line_items): store the array itself as
            # the ExtractedField value (its own confidence already
            # lives in the confidence column below). Every other
            # field is a scalar leaf, so the reconstructed
            # {"value", "confidence"} object is stored.
            if schema.evidence:
                field_value: Any = _stored_array_rows(field.name, value, schema)
            else:
                field_value = value if isinstance(value, list) else []
        elif schema.evidence:
            field_value = _stored_leaf(tool_input.get(field.name), schema)
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
