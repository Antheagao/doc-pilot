"""VLM-based structured extraction from uploaded documents.

Reads a document image/PDF, asks Claude to call the `record_extraction`
tool with structured field data, and persists the result as an Extraction
row plus one ExtractedField row per top-level field. This module is the
real `process_document_job` handler wired into app.worker (see
app/worker.py) -- it replaces the T5 placeholder that raised
NotImplementedError.

Prompt versioning: the extraction prompt lives in prompts/extract_v1.md as
a plain file, not inline in code, so future prompt iterations are versioned
by filename (extract_v2.md, ...) and eval results can be tied to a specific
prompt_version string. See PROMPT_PATH / PROMPT_VERSION below.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.models import Document, ExtractedField, Extraction, Job

# The extraction prompt is a versioned file; the version string is derived
# from its filename so a new prompt (extract_v2.md, ...) automatically
# gets a distinct prompt_version without a separate constant to keep in
# sync. Read once at import time (not per-call): it's a small, static
# file that never changes at runtime, so there's no reason to reread and
# no need for a lazy cache with its own invalidation questions.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_PATH = PROMPTS_DIR / "extract_v1.md"
PROMPT_VERSION = PROMPT_PATH.stem
PROMPT_TEXT = PROMPT_PATH.read_text(encoding="utf-8")

# The schema-repair reprompt (see extract_document / _schema_violations)
# is a separately versioned prompt for the same reason PROMPT_TEXT is:
# eval results and repair-effectiveness metrics should be tied to which
# repair wording was in effect. It contains a `{{violations}}`
# placeholder substituted via str.replace, not str.format -- the prompt
# text itself contains literal JSON braces (`{"value": ...}`) that
# str.format would try (and fail) to interpret as replacement fields.
REPAIR_PROMPT_PATH = PROMPTS_DIR / "repair_v1.md"
REPAIR_PROMPT_VERSION = REPAIR_PROMPT_PATH.stem
REPAIR_PROMPT_TEXT = REPAIR_PROMPT_PATH.read_text(encoding="utf-8")

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
}

# The tool's top-level fields, in the order ExtractedField rows are
# created. Kept as a module constant so process_document_job and tests
# agree on the field set without repeating the literal list.
TOP_LEVEL_FIELDS = (
    "vendor",
    "document_date",
    "line_items",
    "subtotal",
    "tax",
    "total",
    "currency",
)

DEFAULT_MAX_TOKENS = 4096


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
    """

    def __init__(
        self,
        message: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd


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


def _leaf_schema(value_schema: dict[str, Any]) -> dict[str, Any]:
    """A `{"value": <value_schema>, "confidence": number}` leaf, the shape
    every extracted field (and line-item sub-field) uses per the prompt.
    `minimum`/`maximum` on confidence are hints to the model, not an
    enforced guarantee -- tool_choice does not validate input against
    input_schema, so _coerce_leaf still has to clamp defensively.
    """
    return {
        "type": "object",
        "properties": {
            "value": value_schema,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["value", "confidence"],
    }


_STRING_OR_NULL = {"type": ["string", "null"]}
_NUMBER_OR_NULL = {"type": ["number", "null"]}

_LINE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "description": _leaf_schema(_STRING_OR_NULL),
        "quantity": _leaf_schema(_NUMBER_OR_NULL),
        "unit_price": _leaf_schema(_NUMBER_OR_NULL),
        "total": _leaf_schema(_NUMBER_OR_NULL),
    },
    "required": ["description", "quantity", "unit_price", "total"],
}

# The extraction tool. tool_choice forces the model to call this (see
# extract_document), so a tool_use block is guaranteed -- but its input
# is not guaranteed to match this schema (see module docstring), so
# every consumer of tool_input must go through _coerce_leaf.
RECORD_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "record_extraction",
    "description": (
        "Record the structured data extracted from the receipt/invoice "
        "document, with a per-field confidence score."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "vendor": _leaf_schema(_STRING_OR_NULL),
            "document_date": _leaf_schema(_STRING_OR_NULL),
            "line_items": _leaf_schema({"type": "array", "items": _LINE_ITEM_SCHEMA}),
            "subtotal": _leaf_schema(_NUMBER_OR_NULL),
            "tax": _leaf_schema(_NUMBER_OR_NULL),
            "total": _leaf_schema(_NUMBER_OR_NULL),
            "currency": _leaf_schema(_STRING_OR_NULL),
        },
        "required": list(TOP_LEVEL_FIELDS),
    },
}


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
    """
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


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
    except (TypeError, ValueError):
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


_LINE_ITEM_SUBFIELDS = ("description", "quantity", "unit_price", "total")

# _schema_violations truncates its result to this many dotted-path
# entries (plus one trailing summary entry) so a response that's
# malformed in nearly every leaf -- e.g. every line item missing every
# sub-leaf -- can't blow up the repair prompt built from the list.
MAX_SCHEMA_VIOLATIONS = 20

_MISSING = object()


def _schema_violations(tool_input: dict[str, Any]) -> list[str]:
    """Enumerate dotted-path violations of RECORD_EXTRACTION_TOOL's
    schema in `tool_input`, the model's untrusted tool_use.input.

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
    -- one violation per TOP_LEVEL_FIELDS entry -- so this is always safe
    to call on a raw tool_use.input without an isinstance guard at the
    call site.
    """
    if not isinstance(tool_input, dict):
        return list(TOP_LEVEL_FIELDS)

    violations: list[str] = []

    for field_name in TOP_LEVEL_FIELDS:
        raw = tool_input.get(field_name, _MISSING)
        if raw is _MISSING or _leaf_is_malformed(raw):
            violations.append(field_name)
            continue

        if field_name == "line_items":
            items = raw["value"]
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    item_dict = item if isinstance(item, dict) else {}
                    for sub_field in _LINE_ITEM_SUBFIELDS:
                        if _leaf_is_malformed(item_dict.get(sub_field)):
                            violations.append(f"line_items.value[{idx}].{sub_field}")

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


async def _build_document_block(path: Path, mime_type: str) -> dict[str, Any]:
    """Build the image/document content block for the user message.

    Mime-type validation happens before the (threaded) file read so an
    unsupported type fails fast without touching disk. The read itself
    is offloaded via run_in_threadpool so a large file doesn't block the
    event loop the worker shares with everything else in the process.
    OSError from the read (missing file, permission error, ...) is
    wrapped as ExtractionError -- retryable, since the underlying cause
    (e.g. a networked/mounted uploads volume being briefly unavailable)
    is plausibly transient, unlike the deterministic failures that use
    NonRetryableExtractionError.
    """
    if mime_type == PDF_MIME_TYPE:
        content_type = "document"
    elif mime_type in IMAGE_MIME_TYPES:
        content_type = "image"
    else:
        raise NonRetryableExtractionError(f"unsupported mime type for extraction: {mime_type}")

    try:
        raw_bytes = await run_in_threadpool(path.read_bytes)
    except OSError as exc:
        raise ExtractionError(f"failed to read document file {path}: {exc}") from exc

    data = base64.standard_b64encode(raw_bytes).decode("utf-8")
    return {
        "type": content_type,
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    }


async def extract_document(path: str | Path, mime_type: str) -> ExtractionResult:
    """Call Claude with the extraction tool forced, and return the parsed
    result.

    Failure classification: unsupported mime type and an unpriced model
    are validated up front, before any API call, and raise
    NonRetryableExtractionError for free. A refusal or any other
    non-tool_use stop reason (including "max_tokens" truncation) is
    deterministic given the same document/prompt/model, so it also
    raises NonRetryableExtractionError -- but only after the call, since
    it's the API's own response that tells us; those messages include
    the token counts from that (already billed) call so the spend isn't
    invisible in a requeued job's last_error. A missing tool_use block
    despite stop_reason == "tool_use" would mean the forced-tool-choice
    guarantee itself didn't hold -- that's treated as a transient
    anomaly (plain ExtractionError, retryable) rather than classified
    alongside the deterministic cases. A file that can't be read raises
    ExtractionError (see _build_document_block); network/5xx/429 raise
    ExtractionError via the anthropic.APIError branch below.

    Schema-repair reprompt: after a successful tool_use response, if its
    tool_input violates RECORD_EXTRACTION_TOOL's schema (see
    _schema_violations) and settings.schema_repair is enabled, exactly
    one additional call is made asking the model to fix only the
    structure (see REPAIR_PROMPT_TEXT). The repaired tool_input is used
    only if it has strictly fewer violations than the first response's;
    either way both calls' tokens/cost are summed into the returned
    ExtractionResult, and the result's `repaired`/`schema_violations`
    fields report what happened. A failed, refused, or non-tool_use
    repair call never fails the extraction -- it just falls back to the
    first response.
    """
    settings = get_settings()
    _ensure_model_priced(settings.extraction_model)

    document_block = await _build_document_block(Path(path), mime_type)
    client = _build_client(settings)

    user_message = {
        "role": "user",
        "content": [
            document_block,
            {"type": "text", "text": "Extract the document now."},
        ],
    }

    start = time.perf_counter()
    try:
        response = await client.messages.create(
            model=settings.extraction_model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=PROMPT_TEXT,
            tools=[RECORD_EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_extraction"},
            messages=[user_message],
        )
    except anthropic.APIError as exc:
        # Covers rate limits, 5xx, and other transient/network-shaped
        # failures -- the API's own message is preserved so fail_job's
        # stored last_error is actionable. Retryable.
        raise ExtractionError(f"Anthropic API error: {exc}") from exc

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
    violations = _schema_violations(tool_input)
    repaired = False
    total_input_tokens = response.usage.input_tokens
    total_output_tokens = response.usage.output_tokens

    # Schema-repair reprompt: tool_choice guarantees *a* tool_use block,
    # not one that matches RECORD_EXTRACTION_TOOL's schema (see module
    # docstring). If the first response violated it, make exactly one
    # additional call asking the model to fix only the structure. The
    # repair call's cost is billed regardless of whether it's accepted
    # (see docstring below) -- money was spent either way -- but a
    # failed/unusable repair call must never turn an otherwise-usable
    # extraction into a failed job, so any exception, refusal, or missing
    # tool_use block from the repair call just falls back to the first
    # response.
    if violations and settings.schema_repair:
        repair_text = REPAIR_PROMPT_TEXT.replace(
            "{{violations}}", _format_violations_for_prompt(violations)
        )
        try:
            repair_response = await client.messages.create(
                model=settings.extraction_model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=PROMPT_TEXT,
                tools=[RECORD_EXTRACTION_TOOL],
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
                repair_violations = _schema_violations(repair_tool_use_block.input)
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
    cost_usd = _compute_cost_usd(
        settings.extraction_model,
        total_input_tokens,
        total_output_tokens,
    )

    return ExtractionResult(
        tool_input=tool_input,
        model=settings.extraction_model,
        prompt_version=PROMPT_VERSION,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        repaired=repaired,
        schema_violations=len(violations),
    )


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
    document = await session.get(Document, job.document_id)
    if document is None:
        raise ExtractionError(f"document {job.document_id} not found")

    document_id = document.id
    storage_path = document.storage_path
    mime_type = document.mime_type

    await session.rollback()

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

        settings = get_settings()
        for field_name in TOP_LEVEL_FIELDS:
            value, confidence = _coerce_leaf(result.tool_input.get(field_name))
            if field_name == "line_items":
                # line_items is an array field: store the array itself as
                # the ExtractedField value (its own confidence already
                # lives in the confidence column below). Every other
                # field is a scalar leaf, so the reconstructed
                # {"value", "confidence"} object is stored.
                field_value: Any = value if isinstance(value, list) else []
            else:
                field_value = {"value": value, "confidence": confidence}
            session.add(
                ExtractedField(
                    extraction_id=extraction.id,
                    field_name=field_name,
                    value=field_value,
                    confidence=confidence,
                    needs_review=confidence < settings.review_threshold,
                )
            )

        document.status = "extracted"
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
