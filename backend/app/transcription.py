"""Per-page VLM transcription: the text source for the retrieval index.

Extraction (app/extraction.py) answers "what are this document's fields";
retrieval needs "what does this document say", page by page, so a search
hit can cite the page it came from. This module sends each page to the
model with a plain transcription prompt and returns one PageText per page,
each carrying its own tokens/cost/latency so indexing spend is tracked
exactly like extraction spend.

Everything that isn't specific to transcription is shared with extraction
rather than re-implemented: document block encoding, PDF page counting and
splitting, the pricing table, and -- most importantly -- the failure
classification (_classify_api_error), so a 429 during transcription is
requeued with the same retry-after-aware backoff as one during extraction,
and a deterministic 400 fails permanently the same way.

PDFs are always transcribed one page per call (unlike extraction, which
sends up to pdf_max_pages_per_call pages at once): the whole point of this
text is page-accurate citations, and a multi-page response would have to
be split back into pages by trusting the model to mark page boundaries.
"""

import time
from pathlib import Path
from typing import Any

import anthropic

from app.config import get_settings
from app.extraction import (
    PDF_MIME_TYPE,
    PROMPTS_DIR,
    ExtractionError,
    ModelRefusalError,
    NonRetryableExtractionError,
    _build_client,
    _classify_api_error,
    _compute_cost_usd,
    _content_type_for_mime,
    _encode_document_block,
    _ensure_model_priced,
    _read_document_bytes,
)
from app.pdf import count_pdf_pages, split_pdf_pages
from app.retrieval.indexing import PageText

# Versioned the same way the extraction prompt is (see app/extraction.py):
# the filename stem is the prompt_version recorded on every DocumentPage.
TRANSCRIBE_PROMPT_PATH = PROMPTS_DIR / "transcribe_v1.md"
TRANSCRIBE_PROMPT_VERSION = TRANSCRIBE_PROMPT_PATH.stem
TRANSCRIBE_PROMPT_TEXT = TRANSCRIBE_PROMPT_PATH.read_text(encoding="utf-8")

# A dense receipt or invoice page is well under 1,500 output tokens; this
# leaves headroom for a text-heavy page. Hitting it is a non-retryable
# failure, not a silent truncation -- an index built from half a page would
# quietly miss everything below the cut.
TRANSCRIBE_MAX_TOKENS = 4096


async def _transcribe_block(
    client: anthropic.AsyncAnthropic,
    model: str,
    document_block: dict[str, Any],
    page_number: int,
) -> PageText:
    start = time.perf_counter()
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=TRANSCRIBE_MAX_TOKENS,
            system=TRANSCRIBE_PROMPT_TEXT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        document_block,
                        {"type": "text", "text": "Transcribe this page now."},
                    ],
                }
            ],
        )
    except anthropic.AnthropicError as exc:
        raise _classify_api_error(exc) from exc
    latency_ms = int((time.perf_counter() - start) * 1000)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = _compute_cost_usd(model, input_tokens, output_tokens)

    if response.stop_reason != "end_turn":
        error_cls = (
            ModelRefusalError if response.stop_reason == "refusal" else NonRetryableExtractionError
        )
        raise error_cls(
            f"transcription of page {page_number} did not complete "
            f"(stop_reason={response.stop_reason!r}); billed usage: "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    text = "\n".join(block.text for block in response.content if block.type == "text")
    return PageText(
        page_number=page_number,
        text=text.strip(),
        source="transcription",
        model=model,
        prompt_version=TRANSCRIBE_PROMPT_VERSION,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


async def transcribe_document(path: str | Path, mime_type: str) -> list[PageText]:
    """Transcribe every page of a document, in order. Images are one page.

    Pre-flight checks mirror extract_document's and run before any API
    call: an unpriced model, an unsupported mime type, an unreadable or
    empty PDF, or one over pdf_max_pages all fail non-retryably for free.

    Pages are transcribed sequentially (not gathered), for the same
    reason the chunked extraction path is: a 20-page PDF shouldn't burst
    20 requests at the rate limiter. If a page fails, the spend on the
    pages before it is added onto the raised exception, the same
    billed-but-errored convention ExtractionError documents.
    """
    settings = get_settings()
    model = settings.transcription_model
    _ensure_model_priced(model)

    content_type = _content_type_for_mime(mime_type)
    raw_bytes = await _read_document_bytes(Path(path))

    if mime_type == PDF_MIME_TYPE:
        try:
            page_count = count_pdf_pages(raw_bytes)
            if page_count == 0:
                raise NonRetryableExtractionError("PDF has 0 pages; nothing to transcribe")
            if page_count > settings.pdf_max_pages:
                raise NonRetryableExtractionError(
                    f"PDF has {page_count} pages, exceeding pdf_max_pages="
                    f"{settings.pdf_max_pages}; refusing to transcribe before spending "
                    "any API budget"
                )
            page_blobs = split_pdf_pages(raw_bytes) if page_count > 1 else [raw_bytes]
        except ValueError as exc:
            raise NonRetryableExtractionError(
                f"unreadable or encrypted PDF, cannot transcribe: {exc}"
            ) from exc
        blocks = [_encode_document_block(blob, PDF_MIME_TYPE, "document") for blob in page_blobs]
    else:
        blocks = [_encode_document_block(raw_bytes, mime_type, content_type)]

    client = _build_client(settings)
    pages: list[PageText] = []
    for page_number, block in enumerate(blocks, start=1):
        try:
            pages.append(await _transcribe_block(client, model, block, page_number))
        except ExtractionError as exc:
            exc.input_tokens = sum(p.input_tokens for p in pages) + (exc.input_tokens or 0)
            exc.output_tokens = sum(p.output_tokens for p in pages) + (exc.output_tokens or 0)
            exc.cost_usd = _compute_cost_usd(model, exc.input_tokens, exc.output_tokens)
            raise
    return pages
