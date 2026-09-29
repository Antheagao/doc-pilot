"""The /ask agent's tools, and the citation plumbing behind them.

Every tool returns content blocks for a tool_result. Anything the model
might quote comes back as a `search_result` block with citations enabled,
so the API returns verifiable citations -- `cited_text` is copied from
our blocks, never paraphrased by the model -- instead of us asking the
model to write citation markers and hoping they're right.

Each search_result's `source` is a stable doc-pilot URI naming what it came
from (a chunk of a page, a whole page, or a document's extracted record),
and every content block inside it is one line of the page (or one field of
the record). The registry (ToolContext.sources) keeps, per source, which
char span or field each block is, so a citation's block range resolves
back to an exact span of the stored page text -- the same span /search
cites.
"""

import uuid
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentPage
from app.records import DocumentRecord, load_records
from app.retrieval.embeddings import Embedder
from app.retrieval.search import search

SOURCE_SCHEME = "doc-pilot://documents"
SEARCH_K_DEFAULT = 5
SEARCH_K_MAX = 10
# Records listed in full per query_extractions call; the summary line's
# counts and sums always cover every match.
RECORDS_MAX = 20


class ToolError(Exception):
    """A problem the model can fix by calling the tool differently (a bad
    id, a malformed date). Returned to it as an is_error tool_result
    rather than ending the run."""


@dataclass
class Source:
    """What one search_result block stands for."""

    source: str
    kind: Literal["chunk", "page", "record"]
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    # Per content block, in order: the (char_start, char_end) of that line
    # in the page's stored text, or the record field name it shows.
    blocks: list[tuple[int, int] | str]


@dataclass
class ToolContext:
    session: AsyncSession
    embedder: Embedder
    # None searches everything; the agent eval scopes to its own corpus.
    document_ids: list[uuid.UUID] | None = None
    sources: dict[str, Source] = field(default_factory=dict)


# --- definitions ------------------------------------------------------------
#
# strict: schema-valid arguments guaranteed. Strict schemas can't express
# numeric bounds, so k is clamped in code.

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "query_extractions",
        "description": (
            "Query the structured fields extracted from the user's documents (vendor, "
            "document_date, currency, subtotal, tax, total, line items), with human review "
            "corrections applied. All filters are optional and combine with AND; vendor and "
            "item match case-insensitive substrings. Returns a summary line with the number "
            "of matches and their totals summed per currency, then each matching document's "
            "record (at most 20 listed). Use this for amounts, dates, vendors, counts and sums."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor": {"type": "string", "description": "Vendor name or part of it."},
                "item": {"type": "string", "description": "Line item description or part of it."},
                "date_from": {"type": "string", "format": "date", "description": "Earliest document date, inclusive (YYYY-MM-DD)."},
                "date_to": {"type": "string", "format": "date", "description": "Latest document date, inclusive (YYYY-MM-DD)."},
                "currency": {"type": "string", "description": "ISO 4217 code, e.g. USD or EUR."},
                "min_total": {"type": "number", "description": "Minimum document total, inclusive."},
                "max_total": {"type": "number", "description": "Maximum document total, inclusive."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Hybrid semantic + keyword search over the text of the user's documents. Returns "
            "the best-matching passages with their document and page. Use this when the "
            "question describes something in words rather than naming an exact field value."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for, in natural language."},
                "k": {"type": "integer", "description": "Number of passages to return, 1-10 (default 5)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_page",
        "description": (
            "The full text of one page of a document, to check a detail in context. "
            "document_id must come from an earlier search_documents or query_extractions result."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "format": "uuid"},
                "page_number": {"type": "integer", "description": "1-based page number."},
            },
            "required": ["document_id", "page_number"],
            "additionalProperties": False,
        },
    },
]


# --- helpers ----------------------------------------------------------------


def _line_blocks(text: str, offset: int) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """One text block per non-blank line of `text`, plus each line's span
    in the page (offset = where `text` starts within the page)."""
    blocks, spans = [], []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            start = offset + pos + (len(line) - len(line.lstrip()))
            blocks.append({"type": "text", "text": stripped})
            spans.append((start, start + len(stripped)))
        pos += len(line)
    return blocks, spans


def _search_result(source: Source, title: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "search_result",
        "source": source.source,
        "title": title,
        "content": content,
        "citations": {"enabled": True},
    }


def _money(value: Any) -> str:
    return f"{Decimal(str(value)):.2f}" if isinstance(value, int | float) else str(value)


def _record_lines(record: DocumentRecord) -> list[tuple[str, str]]:
    """(field, text) for each line of a record's search_result."""
    fields = record.fields
    lines = [
        ("vendor", f"vendor: {fields.get('vendor') or 'unknown'}"),
        ("document_date", f"date: {fields.get('document_date') or 'unknown'}"),
        ("currency", f"currency: {fields.get('currency') or 'unknown'}"),
    ]
    for name in ("subtotal", "tax", "total"):
        value = fields.get(name)
        lines.append((name, f"{name}: {_money(value) if value is not None else 'not shown'}"))
    for index, item in enumerate(fields.get("line_items") or []):
        parts = [str(item.get("description") or "item")]
        if item.get("quantity") is not None:
            parts.append(f"qty {item['quantity']}")
        if item.get("unit_price") is not None:
            parts.append(f"unit price {_money(item['unit_price'])}")
        if item.get("total") is not None:
            parts.append(f"line total {_money(item['total'])}")
        lines.append((f"line_items[{index}]", "line item: " + ", ".join(parts)))
    if record.unreviewed_low_confidence:
        lines.append(
            (
                "review_status",
                "note: not yet human-reviewed and low-confidence: "
                + ", ".join(record.unreviewed_low_confidence),
            )
        )
    return lines


def _parse_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"{name} must be a date in YYYY-MM-DD form, got {value!r}") from None


def _parse_document_id(value: str, ctx: ToolContext) -> uuid.UUID:
    try:
        document_id = uuid.UUID(value)
    except ValueError:
        raise ToolError(
            "document_id must be a document id from an earlier search_documents or "
            "query_extractions result"
        ) from None
    if ctx.document_ids is not None and document_id not in ctx.document_ids:
        raise ToolError("no document with that id")
    return document_id


# --- tools ------------------------------------------------------------------


async def search_documents(ctx: ToolContext, tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    query = str(tool_input.get("query") or "").strip()
    if not query:
        raise ToolError("query must not be empty")
    k = max(1, min(int(tool_input.get("k") or SEARCH_K_DEFAULT), SEARCH_K_MAX))

    hits = await search(
        ctx.session, query, embedder=ctx.embedder, k=k, mode="hybrid", document_ids=ctx.document_ids
    )
    if not hits:
        return [{"type": "text", "text": "No matching passages found."}]

    results = []
    for hit in hits:
        content, spans = _line_blocks(hit.text, hit.char_start)
        if not content:
            continue
        source = Source(
            source=f"{SOURCE_SCHEME}/{hit.document_id}/pages/{hit.page_number}#chunk={hit.chunk_id}",
            kind="chunk",
            document_id=hit.document_id,
            filename=hit.filename,
            page_number=hit.page_number,
            blocks=list(spans),
        )
        ctx.sources[source.source] = source
        results.append(
            _search_result(
                source, f"{hit.filename}, page {hit.page_number} (document_id {hit.document_id})", content
            )
        )
    return results


async def get_page(ctx: ToolContext, tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    document_id = _parse_document_id(str(tool_input.get("document_id")), ctx)
    page_number = int(tool_input.get("page_number") or 0)
    pages = (
        (
            await ctx.session.execute(
                select(DocumentPage)
                .where(DocumentPage.document_id == document_id)
                .order_by(DocumentPage.page_number)
            )
        )
        .scalars()
        .all()
    )
    if not pages:
        raise ToolError("that document has no indexed pages")
    page = next((p for p in pages if p.page_number == page_number), None)
    if page is None:
        raise ToolError(f"no page {page_number}; that document has pages 1-{len(pages)}")

    content, spans = _line_blocks(page.text, 0)
    if not content:
        return [{"type": "text", "text": f"Page {page_number} has no text."}]

    filename = (await ctx.session.get(Document, document_id)).filename
    source = Source(
        source=f"{SOURCE_SCHEME}/{document_id}/pages/{page_number}",
        kind="page",
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        blocks=list(spans),
    )
    ctx.sources[source.source] = source
    return [_search_result(source, f"{filename}, page {page_number} (full page)", content)]


def _matches(record: DocumentRecord, tool_input: dict[str, Any], date_from, date_to) -> bool:
    fields = record.fields
    vendor = tool_input.get("vendor")
    if vendor and vendor.casefold() not in str(fields.get("vendor") or "").casefold():
        return False
    item = tool_input.get("item")
    if item and not any(
        item.casefold() in str(row.get("description") or "").casefold()
        for row in fields.get("line_items") or []
    ):
        return False
    currency = tool_input.get("currency")
    if currency and str(fields.get("currency") or "").upper() != currency.upper():
        return False
    if date_from or date_to:
        try:
            document_date = date.fromisoformat(str(fields.get("document_date")))
        except ValueError:
            return False
        if (date_from and document_date < date_from) or (date_to and document_date > date_to):
            return False
    total = fields.get("total")
    if tool_input.get("min_total") is not None and (
        not isinstance(total, int | float) or total < tool_input["min_total"]
    ):
        return False
    return tool_input.get("max_total") is None or (
        isinstance(total, int | float) and total <= tool_input["max_total"]
    )


def _summary(matches: list[DocumentRecord]) -> str:
    """Counts and per-currency sums, computed here in Decimal -- the model
    is told never to add amounts itself."""
    sums: dict[str, Decimal] = {}
    missing_total = 0
    for record in matches:
        total = record.fields.get("total")
        if not isinstance(total, int | float):
            missing_total += 1
            continue
        currency = record.fields.get("currency") or "unknown currency"
        sums[currency] = sums.get(currency, Decimal(0)) + Decimal(str(total))
    parts = [f"{len(matches)} matching document(s)."]
    if sums:
        parts.append(
            "Sum of totals: "
            + "; ".join(f"{amount:.2f} {currency}" for currency, amount in sorted(sums.items()))
            + "."
        )
    if missing_total:
        parts.append(f"{missing_total} matching document(s) have no total.")
    if len(matches) > RECORDS_MAX:
        parts.append(f"Listing the first {RECORDS_MAX}; narrow the filters to see the rest.")
    return " ".join(parts)


async def query_extractions(ctx: ToolContext, tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    date_from = _parse_date(tool_input.get("date_from"), "date_from")
    date_to = _parse_date(tool_input.get("date_to"), "date_to")
    records = await load_records(ctx.session, ctx.document_ids)
    matches = [r for r in records if _matches(r, tool_input, date_from, date_to)]

    content: list[dict[str, Any]] = [{"type": "text", "text": _summary(matches)}]
    for record in matches[:RECORDS_MAX]:
        lines = _record_lines(record)
        source = Source(
            source=f"{SOURCE_SCHEME}/{record.document_id}/extraction",
            kind="record",
            document_id=record.document_id,
            filename=record.filename,
            page_number=None,
            blocks=[name for name, _ in lines],
        )
        ctx.sources[source.source] = source
        content.append(
            _search_result(
                source,
                f"{record.filename} (extracted fields, document_id {record.document_id})",
                [{"type": "text", "text": text} for _, text in lines],
            )
        )
    return content


TOOL_HANDLERS = {
    "query_extractions": query_extractions,
    "search_documents": search_documents,
    "get_page": get_page,
}
