"""Tests for H6 (oversized-document chunking): the pure PDF utilities in
app/pdf.py, the pure merge rules in app.extraction.merge_page_tool_inputs,
and the end-to-end process_document_job chunked path.

Kept in its own file (rather than folded into test_extraction.py /
test_pdf.py) so the H6-specific fixtures (a real multi-page PDF built
with Pillow, the chunking-pinned Settings) don't clutter the existing
single-call suite -- and so that suite's "must pass UNMODIFIED" claim
in the task notes is easy to verify by diffing.
"""

from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import extraction as extraction_module
from app.config import Settings
from app.extraction import (
    MERGE_CONFLICT_CONFIDENCE,
    TOP_LEVEL_FIELDS,
    ModelRefusalError,
    NonRetryableExtractionError,
    extract_document,
    merge_page_tool_inputs,
    process_document_job,
)
from app.models import Document, ExtractedField, Extraction, Job
from app.pdf import count_pdf_pages, split_pdf_pages

# --- shared fixtures / fakes -------------------------------------------


def _build_multipage_pdf(n_pages: int) -> bytes:
    """A real, valid multi-page PDF built with Pillow --
    img.save(path, save_all=True, append_images=[...]) is a documented
    Pillow capability, not a workaround. Image content is irrelevant;
    only the page count matters for these tests.
    """
    images = [Image.new("RGB", (10, 10), color=(i * 20, 0, 0)) for i in range(n_pages)]
    buf = BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    return buf.getvalue()


def _build_encrypted_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt(user_password="secret")
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _build_zero_page_pdf() -> bytes:
    """A valid PDF with no pages -- pypdf will write and re-parse this
    without error; the page count just comes out 0.
    """
    writer = PdfWriter()
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeToolUseBlock:
    type = "tool_use"
    name = "record_extraction"

    def __init__(self, tool_input: dict, tool_use_id: str = "toolu_test") -> None:
        self.input = tool_input
        self.id = tool_use_id


class _FakeMessage:
    def __init__(self, stop_reason: str, content: list, usage: _FakeUsage) -> None:
        self.stop_reason = stop_reason
        self.content = content
        self.usage = usage


class _FakeMessagesResource:
    def __init__(self) -> None:
        self.create = AsyncMock()


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessagesResource()


def _patch_client(monkeypatch: pytest.MonkeyPatch, response: _FakeMessage) -> _FakeClient:
    fake_client = _FakeClient()
    fake_client.messages.create.return_value = response
    monkeypatch.setattr(extraction_module, "_build_client", lambda settings: fake_client)
    return fake_client


def _patch_client_side_effect(monkeypatch: pytest.MonkeyPatch, side_effect: list) -> _FakeClient:
    fake_client = _FakeClient()
    fake_client.messages.create.side_effect = side_effect
    monkeypatch.setattr(extraction_module, "_build_client", lambda settings: fake_client)
    return fake_client


async def _make_document(
    db_session: AsyncSession,
    tmp_path,
    *,
    filename: str,
    mime_type: str,
    content: bytes,
) -> Document:
    path = tmp_path / filename
    path.write_bytes(content)
    document = Document(
        filename=filename,
        mime_type=mime_type,
        storage_path=str(path),
        status="uploaded",
    )
    db_session.add(document)
    await db_session.commit()
    return document


async def _make_job(db_session: AsyncSession, document_id) -> Job:
    job = Job(document_id=document_id, state="processing")
    db_session.add(job)
    await db_session.commit()
    return job


async def _extraction_count(db_session: AsyncSession, document_id) -> int:
    return (
        await db_session.execute(
            select(func.count()).select_from(Extraction).where(Extraction.document_id == document_id)
        )
    ).scalar_one()


def _leaf(value, confidence):
    return {"value": value, "confidence": confidence}


def _minimal_page(**overrides) -> dict:
    """A page tool_input with every TOP_LEVEL_FIELDS leaf defaulted to
    {"value": None, "confidence": 0.0} (a well-formed, all-absent leaf
    -- not a malformed one), overridden per-field by kwargs.
    """
    page = {name: _leaf(None, 0.0) for name in TOP_LEVEL_FIELDS}
    page.update(overrides)
    return page


# --- app.pdf round-trip ---------------------------------------------------


def test_count_pdf_pages_matches_actual_page_count() -> None:
    assert count_pdf_pages(_build_multipage_pdf(3)) == 3


def test_split_pdf_pages_returns_one_valid_single_page_pdf_per_page() -> None:
    pages = split_pdf_pages(_build_multipage_pdf(3))

    assert len(pages) == 3
    for page_bytes in pages:
        reader = PdfReader(BytesIO(page_bytes))
        assert len(reader.pages) == 1


def test_count_pdf_pages_raises_valueerror_on_corrupt_bytes() -> None:
    with pytest.raises(ValueError):
        count_pdf_pages(b"not a pdf at all, just garbage bytes")


def test_split_pdf_pages_raises_valueerror_on_corrupt_bytes() -> None:
    with pytest.raises(ValueError):
        split_pdf_pages(b"not a pdf at all, just garbage bytes")


def test_count_pdf_pages_raises_valueerror_on_encrypted_pdf() -> None:
    with pytest.raises(ValueError):
        count_pdf_pages(_build_encrypted_pdf())


def test_count_pdf_pages_is_zero_for_an_empty_pdf() -> None:
    assert count_pdf_pages(_build_zero_page_pdf()) == 0


def test_split_pdf_pages_raises_valueerror_on_encrypted_pdf() -> None:
    with pytest.raises(ValueError):
        split_pdf_pages(_build_encrypted_pdf())


# --- merge_page_tool_inputs: pure merge rules ------------------------------


def test_line_items_concatenated_in_page_order_with_min_confidence() -> None:
    page0 = _minimal_page(
        line_items=_leaf([{"description": _leaf("Widget", 0.9)}], 0.9)
    )
    page1 = _minimal_page(
        line_items=_leaf([{"description": _leaf("Gadget", 0.8)}], 0.6)
    )

    merged = merge_page_tool_inputs([page0, page1])

    assert merged["line_items"]["value"] == [
        {"description": _leaf("Widget", 0.9)},
        {"description": _leaf("Gadget", 0.8)},
    ]
    assert merged["line_items"]["confidence"] == pytest.approx(0.6)


def test_line_items_all_null_gives_empty_list_and_zero_confidence() -> None:
    page0 = _minimal_page()
    page1 = _minimal_page()

    merged = merge_page_tool_inputs([page0, page1])

    assert merged["line_items"] == {"value": [], "confidence": 0.0}


def test_header_field_tie_breaks_to_earliest_page() -> None:
    """vendor is a header field: equal-confidence, disagreeing
    candidates must pick the EARLIEST page's value (and, since they
    disagree, cap the merged confidence at MERGE_CONFLICT_CONFIDENCE).
    """
    page0 = _minimal_page(vendor=_leaf("Acme Corp", 0.9))
    page1 = _minimal_page(vendor=_leaf("Other Co", 0.9))

    merged = merge_page_tool_inputs([page0, page1])

    assert merged["vendor"]["value"] == "Acme Corp"
    assert merged["vendor"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_money_field_tie_breaks_to_latest_page() -> None:
    """total is a money field: equal-confidence, disagreeing candidates
    must pick the LATEST page's value -- the opposite tie-break from
    header fields, since totals conventionally print at the end.
    """
    page0 = _minimal_page(total=_leaf(100.0, 0.9))
    page1 = _minimal_page(total=_leaf(200.0, 0.9))

    merged = merge_page_tool_inputs([page0, page1])

    assert merged["total"]["value"] == 200.0
    assert merged["total"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_all_null_scalar_field_uses_min_confidence_across_pages() -> None:
    page0 = _minimal_page(currency=_leaf(None, 0.4))
    page1 = _minimal_page(currency=_leaf(None, 0.1))
    page2 = _minimal_page(currency=_leaf(None, 0.7))

    merged = merge_page_tool_inputs([page0, page1, page2])

    assert merged["currency"] == {"value": None, "confidence": pytest.approx(0.1)}


def test_conflicting_values_cap_confidence_even_when_winner_was_very_confident() -> None:
    page0 = _minimal_page(vendor=_leaf("Acme Corp", 0.95))
    page1 = _minimal_page(vendor=_leaf("Totally Different Vendor", 0.5))

    merged = merge_page_tool_inputs([page0, page1])

    # Highest-confidence non-null candidate wins the VALUE...
    assert merged["vendor"]["value"] == "Acme Corp"
    # ...but a real disagreement still caps the merged confidence.
    assert merged["vendor"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_agreeing_values_do_not_trigger_the_conflict_cap() -> None:
    """Same value (modulo whitespace/case), different confidences --
    no disagreement, so the winning (higher) confidence is kept as-is.
    """
    page0 = _minimal_page(vendor=_leaf("  Acme Corp  ", 0.6))
    page1 = _minimal_page(vendor=_leaf("acme corp", 0.9))

    merged = merge_page_tool_inputs([page0, page1])

    assert merged["vendor"]["value"] == "acme corp"
    assert merged["vendor"]["confidence"] == pytest.approx(0.9)


def test_agreeing_numeric_values_tolerate_float_noise() -> None:
    page0 = _minimal_page(total=_leaf(10.004, 0.6))
    page1 = _minimal_page(total=_leaf(10.001, 0.9))

    merged = merge_page_tool_inputs([page0, page1])

    # round(10.004, 2) == round(10.001, 2) == 10.0 -- treated as agreement.
    assert merged["total"]["confidence"] == pytest.approx(0.9)


def test_malformed_page_leaf_degrades_instead_of_crashing_the_merge() -> None:
    """A page tool_input that is itself malformed (a bare scalar in
    place of the whole document, missing every field) must not raise --
    _coerce_leaf degrades it to (None, 0.0) for every field on that
    page, same as a single-call malformed response would.
    """
    malformed_page = "not a dict at all"
    well_formed_page = _minimal_page(vendor=_leaf("Acme Corp", 0.9))

    merged = merge_page_tool_inputs([malformed_page, well_formed_page])

    assert merged["vendor"]["value"] == "Acme Corp"
    assert set(merged) == set(TOP_LEVEL_FIELDS)


def test_merge_handles_oversized_int_literal_without_crashing() -> None:
    """FIX 2 regression: a page reporting an oversized int literal
    (valid JSON, but too large for float() to represent -- e.g. from an
    adversarial or corrupted response) must not raise OverflowError out
    of the merge. _values_disagree treats it as a disagreement (the
    safe direction), capping the merged confidence rather than crashing
    or silently trusting an unrepresentable number.
    """
    huge = 10**400
    page0 = _minimal_page(total=_leaf(huge, 0.9))
    page1 = _minimal_page(total=_leaf(100.0, 0.9))

    merged = merge_page_tool_inputs([page0, page1])  # must not raise

    assert merged["total"]["value"] == 100.0  # tie -> latest page wins
    assert merged["total"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_malformed_leaf_within_an_otherwise_valid_page_degrades() -> None:
    page0 = {**_minimal_page(), "subtotal": "bare scalar, not a leaf dict"}
    page1 = _minimal_page(subtotal=_leaf(42.0, 0.9))

    merged = merge_page_tool_inputs([page0, page1])

    assert merged["subtotal"]["value"] == 42.0
    assert merged["subtotal"]["confidence"] == pytest.approx(0.9)


# --- end-to-end process_document_job: chunked PDF path ---------------------


def _page_tool_input(
    *,
    vendor=None,
    document_date=None,
    line_items=None,
    subtotal=None,
    tax=None,
    total=None,
    currency=None,
) -> dict:
    page = _minimal_page()
    overrides = {
        "vendor": vendor,
        "document_date": document_date,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": currency,
    }
    for name, leaf in overrides.items():
        if leaf is not None:
            page[name] = leaf
    return page


async def test_chunked_pdf_extraction_persists_merged_result(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(
            extraction_model="claude-sonnet-5",
            review_threshold=0.8,
            pdf_max_pages_per_call=2,
        ),
    )
    pdf_bytes = _build_multipage_pdf(3)
    document = await _make_document(
        db_session, tmp_path, filename="invoice.pdf", mime_type="application/pdf", content=pdf_bytes
    )
    job = await _make_job(db_session, document.id)

    page_inputs = [
        _page_tool_input(
            vendor=_leaf("Acme Corp", 0.9),
            document_date=_leaf("2026-01-01", 0.9),
            line_items=_leaf(
                [
                    {
                        "description": _leaf("Widget", 0.9),
                        "quantity": _leaf(1, 0.9),
                        "unit_price": _leaf(5.0, 0.9),
                        "total": _leaf(5.0, 0.9),
                    }
                ],
                0.9,
            ),
            currency=_leaf("USD", 0.9),
        ),
        _page_tool_input(
            line_items=_leaf(
                [
                    {
                        "description": _leaf("Gadget", 0.85),
                        "quantity": _leaf(2, 0.85),
                        "unit_price": _leaf(3.0, 0.85),
                        "total": _leaf(6.0, 0.85),
                    }
                ],
                0.85,
            ),
        ),
        _page_tool_input(
            line_items=_leaf([], 0.95),
            subtotal=_leaf(11.0, 0.9),
            tax=_leaf(1.0, 0.9),
            total=_leaf(12.0, 0.9),
        ),
    ]
    responses = [
        _FakeMessage(
            stop_reason="tool_use",
            content=[_FakeToolUseBlock(page_input, tool_use_id=f"toolu_{i}")],
            usage=_FakeUsage(100 + i, 50 + i),
        )
        for i, page_input in enumerate(page_inputs)
    ]
    fake_client = _patch_client_side_effect(monkeypatch, responses)

    await process_document_job(db_session, job)
    await db_session.flush()

    assert fake_client.messages.create.await_count == 3

    extraction = (
        (await db_session.execute(select(Extraction).where(Extraction.document_id == document.id)))
        .scalars()
        .one()
    )

    raw = extraction.raw_response
    assert len(raw["_chunks"]) == 3
    assert [c["page"] for c in raw["_chunks"]] == [1, 2, 3]

    # FIX 4c: each _chunks entry has exactly the documented 7-key shape.
    chunk0 = raw["_chunks"][0]
    assert set(chunk0) == {
        "page",
        "tool_input",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
        "repaired",
    }
    assert chunk0["page"] == 1
    assert chunk0["tool_input"] == page_inputs[0]
    assert chunk0["input_tokens"] == 100
    assert chunk0["output_tokens"] == 50
    assert chunk0["cost_usd"] == pytest.approx((100 * 2.00 + 50 * 10.00) / 1_000_000)
    assert chunk0["latency_ms"] >= 0
    assert chunk0["repaired"] is False

    # Header fields: only page 0 reported a value -- it wins outright.
    assert raw["vendor"]["value"] == "Acme Corp"
    assert raw["document_date"]["value"] == "2026-01-01"
    assert raw["currency"]["value"] == "USD"

    # Money fields: only page 2 reported values.
    assert raw["subtotal"]["value"] == 11.0
    assert raw["tax"]["value"] == 1.0
    assert raw["total"]["value"] == 12.0

    # line_items: concatenated in page order (page 2's empty list
    # contributes nothing but its confidence still enters the min).
    assert raw["line_items"]["value"] == [
        page_inputs[0]["line_items"]["value"][0],
        page_inputs[1]["line_items"]["value"][0],
    ]
    assert raw["line_items"]["confidence"] == pytest.approx(0.85)

    # Tokens/cost summed across all 3 pages.
    expected_input_tokens = 100 + 101 + 102
    expected_output_tokens = 50 + 51 + 52
    assert extraction.input_tokens == expected_input_tokens
    assert extraction.output_tokens == expected_output_tokens
    expected_cost = (expected_input_tokens * 2.00 + expected_output_tokens * 10.00) / 1_000_000
    assert float(extraction.cost_usd) == pytest.approx(expected_cost)

    fields = (
        (
            await db_session.execute(
                select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
            )
        )
        .scalars()
        .all()
    )
    # process_document_job iterates TOP_LEVEL_FIELDS only -- the extra
    # "_chunks" key on raw_response must not spawn its own field row.
    assert {f.field_name for f in fields} == set(TOP_LEVEL_FIELDS)

    await db_session.refresh(document)
    assert document.status == "extracted"


async def test_pdf_over_page_limit_raises_before_any_api_call(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-sonnet-5", review_threshold=0.8, pdf_max_pages=2),
    )
    pdf_bytes = _build_multipage_pdf(3)
    document = await _make_document(
        db_session, tmp_path, filename="invoice.pdf", mime_type="application/pdf", content=pdf_bytes
    )
    job = await _make_job(db_session, document.id)
    response = _FakeMessage("tool_use", [], _FakeUsage(0, 0))
    fake_client = _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError):
        await process_document_job(db_session, job)

    fake_client.messages.create.assert_not_awaited()
    assert await _extraction_count(db_session, document.id) == 0


async def test_zero_page_pdf_raises_before_any_api_call(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 3/4d: a 0-page PDF is a guaranteed-rejection API call
    otherwise -- no page-count threshold catches it (0 is <= every
    positive pdf_max_pages_per_call/pdf_max_pages), so it needs its own
    explicit pre-flight refusal, free, before any call is made.
    """
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-sonnet-5", review_threshold=0.8),
    )
    document = await _make_document(
        db_session,
        tmp_path,
        filename="empty.pdf",
        mime_type="application/pdf",
        content=_build_zero_page_pdf(),
    )
    job = await _make_job(db_session, document.id)
    response = _FakeMessage("tool_use", [], _FakeUsage(0, 0))
    fake_client = _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError):
        await process_document_job(db_session, job)

    fake_client.messages.create.assert_not_awaited()
    assert await _extraction_count(db_session, document.id) == 0


async def test_chunked_page_failure_preserves_exception_type_and_prior_pages_spend(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 4a/4b regression: a page-2 refusal in the middle of a 3-page
    chunked extraction must (a) surface as ModelRefusalError -- the
    exact exception TYPE _extract_one_block raised, not rewrapped into
    a generic ExtractionError by _extract_chunked_pdf's except clause
    -- and (b) carry page 1's already-spent tokens/cost ON TOP of
    page 2's own billed refusal usage, per the billed-but-errored
    convention (see _extract_chunked_pdf's docstring). Page 3's PDF
    page must never be attempted once page 2 raises.
    """
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(
            extraction_model="claude-sonnet-5",
            review_threshold=0.8,
            pdf_max_pages_per_call=2,
        ),
    )
    path = tmp_path / "invoice.pdf"
    path.write_bytes(_build_multipage_pdf(3))

    page1_response = _FakeMessage(
        stop_reason="tool_use",
        content=[
            _FakeToolUseBlock(_minimal_page(vendor=_leaf("Acme", 0.9)), tool_use_id="toolu_1")
        ],
        usage=_FakeUsage(100, 50),
    )
    page2_refusal = _FakeMessage(stop_reason="refusal", content=[], usage=_FakeUsage(20, 10))
    fake_client = _patch_client_side_effect(monkeypatch, [page1_response, page2_refusal])

    with pytest.raises(ModelRefusalError) as exc_info:
        await extract_document(path, "application/pdf")

    # Page 3 never attempted -- the sequential loop stops at the first
    # page that raises.
    assert fake_client.messages.create.await_count == 2

    exc = exc_info.value
    # Page 1's spend (100, 50) PLUS page 2's own billed refusal usage
    # (20, 10) -- not just page 2's usage alone.
    assert exc.input_tokens == 100 + 20
    assert exc.output_tokens == 50 + 10
    expected_cost = ((100 + 20) * 2.00 + (50 + 10) * 10.00) / 1_000_000
    assert exc.cost_usd == pytest.approx(expected_cost)
