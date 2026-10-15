"""Tests for H6 (oversized-document chunking): the pure PDF utilities in
app/pdf.py, the pure merge rules in app.extraction.merge_page_tool_inputs,
and the end-to-end process_document_job chunked path.

Kept in its own file (rather than folded into test_extraction.py /
test_pdf.py) so the H6-specific fixtures (a real multi-page PDF built
with Pillow, the chunking-pinned Settings) don't clutter the existing
single-call suite -- and so that suite's "must pass UNMODIFIED" claim
in the task notes is easy to verify by diffing.
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from pypdf import PdfReader, PdfWriter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from v2_samples import leaf as v2_leaf

from app import extraction as extraction_module
from app.config import Settings
from app.extraction import (
    MERGE_CONFLICT_CONFIDENCE,
    ModelRefusalError,
    NonRetryableExtractionError,
    _evidence_quote,
    extract_document,
    merge_page_tool_inputs,
    process_document_job,
)
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2
from app.models import Document, ExtractedField, Extraction, Job, Reconciliation
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
    """A page tool_input with every SCHEMA_V1.field_names leaf defaulted to
    {"value": None, "confidence": 0.0} (a well-formed, all-absent leaf
    -- not a malformed one), overridden per-field by kwargs.
    """
    page = {name: _leaf(None, 0.0) for name in SCHEMA_V1.field_names}
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

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

    assert merged["line_items"]["value"] == [
        {"description": _leaf("Widget", 0.9)},
        {"description": _leaf("Gadget", 0.8)},
    ]
    assert merged["line_items"]["confidence"] == pytest.approx(0.6)


def test_line_items_all_null_gives_empty_list_and_zero_confidence() -> None:
    page0 = _minimal_page()
    page1 = _minimal_page()

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

    assert merged["line_items"] == {"value": [], "confidence": 0.0}


def test_header_field_tie_breaks_to_earliest_page() -> None:
    """vendor is a header field: equal-confidence, disagreeing
    candidates must pick the EARLIEST page's value (and, since they
    disagree, cap the merged confidence at MERGE_CONFLICT_CONFIDENCE).
    """
    page0 = _minimal_page(vendor=_leaf("Acme Corp", 0.9))
    page1 = _minimal_page(vendor=_leaf("Other Co", 0.9))

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

    assert merged["vendor"]["value"] == "Acme Corp"
    assert merged["vendor"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_money_field_tie_breaks_to_latest_page() -> None:
    """total is a money field: equal-confidence, disagreeing candidates
    must pick the LATEST page's value -- the opposite tie-break from
    header fields, since totals conventionally print at the end.
    """
    page0 = _minimal_page(total=_leaf(100.0, 0.9))
    page1 = _minimal_page(total=_leaf(200.0, 0.9))

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

    assert merged["total"]["value"] == 200.0
    assert merged["total"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_all_null_scalar_field_uses_min_confidence_across_pages() -> None:
    page0 = _minimal_page(currency=_leaf(None, 0.4))
    page1 = _minimal_page(currency=_leaf(None, 0.1))
    page2 = _minimal_page(currency=_leaf(None, 0.7))

    merged = merge_page_tool_inputs([page0, page1, page2], SCHEMA_V1)

    assert merged["currency"] == {"value": None, "confidence": pytest.approx(0.1)}


def test_conflicting_values_cap_confidence_even_when_winner_was_very_confident() -> None:
    page0 = _minimal_page(vendor=_leaf("Acme Corp", 0.95))
    page1 = _minimal_page(vendor=_leaf("Totally Different Vendor", 0.5))

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

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

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

    assert merged["vendor"]["value"] == "acme corp"
    assert merged["vendor"]["confidence"] == pytest.approx(0.9)


def test_agreeing_numeric_values_tolerate_float_noise() -> None:
    page0 = _minimal_page(total=_leaf(10.004, 0.6))
    page1 = _minimal_page(total=_leaf(10.001, 0.9))

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

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

    merged = merge_page_tool_inputs([malformed_page, well_formed_page], SCHEMA_V1)

    assert merged["vendor"]["value"] == "Acme Corp"
    assert set(merged) == set(SCHEMA_V1.field_names)


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

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)  # must not raise

    assert merged["total"]["value"] == 100.0  # tie -> latest page wins
    assert merged["total"]["confidence"] == pytest.approx(MERGE_CONFLICT_CONFIDENCE)


def test_malformed_leaf_within_an_otherwise_valid_page_degrades() -> None:
    page0 = {**_minimal_page(), "subtotal": "bare scalar, not a leaf dict"}
    page1 = _minimal_page(subtotal=_leaf(42.0, 0.9))

    merged = merge_page_tool_inputs([page0, page1], SCHEMA_V1)

    assert merged["subtotal"]["value"] == 42.0
    assert merged["subtotal"]["confidence"] == pytest.approx(0.9)


# The v1 merge output, captured from the merge before evidence existed (B5).
# Four pages: a tie and a conflict, a page of bare scalars, a malformed leaf, a
# non-row in line_items, and a cell that carries an evidence key a v1 merge must
# leave exactly as it came. The comparison is of the serialized text, so key
# order and number types count.
_V1_MERGE_GOLDEN = (
    '{"vendor": {"value": "Acme Corp", "confidence": 0.9}, '
    '"document_date": {"value": "2026-01-01", "confidence": 0.5}, '
    '"line_items": {"value": ['
    '{"description": {"value": "Widget", "confidence": 0.9}, "quantity": {"value": 1, "confidence": 0.9}, '
    '"unit_price": {"value": 10.0, "confidence": 0.9}, "total": {"value": 10.0, "confidence": 0.9}}, '
    '{"description": {"value": "Gadget", "confidence": 0.8}, "quantity": {"value": 2, "confidence": 0.8}, '
    '"unit_price": {"value": 3.0, "confidence": 0.8}, "total": {"value": 6.0, "confidence": 0.8}}, '
    '"not a row", '
    '{"description": {"value": "Doohickey", "confidence": 0.9, "evidence": {"page": 7, "quote": "x"}}}'
    '], "confidence": 0.7}, '
    '"subtotal": {"value": 10.004, "confidence": 0.9}, '
    '"tax": {"value": 2.0, "confidence": 0.5}, '
    '"total": {"value": 12.0, "confidence": 0.5}, '
    '"currency": {"value": "USD", "confidence": 0.95}}'
)


def test_the_v1_merge_output_is_byte_identical_to_the_one_before_evidence() -> None:
    page0 = _minimal_page(
        vendor=_leaf("Acme Corp", 0.9),
        document_date=_leaf("2026-01-01", 0.8),
        currency=_leaf("USD", 0.7),
        subtotal=_leaf(10.0, 0.9),
        tax=_leaf(1.0, 0.6),
        total=_leaf(11.0, 0.9),
        line_items=_leaf(
            [
                {
                    "description": _leaf("Widget", 0.9),
                    "quantity": _leaf(1, 0.9),
                    "unit_price": _leaf(10.0, 0.9),
                    "total": _leaf(10.0, 0.9),
                }
            ],
            0.9,
        ),
    )
    page1 = _minimal_page(
        vendor=_leaf("ACME corp ", 0.9),
        document_date=_leaf("2026-01-02", 0.8),
        currency=_leaf("USD", 0.95),
        subtotal=_leaf(10.004, 0.9),
        tax=_leaf(2.0, 0.6),
        total=_leaf(12.0, 0.9),
        line_items=_leaf(
            [
                {
                    "description": _leaf("Gadget", 0.8),
                    "quantity": _leaf(2, 0.8),
                    "unit_price": _leaf(3.0, 0.8),
                    "total": _leaf(6.0, 0.8),
                },
                "not a row",
            ],
            0.7,
        ),
    )
    page2 = {"vendor": "bare scalar", "subtotal": {"value": 9.0}, "line_items": None, "extra": 1}
    page3 = _minimal_page(
        total=_leaf(12.0, 0.9),
        tax=_leaf(None, 0.1),
        line_items=_leaf(
            [
                {
                    "description": {
                        "value": "Doohickey",
                        "confidence": 0.9,
                        "evidence": {"page": 7, "quote": "x"},
                    }
                }
            ],
            0.95,
        ),
    )

    merged = merge_page_tool_inputs([page0, page1, page2, page3], SCHEMA_V1)

    assert json.dumps(merged) == _V1_MERGE_GOLDEN


# --- merge_page_tool_inputs: evidence pages under schema v2 (B5) -------------
#
# Each chunk call saw one page, so a winner's evidence is cited to its chunk,
# whatever page the model wrote.


def _blank_v2_page(**overrides) -> dict:
    """A v2 page tool_input with every leaf null and no evidence, overridden
    per field by kwargs."""
    page = {
        field.name: {"value": [], "confidence": 0.9}
        if field.kind == "array"
        else v2_leaf(None)
        for field in SCHEMA_V2.fields
    }
    page.update(overrides)
    return page


def test_a_winners_evidence_takes_the_page_of_its_chunk_not_the_one_the_model_wrote() -> None:
    pages = [_blank_v2_page() for _ in range(7)]
    pages[1]["invoice_number"] = v2_leaf("INV-2", "Invoice # INV-2", page=1)
    pages[5]["total"] = v2_leaf(12.0, "Total $12.00", page=1)

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    assert merged["invoice_number"]["evidence"] == {"page": 2, "quote": "Invoice # INV-2"}
    assert merged["total"]["evidence"] == {"page": 6, "quote": "Total $12.00"}


def test_the_tie_break_and_the_conflict_cap_still_pick_the_winner_and_its_page() -> None:
    pages = [_blank_v2_page() for _ in range(3)]
    pages[0]["vendor"] = v2_leaf("Acme", "Acme", page=3)
    pages[2]["vendor"] = v2_leaf("Acme Inc", "Acme Inc", page=1)
    pages[0]["subtotal"] = v2_leaf(10.0, "Subtotal 10.00")
    pages[2]["subtotal"] = v2_leaf(11.0, "Subtotal 11.00")

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    # A tie between pages 1 and 3 goes to the earliest for a header field, the latest for an amount.
    assert merged["vendor"] == {
        "value": "Acme",
        "confidence": MERGE_CONFLICT_CONFIDENCE,
        "evidence": {"page": 1, "quote": "Acme"},
    }
    assert merged["subtotal"] == {
        "value": 11.0,
        "confidence": MERGE_CONFLICT_CONFIDENCE,
        "evidence": {"page": 3, "quote": "Subtotal 11.00"},
    }


def test_a_winner_without_a_usable_quote_and_an_all_null_field_have_null_evidence() -> None:
    pages = [_blank_v2_page() for _ in range(2)]
    pages[0]["vendor"] = {"value": "Acme", "confidence": 0.9, "evidence": None}
    pages[1]["tax"] = {"value": 1.0, "confidence": 0.9, "evidence": {"page": 2, "quote": "  "}}
    pages[1]["total"] = {"value": 2.0, "confidence": 0.9}  # no evidence key at all
    pages[1]["discount"] = {"value": 3.0, "evidence": {"page": 2, "quote": "Discount 3.00"}}  # malformed

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    assert merged["vendor"]["evidence"] is None
    assert merged["tax"]["evidence"] is None
    assert merged["total"]["evidence"] is None
    assert merged["discount"] == {"value": None, "confidence": 0.0, "evidence": None}
    assert merged["tip"] == {"value": None, "confidence": 0.95, "evidence": None}


def test_a_winner_with_no_quote_borrows_the_quote_of_the_best_agreeing_chunk() -> None:
    pages = [_blank_v2_page() for _ in range(4)]
    pages[0]["total"] = v2_leaf(12.0, "Total $12.00", confidence=0.7)
    pages[1]["total"] = v2_leaf(12.0, None, confidence=0.99)  # the winner: nothing to cite
    pages[2]["total"] = v2_leaf(12.0, "Amount due $12.00", confidence=0.8)
    pages[3]["total"] = v2_leaf(13.0, "Total $13.00", confidence=0.95)  # disagrees: never borrowed from

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    assert merged["total"] == {
        "value": 12.0,
        "confidence": MERGE_CONFLICT_CONFIDENCE,  # the page 4 disagreement still caps it
        "evidence": {"page": 3, "quote": "Amount due $12.00"},
    }


def test_a_winner_that_has_a_quote_keeps_it_over_a_more_confident_neighbour_without_one() -> None:
    pages = [_blank_v2_page() for _ in range(2)]
    pages[0]["total"] = v2_leaf(12.0, "Total $12.00", confidence=0.9)
    pages[1]["total"] = v2_leaf(12.0, None, confidence=0.8)

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    assert merged["total"]["evidence"] == {"page": 1, "quote": "Total $12.00"}


def test_two_chunks_that_agree_with_the_winner_unquoted_leave_null_evidence() -> None:
    pages = [_blank_v2_page() for _ in range(2)]
    pages[0]["total"] = v2_leaf(12.0, None, confidence=0.9)
    pages[1]["total"] = v2_leaf(13.0, "Total $13.00", confidence=0.5)

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    assert merged["total"]["value"] == 12.0
    assert merged["total"]["evidence"] is None


def test_concatenated_cells_cite_the_chunk_their_row_came_from() -> None:
    def row(name: str, quote: str, claimed_page: int) -> dict:
        return {"description": v2_leaf(name, quote, page=claimed_page), "total": v2_leaf(1.0)}

    pages = [_blank_v2_page() for _ in range(3)]
    pages[0]["line_items"] = {"value": [row("A", "A 1.00", 1)], "confidence": 0.9}
    pages[2]["line_items"] = {
        "value": [row("B", "B 1.00", 1), row("C", "C 1.00", 9), "not a row"],
        "confidence": 0.8,
    }

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    rows = merged["line_items"]["value"]
    assert [r["description"]["evidence"]["page"] for r in rows[:3]] == [1, 3, 3]
    assert [r["description"]["evidence"]["quote"] for r in rows[:3]] == ["A 1.00", "B 1.00", "C 1.00"]
    assert rows[3] == "not a row"
    assert rows[0]["total"]["evidence"] is None  # null evidence stays null
    assert merged["line_items"]["confidence"] == pytest.approx(0.8)


# Every value is one of a few lowercase tokens, so two leaves that agree (the
# merge's rule for borrowing a quote) are equal and the tests below can compare
# with ==.
_values = st.sampled_from([None, 1, 2, "a", "b"])
_page_claims = st.one_of(
    st.none(), st.integers(-5, 25), st.booleans(), st.floats(allow_nan=False), st.text(max_size=3)
)
_quotes = st.one_of(st.none(), st.integers(), st.text(alphabet="ab ", max_size=6))
_evidence_objects = st.one_of(
    st.none(),
    st.text(max_size=5),
    st.fixed_dictionaries({"page": _page_claims, "quote": _quotes}),
    st.fixed_dictionaries({"quote": _quotes}),
)
_leaves = st.one_of(
    st.fixed_dictionaries(
        {"value": _values, "confidence": st.sampled_from([0.2, 0.5, 0.9]), "evidence": _evidence_objects}
    ),
    st.fixed_dictionaries({"value": _values, "confidence": st.sampled_from([0.2, 0.9])}),
    st.text(max_size=3),
    st.none(),
)
_rows = st.one_of(
    st.just("not a row"), st.fixed_dictionaries({"description": _leaves, "total": _leaves})
)
_ARRAY_NAMES = [f.name for f in SCHEMA_V2.fields if f.kind == "array"]
_SCALAR_NAMES = [f.name for f in SCHEMA_V2.fields if f.kind != "array"]
# Every page carries an array (a page without one cannot expose a cell that
# cites the wrong page) and one or two scalars; everything else is null.
_v2_pages = st.builds(
    lambda array_name, rows, scalars: _blank_v2_page(
        **{array_name: {"value": rows, "confidence": 0.9}}, **scalars
    ),
    st.sampled_from(_ARRAY_NAMES),
    st.lists(_rows, max_size=2),
    st.dictionaries(st.sampled_from(_SCALAR_NAMES), _leaves, min_size=1, max_size=2),
)


@given(pages=st.lists(_v2_pages, min_size=1, max_size=6))
def test_every_merged_evidence_cites_a_page_that_holds_its_quote(pages: list) -> None:
    """A merged quote is cited to a real page of the document, and that page's
    own chunk holds the same value with the same quote."""
    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    def cited_page(evidence, where: str) -> int:
        page = evidence["page"]
        assert isinstance(page, int) and not isinstance(page, bool), f"{where}: page {page!r}"
        assert 1 <= page <= len(pages), f"{where}: page {page} of {len(pages)}"
        assert evidence["quote"] and isinstance(evidence["quote"], str), f"{where}: quote"
        return page

    for name in _SCALAR_NAMES:
        leaf = merged[name]
        if leaf["evidence"] is None:
            continue
        assert leaf["value"] is not None, f"{name}: evidence on a null value"
        page = cited_page(leaf["evidence"], name)
        source = pages[page - 1][name]
        assert isinstance(source, dict), f"{name}: page {page} holds no leaf"
        assert source["value"] == leaf["value"], f"{name}: page {page} read another value"
        assert _evidence_quote(source.get("evidence")) == leaf["evidence"]["quote"], (
            f"{name}: page {page} has another quote"
        )

    for name in _ARRAY_NAMES:
        assert "evidence" not in merged[name]
        for i, row in enumerate(merged[name]["value"]):
            if not isinstance(row, dict):
                continue
            for key, cell in row.items():
                if not (isinstance(cell, dict) and cell.get("evidence") is not None):
                    continue
                where = f"{name}[{i}].{key}"
                page = cited_page(cell["evidence"], where)
                source_rows = pages[page - 1][name]["value"]
                assert any(
                    isinstance(r, dict)
                    and isinstance(r[key], dict)
                    and r[key].get("value") == cell.get("value")
                    and _evidence_quote(r[key].get("evidence")) == cell["evidence"]["quote"]
                    for r in source_rows
                ), f"{where}: page {page} holds no such cell"


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
    # process_document_job iterates SCHEMA_V1.field_names only -- the extra
    # "_chunks" key on raw_response must not spawn its own field row.
    assert {f.field_name for f in fields} == set(SCHEMA_V1.field_names)

    await db_session.refresh(document)
    assert document.status == "extracted"


async def test_a_chunked_v2_extraction_cites_each_chunk_and_its_quote_checks_run(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(
            extraction_model="claude-sonnet-5",
            review_threshold=0.8,
            extraction_prompt_version="extract_v2",
        ),
    )
    document = await _make_document(
        db_session,
        tmp_path,
        filename="seven.pdf",
        mime_type="application/pdf",
        content=_build_multipage_pdf(7),
    )
    job = await _make_job(db_session, document.id)

    # Each page is a one-page PDF to the model, so every page cites page 1.
    pages = [_blank_v2_page() for _ in range(7)]
    pages[0]["vendor"] = v2_leaf("Acme Corp", "Acme Corp")
    pages[0]["currency"] = v2_leaf("USD", "USD")
    pages[1]["invoice_number"] = v2_leaf("INV-2", "Invoice # INV-2")
    pages[2]["line_items"] = {
        "value": [
            {
                "description": v2_leaf("Widget", "Widget"),
                "quantity": v2_leaf(1, "1 x Widget"),
                "unit_price": v2_leaf(100.0, "Widget 1 x 100.00"),
                "total": v2_leaf(100.0, "Widget 100.00"),
            }
        ],
        "confidence": 0.9,
    }
    pages[6]["subtotal"] = v2_leaf(100.0, "Subtotal $100.00")
    pages[6]["tax"] = v2_leaf(8.0, "Tax $8.00")
    pages[6]["total"] = v2_leaf(108.0, "Total $108.00")
    pages[6]["tax_lines"] = {
        "value": [
            {
                "label": v2_leaf("Sales tax", "Sales tax 8% 8.00"),
                "rate": v2_leaf(8, "Sales tax 8% 8.00"),
                "base": v2_leaf(100.0, "Sales tax 8% on 100.00"),
                "amount": v2_leaf(8.0, "Sales tax 8% 8.00"),
            }
        ],
        "confidence": 0.9,
    }
    fake_client = _patch_client_side_effect(
        monkeypatch,
        [
            _FakeMessage("tool_use", [_FakeToolUseBlock(page, tool_use_id=f"toolu_{i}")], _FakeUsage(10, 5))
            for i, page in enumerate(pages)
        ],
    )

    await process_document_job(db_session, job)
    await db_session.flush()

    assert fake_client.messages.create.await_count == 7
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    assert extraction.prompt_version == "extract_v2"
    raw = extraction.raw_response
    assert raw["invoice_number"]["evidence"] == {"page": 2, "quote": "Invoice # INV-2"}
    assert raw["line_items"]["value"][0]["total"]["evidence"] == {"page": 3, "quote": "Widget 100.00"}
    assert raw["total"]["evidence"] == {"page": 7, "quote": "Total $108.00"}
    # The per-chunk provenance keeps what each one-page call said.
    assert raw["_chunks"][1]["tool_input"]["invoice_number"]["evidence"]["page"] == 1

    fields = {
        f.field_name: f
        for f in (
            await db_session.execute(
                select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
            )
        ).scalars()
    }
    assert fields["vendor"].value["evidence"] == {"page": 1, "quote": "Acme Corp"}
    assert fields["invoice_number"].value["evidence"] == {"page": 2, "quote": "Invoice # INV-2"}
    assert fields["subtotal"].value["evidence"] == {"page": 7, "quote": "Subtotal $100.00"}
    assert fields["line_items"].value[0]["description"]["evidence"]["page"] == 3
    assert fields["tax_lines"].value[0]["amount"]["evidence"]["page"] == 7

    reconciliation = (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction.id)
        )
    ).scalar_one()
    quote_checks = {c["id"]: c["status"] for c in reconciliation.report["checks"] if c["kind"] == "quote"}
    for check_id in ("quote:subtotal", "quote:tax", "quote:total"):
        assert quote_checks[check_id] == "pass"
    assert "skipped" not in quote_checks.values()
    assert reconciliation.status == "pass"


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
