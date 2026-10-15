"""A schema v2 tool_input as the model returns it, shared by the v2 tests.

Not a test module (no tests, never collected); pytest puts this directory on
sys.path, so the test files import it as `v2_samples`.
"""

import copy
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.extraction import build_extracted_fields
from app.extraction_fields import SCHEMA_V2
from app.models import Document, ExtractedField, Extraction
from app.review_routing import refresh_reconciliation


def leaf(value: Any, quote: str | None = None, *, page: int = 1, confidence: float = 0.95) -> dict:
    """A v2 leaf: evidence is {page, quote}, or null when there is no quote."""
    evidence = None if quote is None else {"page": page, "quote": quote}
    return {"value": value, "confidence": confidence, "evidence": evidence}


def tax_row(label: str, rate: float, base: float, amount: float, quote: str) -> dict:
    return {
        "label": leaf(label, quote),
        "rate": leaf(rate, quote),
        "base": leaf(base, quote),
        "amount": leaf(amount, quote),
    }


# 100.00 - 10.00 discount + 5.00 shipping + 7.20 tax = 102.20, of which 50.00 is
# paid; 52.20 is due. Every number is quoted as it is printed, label included.
V2_TOOL_INPUT: dict[str, Any] = {
    "vendor": leaf("Acme Corp", "Acme Corp"),
    "invoice_number": leaf("INV-0042", "Invoice # INV-0042"),
    "po_number": leaf("PO-77", "PO: PO-77"),
    "vendor_tax_id": leaf("12-3456789", "EIN 12-3456789"),
    "document_date": leaf("2026-01-01", "Date: 01/01/2026"),
    "due_date": leaf("2026-01-31", "Due: 01/31/2026"),
    "payment_terms": leaf("Net 30", "Terms: Net 30"),
    "payment_method": leaf("VISA", "Paid by VISA"),
    "payment_card_last4": leaf("4242", "VISA ending 4242"),
    "line_items": {
        "value": [
            {
                "description": leaf("Widget", "Widget"),
                "quantity": leaf(2, "2 x Widget"),
                "unit_price": leaf(50.0, "Widget 2 x 50.00"),
                "total": leaf(100.0, "Widget 100.00"),
            }
        ],
        "confidence": 0.9,
        "evidence": None,
    },
    "subtotal": leaf(100.0, "Subtotal $100.00"),
    "discount": leaf(10.0, "Discount -$10.00"),
    "shipping": leaf(5.0, "Shipping $5.00"),
    "tax": leaf(7.2, "Tax $7.20"),
    "tax_lines": {
        "value": [tax_row("State tax", 8, 90.0, 7.2, "State tax 8% on 90.00 7.20")],
        "confidence": 0.9,
        "evidence": None,
    },
    "tip": leaf(None),
    "total": leaf(102.2, "Total $102.20"),
    "amount_paid": leaf(50.0, "Paid $50.00"),
    "amount_due": leaf(52.2, "Balance due $52.20"),
    "currency": leaf("USD", "USD"),
}


def v2_tool_input() -> dict[str, Any]:
    """A fresh copy, so a test can change it freely."""
    return copy.deepcopy(V2_TOOL_INPUT)


_SETTINGS = Settings(review_threshold=0.8)


async def seed_v2(db_session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID, dict[str, ExtractedField]]:
    """A schema v2 extraction (V2_TOOL_INPUT: all 20 fields, evidence
    on every leaf, consistent arithmetic) stored and reconciled the way the
    worker does it."""
    tool_input = v2_tool_input()
    document = Document(
        filename="v2.png", mime_type="image/png", storage_path=f"/tmp/{uuid.uuid4()}.png", status="extracted"
    )
    db_session.add(document)
    await db_session.flush()
    extraction = Extraction(
        document_id=document.id,
        prompt_version=SCHEMA_V2.prompt_version,
        model="claude-sonnet-5",
        raw_response=tool_input,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0,
        latency_ms=1,
    )
    db_session.add(extraction)
    await db_session.flush()
    fields = build_extracted_fields(
        extraction.id, tool_input, _SETTINGS.review_threshold, SCHEMA_V2
    )
    db_session.add_all(fields)
    await db_session.flush()
    await refresh_reconciliation(db_session, extraction.id, _SETTINGS)
    await db_session.commit()
    return document.id, extraction.id, {f.field_name: f for f in fields}
