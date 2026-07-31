import base64
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import extraction as extraction_module
from app.config import Settings
from app.extraction import (
    PROMPT_TEXT,
    PROMPT_VERSION,
    RECORD_EXTRACTION_TOOL,
    TOP_LEVEL_FIELDS,
    ExtractionError,
    NonRetryableExtractionError,
    process_document_job,
)
from app.models import Document, ExtractedField, Extraction, Job

# A minimal valid 1x1 transparent PNG (same fixture used in test_documents.py).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

REALISTIC_TOOL_INPUT = {
    "vendor": {"value": "Acme Corp", "confidence": 0.95},
    "document_date": {"value": "2026-01-01", "confidence": 0.6},
    "line_items": {
        "value": [
            {
                "description": {"value": "Widget", "confidence": 0.9},
                "quantity": {"value": 2, "confidence": 0.9},
                "unit_price": {"value": 5.0, "confidence": 0.9},
                "total": {"value": 10.0, "confidence": 0.9},
            }
        ],
        "confidence": 0.85,
    },
    "subtotal": {"value": 10.0, "confidence": 0.8},
    "tax": {"value": 0.0, "confidence": 0.99},
    "total": {"value": 10.0, "confidence": 0.79},
    "currency": {"value": "USD", "confidence": 1.0},
}


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, tool_input: dict) -> None:
        self.input = tool_input


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, stop_reason: str, content: list, usage: _FakeUsage) -> None:
        self.stop_reason = stop_reason
        self.content = content
        self.usage = usage


class _FakeMessagesResource:
    def __init__(self, response: _FakeMessage) -> None:
        self.create = AsyncMock(return_value=response)


class _FakeClient:
    def __init__(self, response: _FakeMessage) -> None:
        self.messages = _FakeMessagesResource(response)


@pytest.fixture(autouse=True)
def _pinned_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the settings extraction.py reads so a developer's local
    backend/.env (e.g. an EXTRACTION_MODEL or REVIEW_THRESHOLD override)
    can't change what these tests assert. get_settings() is lru_cache'd
    at the app.config level and extraction.py calls it fresh inside each
    function rather than taking it as a parameter, so the only way to
    pin it from a test is to monkeypatch the name extraction.py actually
    looks up (app.extraction.get_settings) -- busting app.config's
    lru_cache directly would leak across other test modules that share
    the process.
    """
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-sonnet-5", review_threshold=0.8),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, response: _FakeMessage) -> _FakeClient:
    fake_client = _FakeClient(response)
    monkeypatch.setattr(extraction_module, "_build_client", lambda settings: fake_client)
    return fake_client


async def _make_document(
    db_session: AsyncSession,
    tmp_path,
    *,
    filename: str = "receipt.png",
    mime_type: str = "image/png",
    content: bytes = TINY_PNG,
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
    # Commit (not just flush): process_document_job now calls
    # session.rollback() before the VLM call (fix 5, keeping the DB
    # connection from sitting idle-in-transaction across it). Under this
    # fixture's join_transaction_mode="create_savepoint", commit() only
    # releases a savepoint -- the real transaction stays open for the
    # fixture's own final rollback -- but it's what makes fixture rows
    # survive a rollback() the handler issues afterward, the same way
    # they'd already be committed for real by the time a real worker
    # picks the job up.
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
            select(func.count())
            .select_from(Extraction)
            .where(Extraction.document_id == document_id)
        )
    ).scalar_one()


async def test_process_document_job_persists_extraction_and_fields(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT)],
        usage=_FakeUsage(1000, 500),
    )
    _patch_client(monkeypatch, response)

    await process_document_job(db_session, job)
    await db_session.flush()

    extraction = (
        (
            await db_session.execute(
                select(Extraction).where(Extraction.document_id == document.id)
            )
        )
        .scalars()
        .one()
    )
    assert extraction.prompt_version == PROMPT_VERSION
    assert extraction.model == "claude-sonnet-5"
    assert extraction.input_tokens == 1000
    assert extraction.output_tokens == 500
    assert extraction.raw_response == REALISTIC_TOOL_INPUT
    assert extraction.latency_ms >= 0
    # cost math matches PRICING_PER_MTOK["claude-sonnet-5"] = ($2.00, $10.00)/MTok
    expected_cost = (1000 * 2.00 + 500 * 10.00) / 1_000_000
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
    by_name = {f.field_name: f for f in fields}
    assert set(by_name) == set(TOP_LEVEL_FIELDS)

    # Scalar leaves store the whole {"value", "confidence"} object.
    assert by_name["vendor"].value == REALISTIC_TOOL_INPUT["vendor"]
    assert by_name["vendor"].confidence == pytest.approx(0.95)
    assert by_name["vendor"].needs_review is False

    # line_items stores the unwrapped array; its own confidence goes in
    # the confidence column.
    assert by_name["line_items"].value == REALISTIC_TOOL_INPUT["line_items"]["value"]
    assert by_name["line_items"].confidence == pytest.approx(0.85)
    assert by_name["line_items"].needs_review is False

    # Confidence-threshold flagging: needs_review true only strictly below
    # 0.8 (settings.review_threshold default), per-field, not global.
    assert by_name["document_date"].needs_review is True  # 0.6
    assert by_name["subtotal"].needs_review is False  # 0.8 == threshold, not below
    assert by_name["tax"].needs_review is False  # 0.99
    assert by_name["total"].needs_review is True  # 0.79
    assert by_name["currency"].needs_review is False  # 1.0

    await db_session.refresh(document)
    assert document.status == "extracted"


async def test_messages_create_called_with_expected_request_shape(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the actual client.messages.create call. Without this, a
    `tool_choise=` typo, a dropped/wrong tool list, or a missing system
    prompt would still pass every other test, since the fake response is
    scripted independently of what was actually sent.
    """
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT)],
        usage=_FakeUsage(10, 5),
    )
    fake_client = _patch_client(monkeypatch, response)

    await process_document_job(db_session, job)

    fake_client.messages.create.assert_awaited_once()
    kwargs = fake_client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_extraction"}
    assert kwargs["tools"] == [RECORD_EXTRACTION_TOOL]
    assert kwargs["system"] == PROMPT_TEXT

    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"


async def test_pdf_document_uses_document_content_block(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _make_document(
        db_session,
        tmp_path,
        filename="invoice.pdf",
        mime_type="application/pdf",
        content=b"%PDF-1.4 not a real pdf, just bytes for the mocked call\n",
    )
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT)],
        usage=_FakeUsage(10, 5),
    )
    fake_client = _patch_client(monkeypatch, response)

    await process_document_job(db_session, job)

    kwargs = fake_client.messages.create.await_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"


async def test_unsupported_mime_raises_non_retryable_before_api_call(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _make_document(
        db_session, tmp_path, filename="notes.txt", mime_type="text/plain", content=b"hello"
    )
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(
        "tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(0, 0)
    )
    fake_client = _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError):
        await process_document_job(db_session, job)

    fake_client.messages.create.assert_not_awaited()
    assert await _extraction_count(db_session, document.id) == 0


async def test_unknown_model_raises_non_retryable_before_api_call(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-does-not-exist", review_threshold=0.8),
    )
    response = _FakeMessage(
        "tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(0, 0)
    )
    fake_client = _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError):
        await process_document_job(db_session, job)

    fake_client.messages.create.assert_not_awaited()
    assert await _extraction_count(db_session, document.id) == 0


async def test_malformed_leaf_shapes_degrade_instead_of_crashing(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tool_choice guarantees a tool_use block, not schema conformance.
    A bare scalar, a null confidence, and a leaf missing "value" must
    all degrade to value=None/confidence=0.0/needs_review=True instead
    of raising AttributeError/TypeError/KeyError.
    """
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    malformed_input = {
        "vendor": "Acme",  # bare scalar, not a {value, confidence} dict
        "document_date": {"value": "2026-01-01", "confidence": None},  # null confidence
        "line_items": {"value": []},  # missing "confidence" key
        "subtotal": {},  # missing "value" key
        "tax": {"value": 1.0, "confidence": 0.9},
        "total": {"value": 1.0, "confidence": 0.9},
        "currency": {"value": "USD", "confidence": 0.9},
    }
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(malformed_input)],
        usage=_FakeUsage(10, 5),
    )
    _patch_client(monkeypatch, response)

    await process_document_job(db_session, job)  # must not raise

    extraction = (
        (
            await db_session.execute(
                select(Extraction).where(Extraction.document_id == document.id)
            )
        )
        .scalars()
        .one()
    )
    fields = (
        (
            await db_session.execute(
                select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
            )
        )
        .scalars()
        .all()
    )
    by_name = {f.field_name: f for f in fields}

    for field_name in ("vendor", "document_date", "subtotal"):
        assert by_name[field_name].value == {"value": None, "confidence": 0.0}
        assert by_name[field_name].confidence == pytest.approx(0.0)
        assert by_name[field_name].needs_review is True

    assert by_name["line_items"].value == []
    assert by_name["line_items"].confidence == pytest.approx(0.0)
    assert by_name["line_items"].needs_review is True

    # Well-formed leaves in the same response are unaffected.
    assert by_name["tax"].value == {"value": 1.0, "confidence": 0.9}
    assert by_name["tax"].needs_review is False

    await db_session.refresh(document)
    assert document.status == "extracted"


async def test_refusal_raises_non_retryable_and_persists_nothing(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(stop_reason="refusal", content=[], usage=_FakeUsage(50, 10))
    _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError) as exc_info:
        await process_document_job(db_session, job)
    # billed usage is surfaced in the error text (fix 2c) so a requeued
    # job's last_error doesn't hide the spend from the failed attempt.
    assert "input_tokens=50" in str(exc_info.value)
    assert "output_tokens=10" in str(exc_info.value)

    assert await _extraction_count(db_session, document.id) == 0
    await db_session.refresh(document)
    assert document.status == "uploaded"


async def test_max_tokens_truncation_is_non_retryable(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(
        stop_reason="max_tokens",
        content=[_FakeTextBlock("(truncated)")],
        usage=_FakeUsage(4000, 4096),
    )
    _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError):
        await process_document_job(db_session, job)

    assert await _extraction_count(db_session, document.id) == 0


async def test_tool_use_stop_reason_without_tool_use_block_is_retryable(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive path: stop_reason claims tool_use but no tool_use block
    is actually present in content. Classified as a transient anomaly
    (plain ExtractionError, retryable) rather than
    NonRetryableExtractionError, since forced tool_choice is documented
    to guarantee a tool_use block whenever stop_reason is "tool_use" --
    if that guarantee doesn't hold, it isn't obviously deterministic the
    way a refusal is.
    """
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeTextBlock("oops")],
        usage=_FakeUsage(10, 5),
    )
    _patch_client(monkeypatch, response)

    with pytest.raises(ExtractionError) as exc_info:
        await process_document_job(db_session, job)
    assert not isinstance(exc_info.value, NonRetryableExtractionError)

    assert await _extraction_count(db_session, document.id) == 0
