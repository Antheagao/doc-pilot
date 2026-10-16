import base64
import copy
import json
import re
import uuid
from io import BytesIO
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest
from anthropic._constants import MODEL_NONSTREAMING_TOKENS
from pypdf import PdfWriter
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from v2_samples import leaf, v2_tool_input

from app import extraction as extraction_module
from app.config import Settings
from app.extraction import (
    ExtractionError,
    ExtractionResult,
    ModelRefusalError,
    NonRetryableExtractionError,
    _coerce_evidence,
    _coerce_leaf,
    _format_violations_for_prompt,
    _leaf_is_malformed,
    _schema_violations,
    build_extracted_fields,
    extract_document,
    mask_card_data,
    mask_evidence_quotes,
    mask_payment_fields,
    merge_page_tool_inputs,
    process_document_job,
)
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2, record_extraction_tool
from app.models import Document, ExtractedField, Extraction, Job, Reconciliation
from app.reconcile import ReconcileInput, reconcile
from app.review_routing import Routing

# A minimal valid 1x1 transparent PNG (same fixture used in test_documents.py).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _minimal_pdf() -> bytes:
    """A genuinely parseable single-page PDF (unlike a hand-typed
    "%PDF-1.4 ..." string): H6 (see app/pdf.py) makes extract_document
    read and page-count every PDF via pypdf before deciding whether to
    take the single-call or chunked path, so any PDF fixture used
    through process_document_job/extract_document must be real enough
    for pypdf to parse, even in tests that only care about the
    single-call request shape.
    """
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()

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
    name = "record_extraction"

    def __init__(self, tool_input: dict, tool_use_id: str = "toolu_test") -> None:
        self.input = tool_input
        self.id = tool_use_id


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


def _patch_client_side_effect(
    monkeypatch: pytest.MonkeyPatch, side_effect: list
) -> _FakeClient:
    """Like _patch_client, but drives client.messages.create via
    AsyncMock's side_effect list instead of a fixed return_value -- for
    the schema-repair tests, where the first and second create() calls
    return different responses (test a/b) or the second raises (test d).
    An exception instance in the list is raised, not returned, when its
    turn comes -- standard unittest.mock side_effect semantics.
    """
    fake_client = _FakeClient(None)  # response unused once side_effect is set below
    fake_client.messages.create.side_effect = side_effect
    monkeypatch.setattr(extraction_module, "_build_client", lambda settings: fake_client)
    return fake_client


async def _extract_from_tmp_file(tmp_path, *, filename: str = "receipt.png") -> ExtractionResult:
    path = tmp_path / filename
    path.write_bytes(TINY_PNG)
    return await extract_document(path, "image/png")


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
    assert extraction.prompt_version == SCHEMA_V1.prompt_version
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
    assert set(by_name) == set(SCHEMA_V1.field_names)

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
    assert kwargs["tools"] == [record_extraction_tool(SCHEMA_V1)]
    assert kwargs["system"] == SCHEMA_V1.prompt_text

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
        content=_minimal_pdf(),
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

    with pytest.raises(ModelRefusalError) as exc_info:
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

    with pytest.raises(NonRetryableExtractionError) as exc_info:
        await process_document_job(db_session, job)
    assert not isinstance(exc_info.value, ModelRefusalError)

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


# --- rate-limit-aware backoff and HTTP-status classification (H5) --------


def _fake_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


async def test_rate_limit_error_with_retry_after_is_retryable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real anthropic.RateLimitError (429) with a `retry-after: 7`
    header must raise a retryable ExtractionError (not
    NonRetryableExtractionError) with retry_after_seconds parsed out of
    the header.
    """
    exc = anthropic.RateLimitError(
        "rate limited", response=_fake_response(429, {"retry-after": "7"}), body=None
    )
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)
    assert exc_info.value.retry_after_seconds == pytest.approx(7.0)


async def test_overloaded_error_is_retryable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """anthropic.OverloadedError (529) -- like RateLimitError, an
    APIStatusError subclass with a fixed status_code -- must also be
    retryable, classified the same way as any other 5xx.
    """
    exc = anthropic.OverloadedError(
        "overloaded", response=_fake_response(529), body=None
    )
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)


async def test_bad_request_error_is_non_retryable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 400 BadRequestError never succeeds on retry (same request, same
    document, same rejection every time) -- must raise
    NonRetryableExtractionError so fail_job doesn't burn all 3 worker
    attempts on it.
    """
    exc = anthropic.BadRequestError("bad request", response=_fake_response(400), body=None)
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(NonRetryableExtractionError):
        await _extract_from_tmp_file(tmp_path)


async def test_413_non_retryable_mentions_document_size_and_chunking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """413 is deterministic (the document itself is too large) so it's
    NonRetryableExtractionError like the other 4xx cases, but its message
    must call out the specific cause -- and that page chunking (H6,
    shipped) has already been tried and still wasn't enough, since a
    bare "413" is not actionable in a requeued job's last_error. This
    fires on both extract_document's single-call path (this test, via
    _extract_from_tmp_file's plain image) and the H6 chunked path
    (where a single page from _extract_one_block is still too large on
    its own) -- the message text must read sensibly for either.
    """
    exc = anthropic.RequestTooLargeError(
        "payload too large", response=_fake_response(413), body=None
    )
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(NonRetryableExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)
    assert "too large" in str(exc_info.value)
    assert "chunking" in str(exc_info.value)


async def test_5xx_status_error_is_retryable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A generic >=500 APIStatusError (InternalServerError -- the SDK's
    catch-all for 5xx codes with no dedicated exception class) must be
    retryable, same as RateLimitError/OverloadedError.
    """
    exc = anthropic.InternalServerError("server error", response=_fake_response(500), body=None)
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)


async def test_api_connection_error_is_retryable_with_no_retry_after(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """anthropic.APIConnectionError (and its subclass APITimeoutError)
    carries no HTTP response, so there's no `retry-after` header to
    parse -- retryable, with retry_after_seconds left at its default None.
    """
    exc = anthropic.APIConnectionError(
        message="connection reset",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)
    assert exc_info.value.retry_after_seconds is None


async def test_api_timeout_error_is_retryable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """APITimeoutError is a subclass of APIConnectionError, not a
    separate except clause in extract_document -- confirm it's still
    caught (isinstance, not exact-type matching) and classified the same
    way as a plain connection error.
    """
    exc = anthropic.APITimeoutError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)


async def test_retry_after_http_date_form_yields_none(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retry-after in the HTTP-date form RFC 7231 also permits (rather
    than delay-seconds) must not crash int() parsing -- it degrades to
    retry_after_seconds=None instead.
    """
    exc = anthropic.RateLimitError(
        "rate limited",
        response=_fake_response(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        body=None,
    )
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert exc_info.value.retry_after_seconds is None


async def test_retry_after_absent_header_yields_none(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exc = anthropic.RateLimitError("rate limited", response=_fake_response(429), body=None)
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert exc_info.value.retry_after_seconds is None


async def test_bare_anthropic_error_is_retryable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-APIError anthropic.AnthropicError (e.g. a validation error
    from the SDK itself, not an HTTP response) must not escape
    extract_document unclassified -- the outermost `except
    anthropic.AnthropicError` branch catches it as a retryable
    ExtractionError.
    """
    exc = anthropic.AnthropicError("something the SDK itself rejected")
    _patch_client_side_effect(monkeypatch, [exc])

    with pytest.raises(ExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)


def test_build_client_passes_max_retries_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_client must forward settings.anthropic_max_retries/
    anthropic_timeout_seconds to anthropic.AsyncAnthropic explicitly,
    rather than relying on the SDK's own defaults -- the simplest honest
    check is to swap out the class itself and inspect the kwargs it was
    constructed with.
    """
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)
    settings = Settings(
        extraction_model="claude-sonnet-5",
        anthropic_max_retries=5,
        anthropic_timeout_seconds=45.0,
    )

    extraction_module._build_client(settings)

    assert captured["max_retries"] == 5
    assert captured["timeout"] == pytest.approx(45.0)


# --- schema-repair reprompt (H3) -----------------------------------------


_MALFORMED_TOOL_INPUT = {
    "vendor": "Acme",  # bare scalar instead of {value, confidence}
    "document_date": {"value": "2026-01-01", "confidence": 0.6},
    "line_items": {
        "value": [
            {
                "description": {"value": "Widget", "confidence": 0.9},
                "quantity": {"value": 2, "confidence": "not-a-number"},  # non-numeric confidence
                "unit_price": {"value": 5.0, "confidence": 0.9},
                "total": {"value": 10.0, "confidence": 0.9},
            }
        ],
        "confidence": 0.85,
    },
    "subtotal": {"value": 10.0, "confidence": 0.8},
    "tax": {},  # missing "value"
    "total": {"value": 10.0, "confidence": 0.79},
    "currency": {"value": "USD", "confidence": 1.0},
}


async def test_repair_reprompt_accepts_strictly_cleaner_response(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First response has three distinct malformed shapes (bare-scalar
    vendor, tax missing "value", a line-item quantity with a
    non-numeric confidence); the repair response is fully clean. Exactly
    one repair call must fire, its (strictly-fewer-violations) tool_input
    must be the one returned, repaired must be True, and tokens/cost must
    be the SUM of both calls.
    """
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    second_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT, tool_use_id="toolu_2")],
        usage=_FakeUsage(200, 100),
    )
    fake_client = _patch_client_side_effect(monkeypatch, [first_response, second_response])

    result = await _extract_from_tmp_file(tmp_path)

    assert fake_client.messages.create.await_count == 2
    assert result.tool_input == REALISTIC_TOOL_INPUT
    assert result.repaired is True
    assert result.schema_violations == 0
    assert result.input_tokens == 1000 + 200
    assert result.output_tokens == 500 + 100
    expected_cost = ((1000 + 200) * 2.00 + (500 + 100) * 10.00) / 1_000_000
    assert result.cost_usd == pytest.approx(expected_cost)

    # Confirm the exact repair-call request shape: same model/max_tokens/
    # system/tools/tool_choice as the first call; messages = [original
    # user turn, assistant turn reconstructed from the tool_use block's
    # id/input, user turn with a tool_result carrying is_error=True and
    # the same tool_use_id].
    first_kwargs = fake_client.messages.create.await_args_list[0].kwargs
    second_kwargs = fake_client.messages.create.await_args_list[1].kwargs
    for key in ("model", "max_tokens", "system", "tools", "tool_choice"):
        assert second_kwargs[key] == first_kwargs[key]

    repair_messages = second_kwargs["messages"]
    assert len(repair_messages) == 3
    assert repair_messages[0] == first_kwargs["messages"][0]  # original user turn, unchanged

    assistant_turn = repair_messages[1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "record_extraction",
            "input": _MALFORMED_TOOL_INPUT,
        }
    ]

    tool_result_turn = repair_messages[2]
    assert tool_result_turn["role"] == "user"
    assert len(tool_result_turn["content"]) == 1
    tool_result_block = tool_result_turn["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "toolu_1"
    assert tool_result_block["is_error"] is True
    assert isinstance(tool_result_block["content"], str) and tool_result_block["content"]


async def test_repair_reprompt_rejected_when_not_strictly_better(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair response has the SAME violation count as the first (still
    a bare-scalar vendor) -- not strictly fewer -- so the first
    response's tool_input must be kept, repaired must be False, and
    both calls' cost/tokens must still be billed (the repair call
    genuinely ran and was billed; it just wasn't accepted).
    """
    # Differs in content from _MALFORMED_TOOL_INPUT (a different bare
    # scalar for vendor) but has the SAME violation count (still a
    # malformed vendor, still a non-numeric line-item confidence, still
    # a tax leaf missing "value") -- so the tool_input equality
    # assertion below actually discriminates "first response kept"
    # from "repair wrongly accepted" instead of being vacuously true.
    no_better_repair_input = {**_MALFORMED_TOOL_INPUT, "vendor": "Widgets Inc"}
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    second_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(no_better_repair_input, tool_use_id="toolu_2")],
        usage=_FakeUsage(200, 100),
    )
    fake_client = _patch_client_side_effect(monkeypatch, [first_response, second_response])

    result = await _extract_from_tmp_file(tmp_path)

    assert fake_client.messages.create.await_count == 2
    assert result.tool_input == _MALFORMED_TOOL_INPUT
    assert result.repaired is False
    assert result.schema_violations == len(_schema_violations(_MALFORMED_TOOL_INPUT, SCHEMA_V1))
    # Both calls are billed regardless of whether the repair was accepted.
    assert result.input_tokens == 1000 + 200
    assert result.output_tokens == 500 + 100


async def test_clean_first_response_never_triggers_repair(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cost regression guard: a schema-clean first response must not
    trigger a second (billed) call.
    """
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT)],
        usage=_FakeUsage(1000, 500),
    )
    fake_client = _patch_client(monkeypatch, response)

    result = await _extract_from_tmp_file(tmp_path)

    fake_client.messages.create.assert_awaited_once()
    assert result.repaired is False
    assert result.schema_violations == 0
    assert result.input_tokens == 1000
    assert result.output_tokens == 500


async def test_repair_call_raising_falls_back_to_first_response(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair call that raises anthropic.APIError must never turn a
    usable first extraction into a failed job: the first response's
    tool_input is still returned, repaired is False, and -- since the
    repair call never returned a usage object -- only the first call's
    cost is counted. (Choice documented here: the failed call may or may
    not have been billed server-side, e.g. a connection error before any
    tokens were processed; extract_document has no way to know, so it
    doesn't guess and simply omits it from the returned totals.)
    """
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    repair_error = anthropic.APIError(
        "rate limited",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        body=None,
    )
    fake_client = _patch_client_side_effect(monkeypatch, [first_response, repair_error])

    result = await _extract_from_tmp_file(tmp_path)

    assert fake_client.messages.create.await_count == 2
    assert result.tool_input == _MALFORMED_TOOL_INPUT
    assert result.repaired is False
    assert result.input_tokens == 1000
    assert result.output_tokens == 500


async def test_schema_repair_disabled_never_makes_a_second_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(
            extraction_model="claude-sonnet-5", review_threshold=0.8, schema_repair=False
        ),
    )
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT)],
        usage=_FakeUsage(1000, 500),
    )
    fake_client = _patch_client(monkeypatch, response)

    result = await _extract_from_tmp_file(tmp_path)

    fake_client.messages.create.assert_awaited_once()
    assert result.repaired is False
    assert result.tool_input == _MALFORMED_TOOL_INPUT
    assert result.schema_violations == len(_schema_violations(_MALFORMED_TOOL_INPUT, SCHEMA_V1))


async def test_repair_reprompt_accepted_when_both_counts_exceed_truncation_cap(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for FINDING 1: a first response with a large violation
    count (32, from 8 malformed line items) and a repair that cuts it
    down to a still-large-but-smaller count (28, from 7 malformed line
    items) must be ACCEPTED -- 28 < 32. Under the bug this guards
    against, _schema_violations used to truncate its OWN return value to
    MAX_SCHEMA_VIOLATIONS (20) + 1 summary entry, so both counts would
    have collapsed to the same length (21) before the acceptance
    comparison ran, making `21 < 21` false and wrongly rejecting a
    repair that was genuinely, substantially better.
    """
    first_input = _tool_input_with_n_bad_items(8)  # 32 violations
    repair_input = _tool_input_with_n_bad_items(7)  # 28 violations -- fewer, but still > cap
    assert len(_schema_violations(first_input, SCHEMA_V1)) == 32
    assert len(_schema_violations(repair_input, SCHEMA_V1)) == 28

    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(first_input, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    second_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(repair_input, tool_use_id="toolu_2")],
        usage=_FakeUsage(200, 100),
    )
    fake_client = _patch_client_side_effect(monkeypatch, [first_response, second_response])

    result = await _extract_from_tmp_file(tmp_path)

    assert fake_client.messages.create.await_count == 2
    assert result.tool_input == repair_input
    assert result.repaired is True
    assert result.schema_violations == 28


async def test_repair_response_with_non_dict_input_falls_back_to_first_response(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINDING 2 regression: a repair call that succeeds but returns a
    tool_use block whose .input is not even a dict (a maximally
    malformed repair reply) must not raise out of _schema_violations --
    the extraction still succeeds, falling back to the first response.
    Both calls are still billed since the repair call itself succeeded
    and returned usage; it just produced an unusable input (7 "field
    missing" violations for a non-dict input is never fewer than the
    first response's 3, so it's correctly rejected too).
    """
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    second_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock("not a dict at all", tool_use_id="toolu_2")],
        usage=_FakeUsage(200, 100),
    )
    fake_client = _patch_client_side_effect(monkeypatch, [first_response, second_response])

    result = await _extract_from_tmp_file(tmp_path)  # must not raise

    assert fake_client.messages.create.await_count == 2
    assert result.tool_input == _MALFORMED_TOOL_INPUT
    assert result.repaired is False
    assert result.input_tokens == 1000 + 200
    assert result.output_tokens == 500 + 100


async def test_repair_response_refusal_falls_back_but_still_bills_tokens(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINDING 6: a repair call that comes back with a non-tool_use
    stop_reason (e.g. a refusal) must fall back to the first response
    without raising -- but unlike a repair call that raises (see
    test_repair_call_raising_falls_back_to_first_response), a refusal
    DID complete and return a usage object, so its tokens are still
    summed into the result -- the same "billed usage is never invisible"
    treatment this module already gives a first-call refusal (see the
    stop_reason != "tool_use" branch near the top of extract_document).
    """
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    refusal_response = _FakeMessage(stop_reason="refusal", content=[], usage=_FakeUsage(200, 100))
    fake_client = _patch_client_side_effect(monkeypatch, [first_response, refusal_response])

    result = await _extract_from_tmp_file(tmp_path)  # must not raise

    assert fake_client.messages.create.await_count == 2
    assert result.tool_input == _MALFORMED_TOOL_INPUT
    assert result.repaired is False
    assert result.input_tokens == 1000 + 200
    assert result.output_tokens == 500 + 100


# --- _schema_violations ----------------------------------------------------


def test_schema_violations_reports_missing_and_malformed_paths() -> None:
    tool_input = {
        "vendor": "Acme",  # malformed: bare scalar
        # document_date: missing entirely
        "line_items": {
            "value": [
                {
                    "description": {"value": "Widget", "confidence": 0.9},
                    "quantity": {"value": 2, "confidence": None},  # malformed
                    "unit_price": {"value": 5.0, "confidence": 0.9},
                    "total": {"value": 10.0, "confidence": 0.9},
                },
                "not-a-dict",  # malformed item -- all 4 sub-leaves malformed
            ],
            "confidence": 0.85,
        },
        "subtotal": {"value": 10.0, "confidence": 0.8},
        "tax": {"value": 0.0, "confidence": 0.99},
        "total": {"value": 10.0, "confidence": 0.79},
        "currency": {"value": "USD", "confidence": 1.0},
    }

    violations = _schema_violations(tool_input, SCHEMA_V1)

    assert "vendor" in violations
    assert "document_date" in violations
    assert "line_items.value[0].quantity" in violations
    for sub_field in ("description", "quantity", "unit_price", "total"):
        assert f"line_items.value[1].{sub_field}" in violations
    # well-formed leaves are not reported
    for clean in (
        "line_items.value[0].description",
        "line_items.value[0].unit_price",
        "line_items.value[0].total",
        "subtotal",
        "tax",
        "total",
        "currency",
    ):
        assert clean not in violations
    assert len(violations) == 2 + 1 + 4  # vendor, document_date, quantity[0], all of item[1]


def test_schema_violations_reports_nothing_for_realistic_input() -> None:
    assert _schema_violations(REALISTIC_TOOL_INPUT, SCHEMA_V1) == []


def _tool_input_with_n_bad_items(n: int) -> dict:
    """Build a tool_input whose only violations are `n` line items, each
    with all 4 sub-fields malformed (a bare string instead of a
    {value, confidence} dict) -- i.e. exactly 4*n violations, all from
    line_items.
    """
    items = [
        {"description": "bad", "quantity": "bad", "unit_price": "bad", "total": "bad"}
        for _ in range(n)
    ]
    return {
        "vendor": {"value": "Acme", "confidence": 0.9},
        "document_date": {"value": "2026-01-01", "confidence": 0.9},
        "line_items": {"value": items, "confidence": 0.9},
        "subtotal": {"value": 1.0, "confidence": 0.9},
        "tax": {"value": 1.0, "confidence": 0.9},
        "total": {"value": 1.0, "confidence": 0.9},
        "currency": {"value": "USD", "confidence": 0.9},
    }


def test_schema_violations_does_not_truncate_the_returned_list() -> None:
    """Regression for FINDING 1: _schema_violations must return the FULL
    list, not a truncated one -- both the repair-acceptance comparison
    and the returned schema_violations count in extract_document depend
    on the true count. 10 items x 4 sub-fields = 40 violations, all
    returned untruncated (no "… (N more)" summary entry here -- that
    only appears in the repair PROMPT TEXT, built by
    _format_violations_for_prompt).
    """
    violations = _schema_violations(_tool_input_with_n_bad_items(10), SCHEMA_V1)

    assert len(violations) == 40
    assert violations == [
        f"line_items.value[{item_idx}].{sub_field}"
        for item_idx in range(10)
        for sub_field in ("description", "quantity", "unit_price", "total")
    ]


def test_schema_violations_defensive_against_non_dict_tool_input() -> None:
    """FINDING 2: a non-dict tool_input (the model returning something
    other than an object at all, e.g. from a malformed repair response)
    must degrade to "every top-level field missing" rather than raising
    AttributeError out of `tool_input.get(...)`.
    """
    for bad_input in (None, "not a dict", [], 42):
        assert _schema_violations(bad_input, SCHEMA_V1) == list(SCHEMA_V1.field_names)


def test_format_violations_for_prompt_truncates_at_20_with_summary_entry() -> None:
    """Truncation (MAX_SCHEMA_VIOLATIONS + "… (N more)") lives ONLY in
    _format_violations_for_prompt, the repair-prompt-text renderer -- not
    in _schema_violations itself (see FINDING 1 regression test above).
    """
    violations = _schema_violations(_tool_input_with_n_bad_items(10), SCHEMA_V1)  # 40 violations
    assert len(violations) == 40

    prompt_text = _format_violations_for_prompt(violations)
    lines = prompt_text.splitlines()

    assert len(lines) == 21  # 20 kept + 1 trailing summary line
    assert lines[-1] == "- … (20 more)"
    assert lines[:20] == [
        f"- line_items.value[{item_idx}].{sub_field}"
        for item_idx in range(5)
        for sub_field in ("description", "quantity", "unit_price", "total")
    ]


def test_format_violations_for_prompt_no_truncation_under_the_cap() -> None:
    violations = ["vendor", "tax"]
    assert _format_violations_for_prompt(violations) == "- vendor\n- tax"


# --- _leaf_is_malformed / _coerce_leaf consistency --------------------------


@pytest.mark.parametrize(
    "leaf",
    [
        "Acme",  # bare scalar
        123,
        None,
        [],
        {},  # missing "value"
        {"confidence": 0.5},  # missing "value"
        {"value": "x"},  # missing "confidence" -> float(None) raises
        {"value": "x", "confidence": "not-a-number"},
        {"value": "x", "confidence": float("nan")},
        {"value": "x", "confidence": float("inf")},
        {"value": None, "confidence": "also-not-a-number"},
        # FIX 2 regression: an oversized int literal (valid JSON) is too
        # large for float() to represent -- float(10**400) raises
        # OverflowError, which must degrade the leaf the same as any
        # other non-numeric confidence, not escape unclassified.
        {"value": "x", "confidence": 10**400},
    ],
)
def test_leaf_is_malformed_implies_coerce_leaf_degrades(leaf: object) -> None:
    """Property: wherever _leaf_is_malformed(x) is True, _coerce_leaf(x)
    must degrade to (None, 0.0) -- the reject rule lives in exactly one
    place and both consult it, so they can never disagree. (The converse
    does NOT hold in general -- see the next test.)
    """
    assert _leaf_is_malformed(leaf) is True
    assert _coerce_leaf(leaf) == (None, 0.0)


def test_leaf_is_malformed_false_for_well_formed_null_value() -> None:
    """A VALID leaf can legitimately hold value=None (the prompt's
    documented way to mark an absent/illegible field) -- _leaf_is_malformed
    must NOT flag it, even though _coerce_leaf's output for it also
    happens to have value=None (with the given confidence, not forced to
    0.0, since the leaf itself is well-formed).
    """
    leaf = {"value": None, "confidence": 0.3}
    assert _leaf_is_malformed(leaf) is False
    assert _coerce_leaf(leaf) == (None, 0.3)


async def _jobs_for(db_session: AsyncSession, document_id) -> list[Job]:
    return list(
        (await db_session.execute(select(Job).where(Job.document_id == document_id)))
        .scalars()
        .all()
    )


async def test_successful_extraction_enqueues_an_index_job(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    index_jobs = [j for j in await _jobs_for(db_session, document.id) if j.kind == "index"]
    assert len(index_jobs) == 1
    assert index_jobs[0].state == "pending"


async def test_index_after_extraction_off_enqueues_nothing(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-sonnet-5", index_after_extraction=False),
    )
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    assert [j.kind for j in await _jobs_for(db_session, document.id)] == ["extract"]


async def test_failed_extraction_enqueues_no_index_job(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, _FakeMessage("refusal", [], _FakeUsage(100, 0)))
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    with pytest.raises(ModelRefusalError):
        await process_document_job(db_session, job)
    await db_session.rollback()

    assert [j.kind for j in await _jobs_for(db_session, document.id)] == ["extract"]


# ---- Reconciliation at persistence ----------------------------------------------


def _bakery_tool_input(subtotal: float) -> dict:
    """Lines 36.60 + 8.65 = 45.25, tax 3.05, total 48.30; the subtotal is the variable."""

    def cell(value, confidence=0.99):
        return {"value": value, "confidence": confidence}

    rows = [
        {
            "description": cell("Sourdough"),
            "quantity": cell(4),
            "unit_price": cell(9.15),
            "total": cell(36.6),
        },
        {
            "description": cell("Rolls"),
            "quantity": cell(1),
            "unit_price": cell(8.65),
            "total": cell(8.65),
        },
    ]
    return {
        "vendor": cell("Bakery"),
        "document_date": cell("2025-03-01"),
        "line_items": {"value": rows, "confidence": 0.99},
        "subtotal": cell(subtotal, 0.9),
        "tax": cell(3.05),
        "total": cell(48.3),
        "currency": cell("USD"),
    }


async def _reconciliation_of(db_session: AsyncSession, document_id) -> Reconciliation | None:
    return (
        await db_session.execute(
            select(Reconciliation)
            .join(Extraction, Reconciliation.extraction_id == Extraction.id)
            .where(Extraction.document_id == document_id)
        )
    ).scalar_one_or_none()


async def test_a_consistent_extraction_is_stored_with_a_passing_reconciliation(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    row = await _reconciliation_of(db_session, document.id)
    assert row is not None
    assert (row.status, row.rules_version) == ("pass", "rules_v2")
    fields = (await db_session.execute(select(ExtractedField))).scalars()
    reasons = {f.field_name: f.review_reason for f in fields if f.extraction_id == row.extraction_id}
    # Only the model's own doubts are flagged: document_date (0.6) and total (0.79).
    assert {n: r for n, r in reasons.items() if r} == {
        "document_date": "confidence",
        "total": "confidence",
    }


async def test_a_failing_reconciliation_flags_the_suspect_field_at_persistence(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(_bakery_tool_input(46.25))], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    row = await _reconciliation_of(db_session, document.id)
    assert row is not None and row.status == "fail"
    assert row.report["suggestion"]["path"] == "subtotal"
    fields = (
        await db_session.execute(
            select(ExtractedField).where(ExtractedField.extraction_id == row.extraction_id)
        )
    ).scalars()
    flagged = {f.field_name: f.review_reason for f in fields if f.needs_review}
    assert flagged == {"subtotal": "reconciliation:lines_sum"}


async def test_a_reconciliation_failure_never_loses_the_billed_extraction(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    async def explode(*args, **kwargs):
        raise RuntimeError("rules bug")

    monkeypatch.setattr("app.review_routing.refresh_reconciliation", explode)
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    with caplog.at_level("ERROR", logger="app.extraction"):
        await process_document_job(db_session, job)
    await db_session.flush()

    # The extraction, its fields (with their confidence flags), and the index
    # job are all there; only the reconciliation is missing, and it was logged.
    assert await _extraction_count(db_session, document.id) == 1
    assert await _reconciliation_of(db_session, document.id) is None
    await db_session.refresh(document)
    assert document.status == "extracted"
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalar_one()
    fields = (
        await db_session.execute(
            select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
        )
    ).scalars()
    assert {f.field_name for f in fields if f.review_reason == "confidence"} == {
        "document_date",
        "total",
    }
    assert [j.kind for j in await _jobs_for(db_session, document.id)].count("index") == 1
    assert "reconciliation failed" in caplog.text


async def test_a_database_error_in_the_reconciliation_rolls_back_only_the_reconciliation(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def break_the_transaction(session, extraction_id, settings):
        # A write the database refuses, inside the savepoint the job opened.
        await session.execute(
            update(ExtractedField)
            .where(ExtractedField.extraction_id == extraction_id)
            .values(needs_review=True, review_reason=None)
        )

    monkeypatch.setattr("app.review_routing.refresh_reconciliation", break_the_transaction)
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    assert await _extraction_count(db_session, document.id) == 1
    field_count = (
        await db_session.execute(
            select(func.count())
            .select_from(ExtractedField)
            .join(Extraction, ExtractedField.extraction_id == Extraction.id)
            .where(Extraction.document_id == document.id)
        )
    ).scalar_one()
    assert field_count == len(SCHEMA_V1.field_names)
    assert await _reconciliation_of(db_session, document.id) is None
    await db_session.refresh(document)
    assert document.status == "extracted"


async def _assert_extraction_kept_with_confidence_flags(
    db_session: AsyncSession, document: Document
) -> None:
    assert await _extraction_count(db_session, document.id) == 1
    assert await _reconciliation_of(db_session, document.id) is None
    await db_session.refresh(document)
    assert document.status == "extracted"
    extraction_id = (
        await db_session.execute(select(Extraction.id).where(Extraction.document_id == document.id))
    ).scalar_one()
    fields = (
        await db_session.execute(
            select(ExtractedField).where(ExtractedField.extraction_id == extraction_id)
        )
    ).scalars().all()
    assert len(fields) == len(SCHEMA_V1.field_names)
    assert {f.field_name: f.review_reason for f in fields if f.needs_review} == {
        "document_date": "confidence",
        "total": "confidence",
    }


async def test_a_routing_error_after_fields_were_touched_keeps_the_extraction(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def route_then_fail(fields, report, review_threshold):
        for field in fields:
            field.needs_review = True
            field.review_reason = "reconciliation:lines_sum"
        raise RuntimeError("routing bug")

    monkeypatch.setattr("app.review_routing.route", route_then_fail)
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    await _assert_extraction_kept_with_confidence_flags(db_session, document)


async def test_an_inconsistent_routing_trips_the_check_inside_the_savepoint(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flag_without_a_reason(fields, report, review_threshold):
        return {f.field_name: Routing(True, None) for f in fields}

    monkeypatch.setattr("app.review_routing.route", flag_without_a_reason)
    _patch_client(
        monkeypatch,
        _FakeMessage("tool_use", [_FakeToolUseBlock(REALISTIC_TOOL_INPUT)], _FakeUsage(100, 50)),
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    await _assert_extraction_kept_with_confidence_flags(db_session, document)


# --- schema v2: evidence, new fields, and the card-number rule (B4) ---------


def _use_prompt_version(monkeypatch: pytest.MonkeyPatch, prompt_version: str) -> None:
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(
            extraction_model="claude-sonnet-5",
            review_threshold=0.8,
            extraction_prompt_version=prompt_version,
        ),
    )


def _v2_response(tool_input: dict) -> _FakeMessage:
    return _FakeMessage("tool_use", [_FakeToolUseBlock(tool_input)], _FakeUsage(1000, 500))


async def _persist(db_session: AsyncSession, tmp_path, monkeypatch, tool_input: dict, **document):
    """Run process_document_job on a fake v2 response; returns the stored
    extraction and its fields by name. `document` overrides the uploaded file
    (see _make_document); the default is a PNG."""
    _use_prompt_version(monkeypatch, "extract_v2")
    _patch_client(monkeypatch, _v2_response(tool_input))
    document = await _make_document(db_session, tmp_path, **document)
    job = await _make_job(db_session, document.id)
    await process_document_job(db_session, job)
    await db_session.flush()
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    fields = (
        await db_session.execute(
            select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
        )
    ).scalars().all()
    return extraction, {f.field_name: f for f in fields}


def _all_leaves(tool: dict):
    """Every leaf of the v2 tool schema: the top-level ones and each array
    cell."""
    for prop in tool["input_schema"]["properties"].values():
        yield prop
        items = prop["properties"]["value"].get("items")
        if items:
            yield from items["properties"].values()


def test_the_v2_tool_asks_every_leaf_for_evidence() -> None:
    tool = record_extraction_tool(SCHEMA_V2)
    schema = tool["input_schema"]

    assert list(schema["properties"]) == list(SCHEMA_V2.field_names)
    assert schema["required"] == list(SCHEMA_V2.field_names)
    assert len(SCHEMA_V2.field_names) == 20
    leaves = list(_all_leaves(tool))
    assert len(leaves) == 20 + 4 + 4  # fields, line item cells, tax line cells
    wrappers = [schema["properties"][name] for name in ("line_items", "tax_lines")]
    for each in wrappers:
        # An array's own leaf has no evidence: its cells cite their own quotes.
        assert each["required"] == ["value", "confidence"]
        assert "evidence" not in each["properties"]
    for each in (lf for lf in leaves if not any(lf is w for w in wrappers)):
        assert each["required"] == ["value", "confidence", "evidence"]
        assert each["properties"]["evidence"] == {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "quote": {"type": "string", "maxLength": 200},
                    },
                    "required": ["page", "quote"],
                },
                {"type": "null"},
            ]
        }
    cells = schema["properties"]["tax_lines"]["properties"]["value"]["items"]
    assert list(cells["properties"]) == ["label", "rate", "base", "amount"]
    assert cells["properties"]["rate"]["properties"]["value"] == {"type": ["number", "null"]}
    assert SCHEMA_V2.max_tokens == 16384
    assert SCHEMA_V2.evidence is True
    assert SCHEMA_V2.prompt_version == "extract_v2"
    assert SCHEMA_V2.repair_prompt_version == "repair_v2"


def test_the_v1_tool_has_no_evidence_anywhere() -> None:
    assert "evidence" not in json.dumps(record_extraction_tool(SCHEMA_V1))


async def test_a_v2_extraction_is_requested_and_persisted_with_evidence(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_prompt_version(monkeypatch, "extract_v2")
    fake_client = _patch_client(monkeypatch, _v2_response(v2_tool_input()))
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    kwargs = fake_client.messages.create.await_args.kwargs
    assert kwargs["tools"] == [record_extraction_tool(SCHEMA_V2)]
    assert kwargs["system"] == SCHEMA_V2.prompt_text
    assert kwargs["max_tokens"] == 16384
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_extraction"}

    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    assert extraction.prompt_version == "extract_v2"
    assert extraction.raw_response == v2_tool_input()
    fields = {
        f.field_name: f
        for f in (
            await db_session.execute(
                select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
            )
        ).scalars()
    }
    assert set(fields) == set(SCHEMA_V2.field_names)
    assert len(fields) == 20

    # Scalar leaves keep their evidence beside value and confidence.
    assert fields["subtotal"].value == {
        "value": 100.0,
        "confidence": 0.95,
        "evidence": {"page": 1, "quote": "Subtotal $100.00"},
    }
    # A null value stores null evidence.
    assert fields["tip"].value == {"value": None, "confidence": 0.95, "evidence": None}
    assert fields["payment_card_last4"].value["value"] == "4242"
    # Arrays store their rows cell by cell, each cell a leaf with its own evidence.
    assert fields["tax_lines"].value == [
        {
            "label": {
                "value": "State tax",
                "confidence": 0.95,
                "evidence": {"page": 1, "quote": "State tax 8% on 90.00 7.20"},
            },
            "rate": {
                "value": 8,
                "confidence": 0.95,
                "evidence": {"page": 1, "quote": "State tax 8% on 90.00 7.20"},
            },
            "base": {
                "value": 90.0,
                "confidence": 0.95,
                "evidence": {"page": 1, "quote": "State tax 8% on 90.00 7.20"},
            },
            "amount": {
                "value": 7.2,
                "confidence": 0.95,
                "evidence": {"page": 1, "quote": "State tax 8% on 90.00 7.20"},
            },
        }
    ]
    line_cells = fields["line_items"].value[0]
    assert line_cells["total"]["evidence"] == {"page": 1, "quote": "Widget 100.00"}
    assert fields["tax_lines"].confidence == pytest.approx(0.9)
    # The consistent document passes every check, quote checks included.
    row = await _reconciliation_of(db_session, document.id)
    assert row is not None and row.status == "pass"
    assert any(check["kind"] == "quote" for check in row.report["checks"])


@pytest.mark.parametrize(
    "bad_evidence",
    [
        {"page": "1", "quote": "Total $5.00"},
        {"page": 0, "quote": "Total $5.00"},
        {"page": -2, "quote": "Total $5.00"},
        {"page": True, "quote": "Total $5.00"},
        {"page": 1.0, "quote": "Total $5.00"},
        {"page": 10**30, "quote": "Total $5.00"},
        {"page": 10_001, "quote": "Total $5.00"},
        {"page": 1, "quote": "x" * 201},
        {"page": 1, "quote": "   "},
        {"page": 1, "quote": 5},
        {"page": 1, "quote": None},
        {"page": 1},
        {"quote": "Total $5.00"},
        "Total $5.00",
        ["Total $5.00"],
        None,
    ],
)
def test_bad_evidence_becomes_none(bad_evidence) -> None:
    assert _coerce_evidence({"value": 5, "confidence": 0.9, "evidence": bad_evidence}) is None


def test_evidence_with_no_page_keeps_its_quote() -> None:
    assert _coerce_evidence({"evidence": {"page": None, "quote": " Total $5.00 "}}) == {
        "page": None,
        "quote": "Total $5.00",
    }


def test_good_evidence_is_kept_with_the_quote_stripped() -> None:
    kept = _coerce_evidence({"evidence": {"page": 3, "quote": "  Total $5.00 \n", "extra": 1}})

    assert kept == {"page": 3, "quote": "Total $5.00"}
    assert _coerce_evidence({"evidence": {"page": 1, "quote": "x" * 200}}) is not None
    assert _coerce_evidence({"evidence": {"page": 10_000, "quote": "x"}}) is not None
    assert _coerce_evidence({"value": 5}) is None
    assert _coerce_evidence("not a leaf") is None


async def test_bad_evidence_is_stored_as_none_not_as_a_violation(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = v2_tool_input()
    tool_input["vendor"]["evidence"] = {"page": 1, "quote": "x" * 500}
    tool_input["subtotal"]["evidence"] = {"page": 1, "quote": "   "}
    tool_input["total"]["evidence"] = {"page": 1}
    tool_input["tax"]["evidence"] = {"page": 1, "quote": 7.2}
    tool_input["tax_lines"]["value"][0]["amount"]["evidence"] = "7.20"

    assert _schema_violations(tool_input, SCHEMA_V2) == []
    _, fields = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    for name in ("vendor", "subtotal", "total", "tax"):
        assert fields[name].value["evidence"] is None
        assert fields[name].value["value"] == tool_input[name]["value"]
    assert fields["tax_lines"].value[0]["amount"]["evidence"] is None
    assert fields["tax_lines"].value[0]["amount"]["value"] == 7.2


# --- evidence pages through the single-call path (B5) -----------------------
#
# A bad page is not a bad quote: the quote is still printed text the checks can
# read, so only the page is dropped. An image is one page, so its evidence is on
# page 1 whatever the model wrote.


async def test_an_image_stores_every_evidence_page_as_1(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = v2_tool_input()
    tool_input["vendor"]["evidence"]["page"] = 7
    tool_input["subtotal"]["evidence"]["page"] = "3"
    tool_input["total"]["evidence"]["page"] = None
    tool_input["tax"]["evidence"]["page"] = True
    tool_input["amount_paid"]["evidence"]["page"] = 0
    tool_input["tax_lines"]["value"][0]["amount"]["evidence"]["page"] = 4
    tool_input["line_items"]["value"][0]["total"]["evidence"]["page"] = 2

    extraction, fields = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    for name in ("vendor", "subtotal", "total", "tax", "amount_paid"):
        assert fields[name].value["evidence"]["page"] == 1
        assert extraction.raw_response[name]["evidence"]["page"] == 1
    assert fields["vendor"].value["evidence"]["quote"] == "Acme Corp"
    assert fields["tax_lines"].value[0]["amount"]["evidence"]["page"] == 1
    assert fields["line_items"].value[0]["total"]["evidence"]["page"] == 1
    # A null evidence stays null: there is no quote to put on a page.
    assert fields["tip"].value["evidence"] is None
    assert extraction.raw_response["tip"]["evidence"] is None


def _pdf_of_pages(count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(count):
        writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


async def test_a_single_call_pdf_drops_a_page_it_does_not_have_and_keeps_the_quote(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_prompt_version(monkeypatch, "extract_v2")
    tool_input = v2_tool_input()
    tool_input["vendor"]["evidence"]["page"] = 2
    tool_input["subtotal"]["evidence"]["page"] = 9
    tool_input["discount"]["evidence"]["page"] = 0
    tool_input["shipping"]["evidence"]["page"] = "1"
    tool_input["tax"]["evidence"]["page"] = True
    tool_input["total"]["evidence"]["page"] = None
    tool_input["tax_lines"]["value"][0]["amount"]["evidence"]["page"] = 9
    tool_input["tax_lines"]["value"][0]["base"]["evidence"]["page"] = 2
    _patch_client(monkeypatch, _v2_response(tool_input))
    document = await _make_document(
        db_session, tmp_path, filename="two.pdf", mime_type="application/pdf", content=_pdf_of_pages(2)
    )
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    fields = {
        f.field_name: f
        for f in (
            await db_session.execute(
                select(ExtractedField).where(ExtractedField.extraction_id == extraction.id)
            )
        ).scalars()
    }
    assert fields["vendor"].value["evidence"] == {"page": 2, "quote": "Acme Corp"}
    assert fields["subtotal"].value["evidence"] == {"page": None, "quote": "Subtotal $100.00"}
    for name in ("discount", "shipping", "tax", "total"):
        assert fields[name].value["evidence"]["page"] is None
        assert fields[name].value["evidence"]["quote"] == tool_input[name]["evidence"]["quote"]
    cells = fields["tax_lines"].value[0]
    assert cells["amount"]["evidence"] == {"page": None, "quote": "State tax 8% on 90.00 7.20"}
    assert cells["base"]["evidence"]["page"] == 2
    assert extraction.raw_response["subtotal"]["evidence"] == {
        "page": None,
        "quote": "Subtotal $100.00",
    }
    # The quotes still read: a page-less quote is checked like any other.
    reconciliation = (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction.id)
        )
    ).scalar_one()
    statuses = {c["id"]: c["status"] for c in reconciliation.report["checks"] if c["kind"] == "quote"}
    assert statuses["quote:subtotal"] == "pass"
    assert "skipped" not in statuses.values()


@pytest.mark.parametrize("bad_page", [0, -1, "1", 9, True, 1.5, None])
async def test_a_payment_field_stores_the_same_evidence_page_as_any_other_field(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch, bad_page
) -> None:
    """The card rule rebuilds the payment leaves; it must not judge the page
    itself, or a bad page would cost them their quote and not the others."""

    def with_bad_page() -> dict:
        tool_input = v2_tool_input()
        for name in ("vendor", "payment_method", "payment_card_last4"):
            tool_input[name]["evidence"]["page"] = bad_page
        return tool_input

    _, on_image = await _persist(db_session, tmp_path, monkeypatch, with_bad_page())
    for name in ("payment_method", "payment_card_last4"):
        assert on_image[name].value["evidence"]["page"] == 1
        assert on_image[name].value["evidence"]["quote"] == v2_tool_input()[name]["evidence"]["quote"]
        assert on_image[name].value["evidence"] == {
            **on_image["vendor"].value["evidence"],
            "quote": v2_tool_input()[name]["evidence"]["quote"],
        }

    _, on_pdf = await _persist(
        db_session,
        tmp_path,
        monkeypatch,
        with_bad_page(),
        filename="two.pdf",
        mime_type="application/pdf",
        content=_pdf_of_pages(2),
    )
    for name in ("payment_method", "payment_card_last4"):
        assert on_pdf[name].value["evidence"]["page"] is None
        assert on_pdf[name].value["evidence"]["quote"] == v2_tool_input()[name]["evidence"]["quote"]
    assert on_pdf["vendor"].value["evidence"]["page"] is None


async def test_the_repair_call_echoes_what_the_model_wrote_and_the_repaired_page_is_fixed(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_prompt_version(monkeypatch, "extract_v2")
    first = v2_tool_input()
    first["vendor"]["evidence"]["page"] = 7
    first["po_number"] = "PO-77"  # a bare scalar: one violation
    repaired = v2_tool_input()
    repaired["vendor"]["evidence"] = {"page": 9, "quote": "Acme Corp (repaired)"}
    fake_client = _patch_client_side_effect(
        monkeypatch,
        [
            _FakeMessage("tool_use", [_FakeToolUseBlock(first, tool_use_id="toolu_1")], _FakeUsage(10, 5)),
            _FakeMessage("tool_use", [_FakeToolUseBlock(repaired, tool_use_id="toolu_2")], _FakeUsage(10, 5)),
        ],
    )
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    assert fake_client.messages.create.await_count == 2
    echoed = fake_client.messages.create.await_args_list[1].kwargs["messages"][1]["content"][0]["input"]
    assert echoed["vendor"]["evidence"]["page"] == 7
    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    # The stored extraction is the repaired one, with its page fixed.
    assert extraction.raw_response["vendor"]["evidence"] == {
        "page": 1,
        "quote": "Acme Corp (repaired)",
    }
    assert extraction.raw_response["po_number"]["evidence"]["page"] == 1


async def test_a_malformed_v2_leaf_loses_its_evidence_with_its_value(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = v2_tool_input()
    tool_input["vendor"] = {"value": "Acme", "evidence": {"page": 1, "quote": "Acme"}}  # no confidence
    tool_input["po_number"] = "PO-77"  # a bare scalar
    tool_input["tax_lines"]["value"][0]["rate"] = {"confidence": 0.9, "evidence": {"page": 1, "quote": "8%"}}

    # The repair call returns the same broken input, so it is not accepted.
    _, fields = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    assert fields["vendor"].value == {"value": None, "confidence": 0.0, "evidence": None}
    assert fields["po_number"].value == {"value": None, "confidence": 0.0, "evidence": None}
    assert fields["tax_lines"].value[0]["rate"] == {"value": None, "confidence": 0.0, "evidence": None}
    assert fields["vendor"].needs_review is True


def test_v2_schema_violations_walk_the_tax_line_cells() -> None:
    tool_input = v2_tool_input()
    del tool_input["tax_lines"]["value"][0]["base"]
    tool_input["tax_lines"]["value"].append("not a row")
    tool_input["line_items"]["value"][0]["total"] = {"value": 1}

    assert _schema_violations(tool_input, SCHEMA_V2) == [
        "line_items.value[0].total",
        "tax_lines.value[0].base",
        "tax_lines.value[1].label",
        "tax_lines.value[1].rate",
        "tax_lines.value[1].base",
        "tax_lines.value[1].amount",
    ]
    # v1 never looks at tax_lines.
    assert _schema_violations(tool_input, SCHEMA_V1) == ["line_items.value[0].total"] + [
        name for name in SCHEMA_V1.field_names if name not in tool_input
    ]


def test_missing_evidence_is_not_a_violation() -> None:
    tool_input = v2_tool_input()
    for each in tool_input.values():
        each.pop("evidence")
    for row in tool_input["tax_lines"]["value"] + tool_input["line_items"]["value"]:
        for cell in row.values():
            cell.pop("evidence")

    assert _schema_violations(tool_input, SCHEMA_V2) == []


# PII: the prompt asks for the last four digits only; the code makes it so.


def _card_input(*, method="VISA", method_quote="Paid by VISA", last4="4242", last4_quote="ending 4242"):
    tool_input = v2_tool_input()
    tool_input["payment_method"] = leaf(method, method_quote)
    tool_input["payment_card_last4"] = leaf(last4, last4_quote)
    return tool_input


def test_mask_payment_fields_keeps_only_the_last_four_digits() -> None:
    masked = mask_payment_fields(
        _card_input(
            method="VISA 4111 1111 1111 1111",
            method_quote="VISA 4111 1111 1111 1111 exp 12/29",
            last4="4111111111111111",
            last4_quote="Card: 4111-1111-1111-1111",
        )
    )

    assert masked["payment_method"]["value"] == "VISA ****1111"
    assert masked["payment_method"]["evidence"]["quote"] == "VISA ****1111 exp 12/29"
    assert masked["payment_card_last4"]["value"] == "1111"
    assert masked["payment_card_last4"]["evidence"] == {"page": 1, "quote": "Card: ****1111"}
    assert masked["payment_method"]["confidence"] == 0.95


def test_mask_payment_fields_leaves_alone_what_is_not_a_card_number() -> None:
    tool_input = _card_input(method="Mastercard", last4="0042", last4_quote="ending 0042")

    assert mask_payment_fields(tool_input) == tool_input
    # Four digits are not a run to mask, even inside a quote.
    assert mask_payment_fields(_card_input(last4_quote="XXXX XXXX XXXX 4242"))[
        "payment_card_last4"
    ]["evidence"]["quote"] == "XXXX XXXX XXXX 4242"


@pytest.mark.parametrize(
    ("last4", "expected"),
    [
        ("123", None),
        ("", None),
        ("abcd", None),
        (None, None),
        (4242, "4242"),
        ("**** 4242", "4242"),
        ("12345", "2345"),
        (True, None),
        (["4242"], None),
    ],
)
def test_payment_card_last4_is_cut_to_four_digits_or_null(last4, expected) -> None:
    masked = mask_payment_fields(_card_input(last4=last4))

    assert masked["payment_card_last4"]["value"] == expected


def test_mask_payment_fields_does_not_change_its_input_or_other_fields() -> None:
    tool_input = _card_input(method="VISA 4111111111111111", last4="4111111111111111")
    tool_input["invoice_number"] = leaf("0004111111111111", "Invoice 0004111111111111")
    before = copy.deepcopy(tool_input)

    masked = mask_payment_fields(tool_input)

    assert tool_input == before
    assert masked["invoice_number"] == before["invoice_number"]
    # Nothing to mask (v1 has no payment fields): the input comes back as it is.
    v1 = REALISTIC_TOOL_INPUT
    assert mask_payment_fields(v1) is v1
    assert mask_payment_fields("not a dict") == "not a dict"
    # A payment field that is not a leaf is dropped, which is the same violation
    # as a bare scalar, and its text never reaches the stored answer.
    assert mask_payment_fields({"payment_method": "VISA 4111111111111111"}) == {
        "payment_method": None
    }


async def test_a_card_number_never_reaches_the_stored_extraction(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = _card_input(
        method="VISA 4111 1111 1111 1111",
        method_quote="VISA 4111 1111 1111 1111",
        last4="4111111111111111",
        last4_quote="Card number 4111111111111111",
    )
    tool_input["invoice_number"] = leaf("0004111111111111", "Invoice # 0004111111111111")

    extraction, fields = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    assert fields["payment_method"].value["value"] == "VISA ****1111"
    assert fields["payment_method"].value["evidence"]["quote"] == "VISA ****1111"
    assert fields["payment_card_last4"].value["value"] == "1111"
    assert fields["payment_card_last4"].value["evidence"]["quote"] == "Card number ****1111"
    # The invoice number is not a card number and stays whole.
    assert fields["invoice_number"].value["value"] == "0004111111111111"
    # Stored whole in raw_response, and masked there too: the only 13+ digit
    # run left is the invoice number.
    raw = json.dumps(extraction.raw_response)
    assert raw.count("4111111111111111") == 0
    assert raw.count("0004111111111111") == 2
    assert not re.search(r"\d{13,}", raw.replace("0004111111111111", ""))
    assert extraction.raw_response["payment_method"]["value"] == "VISA ****1111"
    assert extraction.raw_response["payment_card_last4"]["value"] == "1111"


async def test_no_card_number_survives_anywhere_in_the_raw_response(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = _card_input(
        method="VISA 4111 1111 1111 1111",
        method_quote="VISA 4111 1111 1111 1111",
        last4="4111111111111111",
        last4_quote="4111111111111111",
    )

    extraction, _ = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    assert not re.search(r"\d{13,}", json.dumps(extraction.raw_response))


async def test_a_repair_response_is_masked_before_it_is_accepted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = v2_tool_input()
    broken["vendor"] = "Acme"  # a violation, so a repair call fires
    broken["payment_card_last4"] = leaf("4111111111111111", "4111 1111 1111 1111")
    repaired = _card_input(
        method="VISA 4111111111111111", last4="4111111111111111", last4_quote="4111 1111 1111 1111"
    )
    fake_client = _patch_client_side_effect(
        monkeypatch, [_v2_response(broken), _v2_response(repaired)]
    )
    _use_prompt_version(monkeypatch, "extract_v2")

    path = tmp_path / "receipt.png"
    path.write_bytes(TINY_PNG)
    result = await extract_document(path, "image/png")

    assert result.repaired is True
    assert fake_client.messages.create.await_count == 2
    # The tool_input echoed back to the model in the repair call is masked too.
    echoed = fake_client.messages.create.await_args_list[1].kwargs["messages"][1]["content"][0]["input"]
    assert echoed["payment_card_last4"]["value"] == "1111"
    assert not re.search(r"\d{13,}", json.dumps(echoed))
    assert result.tool_input["payment_card_last4"]["value"] == "1111"
    assert result.tool_input["payment_method"]["value"] == "VISA ****1111"
    assert not re.search(r"\d{13,}", json.dumps(result.tool_input))


async def test_a_v1_extraction_is_unchanged_by_the_masking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, _v2_response(REALISTIC_TOOL_INPUT))

    result = await _extract_from_tmp_file(tmp_path)

    assert result.tool_input is REALISTIC_TOOL_INPUT
    assert result.prompt_version == "extract_v1"


# The worker handler picks the schema from EXTRACTION_PROMPT_VERSION.


async def test_the_default_prompt_version_stays_extract_v1(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Settings().extraction_prompt_version == "extract_v1"
    _patch_client(monkeypatch, _v2_response(REALISTIC_TOOL_INPUT))
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    assert extraction.prompt_version == "extract_v1"
    count = (
        await db_session.execute(
            select(func.count())
            .select_from(ExtractedField)
            .where(ExtractedField.extraction_id == extraction.id)
        )
    ).scalar_one()
    assert count == 7


async def test_the_worker_stores_twenty_fields_under_extract_v2(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Settings reads EXTRACTION_PROMPT_VERSION from the environment, as the
    # compose file sets it for the worker.
    monkeypatch.setenv("EXTRACTION_PROMPT_VERSION", "extract_v2")
    monkeypatch.setattr(
        extraction_module,
        "get_settings",
        lambda: Settings(extraction_model="claude-sonnet-5", review_threshold=0.8),
    )
    _patch_client(monkeypatch, _v2_response(v2_tool_input()))
    document = await _make_document(db_session, tmp_path)
    job = await _make_job(db_session, document.id)

    await process_document_job(db_session, job)
    await db_session.flush()

    extraction = (
        await db_session.execute(select(Extraction).where(Extraction.document_id == document.id))
    ).scalars().one()
    names = (
        await db_session.execute(
            select(ExtractedField.field_name).where(ExtractedField.extraction_id == extraction.id)
        )
    ).scalars().all()
    assert extraction.prompt_version == "extract_v2"
    assert sorted(names) == sorted(SCHEMA_V2.field_names)
    assert len(names) == 20


# The card-number rule against shapes a prompt would never produce.

_LEAK = "4111111111111111"


def _assert_no_card_number(extraction: Extraction) -> None:
    assert not re.search(r"\d{13,}", json.dumps(extraction.raw_response))


@pytest.mark.parametrize(
    "payment_method",
    [
        {"value": ["VISA 4111111111111111"], "confidence": 0.9, "evidence": None},
        {"value": {"brand": "VISA", "number": "4111111111111111"}, "confidence": 0.9, "evidence": None},
        {"value": "VISA", "confidence": 0.9, "evidence": {"page": 1, "quote": ["4111111111111111"]}},
        {"value": "VISA", "confidence": 0.9, "evidence": {"page": 1, "quote": "VISA", "raw": _LEAK}},
        {"value": "VISA", "confidence": 0.9, "evidence": _LEAK},
        {"value": "VISA", "confidence": 0.9, "note": _LEAK, "evidence": None},
        {"value": "VISA", "confidence": [_LEAK], "evidence": None},
        {"value": "VISA", "confidence": _LEAK[:8] + "x", "evidence": None},
        _LEAK,
        {"value": f"VISA {_LEAK}", "confidence": 0.9, "evidence": None},
    ],
    ids=[
        "list-value",
        "dict-value",
        "list-quote",
        "extra-evidence-key",
        "string-evidence",
        "extra-leaf-key",
        "list-confidence",
        "string-confidence",
        "bare-scalar",
        "plain-value",
    ],
)
async def test_a_card_number_in_an_unexpected_shape_is_not_stored(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch, payment_method
) -> None:
    tool_input = v2_tool_input()
    tool_input["payment_method"] = payment_method

    extraction, fields = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    _assert_no_card_number(extraction)
    assert _LEAK not in json.dumps(fields["payment_method"].value)
    assert fields["payment_method"].value["value"] in (None, "VISA", "VISA ****1111")


def test_a_payment_leaf_is_rebuilt_from_its_allowed_keys_only() -> None:
    masked = mask_payment_fields(
        {
            "payment_card_last4": {
                "value": {"number": _LEAK},
                "confidence": 0.8,
                "note": _LEAK,
                "evidence": {"page": 2, "quote": f" Card {_LEAK} ", "raw": _LEAK},
            },
            "payment_method": {"confidence": 0.5},
        }
    )

    assert masked["payment_card_last4"] == {
        "value": None,
        "confidence": 0.8,
        "evidence": {"page": 2, "quote": "Card ****1111"},
    }
    # Missing value stays missing, so the leaf is still a schema violation.
    assert masked["payment_method"] == {"confidence": 0.5}
    assert _schema_violations(masked, SCHEMA_V2)[:1] == ["vendor"]  # the rest of v2 is absent here
    assert "payment_method" in _schema_violations(masked, SCHEMA_V2)


# Card numbers in the quotes of fields that are not about payment.


def test_a_card_number_in_any_quote_is_masked_and_a_value_is_not() -> None:
    tool_input = v2_tool_input()
    tool_input["amount_paid"]["evidence"]["quote"] = "VISA 4111 1111 1111 1111 APPROVED $50.00"
    tool_input["total"]["evidence"]["quote"] = "Total $102.20 paid with 4111-1111-1111-1111"
    tool_input["invoice_number"] = leaf("4111111111111111", "Invoice 4111111111111111 ")  # Luhn-valid
    tool_input["tax_lines"]["value"][0]["amount"]["evidence"]["quote"] = "Tax 7.20 card 4111111111111111"
    tool_input["po_number"] = leaf("1234567890123456", "PO 1234 5678 9012 3456")  # not Luhn-valid
    before = copy.deepcopy(tool_input)

    masked = mask_evidence_quotes(tool_input)

    assert tool_input == before
    assert masked["amount_paid"]["evidence"]["quote"] == "VISA ****1111 APPROVED $50.00"
    assert masked["total"]["evidence"]["quote"] == "Total $102.20 paid with ****1111"
    assert masked["tax_lines"]["value"][0]["amount"]["evidence"]["quote"] == "Tax 7.20 card ****1111"
    # Values are never touched, and a reference that is not a card stays whole.
    assert masked["invoice_number"]["value"] == "4111111111111111"
    assert masked["po_number"] == before["po_number"]
    assert masked["vendor"] == before["vendor"]
    # Nothing to mask: the same object comes back.
    clean = v2_tool_input()
    assert mask_evidence_quotes(clean) is clean
    assert mask_evidence_quotes(REALISTIC_TOOL_INPUT) is REALISTIC_TOOL_INPUT
    assert mask_card_data(REALISTIC_TOOL_INPUT) is REALISTIC_TOOL_INPUT


async def test_a_masked_quote_keeps_its_quote_check_passing(
    db_session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = v2_tool_input()
    tool_input["amount_paid"]["evidence"]["quote"] = "VISA 4111 1111 1111 1111 APPROVED $50.00"
    tool_input["invoice_number"] = leaf("4111111111111111", "Invoice # 4111111111111111")

    extraction, fields = await _persist(db_session, tmp_path, monkeypatch, tool_input)

    assert fields["amount_paid"].value["evidence"]["quote"] == "VISA ****1111 APPROVED $50.00"
    assert extraction.raw_response["amount_paid"]["evidence"]["quote"] == "VISA ****1111 APPROVED $50.00"
    assert fields["invoice_number"].value["value"] == "4111111111111111"
    assert extraction.raw_response["invoice_number"]["evidence"]["quote"] == "Invoice # ****1111"
    reconciliation = (
        await db_session.execute(
            select(Reconciliation).where(Reconciliation.extraction_id == extraction.id)
        )
    ).scalar_one()
    checks = {check["id"]: check for check in reconciliation.report["checks"]}
    assert checks["quote:amount_paid"]["status"] == "pass"
    assert reconciliation.status == "pass"


def test_build_extracted_fields_masks_for_every_writer() -> None:
    tool_input = v2_tool_input()
    tool_input["payment_method"] = leaf("VISA 4111 1111 1111 1111", "VISA 4111 1111 1111 1111")
    tool_input["amount_paid"]["evidence"]["quote"] = "4111111111111111 $50.00"

    fields = {
        f.field_name: f
        for f in build_extracted_fields(uuid.uuid4(), tool_input, 0.8, SCHEMA_V2)
    }

    assert fields["payment_method"].value["value"] == "VISA ****1111"
    assert fields["payment_method"].value["evidence"]["quote"] == "VISA ****1111"
    assert fields["amount_paid"].value["evidence"]["quote"] == "****1111 $50.00"
    assert not re.search(r"\d{13,}", json.dumps([f.value for f in fields.values()]))


def test_the_v2_budget_stays_under_the_sdk_nonstreaming_limit() -> None:
    client = anthropic.AsyncAnthropic(api_key="not-a-key")

    # Raises ValueError ("Streaming is required...") above about 21,000 tokens.
    timeout = client._calculate_nonstreaming_timeout(
        SCHEMA_V2.max_tokens, MODEL_NONSTREAMING_TOKENS.get("claude-sonnet-5")
    )

    assert timeout is not None
    assert SCHEMA_V2.max_tokens == 4 * SCHEMA_V1.max_tokens


def test_a_chunked_v2_pdf_keeps_the_winners_scalar_evidence_with_its_chunks_page() -> None:
    pages = [v2_tool_input(), v2_tool_input()]

    merged = merge_page_tool_inputs(pages, SCHEMA_V2)

    # Both pages agree at the same confidence: an amount goes to the latest page.
    assert merged["subtotal"]["evidence"] == {"page": 2, "quote": "Subtotal $100.00"}
    # A header field goes to the earliest.
    assert merged["vendor"]["evidence"] == {"page": 1, "quote": "Acme Corp"}
    fields = {f.field_name: f for f in build_extracted_fields(uuid.uuid4(), merged, 0.8, SCHEMA_V2)}
    assert fields["subtotal"].value["evidence"] == {"page": 2, "quote": "Subtotal $100.00"}
    assert fields["subtotal"].value["value"] == 100.0
    report = reconcile(ReconcileInput.from_fields(fields.values()))
    quote_checks = [c for c in report.checks if c.kind == "quote" and c.id == "quote:subtotal"]
    assert [c.status for c in quote_checks] == ["pass"]
    # The concatenated arrays' cells cite the chunk each row came from.
    assert [row["amount"]["evidence"]["page"] for row in fields["tax_lines"].value] == [1, 2]


# --- stop reasons ---------------------------------------------------------------


async def test_a_clean_single_call_records_its_stop_reason(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT)],
        usage=_FakeUsage(100, 50),
    )
    _patch_client(monkeypatch, response)

    result = await _extract_from_tmp_file(tmp_path)

    assert result.stop_reasons == [{"first": "tool_use", "repair": None}]


async def test_a_repair_call_cut_off_at_max_tokens_is_recorded_not_swallowed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    cut_off = _FakeMessage(
        stop_reason="max_tokens", content=[_FakeTextBlock("(cut)")], usage=_FakeUsage(200, 4096)
    )
    _patch_client_side_effect(monkeypatch, [first_response, cut_off])

    result = await _extract_from_tmp_file(tmp_path)

    assert result.repaired is False
    assert result.stop_reasons == [{"first": "tool_use", "repair": "max_tokens"}]


async def test_an_accepted_repair_records_both_stop_reasons(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(_MALFORMED_TOOL_INPUT, tool_use_id="toolu_1")],
        usage=_FakeUsage(1000, 500),
    )
    second_response = _FakeMessage(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(REALISTIC_TOOL_INPUT, tool_use_id="toolu_2")],
        usage=_FakeUsage(200, 100),
    )
    _patch_client_side_effect(monkeypatch, [first_response, second_response])

    result = await _extract_from_tmp_file(tmp_path)

    assert result.stop_reasons == [{"first": "tool_use", "repair": "tool_use"}]


async def test_a_first_call_cut_off_at_max_tokens_carries_its_stop_reason_on_the_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _FakeMessage(
        stop_reason="max_tokens", content=[_FakeTextBlock("(cut)")], usage=_FakeUsage(4000, 4096)
    )
    _patch_client(monkeypatch, response)

    with pytest.raises(NonRetryableExtractionError) as exc_info:
        await _extract_from_tmp_file(tmp_path)

    assert exc_info.value.stop_reasons == [{"first": "max_tokens", "repair": None}]


@pytest.mark.parametrize(
    ("model", "dollars"),
    [
        # Input plus output price per million tokens, from the published
        # pricing page (checked 2026-10-15): a million tokens of each.
        ("claude-sonnet-5", 12.00),
        ("claude-haiku-4-5", 6.00),
        ("claude-opus-5", 30.00),
        ("claude-opus-5-5", 24.00),
        ("claude-sonnet-5-5", 12.00),
        ("claude-opus-4-8", 30.00),
    ],
)
def test_the_pricing_table_matches_the_published_prices(model: str, dollars: float) -> None:
    assert extraction_module._compute_cost_usd(model, 1_000_000, 1_000_000) == pytest.approx(dollars)

