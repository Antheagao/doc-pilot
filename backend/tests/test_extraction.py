import base64
from unittest.mock import AsyncMock

import anthropic
import httpx
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
    ExtractionResult,
    ModelRefusalError,
    NonRetryableExtractionError,
    _coerce_leaf,
    _format_violations_for_prompt,
    _leaf_is_malformed,
    _schema_violations,
    extract_document,
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
    assert result.schema_violations == len(_schema_violations(_MALFORMED_TOOL_INPUT))
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
    assert result.schema_violations == len(_schema_violations(_MALFORMED_TOOL_INPUT))


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
    assert len(_schema_violations(first_input)) == 32
    assert len(_schema_violations(repair_input)) == 28

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

    violations = _schema_violations(tool_input)

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
    assert _schema_violations(REALISTIC_TOOL_INPUT) == []


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
    violations = _schema_violations(_tool_input_with_n_bad_items(10))

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
        assert _schema_violations(bad_input) == list(TOP_LEVEL_FIELDS)


def test_format_violations_for_prompt_truncates_at_20_with_summary_entry() -> None:
    """Truncation (MAX_SCHEMA_VIOLATIONS + "… (N more)") lives ONLY in
    _format_violations_for_prompt, the repair-prompt-text renderer -- not
    in _schema_violations itself (see FINDING 1 regression test above).
    """
    violations = _schema_violations(_tool_input_with_n_bad_items(10))  # 40 violations
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
