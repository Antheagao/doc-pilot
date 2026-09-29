"""app.transcription with a fake Anthropic client (no network), plus the
index job that consumes it."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest
from pypdf import PdfWriter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import transcription as transcription_module
from app.config import Settings
from app.extraction import (
    ExtractionError,
    ModelRefusalError,
    NonRetryableExtractionError,
)
from app.models import Document, DocumentChunk, DocumentPage, Job
from app.retrieval import index_job as index_job_module
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.indexing import PageText
from app.transcription import (
    TRANSCRIBE_PROMPT_TEXT,
    TRANSCRIBE_PROMPT_VERSION,
    transcribe_document,
)

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _pdf(pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str, stop_reason: str = "end_turn", usage=(1000, 200)) -> None:
        self.content = [_Text(text)] if text else []
        self.stop_reason = stop_reason
        self.usage = _Usage(*usage)


class _Client:
    def __init__(self, side_effect: list) -> None:
        self.messages = type("M", (), {})()
        self.messages.create = AsyncMock(side_effect=side_effect)


@pytest.fixture(autouse=True)
def _pinned_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transcription_module,
        "get_settings",
        lambda: Settings(transcription_model="claude-haiku-4-5", pdf_max_pages=5),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, side_effect: list) -> _Client:
    client = _Client(side_effect)
    monkeypatch.setattr(transcription_module, "_build_client", lambda settings: client)
    return client


def _status_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls(f"status {status}", response=response, body=None)


async def test_image_is_one_page_with_cost_and_prompt_version(tmp_path, monkeypatch) -> None:
    client = _patch_client(monkeypatch, [_Message("  Cobblestone Bakery\nTotal: $21.71\n")])
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    pages = await transcribe_document(path, "image/png")

    assert len(pages) == 1
    page = pages[0]
    assert page.page_number == 1
    assert page.text == "Cobblestone Bakery\nTotal: $21.71"
    assert page.source == "transcription"
    assert page.model == "claude-haiku-4-5"
    assert page.prompt_version == TRANSCRIBE_PROMPT_VERSION == "transcribe_v1"
    # claude-haiku-4-5 is $1/$5 per MTok: 1000 in + 200 out.
    assert page.cost_usd == pytest.approx(0.002)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == TRANSCRIBE_PROMPT_TEXT
    image_block = kwargs["messages"][0]["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"


async def test_multi_page_pdf_is_transcribed_one_call_per_page(tmp_path, monkeypatch) -> None:
    client = _patch_client(
        monkeypatch, [_Message("page one"), _Message("page two"), _Message("page three")]
    )
    path = tmp_path / "doc.pdf"
    path.write_bytes(_pdf(3))

    pages = await transcribe_document(path, "application/pdf")

    assert [(p.page_number, p.text) for p in pages] == [
        (1, "page one"),
        (2, "page two"),
        (3, "page three"),
    ]
    assert client.messages.create.await_count == 3
    for call in client.messages.create.call_args_list:
        assert call.kwargs["messages"][0]["content"][0]["type"] == "document"


async def test_blank_page_transcribes_to_empty_text(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, [_Message("")])
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    pages = await transcribe_document(path, "image/png")

    assert pages[0].text == ""


async def test_refusal_is_non_retryable_and_carries_billed_cost(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, [_Message("", stop_reason="refusal")])
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    with pytest.raises(ModelRefusalError) as exc_info:
        await transcribe_document(path, "image/png")

    assert exc_info.value.cost_usd == pytest.approx(0.002)


async def test_truncated_transcription_is_non_retryable(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, [_Message("partial", stop_reason="max_tokens")])
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    with pytest.raises(NonRetryableExtractionError, match="max_tokens"):
        await transcribe_document(path, "image/png")


async def test_rate_limit_is_retryable_and_keeps_retry_after(tmp_path, monkeypatch) -> None:
    _patch_client(
        monkeypatch, [_status_error(anthropic.RateLimitError, 429, {"retry-after": "12"})]
    )
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    with pytest.raises(ExtractionError) as exc_info:
        await transcribe_document(path, "image/png")

    assert not isinstance(exc_info.value, NonRetryableExtractionError)
    assert exc_info.value.retry_after_seconds == 12.0


async def test_bad_request_is_non_retryable(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, [_status_error(anthropic.BadRequestError, 400)])
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)

    with pytest.raises(NonRetryableExtractionError):
        await transcribe_document(path, "image/png")


async def test_failure_mid_pdf_carries_spend_from_earlier_pages(tmp_path, monkeypatch) -> None:
    _patch_client(
        monkeypatch,
        [_Message("page one", usage=(1000, 200)), _status_error(anthropic.InternalServerError, 500)],
    )
    path = tmp_path / "doc.pdf"
    path.write_bytes(_pdf(2))

    with pytest.raises(ExtractionError) as exc_info:
        await transcribe_document(path, "application/pdf")

    assert exc_info.value.input_tokens == 1000
    assert exc_info.value.output_tokens == 200
    assert exc_info.value.cost_usd == pytest.approx(0.002)


@pytest.mark.parametrize(
    "settings,filename,mime,content",
    [
        (Settings(transcription_model="not-a-model"), "r.png", "image/png", TINY_PNG),
        (Settings(transcription_model="claude-haiku-4-5"), "r.txt", "text/plain", b"hi"),
        (Settings(transcription_model="claude-haiku-4-5", pdf_max_pages=2), "d.pdf", "application/pdf", _pdf(3)),
        (Settings(transcription_model="claude-haiku-4-5"), "d.pdf", "application/pdf", b"%PDF-garbage"),
    ],
    ids=["unpriced-model", "unsupported-mime", "too-many-pages", "unreadable-pdf"],
)
async def test_preflight_failures_are_free_and_non_retryable(
    tmp_path, monkeypatch, settings, filename, mime, content
) -> None:
    monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)
    client = _patch_client(monkeypatch, [])
    path = tmp_path / filename
    path.write_bytes(content)

    with pytest.raises(NonRetryableExtractionError):
        await transcribe_document(path, mime)

    client.messages.create.assert_not_awaited()


# --- index job --------------------------------------------------------------


async def _document(db_session: AsyncSession, tmp_path) -> Document:
    path = tmp_path / "r.png"
    path.write_bytes(TINY_PNG)
    document = Document(
        filename="r.png", mime_type="image/png", storage_path=str(path), status="extracted"
    )
    db_session.add(document)
    await db_session.commit()
    return document


@pytest.fixture
def _hashing_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_job_module, "get_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(
        index_job_module,
        "get_settings",
        lambda: Settings(chunk_max_chars=0, chunk_overlap_chars=0, chunk_context_headers=True),
    )


async def test_index_job_transcribes_and_indexes(
    db_session: AsyncSession, tmp_path, monkeypatch, _hashing_embedder
) -> None:
    document = await _document(db_session, tmp_path)
    job = Job(document_id=document.id, kind="index", state="processing")
    db_session.add(job)
    await db_session.commit()

    async def fake_transcribe(path, mime_type):
        return [
            PageText(1, "Cobblestone Bakery\nHerbal Tea Sampler  1  $8.05", "transcription",
                     model="claude-haiku-4-5", prompt_version="transcribe_v1",
                     input_tokens=900, output_tokens=40, cost_usd=0.0011, latency_ms=800)
        ]

    monkeypatch.setattr(index_job_module, "transcribe_document", fake_transcribe)

    await index_job_module.process_index_job(db_session, job)

    page = (
        await db_session.execute(select(DocumentPage).where(DocumentPage.document_id == document.id))
    ).scalar_one()
    assert page.source == "transcription"
    assert float(page.cost_usd) == pytest.approx(0.0011)
    chunk = (
        await db_session.execute(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
    ).scalar_one()
    assert chunk.context == "Cobblestone Bakery (page 1 of 1)"
    assert chunk.text == page.text


async def test_index_failure_after_transcription_reports_billed_spend(
    db_session: AsyncSession, tmp_path, monkeypatch
) -> None:
    document = await _document(db_session, tmp_path)
    job = Job(document_id=document.id, kind="index", state="processing")
    db_session.add(job)
    await db_session.commit()

    async def fake_transcribe(path, mime_type):
        return [PageText(1, "text", "transcription", input_tokens=500, output_tokens=50,
                         cost_usd=0.00075)]

    class BrokenEmbedder(HashingEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("model download failed")

    monkeypatch.setattr(index_job_module, "transcribe_document", fake_transcribe)
    monkeypatch.setattr(index_job_module, "get_embedder", lambda: BrokenEmbedder())

    with pytest.raises(ExtractionError, match="billed transcription") as exc_info:
        await index_job_module.process_index_job(db_session, job)

    assert not isinstance(exc_info.value, NonRetryableExtractionError)
    assert exc_info.value.cost_usd == pytest.approx(0.00075)
