"""The scripted model (app/scripted_model.py): MODEL_PROVIDER=scripted, the
Settings guard around it, and its four kinds of request.

Nothing here can spend: the autouse fixture makes building an Anthropic client
fail the test, and every client is the scripted one. Requests are the ones the
app's real code makes -- extract_document, transcribe_document, answer_question,
grade_groundedness run unchanged on a recording scripted client -- so a
malformed-request test breaks a captured real request instead of hand-writing a
shape the app never sends.
"""

import asyncio
import base64
import copy
import json
import logging
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from pydantic import ValidationError
from scripted_helpers import jpeg_with_comment, png_with_text_chunk
from sqlalchemy.ext.asyncio import AsyncSession

from app import extraction as extraction_module
from app import transcription as transcription_module
from app.agent import loop as loop_module
from app.agent.loop import answer_question, call_cost_usd
from app.agent.requests import InvalidRequestError, validate_request
from app.agent.tools import TOOL_DEFINITIONS, ToolContext
from app.config import SCRIPTED_MODEL, Settings
from app.evals.corpus import seed_labeled_corpus
from app.evals.dataset import load_cases
from app.evals.judge import grade_groundedness, judge_answer
from app.evals.replay import (
    REPLAY_ARTIFACT,
    RESULTS_DIR,
    TRANSCRIPTION_PROXY_ARTIFACT,
    _replayed_tool_input,
    scored_replay,
)
from app.evals.retrieval import load_gold_corpus
from app.extraction import (
    PRICING_PER_MTOK,
    ModelPrice,
    NonRetryableExtractionError,
    _classify_api_error,
    _compute_cost_usd,
    _schema_violations,
    _usage_cost_usd,
    check_model_provider,
    extract_document,
)
from app.extraction_fields import (
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMAS,
    ExtractionSchema,
)
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.indexing import ChunkingConfig
from app.routers import ask as ask_module
from app.scripted_model import (
    DOCUMENT_BLOCK_TOKENS,
    GENERIC_CONFIDENCE,
    GENERIC_PAGE_TEXT,
    JUDGE_EXPLANATION,
    LABEL_CONFIDENCE,
    ScriptedCorpus,
    ScriptedModelClient,
    content_digest,
    default_corpus,
    estimate_tokens,
    request_input_tokens,
    require_corpus,
)
from app.transcription import transcribe_document

EMBEDDER = HashingEmbedder()
SCRIPTED = {
    "model_provider": "scripted",
    "extraction_model": SCRIPTED_MODEL,
    "transcription_model": SCRIPTED_MODEL,
    "agent_model": SCRIPTED_MODEL,
    "judge_model": SCRIPTED_MODEL,
    "anthropic_api_key": None,
}
CASES = {case.doc_id: case for case in load_cases()}
COFFEE = CASES["001-clean-coffee-receipt"]
JPEG_CASE = next(case for case in CASES.values() if case.mime_type == "image/jpeg")
PDF_CASE = next(case for case in CASES.values() if case.mime_type == "application/pdf")
SEEDED = (
    "001-clean-coffee-receipt",
    "018-dense-office-outfitters",
    "024-eur-bakery-berlin",
    "025-adversarial-office-order-injection",
)


def scripted_settings(**overrides: Any) -> Settings:
    return Settings(**{**SCRIPTED, **overrides})


@pytest.fixture(autouse=True)
def _no_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building an Anthropic client, sync or async, fails the test: nothing in
    this module may reach the real SDK, whatever key backend/.env holds."""

    def refuse(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a real Anthropic client was built")

    monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", refuse)
    monkeypatch.setattr(anthropic.Anthropic, "__init__", refuse)


class RecordingClient(ScriptedModelClient):
    """The scripted client, remembering each request it was sent."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        self.requests: list[dict[str, Any]] = []

    async def respond(self, request: dict[str, Any], *, beta: bool):
        self.requests.append(copy.deepcopy(request))
        return await super().respond(request, beta=beta)


class Sleeps:
    """An injected sleep that records its delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def document_block(raw: bytes, media_type: str) -> dict[str, Any]:
    kind = "document" if media_type == "application/pdf" else "image"
    data = base64.standard_b64encode(raw).decode()
    return {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}


async def extract(
    case_or_bytes, settings: Settings | None = None, *, monkeypatch, schema=SCHEMA_V1, tmp_path: Path
):
    """extract_document on a recording scripted client: (result, client). The
    document is a corpus case's image or raw bytes written to a temp file."""
    settings = settings or scripted_settings()
    client = RecordingClient(settings)
    if isinstance(case_or_bytes, bytes):
        path = tmp_path / "upload.png"
        path.write_bytes(case_or_bytes)
        mime = "image/png"
    else:
        path, mime = case_or_bytes.image_path, case_or_bytes.mime_type
    monkeypatch.setattr(extraction_module, "get_settings", lambda: settings)
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: client)
    return await extract_document(path, mime, schema=schema), client


# --- the Settings guard ---------------------------------------------------------


def test_the_default_provider_is_anthropic_and_the_defaults_pass() -> None:
    settings = Settings()

    assert settings.model_provider == "anthropic"
    assert settings.scripted_model_profile == "zero"
    assert settings.scripted_model_latency_scale == 1.0
    assert settings.scripted_agent_step_ms == 3000
    assert settings.scripted_judge_ms == 4000
    assert settings.scripted_model_allow_api_key is False


def test_scripted_with_the_four_scripted_names_and_no_key_passes() -> None:
    settings = scripted_settings()

    assert settings.model_provider == "scripted"
    assert "scripted" in settings.model_provider_summary()
    assert "no model API calls" in settings.model_provider_summary()


@pytest.mark.parametrize(
    "name", ["extraction_model", "transcription_model", "agent_model", "judge_model"]
)
def test_scripted_refuses_each_real_model_name(name: str) -> None:
    with pytest.raises(ValidationError, match=name.upper()):
        scripted_settings(**{name: "claude-sonnet-5"})


@pytest.mark.parametrize(
    "name", ["extraction_model", "transcription_model", "agent_model", "judge_model"]
)
def test_anthropic_refuses_each_scripted_model_name(name: str) -> None:
    with pytest.raises(ValidationError, match=name.upper()):
        Settings(**{name: SCRIPTED_MODEL})


def test_scripted_refuses_a_key_and_never_prints_it() -> None:
    with pytest.raises(ValidationError) as caught:
        scripted_settings(anthropic_api_key="sk-ant-do-not-print")

    assert "ANTHROPIC_API_KEY" in str(caught.value)
    assert "sk-ant-do-not-print" not in str(caught.value)
    # Pydantic would otherwise echo the whole settings dict it refused.
    assert "input_value" not in str(caught.value)


def test_scripted_accepts_a_key_only_when_allowed() -> None:
    settings = scripted_settings(anthropic_api_key="sk-test", scripted_model_allow_api_key=True)

    assert settings.model_provider == "scripted"


def test_scripted_accepts_an_empty_key() -> None:
    assert scripted_settings(anthropic_api_key="").model_provider == "scripted"


def test_the_environment_reaches_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "MODEL_PROVIDER": "scripted",
        "EXTRACTION_MODEL": "scripted",
        "TRANSCRIPTION_MODEL": "scripted",
        "AGENT_MODEL": "scripted",
        "JUDGE_MODEL": "scripted",
        "ANTHROPIC_API_KEY": "sk-test",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings()

    monkeypatch.setenv("SCRIPTED_MODEL_ALLOW_API_KEY", "true")
    assert Settings().model_provider == "scripted"


def test_the_latency_settings_are_bounded() -> None:
    for name in ("scripted_model_latency_scale", "scripted_agent_step_ms", "scripted_judge_ms"):
        with pytest.raises(ValidationError):
            scripted_settings(**{name: -1})
    with pytest.raises(ValidationError):
        scripted_settings(scripted_model_profile="fast")


@pytest.mark.parametrize("scale", [float("inf"), float("nan")])
def test_the_latency_scale_refuses_a_value_that_is_not_a_finite_number(scale: float) -> None:
    # inf passes ge=0 and would make every replayed latency sleep forever.
    with pytest.raises(ValidationError):
        scripted_settings(scripted_model_latency_scale=scale)


def test_the_scripted_model_is_priced_at_zero_by_the_real_pricing_code() -> None:
    assert PRICING_PER_MTOK[SCRIPTED_MODEL] == ModelPrice(0, 0, 0)
    assert _compute_cost_usd(SCRIPTED_MODEL, 10**9, 10**9, 10**9, 10**9) == 0.0


# --- the client factories -------------------------------------------------------


def test_build_client_returns_the_scripted_client_without_building_a_real_one() -> None:
    client = extraction_module._build_client(scripted_settings())

    assert isinstance(client, ScriptedModelClient)


def test_the_ask_dependency_builds_the_scripted_client_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ask_module, "_client", None)
    settings = scripted_settings()

    first = ask_module.get_anthropic_client(settings)
    second = ask_module.get_anthropic_client(settings)

    assert isinstance(first, ScriptedModelClient) and first is second
    monkeypatch.setattr(ask_module, "_client", None)
    assert ask_module.get_anthropic_client(Settings(anthropic_api_key=None)) is None


# --- recognizing a document -----------------------------------------------------


def test_a_png_text_chunk_does_not_change_its_pixel_digest() -> None:
    raw = COFFEE.image_path.read_bytes()
    nonced = png_with_text_chunk(raw, b"upload 17")

    assert nonced != raw
    assert content_digest(nonced, "image/png") == content_digest(raw, "image/png")
    assert content_digest(png_with_text_chunk(raw, b"upload 18"), "image/png") == content_digest(
        nonced, "image/png"
    )


def test_a_jpeg_comment_does_not_change_its_pixel_digest() -> None:
    raw = JPEG_CASE.image_path.read_bytes()

    assert content_digest(jpeg_with_comment(raw, b"upload 3"), "image/jpeg") == content_digest(
        raw, "image/jpeg"
    )


def test_different_pixels_and_corrupt_bytes_have_their_own_digests() -> None:
    raw = COFFEE.image_path.read_bytes()
    other = CASES["002-clean-hardware-invoice"].image_path.read_bytes()
    truncated = raw[: len(raw) // 2]

    assert content_digest(raw, "image/png") != content_digest(other, "image/png")
    # Not a whole PNG: every byte counts, so it is simply not recognized.
    assert content_digest(truncated, "image/png") != content_digest(raw, "image/png")
    assert content_digest(b"junk", "image/png") == content_digest(b"junk", "image/png")


def test_the_corpus_recognizes_every_committed_document_through_a_nonce() -> None:
    corpus = default_corpus()
    for case in CASES.values():
        raw = case.image_path.read_bytes()
        if case.mime_type == "image/png":
            raw = png_with_text_chunk(raw, b"nonce")
        elif case.mime_type == "image/jpeg":
            raw = jpeg_with_comment(raw, b"nonce")
        assert corpus.recognize(raw, case.mime_type).doc_id == case.doc_id, case.doc_id
    assert corpus.recognize(b"not a document", "image/png") is None


# --- extraction -----------------------------------------------------------------


async def test_extraction_replays_the_committed_sonnet_answer(monkeypatch, tmp_path) -> None:
    result, client = await extract(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)

    artifact = json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))
    entry = next(e for e in artifact["per_doc"] if e["doc_id"] == COFFEE.doc_id)
    expected, _ = _replayed_tool_input(entry, COFFEE, SCHEMA_V1)
    assert result.tool_input == expected
    assert result.model == SCRIPTED_MODEL
    assert result.prompt_version == "extract_v1"
    assert result.cost_usd == 0.0
    assert result.repaired is False and result.schema_violations == 0
    assert result.stop_reasons == [{"first": "tool_use", "repair": None}]
    assert len(client.requests) == 1


async def test_extraction_returns_the_sdks_own_types(monkeypatch, tmp_path) -> None:
    _, client = await extract(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)
    from anthropic.types import Message, ToolUseBlock, Usage

    response = await client.messages.create(**client.requests[0])

    assert isinstance(response, Message)
    assert isinstance(response.usage, Usage)
    (block,) = response.content
    assert isinstance(block, ToolUseBlock) and block.name == "record_extraction"
    assert response.stop_reason == "tool_use" and response.model == SCRIPTED_MODEL
    assert _usage_cost_usd(SCRIPTED_MODEL, response.usage) == 0.0


async def test_a_nonced_upload_is_read_as_its_corpus_document(monkeypatch, tmp_path) -> None:
    plain, _ = await extract(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)
    nonced_bytes = png_with_text_chunk(COFFEE.image_path.read_bytes(), b"upload 9")

    nonced, _ = await extract(nonced_bytes, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert nonced.tool_input == plain.tool_input


async def test_unknown_bytes_get_the_generic_answer_at_half_confidence(
    monkeypatch, tmp_path
) -> None:
    unknown = png_with_text_chunk(
        CASES["002-clean-hardware-invoice"].image_path.read_bytes(), b"x"
    )[:-20]  # truncated: not any corpus picture

    result, _ = await extract(unknown, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert result.tool_input["vendor"] == {"value": "Scripted Vendor", "confidence": GENERIC_CONFIDENCE}
    assert result.schema_violations == 0
    assert all(
        leaf["confidence"] == GENERIC_CONFIDENCE
        for name, leaf in result.tool_input.items()
        if name != "line_items"
    )


@pytest.mark.parametrize("schema", list(SCHEMAS.values()), ids=lambda s: s.prompt_version)
async def test_every_answer_is_valid_for_every_schema(schema: ExtractionSchema, monkeypatch, tmp_path) -> None:
    """The committed replays, the labels, and the generic answer all parse as
    the schema asked for (no repair needed): 32 documents and unknown bytes."""
    for case in CASES.values():
        result, _ = await extract(case, monkeypatch=monkeypatch, schema=schema, tmp_path=tmp_path)
        assert _schema_violations(result.tool_input, schema) == [], case.doc_id
        assert result.prompt_version == schema.prompt_version
    generic, _ = await extract(b"junk", monkeypatch=monkeypatch, schema=schema, tmp_path=tmp_path)
    assert _schema_violations(generic.tool_input, schema) == []


async def test_the_request_names_its_schema_v2_and_v3_differ_by_prompt(
    monkeypatch, tmp_path
) -> None:
    # 001 is in the committed v3 run and in no v2 run: v3 replays it, v2 gets the
    # label at LABEL_CONFIDENCE, so the two prompts were told apart.
    v3, _ = await extract(COFFEE, monkeypatch=monkeypatch, schema=SCHEMA_V3, tmp_path=tmp_path)
    v2, _ = await extract(COFFEE, monkeypatch=monkeypatch, schema=SCHEMA_V2, tmp_path=tmp_path)

    assert v3.tool_input["vendor"]["confidence"] == 0.98
    assert v2.tool_input["vendor"]["confidence"] == LABEL_CONFIDENCE
    assert v2.tool_input["vendor"]["evidence"] == COFFEE.evidence["vendor"]
    assert v2.prompt_version == "extract_v2" and v3.prompt_version == "extract_v3"


async def test_a_document_no_run_scored_is_its_label_at_label_confidence(
    monkeypatch, tmp_path
) -> None:
    case = CASES["026-invoice-discount-shipping-two-taxes"]

    result, _ = await extract(case, monkeypatch=monkeypatch, schema=SCHEMA_V1, tmp_path=tmp_path)

    assert result.tool_input["vendor"] == {"value": case.fields["vendor"], "confidence": LABEL_CONFIDENCE}


async def test_the_repair_reprompt_is_answered_with_the_same_input(monkeypatch, tmp_path) -> None:
    _, client = await extract(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)
    first = client.requests[0]
    answer = await client.messages.create(**first)
    repair = copy.deepcopy(first)
    tool_use = answer.content[0]
    repair["messages"] += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use.id, "name": tool_use.name, "input": tool_use.input}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "fix", "is_error": True}]},
    ]

    again = await client.messages.create(**repair)

    assert again.content[0].input == tool_use.input
    assert again.content[0].id != tool_use.id  # a different turn of the conversation


async def test_extraction_cost_is_zero_and_tokens_follow_the_formula(monkeypatch, tmp_path) -> None:
    result, client = await extract(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)

    request = client.requests[0]
    assert result.input_tokens == request_input_tokens(request)
    assert result.input_tokens >= DOCUMENT_BLOCK_TOKENS
    assert result.output_tokens == estimate_tokens(json.dumps(result.tool_input, sort_keys=True))
    assert result.cost_usd == 0.0


async def test_zero_profile_never_sleeps(monkeypatch, tmp_path) -> None:
    sleeps = Sleeps()
    settings = scripted_settings()
    client = ScriptedModelClient(settings, sleep=sleeps)
    monkeypatch.setattr(extraction_module, "get_settings", lambda: settings)
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: client)

    await extract_document(COFFEE.image_path, COFFEE.mime_type)

    assert sleeps.delays == []


async def test_replay_profile_takes_the_committed_latency_and_tokens(monkeypatch, tmp_path) -> None:
    sleeps = Sleeps()
    settings = scripted_settings(scripted_model_profile="replay", scripted_model_latency_scale=0.5)
    client = ScriptedModelClient(settings, sleep=sleeps)
    monkeypatch.setattr(extraction_module, "get_settings", lambda: settings)
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: client)

    result = await extract_document(COFFEE.image_path, COFFEE.mime_type)

    artifact = json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))
    entry = next(e for e in artifact["per_doc"] if e["doc_id"] == COFFEE.doc_id)
    assert sleeps.delays == [entry["latency_ms"] * 0.5 / 1000]
    assert (result.input_tokens, result.output_tokens) == (entry["input_tokens"], entry["output_tokens"])
    assert result.cost_usd == 0.0


async def test_replay_profile_gives_an_unknown_document_the_runs_median(monkeypatch, tmp_path) -> None:
    sleeps = Sleeps()
    settings = scripted_settings(scripted_model_profile="replay", scripted_model_latency_scale=1)
    client = ScriptedModelClient(settings, sleep=sleeps)
    path = tmp_path / "x.png"
    path.write_bytes(b"junk")
    monkeypatch.setattr(extraction_module, "get_settings", lambda: settings)
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: client)

    await extract_document(path, "image/png")

    artifact = json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))
    assert sleeps.delays == [artifact["latency_p50_ms"] / 1000]


async def test_a_zero_scale_replays_numbers_without_waiting(monkeypatch, tmp_path) -> None:
    sleeps = Sleeps()
    settings = scripted_settings(scripted_model_profile="replay", scripted_model_latency_scale=0)
    client = ScriptedModelClient(settings, sleep=sleeps)
    monkeypatch.setattr(extraction_module, "get_settings", lambda: settings)
    monkeypatch.setattr(extraction_module, "_build_client", lambda s: client)

    await extract_document(COFFEE.image_path, COFFEE.mime_type)

    assert sleeps.delays == []


def test_scored_replay_follows_the_requested_schema() -> None:
    primary = json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))
    doc = COFFEE.doc_id

    v1 = scored_replay(doc, SCHEMA_V1, primary)
    v3 = scored_replay(doc, SCHEMA_V3, primary)

    assert v1.artifact == REPLAY_ARTIFACT and v1.entry["doc_id"] == doc
    assert v3.artifact.name.endswith("claude-sonnet-5_extract_v3.json")
    assert scored_replay(doc, SCHEMA_V2, primary) is None
    assert scored_replay("026-invoice-discount-shipping-two-taxes", SCHEMA_V1, primary) is None
    assert scored_replay(doc, SCHEMA_V3, primary, RESULTS_DIR / "nowhere") is None


# --- transcription --------------------------------------------------------------


async def transcribe(raw_or_case, settings=None, *, monkeypatch, tmp_path, client=None):
    settings = settings or scripted_settings()
    client = client or RecordingClient(settings)
    monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)
    monkeypatch.setattr(transcription_module, "_build_client", lambda s: client)
    if isinstance(raw_or_case, bytes):
        path = tmp_path / "page.png"
        path.write_bytes(raw_or_case)
        return await transcribe_document(path, "image/png"), client
    return await transcribe_document(raw_or_case.image_path, raw_or_case.mime_type), client


async def test_transcription_is_the_gold_text_plus_a_reference_line(monkeypatch, tmp_path) -> None:
    gold = {doc.doc_id: doc.text for doc in load_gold_corpus(list(CASES.values()))}

    (page,), client = await transcribe(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)

    body, ref = page.text.rsplit("\n", 1)
    assert body == gold[COFFEE.doc_id].strip()
    assert ref.startswith("Ref ") and len(ref) == len("Ref ") + 12
    assert (page.model, page.cost_usd, page.source) == (SCRIPTED_MODEL, 0.0, "transcription")
    assert page.input_tokens == request_input_tokens(client.requests[0])


async def test_every_upload_of_a_picture_transcribes_to_text_of_its_own(monkeypatch, tmp_path) -> None:
    raw = COFFEE.image_path.read_bytes()

    (first,), _ = await transcribe(png_with_text_chunk(raw, b"1"), monkeypatch=monkeypatch, tmp_path=tmp_path)
    (second,), _ = await transcribe(png_with_text_chunk(raw, b"2"), monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert first.text != second.text
    assert first.text.rsplit("\n", 1)[0] == second.text.rsplit("\n", 1)[0]


async def test_unknown_bytes_transcribe_to_the_generic_page(monkeypatch, tmp_path) -> None:
    (page,), _ = await transcribe(b"junk", monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert page.text.startswith(GENERIC_PAGE_TEXT)


async def test_a_pdf_page_with_no_gold_text_gets_the_generic_page(monkeypatch, tmp_path) -> None:
    pages, _ = await transcribe(PDF_CASE, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert pages and all(page.text.startswith(GENERIC_PAGE_TEXT) for page in pages)


async def test_replay_profile_transcription_uses_the_haiku_proxy(monkeypatch, tmp_path) -> None:
    sleeps = Sleeps()
    settings = scripted_settings(scripted_model_profile="replay", scripted_model_latency_scale=0.1)
    client = ScriptedModelClient(settings, sleep=sleeps)

    (page,), _ = await transcribe(
        COFFEE, settings, monkeypatch=monkeypatch, tmp_path=tmp_path, client=client
    )

    proxy = json.loads(TRANSCRIPTION_PROXY_ARTIFACT.read_text(encoding="utf-8"))
    entry = next(e for e in proxy["per_doc"] if e["doc_id"] == COFFEE.doc_id)
    assert sleeps.delays == [entry["latency_ms"] * 0.1 / 1000]
    assert (page.input_tokens, page.output_tokens) == (entry["input_tokens"], entry["output_tokens"])
    assert page.cost_usd == 0.0


# --- the agent ------------------------------------------------------------------


async def seeded_context(db_session: AsyncSession) -> ToolContext:
    cases = [CASES[doc_id] for doc_id in SEEDED]
    ids = await seed_labeled_corpus(
        db_session, cases, load_gold_corpus(cases), EMBEDDER, ChunkingConfig(200, 80, True)
    )
    return ToolContext(session=db_session, embedder=EMBEDDER, document_ids=list(ids.values()))


async def test_the_agent_follows_its_policy_and_its_citations_resolve(db_session: AsyncSession) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()
    client = RecordingClient(settings)

    result = await answer_question(
        ctx, "Which store's receipt came to $425.58?", settings, client
    )

    assert result.status == "answered"
    assert [call.name for call in result.tool_calls] == ["query_extractions", "get_page"]
    assert result.tool_calls[0].input == {"min_total": 425.58, "max_total": 425.58}
    assert result.steps == 3 and len(client.requests) == 3
    assert result.unresolved_citations == 0
    assert len(result.citations) == 2
    assert result.citations[0].kind == "record"
    assert result.citations[0].cited_text == "total: 425.58"
    assert result.citations[0].fields == ["total"]
    assert result.citations[1].kind == "page" and result.citations[1].page_number == 1
    assert result.answer.startswith("According to your documents: total: 425.58. [1]")
    assert result.answer.endswith("[2]")
    assert (result.model, result.cost_usd) == (SCRIPTED_MODEL, 0.0)
    assert result.input_tokens > 0 and result.output_tokens > 0


async def test_a_question_without_an_amount_searches_then_opens_the_top_page(
    db_session: AsyncSession,
) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()

    result = await answer_question(
        ctx, "Where did I buy a light for my workspace?", settings, RecordingClient(settings)
    )

    assert [call.name for call in result.tool_calls] == ["search_documents", "get_page"]
    assert result.tool_calls[0].input == {"query": "Where did I buy a light for my workspace?"}
    assert result.status == "answered" and result.unresolved_citations == 0
    assert [c.kind for c in result.citations] == ["chunk", "page"]
    assert result.citations[0].document_id == result.tool_calls[1].input["document_id"]
    assert result.citations[0].char_start is not None


async def test_an_aggregate_question_queries_the_extractions(db_session: AsyncSession) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()

    result = await answer_question(
        ctx, "How much have I spent in total?", settings, RecordingClient(settings)
    )

    assert result.tool_calls[0].name == "query_extractions" and result.tool_calls[0].input == {}
    assert result.tool_calls[0].result_summary.startswith("4 source(s); 4 matching document(s)")
    assert result.status == "answered" and result.citations


async def test_a_question_with_no_match_says_so_without_citations(db_session: AsyncSession) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()

    result = await answer_question(
        ctx, "Which receipt came to $9999.99?", settings, RecordingClient(settings)
    )

    assert result.status == "answered" and result.citations == []
    assert result.answer == "I could not find a document that answers that question."
    assert result.steps == 2


async def test_a_chat_turn_with_history_still_starts_with_its_first_tool_call(
    db_session: AsyncSession,
) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()
    history = [loop_module.Turn("What did I buy?", "Some office supplies. [1]")]

    result = await answer_question(
        ctx, "Which receipt came to $425.58?", settings, RecordingClient(settings), history=history
    )

    assert [call.name for call in result.tool_calls] == ["query_extractions", "get_page"]
    assert result.citations


async def test_agent_latency_is_the_step_setting_scaled_per_call(db_session: AsyncSession) -> None:
    ctx = await seeded_context(db_session)
    sleeps = Sleeps()
    settings = scripted_settings(
        scripted_model_profile="replay", scripted_model_latency_scale=0.1, scripted_agent_step_ms=2500
    )

    await answer_question(
        ctx, "Which receipt came to $425.58?", settings, ScriptedModelClient(settings, sleep=sleeps)
    )

    assert sleeps.delays == [0.25, 0.25, 0.25]


async def test_the_agent_request_passes_the_shared_request_check(db_session: AsyncSession) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()
    client = RecordingClient(settings)
    await answer_question(ctx, "Which receipt came to $425.58?", settings, client)

    assert client.requests[0]["tools"] == TOOL_DEFINITIONS
    for request in client.requests:
        validate_request(request)  # the scripted client ran it too
    assert "output_config" not in client.requests[0] and "thinking" not in client.requests[0]


async def test_the_agent_costs_zero_through_the_real_cost_function(db_session: AsyncSession) -> None:
    ctx = await seeded_context(db_session)
    settings = scripted_settings()
    client = RecordingClient(settings)
    await answer_question(ctx, "Which receipt came to $425.58?", settings, client)

    response = await client.beta.messages.create(**client.requests[0])

    assert call_cost_usd(SCRIPTED_MODEL, response) == 0.0


# --- the judge ------------------------------------------------------------------


async def test_the_groundedness_verdict_is_read_by_the_real_parser() -> None:
    settings = scripted_settings()
    client = RecordingClient(settings)

    verdict = await grade_groundedness("q?", "evidence", "answer", settings, client=client)

    assert verdict.error is None
    assert (verdict.grounded, verdict.answers_question) == (True, True)
    assert verdict.claims == [] and verdict.explanation == JUDGE_EXPLANATION
    assert (verdict.model, verdict.cost_usd) == (SCRIPTED_MODEL, 0.0)
    assert client.requests[0]["output_config"]["format"]["type"] == "json_schema"


async def test_the_reference_verdict_fills_the_whole_schema() -> None:
    settings = scripted_settings()

    verdict = await judge_answer("q?", "facts", "evidence", "answer", settings, client=RecordingClient(settings))

    assert verdict.error is None
    assert (verdict.correct, verdict.grounded, verdict.score) == (True, True, 5)
    assert verdict.claims == []


async def test_judge_latency_is_its_setting_scaled() -> None:
    sleeps = Sleeps()
    settings = scripted_settings(
        scripted_model_profile="replay", scripted_model_latency_scale=2, scripted_judge_ms=1500
    )

    await grade_groundedness(
        "q?", "e", "a", settings, client=ScriptedModelClient(settings, sleep=sleeps)
    )

    assert sleeps.delays == [3.0]


async def test_the_judge_is_built_from_whatever_schema_it_is_given() -> None:
    settings = scripted_settings()
    client = ScriptedModelClient(settings)
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "label": {"type": "string", "enum": ["fine", "bad"]},
            "n": {"type": ["integer", "null"]},
        },
    }

    response = await client.beta.messages.create(
        model=SCRIPTED_MODEL,
        max_tokens=100,
        system="s",
        messages=[{"role": "user", "content": "u"}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )

    assert json.loads(response.content[0].text) == {"ok": True, "tags": [], "label": "fine", "n": 1}


# --- a malformed request fails the way the API does -----------------------------


@pytest.fixture
async def captured(monkeypatch, tmp_path, db_session: AsyncSession) -> dict[str, dict[str, Any]]:
    """One real request of each kind, and the client that answered them."""
    settings = scripted_settings()
    client = RecordingClient(settings)
    _, extraction_client = await extract(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)
    _, transcription_client = await transcribe(COFFEE, monkeypatch=monkeypatch, tmp_path=tmp_path)
    ctx = await seeded_context(db_session)
    await answer_question(ctx, "Which receipt came to $425.58?", settings, client)
    judge_client = RecordingClient(settings)
    await grade_groundedness("q", "e", "a", settings, client=judge_client)
    return {
        "extraction": extraction_client.requests[0],
        "transcription": transcription_client.requests[0],
        "agent_first": client.requests[0],
        "agent_last": client.requests[-1],
        "judge": judge_client.requests[0],
    }


async def send(request: dict[str, Any], *, beta: bool = False):
    client = ScriptedModelClient(scripted_settings())
    endpoint = client.beta.messages if beta else client.messages
    return await endpoint.create(**request)


async def refused(request: dict[str, Any], *, beta: bool = False) -> anthropic.BadRequestError:
    with pytest.raises(anthropic.BadRequestError) as caught:
        await send(request, beta=beta)
    assert caught.value.status_code == 400
    assert isinstance(caught.value.body, dict) and caught.value.body["type"] == "error"
    return caught.value


async def test_the_captured_requests_are_accepted(captured) -> None:
    await send(captured["extraction"])
    await send(captured["transcription"])
    await send(captured["agent_first"], beta=True)
    await send(captured["agent_last"], beta=True)
    await send(captured["judge"], beta=True)


async def test_the_refusal_is_classified_as_permanent_like_the_real_one(captured) -> None:
    request = {**captured["extraction"], "model": "claude-sonnet-5"}

    error = await refused(request)

    assert isinstance(_classify_api_error(error), NonRetryableExtractionError)


async def test_each_kind_goes_to_its_own_endpoint(captured) -> None:
    await refused(captured["agent_first"])
    await refused(captured["judge"])
    await refused(captured["extraction"], beta=True)
    await refused(captured["transcription"], beta=True)


@pytest.mark.parametrize(
    ("kind", "change", "message"),
    [
        ("extraction", {"model": "claude-sonnet-5"}, "unknown model"),
        ("extraction", {"max_tokens": 0}, "max_tokens"),
        ("extraction", {"messages": []}, "messages"),
        ("extraction", {"temperature": 0}, "unexpected parameters: temperature"),
        ("extraction", {"stream": True}, "does not stream"),
        ("extraction", {"tool_choice": {"type": "auto"}}, "tool_choice"),
        ("extraction", {"system": "an unknown prompt"}, "no known extraction schema"),
        ("extraction", {"tools": [{"name": "record_extraction", "input_schema": {}}]}, "no known extraction schema"),
        ("transcription", {"system": "write a poem"}, "matches no kind"),
        ("transcription", {"tools": []}, "tools match neither"),
        ("judge", {"output_config": {"format": {"type": "text"}}}, "json_schema"),
        ("judge", {"output_config": {"effort": "high", "format": {"type": "json_schema", "schema": {"properties": {}}}}}, "json_schema"),
        ("judge", {"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}}, "properties"),
        ("judge", {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}, "one user message"),
        ("agent_first", {"tools": TOOL_DEFINITIONS[:1]}, "tools match neither"),
        ("agent_first", {"thinking": {"type": "adaptive"}}, "unexpected parameters: thinking"),
    ],
)
async def test_a_malformed_request_raises_bad_request(captured, kind, change, message) -> None:
    request = {**captured[kind], **change}
    beta = kind.startswith(("agent", "judge"))

    error = await refused(request, beta=beta)

    assert message in str(error)


async def test_a_missing_parameter_is_refused(captured) -> None:
    request = {k: v for k, v in captured["extraction"].items() if k != "max_tokens"}

    assert "missing required parameters: max_tokens" in str(await refused(request))


async def test_a_bad_document_is_refused(captured) -> None:
    request = copy.deepcopy(captured["extraction"])
    block = request["messages"][0]["content"][0]

    block["source"]["data"] = "not base64!"
    assert "not valid base64" in str(await refused(request))

    block["source"] = {"type": "base64", "media_type": "image/bmp", "data": ""}
    assert "unsupported image source" in str(await refused(request))

    request["messages"][0]["content"] = [request["messages"][0]["content"][1]]
    assert "exactly one image or document" in str(await refused(request))


async def test_an_agent_request_with_no_question_is_refused(captured) -> None:
    request = copy.deepcopy(captured["agent_first"])
    request["messages"] = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]

    assert "needs the user's question" in str(await refused(request, beta=True))


async def test_a_conversation_of_the_wrong_length_is_refused(captured) -> None:
    request = copy.deepcopy(captured["extraction"])
    request["messages"] = request["messages"] * 2

    assert "one message, or three" in str(await refused(request))
    request = copy.deepcopy(captured["transcription"])
    request["messages"] = request["messages"] * 2
    assert "one message" in str(await refused(request))


async def test_an_agent_conversation_the_api_would_reject_is_refused(captured) -> None:
    broken = copy.deepcopy(captured["agent_last"])
    broken["messages"][2]["content"][0]["tool_use_id"] = "toolu_other"
    assert "tool_use ids" in str(await refused(broken, beta=True))

    broken = copy.deepcopy(captured["agent_last"])
    broken["messages"][1:2] = []  # two user messages in a row, no assistant between
    assert "roles must alternate" in str(await refused(broken, beta=True))

    broken = copy.deepcopy(captured["agent_last"])
    broken["messages"][2]["content"][0]["content"] = [{"type": "text", "text": " "}]
    assert "invalid tool_result" in str(await refused(broken, beta=True))


# --- the shared request check ---------------------------------------------------


def conversation(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"messages": list(messages)}


def tool_use_turn(*ids: str) -> dict[str, Any]:
    blocks = [SimpleNamespace(type="tool_use", id=i, name="get_page", input={}) for i in ids]
    return {"role": "assistant", "content": blocks}


def results_turn(*ids: str) -> dict[str, Any]:
    content = [
        {"type": "tool_result", "tool_use_id": i, "content": [{"type": "text", "text": "ok"}]}
        for i in ids
    ]
    return {"role": "user", "content": content}


def test_validate_request_accepts_a_tool_conversation_of_objects_or_dicts() -> None:
    user = {"role": "user", "content": "q"}
    validate_request(conversation(user, tool_use_turn("a", "b"), results_turn("a", "b")))
    as_dicts = {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "x", "input": {}}]}
    validate_request(conversation(user, as_dicts, results_turn("a")))
    validate_request(conversation(user, {"role": "assistant", "content": "plain"}))


@pytest.mark.parametrize(
    ("messages", "message"),
    [
        ([], "first message must be from the user"),
        ([{"role": "assistant", "content": "x"}], "first message must be from the user"),
        ([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], "roles must alternate"),
        ([{"role": "user", "content": "q"}, tool_use_turn("a")], "needs its tool_results"),
        (
            [{"role": "user", "content": "q"}, tool_use_turn("a"), {"role": "user", "content": "text"}],
            "must be a list",
        ),
        (
            [{"role": "user", "content": "q"}, tool_use_turn("a", "b"), results_turn("a")],
            "one tool_result per tool_use",
        ),
        (
            [{"role": "user", "content": "q"}, tool_use_turn("a", "b"), results_turn("b", "a")],
            "in order",
        ),
        (
            [
                {"role": "user", "content": "q"},
                tool_use_turn("a"),
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": []}]},
            ],
            "invalid tool_result",
        ),
    ],
)
def test_validate_request_names_what_the_api_would_reject(messages, message) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        validate_request({"messages": messages})


# --- the corpus -----------------------------------------------------------------


def test_the_corpus_can_be_built_from_a_subset() -> None:
    primary = json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8"))
    proxy = json.loads(TRANSCRIPTION_PROXY_ARTIFACT.read_text(encoding="utf-8"))
    corpus = ScriptedCorpus([COFFEE], {}, primary, proxy)

    assert corpus.recognize(COFFEE.image_path.read_bytes(), "image/png").gold_text is None
    assert corpus.recognize(JPEG_CASE.image_path.read_bytes(), "image/jpeg") is None


# --- the startup log ------------------------------------------------------------


async def test_the_api_lifespan_logs_the_provider_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app import main as main_module

    monkeypatch.setattr(main_module, "warm_up_embedder", lambda: None)
    monkeypatch.setattr(main_module, "get_settings", lambda: scripted_settings())
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    async with main_module.app.router.lifespan_context(main_module.app):
        pass

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("model provider:")]
    assert lines == [scripted_settings().model_provider_summary()]


def stub_worker_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, worker_module: Any
) -> Settings:
    """Everything worker.main() touches outside the provider check, replaced, and
    the scripted settings it reads."""
    from contextlib import asynccontextmanager

    async def nothing(*args: Any, **kwargs: Any) -> None:
        return None

    @asynccontextmanager
    async def no_queue(*args: Any, **kwargs: Any):
        yield None

    settings = scripted_settings(worker_heartbeat_file=str(tmp_path / "heartbeat"))
    monkeypatch.setattr(worker_module, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_module, "verify_tenant_isolation", nothing)
    monkeypatch.setattr(worker_module, "queue_session", no_queue)
    monkeypatch.setattr(worker_module, "reclaim_orphaned_jobs", nothing)
    monkeypatch.setattr(worker_module, "warm_up_embedder", lambda: None)
    monkeypatch.setattr(worker_module, "run_worker", nothing)
    # main() flushes and shuts down the process-wide tracer provider on the way
    # out, which would silence every later test that reads spans.
    monkeypatch.setattr(worker_module, "shutdown_tracing", lambda: None)
    monkeypatch.setattr(worker_module, "engine", SimpleNamespace(dispose=nothing))
    return settings


async def test_the_worker_logs_the_provider_once_at_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    from app import worker as worker_module

    settings = stub_worker_startup(monkeypatch, tmp_path, worker_module)
    caplog.set_level(logging.INFO, logger=worker_module.logger.name)

    try:
        await asyncio.wait_for(worker_module.main(), timeout=10)
    finally:
        # main() installs a SIGTERM handler on the suite's one event loop.
        asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("model provider:")]
    assert lines == [settings.model_provider_summary()]


# --- a scripted process refuses to start without its corpus ------------------------


@pytest.fixture
def no_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The production image's view: no evals/ directory beside app/. The cached
    corpus is dropped so the load really runs (a failed load is never cached)."""
    monkeypatch.setattr("app.evals.dataset.LABELS_DIR", tmp_path / "no-evals" / "labels")
    monkeypatch.setattr("app.scripted_model.REPLAY_ARTIFACT", tmp_path / "no-evals" / "run.json")
    default_corpus.cache_clear()
    yield
    default_corpus.cache_clear()


def test_a_missing_corpus_is_a_clear_error_at_startup(no_corpus: None) -> None:
    with pytest.raises(RuntimeError, match="MODEL_PROVIDER=scripted needs the committed eval corpus"):
        require_corpus()


def test_the_real_provider_needs_no_corpus(no_corpus: None) -> None:
    check_model_provider(Settings(anthropic_api_key="not-a-real-key"))


async def test_the_api_refuses_to_start_without_the_corpus(
    monkeypatch: pytest.MonkeyPatch, no_corpus: None
) -> None:
    from app import main as main_module

    monkeypatch.setattr(main_module, "warm_up_embedder", lambda: None)
    monkeypatch.setattr(main_module, "get_settings", lambda: scripted_settings())

    with pytest.raises(RuntimeError, match="needs the committed eval corpus"):
        async with main_module.app.router.lifespan_context(main_module.app):
            pytest.fail("the API served without its corpus")


async def test_the_worker_refuses_to_start_without_the_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_corpus: None
) -> None:
    from app import worker as worker_module

    stub_worker_startup(monkeypatch, tmp_path, worker_module)

    with pytest.raises(RuntimeError, match="needs the committed eval corpus"):
        await asyncio.wait_for(worker_module.main(), timeout=10)
