"""The /ask agent: its tools against a seeded corpus, and the loop against
a scripted fake client (no network)."""

import uuid
from decimal import Decimal as D
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from v2_samples import seed_v2

from app.agent import loop as loop_module
from app.agent import tools as tools_module
from app.agent.loop import (
    AGENT_PROMPT_TEXT,
    FALLBACK_BETA,
    _assemble_answer,
    _echoable,
    answer_question,
    call_cost_usd,
)
from app.agent.tools import (
    SUMMARY_SOURCE,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    MalformedToolResult,
    ToolContext,
    ToolError,
    _matches,
    assert_tool_result_content,
    extraction_summary,
    get_page,
    query_extractions,
    record_lines,
    search_documents,
)
from app.config import Settings
from app.evals.corpus import seed_labeled_corpus
from app.evals.dataset import load_cases
from app.evals.retrieval import load_gold_corpus
from app.main import app
from app.records import DocumentRecord, load_records
from app.retrieval.embeddings import HashingEmbedder, get_embedder
from app.retrieval.indexing import ChunkingConfig
from app.routers.ask import get_anthropic_client

EMBEDDER = HashingEmbedder()
DOCS = (
    "001-clean-coffee-receipt",
    "018-dense-office-outfitters",
    "024-eur-bakery-berlin",
    "025-adversarial-office-order-injection",
)


async def _seed(db_session: AsyncSession) -> tuple[ToolContext, dict[str, uuid.UUID]]:
    cases = [case for case in load_cases() if case.doc_id in DOCS]
    ids = await seed_labeled_corpus(
        db_session, cases, load_gold_corpus(cases), EMBEDDER, ChunkingConfig(200, 80, True)
    )
    return ToolContext(session=db_session, embedder=EMBEDDER, document_ids=list(ids.values())), ids


def _texts(content) -> list[str]:
    return [block["text"] for block in content if block["type"] == "text"]


def _results(content) -> list[dict]:
    """The document search_results, without a query's match summary."""
    return [
        block
        for block in content
        if block["type"] == "search_result" and block["source"] != SUMMARY_SOURCE
    ]


def _summary(content) -> str:
    """A query_extractions match summary; it is a search_result because the
    API takes a tool result as all search_results or none."""
    (block,) = [b for b in content if b["type"] == "search_result" and b["source"] == SUMMARY_SOURCE]
    return block["content"][0]["text"]


# --- tools ------------------------------------------------------------------


def test_tool_definitions_are_strict_with_closed_schemas() -> None:
    for tool in TOOL_DEFINITIONS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False


async def test_query_extractions_sums_per_currency_in_code(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)

    content = await query_extractions(ctx, {"vendor": "northgate"})

    assert _summary(content) == "2 matching document(s). Sum of totals: 488.23 USD."
    assert all(block["type"] == "search_result" for block in content)
    results = _results(content)
    assert len(results) == 2
    assert all(result["citations"] == {"enabled": True} for result in results)
    source = ctx.sources[results[0]["source"]]
    assert source.kind == "record" and source.page_number is None
    lines = [block["text"] for block in results[0]["content"]]
    assert lines[0] == "vendor: Northgate Office Outfitters"
    assert source.blocks[lines.index("total: 425.58")] == "total"


async def test_query_extractions_filters_combine(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)

    eur = await query_extractions(ctx, {"currency": "eur"})
    assert _summary(eur) == "1 matching document(s). Sum of totals: 27.82 EUR."

    item = await query_extractions(ctx, {"item": "led desk lamp", "max_total": 500})
    assert _summary(item).startswith("1 matching document(s).")

    none = await query_extractions(ctx, {"vendor": "Target"})
    assert _texts(none) == ["0 matching document(s)."]


async def test_query_extractions_date_range_and_bad_dates(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)

    content = await query_extractions(ctx, {"date_from": "2026-05-01", "date_to": "2026-05-31"})
    assert "27.82 EUR" in _summary(content)

    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        await query_extractions(ctx, {"date_from": "May 2026"})


def _record(total, currency="USD", **fields) -> DocumentRecord:
    return DocumentRecord(
        uuid.uuid4(), "r.png", {"total": total, "currency": currency, **fields}, []
    )


def test_a_string_total_is_counted_summed_and_matched() -> None:
    # A stored human correction will be a canonical string, not a float.
    records = [_record("425.58"), _record(62.65)]

    assert extraction_summary(records) == "2 matching document(s). Sum of totals: 488.23 USD."

    assert _matches(records[0], {}, None, None, D("425.58"), None)
    assert _matches(records[0], {}, None, None, None, D("425.58"))
    assert not _matches(records[0], {}, None, None, D("425.59"), None)
    assert not _matches(records[0], {}, None, None, None, D("425.57"))


def test_an_unparseable_total_counts_as_missing_not_a_crash() -> None:
    records = [_record("425.58"), _record("about 60"), _record(float("nan")), _record(None)]

    assert extraction_summary(records) == (
        "4 matching document(s). Sum of totals: 425.58 USD. 3 matching document(s) have no total."
    )
    assert not _matches(records[1], {}, None, None, D(0), None)
    assert not _matches(records[1], {}, None, None, None, D(1000))
    assert _matches(records[1], {}, None, None, None, None)


def test_a_total_beyond_the_amount_bound_is_not_summed() -> None:
    records = [_record(1e12), _record("999999999999.99"), _record(0.01)]

    assert extraction_summary(records) == (
        "3 matching document(s). Sum of totals: 1000000000000.00 USD. "
        "1 matching document(s) have no total."
    )
    assert not _matches(records[0], {}, None, None, None, D("1E+13"))


def test_sums_keep_every_digit_instead_of_rounding_to_the_decimal_context() -> None:
    records = [_record("1.0000000000000000000000000000001"), _record(1)]

    assert extraction_summary(records) == (
        "2 matching document(s). Sum of totals: 2.0000000000000000000000000000001 USD."
    )


async def test_a_bad_total_filter_is_a_tool_error(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)

    with pytest.raises(ToolError, match="min_total"):
        await query_extractions(ctx, {"min_total": True})
    with pytest.raises(ToolError, match="max_total"):
        await query_extractions(ctx, {"max_total": float("nan")})


def test_sums_and_lines_use_each_currencys_digits() -> None:
    yen = [_record(1500, "JPY"), _record("2500", "jpy")]
    assert extraction_summary(yen) == "2 matching document(s). Sum of totals: 4000 JPY."

    mixed = [_record(1500, "JPY"), _record(10.5, "EUR"), _record(1.5, "KWD"), _record(2, "XYZ")]
    assert extraction_summary(mixed) == (
        "4 matching document(s). Sum of totals: 10.50 EUR; 1500 JPY; 1.500 KWD; 2.00 XYZ."
    )

    lines = dict(
        record_lines(
            _record(
                1500,
                "JPY",
                subtotal="1364",
                tax=136,
                line_items=[{"description": "Tea", "quantity": 2, "unit_price": 682, "total": 1364.0}],
            )
        )
    )
    assert lines["total"] == "total: 1500"
    assert lines["subtotal"] == "subtotal: 1364"
    assert lines["line_items[0]"] == "line item: Tea, qty 2, unit price 682, line total 1364"
    assert dict(record_lines(_record(46.2)))["total"] == "total: 46.20"


def test_a_v1_records_lines_are_exactly_the_v1_lines() -> None:
    record = DocumentRecord(
        uuid.uuid4(),
        "r.png",
        {
            "vendor": "Acme",
            "document_date": "2026-01-01",
            "currency": "EUR",
            "subtotal": "10.5",
            "tax": None,
            "total": 12,
            "line_items": [
                {"description": "Tea", "quantity": 2, "unit_price": 5.25, "total": 10.5},
                {"description": None, "quantity": None, "unit_price": None, "total": None},
            ],
        },
        ["total", "tax"],
    )

    assert record.prompt_version == "extract_v1"
    assert record_lines(record) == [
        ("vendor", "vendor: Acme"),
        ("document_date", "date: 2026-01-01"),
        ("currency", "currency: EUR"),
        ("subtotal", "subtotal: 10.50"),
        ("tax", "tax: not shown"),
        ("total", "total: 12.00"),
        ("line_items[0]", "line item: Tea, qty 2, unit price 5.25, line total 10.50"),
        ("line_items[1]", "line item: item"),
        ("review_status", "note: not yet human-reviewed and low-confidence: total, tax"),
    ]


def test_a_v2_record_without_the_new_fields_has_exactly_the_v1_lines() -> None:
    fields = {"vendor": "Acme", "document_date": None, "currency": "USD", "total": 5}
    v1 = DocumentRecord(uuid.uuid4(), "r.png", dict(fields), [])
    # The v2 fields it lacks, whether absent or null, add no line.
    v2 = DocumentRecord(
        uuid.uuid4(),
        "r.png",
        {**fields, "invoice_number": None, "tax_lines": [], "discount": None},
        [],
        "extract_v2",
    )

    assert record_lines(v2) == record_lines(v1)


async def test_a_v2_record_adds_a_line_for_each_v2_field_it_has(db_session: AsyncSession) -> None:
    document_id, _, _ = await seed_v2(db_session)

    (record,) = await load_records(db_session, [document_id])

    assert record.prompt_version == "extract_v2"
    assert record_lines(record) == [
        ("vendor", "vendor: Acme Corp"),
        ("document_date", "date: 2026-01-01"),
        ("currency", "currency: USD"),
        ("subtotal", "subtotal: 100.00"),
        ("tax", "tax: 7.20"),
        ("total", "total: 102.20"),
        ("line_items[0]", "line item: Widget, qty 2, unit price 50.00, line total 100.00"),
        ("invoice_number", "invoice_number: INV-0042"),
        ("po_number", "po_number: PO-77"),
        ("vendor_tax_id", "vendor_tax_id: 12-3456789"),
        ("due_date", "due_date: 2026-01-31"),
        ("payment_terms", "payment_terms: Net 30"),
        ("payment_method", "payment_method: VISA"),
        ("payment_card_last4", "payment_card_last4: 4242"),
        ("discount", "discount: 10.00"),
        ("shipping", "shipping: 5.00"),
        ("tax_lines[0]", "tax line: State tax, rate 8%, base 90.00, amount 7.20"),
        ("amount_paid", "amount_paid: 50.00"),
        ("amount_due", "amount_due: 52.20"),
    ]


async def test_the_labeled_corpus_loads_as_v1_records(db_session: AsyncSession) -> None:
    _, ids = await _seed(db_session)

    records = await load_records(db_session, list(ids.values()))

    assert {record.prompt_version for record in records} == {"extract_v1"}


async def test_search_documents_returns_line_level_citable_results(
    db_session: AsyncSession,
) -> None:
    ctx, _ = await _seed(db_session)

    content = await search_documents(ctx, {"query": "LED Desk Lamp", "k": 3})

    results = _results(content)
    assert results and "page 1" in results[0]["title"]
    source = ctx.sources[results[0]["source"]]
    assert source.kind == "chunk"
    pages = {}
    from sqlalchemy import select

    from app.models import DocumentPage

    for page in (await db_session.execute(select(DocumentPage))).scalars():
        pages[(page.document_id, page.page_number)] = page.text
    page_text = pages[(source.document_id, source.page_number)]
    for block, (start, end) in zip(results[0]["content"], source.blocks, strict=True):
        assert page_text[start:end] == block["text"]


async def test_search_documents_clamps_k_and_rejects_empty(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)

    content = await search_documents(ctx, {"query": "receipt", "k": 500})
    assert len(_results(content)) <= 10
    with pytest.raises(ToolError):
        await search_documents(ctx, {"query": "   "})


async def test_get_page_validates_ids_and_page_numbers(db_session: AsyncSession) -> None:
    ctx, ids = await _seed(db_session)
    document_id = str(ids["018-dense-office-outfitters"])

    content = await get_page(ctx, {"document_id": document_id, "page_number": 1})
    assert _results(content)[0]["content"][0]["text"] == "Northgate Office Outfitters"

    with pytest.raises(ToolError, match="pages 1-1"):
        await get_page(ctx, {"document_id": document_id, "page_number": 3})
    with pytest.raises(ToolError, match="document id"):
        await get_page(ctx, {"document_id": "not-a-uuid", "page_number": 1})
    with pytest.raises(ToolError, match="no document"):
        await get_page(ctx, {"document_id": str(uuid.uuid4()), "page_number": 1})


# --- scripted model ---------------------------------------------------------


def _usage(input_tokens=1000, output_tokens=200, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def _message(content, stop_reason, usage=None, model="claude-opus-5-5", stop_details=None):
    return SimpleNamespace(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=usage or _usage(),
    )


def _tool_use(name, tool_input, tool_id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _text(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def validate_request(kwargs: dict) -> None:
    """The request rules the real API enforces and a fake would not: roles
    alternate starting with the user, and the user message after a tool_use
    turn opens with exactly one tool_result per tool_use id, in order, each
    with a valid body (assert_tool_result_content)."""
    messages = kwargs["messages"]
    assert messages and messages[0]["role"] == "user"
    for before, after in pairwise(messages):
        assert before["role"] != after["role"], "roles must alternate"
    for index, message in enumerate(messages):
        content = message["content"]
        if message["role"] == "assistant" and not isinstance(content, str):
            tool_ids = [b.id for b in content if getattr(b, "type", None) == "tool_use"]
            if not tool_ids:
                continue
            assert index + 1 < len(messages), "a tool_use turn needs its tool_results"
            results = messages[index + 1]["content"]
            assert isinstance(results, list)
            head = results[: len(tool_ids)]
            assert [r["type"] for r in head] == ["tool_result"] * len(tool_ids)
            assert [r["tool_use_id"] for r in head] == tool_ids
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    assert_tool_result_content(block["content"])


def _scripted_create(scripted: list):
    """A fake `messages.create`: validates each request, then plays the next
    scripted step (an exception to raise, a callable given the request, or a
    response)."""

    async def create(**kwargs):
        validate_request(kwargs)
        step = scripted.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(**kwargs) if callable(step) else step

    return create


def _client(responses) -> SimpleNamespace:
    create = AsyncMock(side_effect=_scripted_create(list(responses or [])))
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def _cite_first_record(_unused, field_line: str):
    """Build a citation the way the API would: pointing at the line
    `field_line` in the search_result of the most recent tool result that
    contains it."""

    def respond(**kwargs):
        tool_result = kwargs["messages"][-1]["content"][0]
        record = next(
            b
            for b in tool_result["content"]
            if b["type"] == "search_result" and any(c["text"] == field_line for c in b["content"])
        )
        index = next(i for i, b in enumerate(record["content"]) if b["text"] == field_line)
        citation = SimpleNamespace(
            type="search_result_location",
            source=record["source"],
            title=record["title"],
            cited_text=field_line,
            search_result_index=0,
            start_block_index=index,
            end_block_index=index + 1,
        )
        return _message(
            [_text("Your Northgate order came to $425.58.", [citation])], "end_turn"
        )

    return respond


SETTINGS = Settings(
    agent_model="claude-opus-5-5",
    agent_effort="medium",
    anthropic_api_key="test",
    daily_budget_usd=0,
    ask_rate_limit_per_minute=0,
)


async def test_agent_answers_with_verified_citations(db_session: AsyncSession) -> None:
    ctx, ids = await _seed(db_session)
    first = _message(
        [_tool_use("query_extractions", {"vendor": "Northgate", "max_total": 500})], "tool_use"
    )
    scripted = [first, _cite_first_record(None, "total: 425.58")]

    client = _client(scripted)

    result = await answer_question(ctx, "How much was my Northgate order?", SETTINGS, client)

    assert result.status == "answered"
    assert result.answer == "Your Northgate order came to $425.58. [1]"
    (citation,) = result.citations
    assert citation.kind == "record"
    assert citation.fields == ["total"]
    assert citation.cited_text == "total: 425.58"
    assert uuid.UUID(citation.document_id) in ids.values()
    assert result.unresolved_citations == 0
    assert result.steps == 2
    assert [call.name for call in result.tool_calls] == ["query_extractions"]
    assert result.tool_calls[0].result_summary.startswith("2 source(s); 2 matching document(s)")
    # claude-opus-5-5 at $4/$20 per MTok, two calls of 1000 in / 200 out
    assert result.cost_usd == pytest.approx(2 * (1000 * 4 + 200 * 20) / 1e6)

    request = client.beta.messages.create.await_args_list[0].kwargs
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == AGENT_PROMPT_TEXT
    assert request["output_config"] == {"effort": "medium"}
    assert request["thinking"] == {"type": "adaptive"}
    assert request["betas"] == [FALLBACK_BETA] and request["fallbacks"] == "default"
    assert request["cache_control"] == {"type": "ephemeral"}
    assert "tool_choice" not in request  # forced tool use 400s on this model
    roles = [m["role"] for m in result.messages]
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_tool_errors_go_back_to_the_model_and_the_run_continues(
    db_session: AsyncSession,
) -> None:
    ctx, _ = await _seed(db_session)
    client = _client(
        [
            _message([_tool_use("get_page", {"document_id": "nope", "page_number": 1})], "tool_use"),
            _message([_text("I couldn't open that page.")], "end_turn"),
        ]
    )

    result = await answer_question(ctx, "Show me page 1", SETTINGS, client)

    assert result.status == "answered"
    assert result.tool_calls[0].is_error is True
    # The mock keeps a reference to the (append-only) messages list, so read
    # the tool result by position rather than as "the last message".
    tool_result = client.beta.messages.create.await_args_list[1].kwargs["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "document id" in tool_result["content"][0]["text"]


async def test_unknown_tool_is_an_error_result_not_a_crash(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)
    client = _client(
        [
            _message([_tool_use("delete_everything", {})], "tool_use"),
            _message([_text("Done.")], "end_turn"),
        ]
    )

    result = await answer_question(ctx, "q", SETTINGS, client)

    assert result.tool_calls[0].is_error is True
    assert result.status == "answered"


async def test_refusal_is_a_status_not_an_exception(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)
    client = _client(
        [_message([], "refusal", stop_details=SimpleNamespace(category="cyber", explanation=""))]
    )

    result = await answer_question(ctx, "q", SETTINGS, client)

    assert result.status == "refused"
    assert result.refusal_category == "cyber"
    assert result.answer == ""


async def test_truncated_answer_is_marked(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)
    client = _client([_message([_text("Partial answ")], "max_tokens")])

    result = await answer_question(ctx, "q", SETTINGS, client)

    assert result.status == "truncated"
    assert result.answer == "Partial answ"


async def test_step_limit_stops_a_tool_loop(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)
    settings = SETTINGS.model_copy(update={"agent_max_steps": 3})
    client = _client(
        [_message([_tool_use("query_extractions", {}, f"t{i}")], "tool_use") for i in range(3)]
    )

    result = await answer_question(ctx, "q", settings, client)

    assert result.status == "step_limit"
    assert result.steps == 3
    assert client.beta.messages.create.await_count == 3


async def test_cost_budget_stops_before_running_more_tools(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)
    settings = SETTINGS.model_copy(update={"agent_max_cost_usd": 0.01})
    client = _client(
        [_message([_tool_use("query_extractions", {})], "tool_use", _usage(5000, 1000))]
    )

    result = await answer_question(ctx, "q", settings, client)

    assert result.status == "budget_exceeded"
    assert result.tool_calls == []


def test_cost_prices_the_serving_model_and_cache_tokens() -> None:
    response = _message([], "end_turn", _usage(1000, 100, cache_read=10_000, cache_write=2000))

    # Opus 5.5: $4 in / $20 out, cache reads at 0.05x, writes at 1.25x.
    assert call_cost_usd("claude-opus-5-5", response) == pytest.approx(
        (4 * (1000 + 2000 * 1.25 + 10_000 * 0.05) + 20 * 100) / 1e6
    )
    served_by_fallback = _message([], "end_turn", _usage(1000, 100), model="claude-opus-4-8")
    assert call_cost_usd("claude-opus-5-5", served_by_fallback) == pytest.approx(
        (5 * 1000 + 25 * 100) / 1e6
    )


def test_echo_drops_declined_blocks_before_a_fallback_boundary() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="", signature="x")
    declined_tool = _tool_use("search_documents", {"query": "x"})
    boundary = SimpleNamespace(type="fallback")
    after = _text("continuing")

    assert _echoable([thinking, declined_tool, boundary, after]) == [boundary, after]
    plain = [thinking, _text("normal")]
    assert _echoable(plain) is plain


def test_models_without_effort_get_plain_requests() -> None:
    options = loop_module._request_options(
        Settings(agent_model="claude-haiku-4-5", agent_refusal_fallback=True)
    )
    assert options == {}
    no_fallback = loop_module._request_options(
        Settings(agent_model="claude-opus-5-5", agent_refusal_fallback=False)
    )
    assert "fallbacks" not in no_fallback and no_fallback["output_config"] == {"effort": "medium"}


def test_agent_max_tokens_stays_under_the_sdks_non_streaming_ceiling() -> None:
    assert Settings(agent_max_tokens=21_000).agent_max_tokens == 21_000
    with pytest.raises(ValueError, match="agent_max_tokens"):
        Settings(agent_max_tokens=21_001)


async def test_a_multi_block_citation_keeps_one_line_per_block(db_session: AsyncSession) -> None:
    """The API's cited_text runs the cited blocks together ("subtotal:
    393.15tax: 32.43"); the stored citation shows each line on its own."""
    ctx, _ = await _seed(db_session)
    content = await query_extractions(ctx, {"vendor": "northgate", "min_total": 400})
    (record,) = _results(content)
    texts = [block["text"] for block in record["content"]]
    start = next(i for i, text in enumerate(texts) if text.startswith("subtotal:"))
    cited = texts[start : start + 2]
    citation = SimpleNamespace(
        type="search_result_location",
        source=record["source"],
        title=record["title"],
        cited_text="".join(cited),
        search_result_index=0,
        start_block_index=start,
        end_block_index=start + 2,
    )

    _, (resolved,), unresolved = _assemble_answer([_text("Before tax.", [citation])], ctx.sources)

    assert resolved.cited_text == "\n".join(cited) == "subtotal: 393.15\ntax: 32.43"
    assert resolved.fields == ["subtotal", "tax"] and unresolved == 0


def test_unresolvable_citations_are_counted_not_trusted() -> None:
    stray = SimpleNamespace(
        source="https://elsewhere", cited_text="x", start_block_index=0, end_block_index=1
    )

    answer, citations, unresolved = _assemble_answer([_text("Claim.", [stray])], {})

    assert answer == "Claim."
    assert citations == [] and unresolved == 1


def test_a_match_summary_citation_is_skipped_not_counted() -> None:
    summary = SimpleNamespace(
        source=SUMMARY_SOURCE, cited_text="1 matching document(s).", start_block_index=0, end_block_index=1
    )

    answer, citations, unresolved = _assemble_answer([_text("One match.", [summary])], {})

    assert answer == "One match."
    assert citations == [] and unresolved == 0


# --- tool results the API accepts -----------------------------------------------


def _line(text="x"):
    return {"type": "text", "text": text}


def _result(**overrides):
    block = {
        "type": "search_result",
        "source": "doc-pilot://documents/1",
        "title": "t",
        "content": [_line()],
        "citations": {"enabled": True},
    }
    return {**block, **overrides}


@pytest.mark.parametrize(
    "content",
    [
        [],
        None,
        [_line("")],
        [_line("   ")],
        [{"type": "image"}],
        [_line(), _result()],
        [_result(source="")],
        [_result(title=None)],
        [_result(content=[])],
        [_result(content=[_line("")])],
        [_result(citations={})],
        [_result(citations={"enabled": "yes"})],
        [_result(citations=None)],
    ],
)
def test_malformed_tool_results_are_rejected(content) -> None:
    with pytest.raises(MalformedToolResult):
        assert_tool_result_content(content)


def test_well_formed_tool_results_pass() -> None:
    assert_tool_result_content([_line()])
    assert_tool_result_content([_result(), _result(citations={"enabled": False})])


# Every tool, through the loop's own _run_tool, with inputs that give each
# kind of result: found, nothing found, a tool error, a truncated listing.
# A new tool without a row here fails test_every_tool_has_a_result_case.
RESULT_CASES = [
    ("query_extractions", "found", lambda ids: {"vendor": "northgate"}, False),
    ("query_extractions", "none", lambda ids: {"vendor": "Target"}, False),
    ("query_extractions", "bad-date", lambda ids: {"date_from": "May 2026"}, True),
    ("query_extractions", "truncated", lambda ids: {}, False),
    ("search_documents", "found", lambda ids: {"query": "desk lamp"}, False),
    ("search_documents", "empty-query", lambda ids: {"query": " "}, True),
    ("get_page", "found", lambda ids: {"document_id": str(ids[DOCS[1]]), "page_number": 1}, False),
    ("get_page", "no-such-page", lambda ids: {"document_id": str(ids[DOCS[1]]), "page_number": 99}, True),
    ("get_page", "bad-id", lambda ids: {"document_id": "nope", "page_number": 1}, True),
]


def test_every_tool_has_a_result_case() -> None:
    assert {tool for tool, *_ in RESULT_CASES} == set(TOOL_HANDLERS)


@pytest.mark.parametrize(
    ("tool", "case", "build_input", "is_error"),
    RESULT_CASES,
    ids=[f"{tool}-{case}" for tool, case, *_ in RESULT_CASES],
)
async def test_every_tool_result_is_valid_for_the_api(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, tool, case, build_input, is_error
) -> None:
    if case == "truncated":
        monkeypatch.setattr(tools_module, "RECORDS_MAX", 1)
    ctx, ids = await _seed(db_session)
    block = _tool_use(tool, build_input(ids))

    result, call = await loop_module._run_tool(ctx, block)

    content = result["content"]
    assert_tool_result_content(content)
    assert len({b["type"] for b in content}) == 1
    assert bool(result.get("is_error")) is is_error and call.is_error is is_error
    if case == "truncated":
        assert len(_results(content)) == 1
        assert "Listing the first 1" in _summary(content)


async def test_an_unknown_tool_is_an_error_result(db_session: AsyncSession) -> None:
    ctx, _ = await _seed(db_session)

    result, call = await loop_module._run_tool(ctx, _tool_use("nope", {}))

    assert result["is_error"] is True and call.is_error
    assert_tool_result_content(result["content"])


async def test_a_tool_that_builds_a_bad_result_fails_its_call_not_the_run(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def mixed(ctx, tool_input):
        return [_line(), _result()]

    monkeypatch.setitem(TOOL_HANDLERS, "mixed", mixed)
    ctx, _ = await _seed(db_session)

    result, call = await loop_module._run_tool(ctx, _tool_use("mixed", {}))

    assert result["is_error"] is True and call.is_error
    assert_tool_result_content(result["content"])
    assert "mixes text and search_result" in caplog.text


async def test_search_with_every_hit_skipped_says_nothing_matched(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_module, "_line_blocks", lambda text, start: ([], []))
    ctx, _ = await _seed(db_session)

    content = await search_documents(ctx, {"query": "desk lamp"})

    assert _texts(content) == ["No matching passages found."]


# --- API --------------------------------------------------------------------


async def test_ask_endpoint_requires_an_api_key(client: AsyncClient) -> None:
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(anthropic_api_key=None, ask_rate_limit_per_minute=0)

    response = await client.post("/ask", json={"question": "anything"})

    assert response.status_code == 503


async def test_ask_endpoint_returns_answer_citations_and_cost(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    await _seed(db_session)
    app.dependency_overrides[get_settings] = lambda: SETTINGS
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    scripted = [
        _message([_tool_use("query_extractions", {"vendor": "Northgate"})], "tool_use"),
        _cite_first_record(None, "total: 425.58"),
    ]

    fake = _client(scripted)
    app.dependency_overrides[get_anthropic_client] = lambda: fake

    response = await client.post("/ask", json={"question": "How much was my Northgate order?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"].endswith("[1]")
    assert body["citations"][0]["fields"] == ["total"]
    assert body["tool_calls"][0]["name"] == "query_extractions"
    assert body["cost_usd"] > 0


def _northgate_client() -> SimpleNamespace:
    scripted = [
        _message([_tool_use("query_extractions", {"vendor": "Northgate"})], "tool_use"),
        _cite_first_record(None, "total: 425.58"),
    ]

    return _client(scripted)


async def _ask(client: AsyncClient, db_session: AsyncSession, settings: Settings, fake) -> dict:
    from app.config import get_settings

    await _seed(db_session)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    app.dependency_overrides[get_anthropic_client] = lambda: fake
    response = await client.post("/ask", json={"question": "How much was my Northgate order?"})
    assert response.status_code == 200
    return response.json()


async def test_every_answer_is_stored_with_its_evidence(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    from app.models import AskRun, Job

    body = await _ask(client, db_session, SETTINGS, _northgate_client())

    stored = (await client.get(f"/ask/runs/{body['id']}")).json()
    assert stored == body
    assert stored["question"] == "How much was my Northgate order?"
    assert stored["citations"][0]["fields"] == ["total"]
    assert stored["feedback"] is None and stored["judgment"] is None
    run = await db_session.get(AskRun, uuid.UUID(body["id"]))
    # What the grader will read: the tool results as the agent saw them.
    assert "total: 425.58" in run.evidence
    # Sampling is off by default: no judge job, no model call nobody asked for.
    assert run.judge_sampled is False
    jobs = (await db_session.execute(select(Job).where(Job.ask_run_id == run.id))).scalars().all()
    assert jobs == []


async def test_a_sampled_answer_gets_a_judge_job_on_its_own_trace(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    from app.models import JOB_KIND_JUDGE, AskRun, Job

    settings = SETTINGS.model_copy(update={"ask_judge_sample_rate": 1.0})
    body = await _ask(client, db_session, settings, _northgate_client())

    run = await db_session.get(AskRun, uuid.UUID(body["id"]))
    (job,) = (await db_session.execute(select(Job).where(Job.ask_run_id == run.id))).scalars().all()
    assert body["judge_sampled"] is True
    assert (job.kind, job.state, job.document_id) == (JOB_KIND_JUDGE, "pending", None)
    assert job.traceparent == run.traceparent


async def test_a_refusal_is_stored_but_never_graded(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    settings = SETTINGS.model_copy(update={"ask_judge_sample_rate": 1.0})
    refusal = _client([_message([], "refusal", stop_details=SimpleNamespace(category="cyber"))])

    body = await _ask(client, db_session, settings, refusal)

    assert body["status"] == "refused" and body["judge_sampled"] is False
    assert (await client.get(f"/ask/runs/{body['id']}")).json()["status"] == "refused"


async def test_feedback_is_recorded_changed_and_validated(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    body = await _ask(client, db_session, SETTINGS, _northgate_client())
    url = f"/ask/runs/{body['id']}/feedback"

    up = await client.post(url, json={"rating": "up", "note": "exactly right"})
    assert up.status_code == 200
    assert (up.json()["feedback"], up.json()["feedback_note"]) == ("up", "exactly right")

    # The latest rating wins, and replaces the note.
    down = (await client.post(url, json={"rating": "down"})).json()
    assert (down["feedback"], down["feedback_note"]) == ("down", None)
    assert (await client.get(f"/ask/runs/{body['id']}")).json()["feedback"] == "down"

    assert (await client.post(url, json={"rating": "meh"})).status_code == 422
    assert (await client.post(url, json={"rating": "up", "note": "x" * 1001})).status_code == 422
    missing = f"/ask/runs/{uuid.uuid4()}/feedback"
    assert (await client.post(missing, json={"rating": "up"})).status_code == 404
    assert (await client.get(f"/ask/runs/{uuid.uuid4()}")).status_code == 404


async def test_run_history_is_newest_first(client: AsyncClient, db_session: AsyncSession) -> None:
    from datetime import UTC, datetime, timedelta

    from app.models import AskRun

    now = datetime.now(UTC)
    older, newer = (
        AskRun(
            question=question, status="answered", answer="a", citations=[], tool_calls=[],
            evidence="e", steps=1, model="claude-opus-5-5", prompt_version="agent_v1",
            created_at=now + timedelta(seconds=offset),
        )
        for question, offset in (("older?", 0), ("newer?", 1))
    )
    db_session.add_all([older, newer])
    await db_session.commit()

    listed = (await client.get("/ask/runs", params={"limit": 100})).json()

    ids = [row["id"] for row in listed]
    assert ids.index(str(newer.id)) < ids.index(str(older.id))
    row = listed[ids.index(str(older.id))]
    assert (row["question"], row["judge_sampled"], row["judge_grounded"]) == ("older?", False, None)
    assert (await client.get("/ask/runs", params={"limit": 0})).status_code == 422


def _sse_events(body: str) -> list[dict]:
    import json

    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        event = json.loads(lines["data"])
        assert event["type"] == lines["event"]
        events.append(event)
    return events


async def _stream(client: AsyncClient, db_session: AsyncSession, settings: Settings, fake):
    from app.config import get_settings

    await _seed(db_session)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    app.dependency_overrides[get_anthropic_client] = lambda: fake
    return await client.post("/ask/stream", json={"question": "How much was my Northgate order?"})


async def test_stream_shows_each_step_then_the_stored_answer(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    response = await _stream(client, db_session, SETTINGS, _northgate_client())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert [e["type"] for e in events] == ["model_call", "tool_start", "tool_call", "model_call", "answer"]
    first_call, tool_start, tool_call, last_call, answer = events
    assert (first_call["step"], first_call["stop_reason"]) == (1, "tool_use")
    assert tool_start == {"type": "tool_start", "name": "query_extractions", "input": {"vendor": "Northgate"}}
    assert tool_call["name"] == "query_extractions" and tool_call["is_error"] is False
    assert (last_call["step"], last_call["stop_reason"]) == (2, "end_turn")
    assert last_call["cost_usd"] > first_call["cost_usd"] > 0
    # The final event is the stored run -- exactly what GET /ask/runs/{id} returns.
    run = answer["run"]
    assert run["status"] == "answered" and run["citations"][0]["fields"] == ["total"]
    assert (await client.get(f"/ask/runs/{run['id']}")).json() == run


async def test_stream_turns_api_errors_into_an_error_event(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    import anthropic
    import httpx

    from app.models import AskRun

    overloaded = anthropic.InternalServerError(
        "overloaded", response=httpx.Response(529, request=httpx.Request("POST", "https://x")), body=None
    )
    before = (await db_session.execute(select(func.count()).select_from(AskRun))).scalar_one()

    response = await _stream(client, db_session, SETTINGS, _client([overloaded]))

    (event,) = _sse_events(response.text)
    assert event["type"] == "error" and event["status"] == 503
    assert "model unavailable" in event["detail"]
    assert (await db_session.execute(select(func.count()).select_from(AskRun))).scalar_one() == before


async def test_stream_refusals_happen_before_streaming(client: AsyncClient) -> None:
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(anthropic_api_key=None, ask_rate_limit_per_minute=0)
    assert (await client.post("/ask/stream", json={"question": "q"})).status_code == 503
    assert (await client.post("/ask/stream", json={"question": ""})).status_code == 422


async def test_a_reader_that_leaves_early_does_not_lose_the_run(db_session: AsyncSession) -> None:
    """The steps already taken are billed, so the run finishes and is
    stored (and counted against the daily budget) even with no reader."""
    from contextlib import asynccontextmanager

    from app.models import AskRun
    from app.routers.ask import stream_answer

    await _seed(db_session)

    @asynccontextmanager
    async def shared_session():
        yield db_session

    question = f"Northgate total? {uuid.uuid4()}"
    stream = stream_answer(shared_session, EMBEDDER, SETTINGS, _northgate_client(), question)

    first = await anext(stream)
    await stream.aclose()

    assert first.startswith("event: model_call")
    stored = (await db_session.execute(select(AskRun).where(AskRun.question == question))).scalar_one()
    assert stored.status == "answered" and stored.steps == 2


async def test_a_tool_whose_sql_fails_does_not_lose_the_answer(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed statement rolls back to the tool call's savepoint, so the
    transaction can still store the (paid-for) answer."""
    from sqlalchemy import text

    from app.agent.tools import TOOL_HANDLERS

    async def broken(ctx, tool_input):
        await ctx.session.execute(text("SELECT 1 / 0"))

    monkeypatch.setitem(TOOL_HANDLERS, "query_extractions", broken)
    fake = _client([
        _message([_tool_use("query_extractions", {"vendor": "Northgate"})], "tool_use"),
        _message([_text("I couldn't look that up.")], "end_turn"),
    ])

    body = await _ask(client, db_session, SETTINGS, fake)

    assert body["status"] == "answered" and body["tool_calls"][0]["is_error"] is True
    assert (await client.get(f"/ask/runs/{body['id']}")).status_code == 200


async def test_streamed_runs_use_a_session_of_their_own(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Not the request's: FastAPI closes that when the response ends, and a
    run outlives a client that disconnects mid-stream."""
    from contextlib import asynccontextmanager

    from app.db import get_session_factory

    opened = []

    @asynccontextmanager
    async def recording_factory():
        opened.append(1)
        yield db_session

    app.dependency_overrides[get_session_factory] = lambda: recording_factory

    response = await _stream(client, db_session, SETTINGS, _northgate_client())

    assert _sse_events(response.text)[-1]["type"] == "answer"
    assert opened == [1]


async def test_ask_endpoint_validates_the_question(client: AsyncClient) -> None:
    assert (await client.post("/ask", json={"question": ""})).status_code == 422
    assert (await client.post("/ask", json={"question": "x" * 1001})).status_code == 422


def test_ask_client_is_built_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.routers import ask as ask_module

    built = []
    monkeypatch.setattr(ask_module, "_client", None)
    monkeypatch.setattr(ask_module, "_build_client", lambda settings: built.append(1) or object())

    first = ask_module.get_anthropic_client(SETTINGS)
    second = ask_module.get_anthropic_client(SETTINGS)

    assert first is second and built == [1]
    assert ask_module.get_anthropic_client(Settings(anthropic_api_key=None)) is None


def test_a_tax_rate_prints_without_trailing_zeros() -> None:
    record = DocumentRecord(
        uuid.uuid4(),
        "r.png",
        {
            "currency": "USD",
            "tax_lines": [
                {"label": "State", "rate": 8.0, "base": "90", "amount": 7.2},
                {"label": "City", "rate": "8.875", "base": None, "amount": None},
                {"label": None, "rate": "n/a", "base": None, "amount": None},
            ],
        },
        [],
        "extract_v2",
    )

    lines = dict(record_lines(record))

    assert lines["tax_lines[0]"] == "tax line: State, rate 8%, base 90.00, amount 7.20"
    assert lines["tax_lines[1]"] == "tax line: City, rate 8.875%"
    assert lines["tax_lines[2]"] == "tax line: tax, rate n/a%"
