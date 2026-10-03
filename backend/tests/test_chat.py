"""The per-document chat (app/routers/chat.py): turns scoped to one
document, history rebuilt from stored turns, and everything /ask enforces
-- against a scripted fake client (no network)."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import (
    AGENT_PROMPT_TEXT,
    DOCUMENT_CHAT_PROMPT,
    Turn,
    history_messages,
)
from app.budget import spent_today_usd
from app.config import Settings, get_settings
from app.evals import online
from app.evals.corpus import seed_labeled_corpus
from app.evals.dataset import load_cases
from app.evals.online import judge_question, process_judge_job
from app.evals.retrieval import load_gold_corpus
from app.main import app
from app.models import JOB_KIND_JUDGE, AskRun, Document, Job
from app.retrieval.embeddings import HashingEmbedder, get_embedder
from app.retrieval.indexing import ChunkingConfig
from app.routers.ask import get_anthropic_client

EMBEDDER = HashingEmbedder()
NORTHGATE = "018-dense-office-outfitters"
DOCS = ("001-clean-coffee-receipt", NORTHGATE, "024-eur-bakery-berlin")

SETTINGS = Settings(
    agent_model="claude-opus-5-5",
    agent_effort="medium",
    anthropic_api_key="test",
    daily_budget_usd=0,
    ask_rate_limit_per_minute=0,
)


def _usage():
    return SimpleNamespace(
        input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )


def _message(content, stop_reason):
    return SimpleNamespace(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        model="claude-opus-5-5",
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=_usage(),
    )


def _tool_use(name, tool_input):
    return SimpleNamespace(type="tool_use", id=f"toolu_{uuid.uuid4().hex[:6]}", name=name, input=tool_input)


def _text(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def _cite_line(field_line: str, answer: str):
    """The model's final turn, citing the line `field_line` of the latest
    tool result -- built the way the API builds a citation."""

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
        return _message([_text(answer, [citation])], "end_turn")

    return respond


class FakeClient:
    """Plays scripted turns and records every request."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        # A snapshot: the loop appends to the same list after the call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        step = self.steps.pop(0)
        return step(**kwargs) if callable(step) else step


def _total_turn(answer="The total was $425.58."):
    return [
        _message([_tool_use("query_extractions", {})], "tool_use"),
        _cite_line("total: 425.58", answer),
    ]


async def _seed(db_session: AsyncSession) -> dict[str, uuid.UUID]:
    cases = [case for case in load_cases() if case.doc_id in DOCS]
    return await seed_labeled_corpus(
        db_session, cases, load_gold_corpus(cases), EMBEDDER, ChunkingConfig(200, 80, True)
    )


def _use(fake, settings: Settings = SETTINGS) -> None:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    app.dependency_overrides[get_anthropic_client] = lambda: fake


def _sse_events(body: str) -> list[dict]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append(json.loads(lines["data"]))
    return events


# --- history ----------------------------------------------------------------


def test_history_is_plain_alternating_text_without_citation_markers() -> None:
    turns = [
        Turn("What's the total?", "It came to $425.58. [1]"),
        Turn("Who refused?", ""),  # no answer: left out, alternation kept
        Turn("And the tax?", "The tax was $31.52 [1] [2]."),
    ]

    assert history_messages(turns) == [
        {"role": "user", "content": "What's the total?"},
        {"role": "assistant", "content": "It came to $425.58."},
        {"role": "user", "content": "And the tax?"},
        {"role": "assistant", "content": "The tax was $31.52."},
    ]


def test_the_grader_reads_a_follow_up_with_its_conversation() -> None:
    assert judge_question("q", []) == "q"

    text = judge_question("And the tax?", [Turn("What's the total?", "$425.58. [1]")])

    assert text.startswith("Earlier in this conversation (context only, not evidence):")
    assert "user: What's the total?\nassistant: $425.58." in text
    assert text.endswith("The question being answered now:\nAnd the tax?")


# --- a turn -----------------------------------------------------------------


async def test_a_turn_is_scoped_to_the_document_and_cites_its_field(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    fake = FakeClient(*_total_turn())
    _use(fake)

    response = await client.post(
        f"/documents/{ids[NORTHGATE]}/chat", json={"question": "What's the total?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered" and body["answer"].endswith("[1]")
    assert body["document_id"] == str(ids[NORTHGATE]) and body["conversation_id"]
    (citation,) = body["citations"]
    assert (citation["kind"], citation["fields"]) == ("record", ["total"])
    assert citation["document_id"] == str(ids[NORTHGATE])
    # Unfiltered, query_extractions saw only this document of the three.
    run = await db_session.get(AskRun, uuid.UUID(body["id"]))
    assert NORTHGATE in run.evidence
    assert not any(other in run.evidence for other in DOCS if other != NORTHGATE)
    assert "1 matching document(s)" in run.evidence
    # The model was told: the agent prompt, then the document-chat block.
    request = fake.requests[0]
    assert [block["text"] for block in request["system"]] == [
        AGENT_PROMPT_TEXT, DOCUMENT_CHAT_PROMPT.system[1]["text"]
    ]
    assert run.prompt_version == DOCUMENT_CHAT_PROMPT.version == "agent_v1+document_chat_v1"


async def test_a_follow_up_carries_the_conversation(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    url = f"/documents/{ids[NORTHGATE]}/chat"
    _use(FakeClient(*_total_turn()))
    first = (await client.post(url, json={"question": "What's the total?"})).json()

    follow_up = FakeClient(
        _message([_tool_use("query_extractions", {})], "tool_use"),
        _cite_line("tax: 32.43", "The tax was $32.43."),
    )
    _use(follow_up)
    response = await client.post(
        url, json={"question": "And the tax?", "conversation_id": first["conversation_id"]}
    )

    assert response.status_code == 200
    second = response.json()
    assert second["conversation_id"] == first["conversation_id"]
    assert second["citations"][0]["fields"] == ["tax"]
    # The earlier turn, as text with its [n] markers gone, then the new question.
    assert follow_up.requests[0]["messages"] == [
        {"role": "user", "content": "What's the total?"},
        {"role": "assistant", "content": "The total was $425.58."},
        {"role": "user", "content": "And the tax?"},
    ]

    turns = (
        await client.get(f"{url}/conversations/{first['conversation_id']}")
    ).json()
    assert [t["question"] for t in turns] == ["What's the total?", "And the tax?"]
    assert turns[1] == second
    (summary,) = (await client.get(f"{url}/conversations")).json()
    assert summary["conversation_id"] == first["conversation_id"]
    assert (summary["first_question"], summary["turns"]) == ("What's the total?", 2)


async def test_another_documents_conversation_is_not_found(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    fake = FakeClient(*_total_turn())
    _use(fake)
    first = (
        await client.post(f"/documents/{ids[NORTHGATE]}/chat", json={"question": "Total?"})
    ).json()
    other = ids["024-eur-bakery-berlin"]

    for conversation_id in (first["conversation_id"], str(uuid.uuid4())):
        for route in ("chat", "chat/stream"):
            response = await client.post(
                f"/documents/{other}/chat" if route == "chat" else f"/documents/{other}/chat/stream",
                json={"question": "And the tax?", "conversation_id": conversation_id},
            )
            assert response.status_code == 404
    assert (
        await client.get(f"/documents/{other}/chat/conversations/{first['conversation_id']}")
    ).status_code == 404
    assert (await client.get(f"/documents/{other}/chat/conversations")).json() == []
    assert fake.steps == []  # only the first turn called the model


async def test_unknown_and_unextracted_documents_are_refused_before_the_model(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    create = AsyncMock()
    _use(SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))))
    pending = Document(filename="p.png", mime_type="image/png", storage_path="/tmp/p", status="processing")
    db_session.add(pending)
    await db_session.commit()

    missing = await client.post(f"/documents/{uuid.uuid4()}/chat", json={"question": "q"})
    unextracted = await client.post(f"/documents/{pending.id}/chat", json={"question": "q"})
    empty = await client.post(f"/documents/{pending.id}/chat", json={"question": ""})

    assert missing.status_code == 404
    assert unextracted.status_code == 409 and "processing" in unextracted.json()["detail"]
    assert empty.status_code == 422
    assert (await client.get(f"/documents/{uuid.uuid4()}/chat/conversations")).status_code == 404
    create.assert_not_awaited()


async def test_no_api_key_is_a_503(client: AsyncClient, db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    app.dependency_overrides[get_settings] = lambda: Settings(
        anthropic_api_key=None, ask_rate_limit_per_minute=0, daily_budget_usd=0
    )

    response = await client.post(f"/documents/{ids[NORTHGATE]}/chat", json={"question": "q"})

    assert response.status_code == 503


async def test_the_daily_budget_applies_to_chat(
    client: AsyncClient, db_session: AsyncSession, tmp_path
) -> None:
    ids = await _seed(db_session)
    spent = await spent_today_usd(db_session)
    db_session.add(AskRun(
        question="q", status="answered", answer="a", citations=[], tool_calls=[], evidence="e",
        steps=1, model="claude-opus-5-5", prompt_version="agent_v1", cost_usd=0.25,
    ))
    await db_session.commit()
    create = AsyncMock()
    _use(
        SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))),
        SETTINGS.model_copy(update={"daily_budget_usd": spent + 0.20}),
    )

    for route in ("chat", "chat/stream"):
        response = await client.post(f"/documents/{ids[NORTHGATE]}/{route}", json={"question": "q"})
        assert response.status_code == 429 and "retry-after" in response.headers
    create.assert_not_awaited()


async def test_a_streamed_turn_ends_with_the_stored_run(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    _use(FakeClient(*_total_turn()))

    response = await client.post(
        f"/documents/{ids[NORTHGATE]}/chat/stream", json={"question": "What's the total?"}
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert [e["type"] for e in events] == ["model_call", "tool_start", "tool_call", "model_call", "answer"]
    run = events[-1]["run"]
    assert run["document_id"] == str(ids[NORTHGATE]) and run["conversation_id"]
    assert (await client.get(f"/ask/runs/{run['id']}")).json() == run


async def test_tools_cannot_reach_outside_the_document(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    other = ids["024-eur-bakery-berlin"]
    _use(FakeClient(
        _message([_tool_use("get_page", {"document_id": str(other), "page_number": 1})], "tool_use"),
        _message([_text("I can only see this document.")], "end_turn"),
    ))

    body = (
        await client.post(f"/documents/{ids[NORTHGATE]}/chat", json={"question": "The bakery?"})
    ).json()

    (call,) = body["tool_calls"]
    assert call["name"] == "get_page" and call["is_error"] is True
    run = await db_session.get(AskRun, uuid.UUID(body["id"]))
    assert "Berlin" not in run.evidence


# --- grading and storage ----------------------------------------------------


async def test_a_sampled_follow_up_is_graded_with_its_conversation(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = await _seed(db_session)
    url = f"/documents/{ids[NORTHGATE]}/chat"
    _use(FakeClient(*_total_turn()))
    first = (await client.post(url, json={"question": "What's the total?"})).json()
    _use(
        FakeClient(*_total_turn("Still $425.58.")),
        SETTINGS.model_copy(update={"ask_judge_sample_rate": 1.0}),
    )
    second = (
        await client.post(
            url, json={"question": "Are you sure?", "conversation_id": first["conversation_id"]}
        )
    ).json()
    (job,) = (
        await db_session.execute(select(Job).where(Job.ask_run_id == uuid.UUID(second["id"])))
    ).scalars().all()
    assert job.kind == JOB_KIND_JUDGE

    verdict = {
        "unsupported_claims": [], "grounded": True, "answers_question": True, "explanation": "ok",
    }
    grader = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(
            id="j", model="claude-sonnet-5-5", stop_reason="end_turn", usage=_usage(),
            content=[SimpleNamespace(type="text", text=json.dumps(verdict))],
        )
    ))))
    monkeypatch.setattr(online, "_build_client", lambda settings: grader)

    await process_judge_job(db_session, job)

    user_text = grader.beta.messages.create.await_args.kwargs["messages"][0]["content"]
    assert "user: What's the total?\nassistant: The total was $425.58." in user_text
    assert "The question being answered now:\nAre you sure?" in user_text


async def test_a_chat_turn_names_both_its_document_and_conversation(db_session: AsyncSession) -> None:
    db_session.add(AskRun(
        question="q", status="answered", answer="a", citations=[], tool_calls=[], evidence="e",
        steps=1, model="claude-opus-5-5", prompt_version="agent_v1", conversation_id=uuid.uuid4(),
    ))

    with pytest.raises(IntegrityError, match="ck_ask_runs_chat_scope"):
        await db_session.flush()
