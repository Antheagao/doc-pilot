"""Online evaluation of /ask: the sampling rule, the background 'judge'
job that grades a stored answer, and the production numbers /stats
reports from it (scripted grader, no network)."""

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import AgentResult
from app.config import Settings
from app.evals import online
from app.evals.judge import GROUNDEDNESS_PROMPT_VERSION
from app.evals.online import process_judge_job, should_judge
from app.extraction import ExtractionError, NonRetryableExtractionError
from app.models import JOB_KIND_JUDGE, AskRun, Document, DocumentPage, Job
from app.worker import HANDLERS


def _run(**overrides) -> AskRun:
    fields = {
        "question": "How much was my Northgate order?",
        "status": "answered",
        "answer": "It came to $425.58. [1]",
        "citations": [],
        "tool_calls": [],
        "evidence": "## northgate.png (extracted fields)\ntotal: 425.58",
        "steps": 2,
        "model": "claude-opus-5-5",
        "prompt_version": "agent_v1",
    }
    return AskRun(**{**fields, **overrides})


def _grader(payload=None, stop_reason="end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    text = json.dumps(payload) if payload is not None else ""
    response = SimpleNamespace(
        id="j", model="claude-sonnet-5-5", stop_reason=stop_reason, usage=usage,
        content=[SimpleNamespace(type="text", text=text)],
    )
    return SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))
    )


GROUNDED = {
    "unsupported_claims": [],
    "grounded": True,
    "answers_question": True,
    "explanation": "The total is in the evidence.",
}


async def _stored(db_session: AsyncSession, run: AskRun) -> Job:
    db_session.add(run)
    await db_session.flush()
    job = Job(kind=JOB_KIND_JUDGE, ask_run_id=run.id)
    db_session.add(job)
    # Committed (to the test's savepoint): the handler rolls back its read
    # transaction before the model call, which must not discard the rows.
    await db_session.commit()
    return job


# --- sampling -------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,rate,draw,expected",
    [
        ("answered", 0.0, 0.0, False),  # off by default, even for the luckiest draw
        ("answered", 0.25, 0.10, True),
        ("answered", 0.25, 0.30, False),
        ("answered", 1.0, 0.999, True),
        ("refused", 1.0, 0.0, False),  # nothing complete to ground
        ("budget_exceeded", 1.0, 0.0, False),
    ],
)
def test_only_answered_questions_are_sampled_at_the_configured_rate(status, rate, draw, expected) -> None:
    result = AgentResult(status, "a", [], [], 1, "claude-opus-5-5", "agent_v1")

    assert should_judge(result, Settings(ask_judge_sample_rate=rate), draw=lambda: draw) is expected


def test_sample_rate_is_a_probability() -> None:
    with pytest.raises(ValueError):
        Settings(ask_judge_sample_rate=1.5)


# --- the judge job ----------------------------------------------------------------


def test_the_worker_routes_judge_jobs() -> None:
    assert HANDLERS[JOB_KIND_JUDGE] is process_judge_job


async def test_judge_job_writes_the_verdict_onto_the_run(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _grader({**GROUNDED, "grounded": False, "unsupported_claims": ["a vendor not in evidence"]})
    monkeypatch.setattr(online, "_build_client", lambda settings: client)
    run = _run()
    job = await _stored(db_session, run)

    await process_judge_job(db_session, job)

    await db_session.refresh(run)
    assert (run.judge_grounded, run.judge_answers_question) == (False, True)
    assert run.judge_claims == ["a vendor not in evidence"]
    assert run.judge_prompt_version == GROUNDEDNESS_PROMPT_VERSION
    assert run.judge_model == "claude-sonnet-5-5"
    assert float(run.judge_cost_usd) == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert run.judge_error is None and run.judged_at is not None
    # The grader saw the stored question, evidence and answer -- and no
    # reference facts, because a live question has none.
    request = client.beta.messages.create.await_args.kwargs
    user_text = request["messages"][0]["content"]
    assert "total: 425.58" in user_text and "$425.58" in user_text
    assert "<reference_facts>" not in user_text
    assert request["output_config"]["format"]["schema"]["required"] == [
        "unsupported_claims", "grounded", "answers_question", "explanation"
    ]

    # Re-delivered job: the verdict stands and nothing is billed twice.
    await process_judge_job(db_session, job)
    assert client.beta.messages.create.await_count == 1


async def test_an_unreadable_verdict_is_recorded_not_retried(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(online, "_build_client", lambda settings: _grader(stop_reason="refusal"))
    run = _run()
    job = await _stored(db_session, run)

    await process_judge_job(db_session, job)

    await db_session.refresh(run)
    assert run.judge_error == "refusal" and run.judge_grounded is None
    assert run.judged_at is not None and float(run.judge_cost_usd) > 0


async def test_api_errors_raise_so_the_worker_retries(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anthropic
    import httpx

    overloaded = anthropic.InternalServerError(
        "overloaded", response=httpx.Response(529, request=httpx.Request("POST", "https://x")), body=None
    )
    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=overloaded)))
    )
    monkeypatch.setattr(online, "_build_client", lambda settings: client)
    run = _run()
    job = await _stored(db_session, run)

    with pytest.raises(ExtractionError) as raised:
        await process_judge_job(db_session, job)

    assert not isinstance(raised.value, NonRetryableExtractionError)
    await db_session.refresh(run)
    assert run.judged_at is None


async def test_a_missing_run_fails_the_job_permanently(db_session: AsyncSession) -> None:
    job = Job(kind=JOB_KIND_JUDGE, ask_run_id=uuid.uuid4())

    with pytest.raises(NonRetryableExtractionError, match="not found"):
        await process_judge_job(db_session, job)


async def test_every_job_has_a_target(db_session: AsyncSession) -> None:
    db_session.add(Job(kind=JOB_KIND_JUDGE))

    with pytest.raises(IntegrityError, match="ck_jobs_has_target"):
        await db_session.flush()


# --- /stats ---------------------------------------------------------------------


async def test_stats_report_spend_by_stage_and_online_quality(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    before = (await client.get("/stats")).json()

    document = Document(filename="r.png", mime_type="image/png", storage_path="/tmp/r", status="extracted")
    db_session.add(document)
    await db_session.flush()
    db_session.add_all([
        DocumentPage(document_id=document.id, page_number=1, text="t", source="transcription", cost_usd=0.004),
        # Gold text (the evals) costs nothing and isn't a transcription.
        DocumentPage(document_id=document.id, page_number=2, text="t", source="gold", cost_usd=0),
    ])
    judged_at = datetime.now(UTC)
    db_session.add_all([
        _run(cost_usd=0.05, latency_ms=4000, feedback="up", judge_sampled=True,
             judge_grounded=True, judge_answers_question=True, judge_cost_usd=0.002, judged_at=judged_at),
        _run(cost_usd=0.07, latency_ms=6000, feedback="down", judge_sampled=True,
             judge_grounded=False, judge_answers_question=True, judge_cost_usd=0.003, judged_at=judged_at),
        # An unreadable verdict: paid for, but not in the rates.
        _run(cost_usd=0.01, judge_sampled=True, judge_error="refusal", judge_cost_usd=0.001,
             judged_at=judged_at),
        _run(status="refused", cost_usd=0.02),
    ])
    await db_session.commit()

    after = (await client.get("/stats")).json()

    spend, was = after["spend"], before["spend"]
    assert spend["transcription_usd"] == pytest.approx(was["transcription_usd"] + 0.004, abs=1e-6)
    assert spend["agent_usd"] == pytest.approx(was["agent_usd"] + 0.15, abs=1e-6)
    assert spend["judge_usd"] == pytest.approx(was["judge_usd"] + 0.006, abs=1e-6)
    assert spend["total_usd"] == pytest.approx(
        spend["extraction_usd"] + spend["transcription_usd"] + spend["agent_usd"] + spend["judge_usd"]
    )
    ask, was = after["ask"], before["ask"]
    assert ask["runs"] == was["runs"] + 4
    assert ask["answered"] == was["answered"] + 3
    assert (ask["feedback_up"], ask["feedback_down"]) == (was["feedback_up"] + 1, was["feedback_down"] + 1)
    assert ask["judge_sampled"] == was["judge_sampled"] + 3
    assert ask["judged"] == was["judged"] + 2
    assert ask["latency_p50_ms"] is not None and ask["mean_cost_usd"] is not None
    if was["judged"] == 0:  # exact only on a DB with no earlier verdicts
        assert ask["judge_grounded_rate"] == pytest.approx(0.5)
        assert ask["judge_answers_rate"] == pytest.approx(1.0)


async def test_a_judge_job_that_fails_for_good_marks_its_run(db_session: AsyncSession) -> None:
    """Otherwise the run reads 'queued for grading' forever."""
    from app.worker import MAX_ATTEMPTS, fail_job

    run = _run(judge_sampled=True)
    job = await _stored(db_session, run)
    job.attempts = MAX_ATTEMPTS
    await db_session.commit()

    await fail_job(db_session, job, NonRetryableExtractionError("unpriced model"))

    await db_session.refresh(run)
    assert run.judged_at is not None and run.judge_grounded is None
    assert run.judge_error == "grading failed: NonRetryableExtractionError: unpriced model"
