"""Online evaluation: grading the answers /ask gives real users.

The offline evals (app/evals/agent.py, judge.py) grade the agent on
questions whose answers are known. Live questions have no answer key, so
two other signals are collected per stored answer (app.models.AskRun):

- a person's thumbs up / down (POST /ask/runs/{id}/feedback), and
- for a sampled share of answered questions (ASK_JUDGE_SAMPLE_RATE), the
  reference-free groundedness grader (judge.grade_groundedness): is every
  claim supported by the evidence the agent retrieved, and does the
  answer respond to the question. It runs as a 'judge' job on the worker,
  off the request path, and its span joins the question's own trace.

The grader is scored against the same human groundedness labels as the
offline judge (`run_judge_calibration.py --judge --grader groundedness`);
its production rate is worth what that agreement says it is.
"""

import dataclasses
import logging
import random
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import AgentResult
from app.config import Settings, get_settings
from app.evals.judge import grade_groundedness, render_evidence_from_messages
from app.extraction import NonRetryableExtractionError, _build_client
from app.models import JOB_KIND_JUDGE, AskRun, Job

logger = logging.getLogger(__name__)


def ask_run_from_result(question: str, result: AgentResult) -> AskRun:
    """The row to store for one /ask call. The evidence is rendered now,
    from the conversation, because the conversation itself isn't kept."""
    return AskRun(
        question=question,
        status=result.status,
        answer=result.answer,
        citations=[dataclasses.asdict(c) for c in result.citations],
        tool_calls=[dataclasses.asdict(t) for t in result.tool_calls],
        evidence=render_evidence_from_messages(result.messages),
        steps=result.steps,
        model=result.model,
        prompt_version=result.prompt_version,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        refusal_category=result.refusal_category,
        unresolved_citations=result.unresolved_citations,
        trace_id=result.trace_id,
        traceparent=result.traceparent,
    )


def should_judge(
    result: AgentResult, settings: Settings, draw: Callable[[], float] = random.random
) -> bool:
    """Sample answered questions at ASK_JUDGE_SAMPLE_RATE. Refusals,
    truncations and budget stops aren't graded: there's no complete
    answer to ground, and their status already says what went wrong."""
    rate = settings.ask_judge_sample_rate
    return result.status == "answered" and rate > 0 and draw() < rate


async def record_ask_run(
    session: AsyncSession, question: str, result: AgentResult, settings: Settings
) -> AskRun:
    """Store the run and, if sampled, its judge job -- one transaction, so
    a sampled answer can't be stored without the job that grades it."""
    run = ask_run_from_result(question, result)
    session.add(run)
    await session.flush()
    if should_judge(result, settings):
        run.judge_sampled = True
        session.add(Job(kind=JOB_KIND_JUDGE, ask_run_id=run.id, traceparent=result.traceparent))
    await session.commit()
    return run


async def process_judge_job(session: AsyncSession, job: Job) -> None:
    """Worker handler for a 'judge' job: grade the stored answer and write
    the verdict onto its AskRun. Idempotent -- a run that already has a
    verdict (a re-delivered job) is left alone. A refusal or unreadable
    verdict is stored as judge_error rather than retried: the same input
    would get the same output. API errors raise and retry like any job."""
    run = await session.get(AskRun, job.ask_run_id)
    if run is None:
        raise NonRetryableExtractionError(f"ask run {job.ask_run_id} not found")
    if run.judged_at is not None:
        logger.info("ask run %s already judged; skipping", run.id)
        return

    run_id, question, evidence, answer = run.id, run.question, run.evidence, run.answer
    # Release the connection during the model call, like the other handlers.
    await session.rollback()

    settings = get_settings()
    # No traceparent: the job span (already a child of the question's
    # trace, via Job.traceparent) is the current context.
    verdict = await grade_groundedness(
        question, evidence, answer, settings, client=_build_client(settings)
    )

    run = await session.get(AskRun, run_id)
    if run is None:
        raise NonRetryableExtractionError(f"ask run {run_id} vanished during grading")
    run.judge_grounded = verdict.grounded
    run.judge_answers_question = verdict.answers_question
    run.judge_claims = verdict.claims
    run.judge_explanation = verdict.explanation or None
    run.judge_model = verdict.model
    run.judge_prompt_version = verdict.prompt_version
    run.judge_cost_usd = round(verdict.cost_usd, 6)
    run.judge_error = verdict.error
    run.judged_at = datetime.now(UTC)
    await session.flush()
