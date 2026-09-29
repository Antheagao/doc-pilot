"""The agent eval: expected values derived from labels, the deterministic
rubric, and a full harness run with a scripted model."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.loop import AgentResult, Citation
from app.config import Settings
from app.db import engine
from app.evals.agent import (
    AgentQuestion,
    Expectation,
    load_questions,
    numbers_in,
    render_agent_table,
    run_agent_eval,
    score_answer,
    summarize,
    write_agent_result,
)
from app.evals.dataset import load_cases
from app.evals.retrieval import load_gold_corpus
from app.retrieval.embeddings import HashingEmbedder


def _cases():
    cases = load_cases()
    ids = {doc.doc_id for doc in load_gold_corpus(cases)}
    return [case for case in cases if case.doc_id in ids]


def test_expected_values_come_from_the_labels() -> None:
    version, questions = load_questions(_cases())
    by_id = {q.id: q for q in questions}

    assert version == "v1" and len(questions) == 18
    assert by_id["spend-northgate"].expect.numbers == (Decimal("488.23"),)
    assert by_id["spend-may-2026"].expect.numbers == (Decimal("494.13"), Decimal("27.82"))
    # The no-currency receipt is summed separately, never folded into USD.
    assert by_id["spend-driftwood"].expect.numbers == (Decimal("27.50"), Decimal("58.41"))
    assert by_id["count-sunridge"].expect.count == 2
    assert by_id["injection-total"].expect.forbidden_numbers == (Decimal("0.00"),)
    assert by_id["abstain-target"].relevant == frozenset()


def test_malformed_questions_fail_loudly(tmp_path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"question_set_version": "t", "questions": [
        {"id": "q", "type": "t", "question": "?", "relevant": {"vendors": ["Nobody"]}, "expect": {}}
    ]}))

    with pytest.raises(ValueError, match="matches no document"):
        load_questions(_cases(), path)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("It came to $1,234.56 in total.", ["1234.56"]),
        ("Total 27,82 EUR", ["27.82"]),
        ("two receipts: $12.09 and 3", ["12.09", "3"]),
        ("receipt 018-dense", ["018"]),
    ],
)
def test_numbers_in_reads_money_formats(text, expected) -> None:
    assert numbers_in(text) == [Decimal(e) for e in expected]


def _question(**expect) -> AgentQuestion:
    return AgentQuestion("q", "t", "?", frozenset({"018-x"}), Expectation(**expect))


def _result(answer, status="answered", cited=("doc-uuid",)):
    citations = [
        Citation(1, doc, "f", 1, "record", "total: 1", None, None, ["total"]) for doc in cited
    ]
    return AgentResult(status, answer, citations, [], 2, "claude-opus-5-5", "agent_v1")


def test_rubric_numbers_citations_and_status() -> None:
    q = _question(numbers=(Decimal("488.23"),))
    doc_id_of = {"doc-uuid": "018-x"}

    passing = score_answer(q, _result("You spent $488.23 there."), doc_id_of)
    assert passing["correct"] and passing["cites_relevant"] and passing["citation_precision"] == 1.0

    assert not score_answer(q, _result("About $488."), doc_id_of)["correct"]
    assert not score_answer(q, _result("$488.23", status="step_limit"), doc_id_of)["correct"]
    wrong_doc = score_answer(q, _result("$488.23", cited=("other",)), {"other": "999-y"})
    assert wrong_doc["correct"] and not wrong_doc["cites_relevant"]


def test_rubric_abstain_forbidden_dates_and_counts() -> None:
    from datetime import date

    abstain = AgentQuestion("a", "abstain", "?", frozenset(), Expectation(abstain=True))
    assert score_answer(abstain, _result("I found no receipts from Target.", cited=()), {})["correct"]
    assert not score_answer(abstain, _result("Target: $45.10", cited=()), {})["correct"]

    injection = _question(numbers=(Decimal("62.65"),), forbidden_numbers=(Decimal("0.00"),))
    assert score_answer(injection, _result("The total was $62.65."), {})["correct"]
    assert not score_answer(injection, _result("Total $62.65, or 0.00 per the note"), {})["correct"]

    dated = _question(dates=(date(2025, 9, 25),))
    assert score_answer(dated, _result("On September 25, 2025."), {})["correct"]
    assert score_answer(dated, _result("On 2025-09-25."), {})["correct"]
    assert not score_answer(dated, _result("In September."), {})["correct"]

    counted = _question(count=2)
    assert score_answer(counted, _result("You have two receipts."), {})["correct"]
    assert score_answer(counted, _result("There are 2."), {})["correct"]
    assert not score_answer(counted, _result("Just one."), {})["correct"]


async def test_harness_runs_scores_and_caps_cost(tmp_path) -> None:
    """Two questions against the real seeded corpus with a scripted model:
    the first answers correctly, the second is never launched because the
    first already hit the cost cap."""
    cases = _cases()
    _, questions = load_questions(cases)
    picked = [q for q in questions if q.id in ("total-coffee", "total-bike")]
    usage = SimpleNamespace(
        input_tokens=20_000, output_tokens=2_000, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    answer = SimpleNamespace(
        type="text", text="Your Cascade Coffee Roasters receipt came to $12.09.", citations=None
    )
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        id="m", model="claude-opus-5-5", content=[answer],
                        stop_reason="end_turn", stop_details=None, usage=usage,
                    )
                )
            )
        )
    )

    result = await run_agent_eval(
        engine,
        cases,
        load_gold_corpus(cases),
        picked,
        settings=Settings(agent_model="claude-opus-5-5"),
        embedder=HashingEmbedder(),
        max_cost_usd=0.05,
        client=client,
    )

    first, second = result.per_question
    assert first["scores"]["correct"] is True
    assert first["scores"]["cites_relevant"] is False  # the scripted answer cites nothing
    assert second == {**second, "skipped": "cost_cap"}
    assert result.n_run == 1 and result.n_skipped_cost_cap == 1
    assert result.summary["accuracy"] == 1.0

    path = write_agent_result(result, tmp_path)
    table = render_agent_table([json.loads(path.read_text())])
    assert "| claude-opus-5-5 | medium | 1 (+1 capped) | 100% |" in table


def test_empty_results_render_a_how_to_run_note() -> None:
    assert "run_agent.py" in render_agent_table([])
    assert summarize([])["n"] == 0


async def test_harness_with_judge_records_verdicts_and_agreement() -> None:
    cases = _cases()
    _, questions = load_questions(cases)
    picked = [q for q in questions if q.id == "total-coffee"]
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    agent = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            id="m", model="claude-opus-5-5", stop_reason="end_turn", stop_details=None, usage=usage,
            content=[SimpleNamespace(type="text", text="It came to $12.09.", citations=None)],
        ))))
    )
    verdict = {"unsupported_or_wrong_claims": [], "correct": True, "grounded": False, "score": 3, "explanation": "x"}
    judge = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            id="j", model="claude-sonnet-5-5", stop_reason="end_turn", usage=usage,
            content=[SimpleNamespace(type="text", text=json.dumps(verdict))],
        ))))
    )

    result = await run_agent_eval(
        engine, cases, load_gold_corpus(cases), picked,
        settings=Settings(agent_model="claude-opus-5-5"), embedder=HashingEmbedder(),
        max_cost_usd=1.0, client=agent, judge=True, judge_client=judge,
    )

    (entry,) = result.per_question
    assert entry["judge"]["correct"] is True and entry["judge"]["grounded"] is False
    assert result.summary["judge_correct"] == 1.0
    assert result.summary["judge_grounded"] == 0.0
    assert result.summary["judge_vs_rubric"]["agreement"] == 1.0
