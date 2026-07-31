"""Tests for app.evals.runner.

Deliberately has no db_session fixture -- run_eval/write_result never
touch a DB (the mock extract_fn used here has zero network access, and
even the real extract_document path never opens a session), so this
suite must stay runnable with Postgres stopped, matching the sibling
test_evals_scoring.py.
"""

import json
import re
from pathlib import Path

import pytest

from app.evals.dataset import load_cases
from app.evals.runner import (
    RunResult,
    build_mock_extract_fn,
    predicted_mock_accuracy,
    run_eval,
    write_result,
)
from app.extraction import ExtractionError, ExtractionResult


class TestMockRun:
    async def test_overall_accuracy_matches_predicted_constant(self):
        cases = load_cases()
        extract_fn = build_mock_extract_fn(cases)

        result = await run_eval(cases, extract_fn=extract_fn)

        predicted = predicted_mock_accuracy(cases)
        # 25 docs x 7 fields = 175 slots, 2 wrong (tax perturbation on the
        # 3rd case, vendor->None on the 7th) -> 173/175.
        assert predicted == pytest.approx(173 / 175)
        assert result.summary["overall_accuracy"] == predicted
        assert result.total_cost_usd == 0.0
        assert result.mean_cost_per_doc == 0.0
        assert result.n_scored == 25
        assert result.n_errors == 0
        assert result.n_skipped_cost_cap == 0
        assert result.model == "mock-model"
        assert result.dataset_version == "v1"

    async def test_limit_changes_predicted_constant_but_still_runs(self):
        cases = load_cases(limit=5)
        extract_fn = build_mock_extract_fn(cases)

        result = await run_eval(cases, extract_fn=extract_fn)

        # Only the 3rd case (tax perturbation) falls within the first 5;
        # the 7th (vendor perturbation) is excluded from this subset.
        assert result.n_scored == 5
        predicted = predicted_mock_accuracy(cases)
        assert predicted == pytest.approx((5 * 7 - 1) / (5 * 7))
        assert result.summary["overall_accuracy"] == predicted


class TestCostCap:
    async def test_stops_early_and_records_skips(self):
        cases = load_cases()
        call_count = 0

        async def paid_extract_fn(path, mime_type):
            nonlocal call_count
            call_count += 1
            return ExtractionResult(
                tool_input={},
                model="paid-mock",
                prompt_version="v1",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.10,
                latency_ms=1,
            )

        result = await run_eval(
            cases, extract_fn=paid_extract_fn, concurrency=3, max_cost_usd=0.25
        )

        assert call_count < len(cases)
        assert result.n_skipped_cost_cap > 0
        assert result.n_scored + result.n_errors + result.n_skipped_cost_cap == len(
            cases
        )
        assert result.total_cost_usd == pytest.approx(call_count * 0.10)

    async def test_billed_but_errored_calls_trip_the_cap_too(self):
        # Simulates app.extraction's refusal/max_tokens/persistence-failure
        # paths: the API call succeeded and was billed, but extract_fn
        # still raises. That cost must count against the cap (it's real
        # spend) even though the doc never gets scored, and must show up
        # in total_error_cost_usd, separate from total_cost_usd (which
        # stays scored-docs-only).
        cases = load_cases()
        call_count = 0

        async def always_errors_but_billed(path, mime_type):
            nonlocal call_count
            call_count += 1
            raise ExtractionError(
                "simulated refusal after a billed call",
                input_tokens=500,
                output_tokens=10,
                cost_usd=0.10,
            )

        result = await run_eval(
            cases, extract_fn=always_errors_but_billed, concurrency=3, max_cost_usd=0.25
        )

        assert call_count < len(cases)
        assert result.n_scored == 0
        assert result.n_errors == call_count
        assert result.n_skipped_cost_cap > 0
        assert result.total_cost_usd == 0.0  # no doc was ever scored
        assert result.total_error_cost_usd == pytest.approx(call_count * 0.10)
        assert result.errors[0]["cost_usd"] == pytest.approx(0.10)
        assert result.errors[0]["input_tokens"] == 500
        assert result.errors[0]["output_tokens"] == 10


class TestConcurrencyValidation:
    async def test_zero_concurrency_raises(self):
        cases = load_cases(limit=1)
        with pytest.raises(ValueError, match="concurrency"):
            await run_eval(cases, extract_fn=build_mock_extract_fn(cases), concurrency=0)

    async def test_negative_concurrency_raises(self):
        cases = load_cases(limit=1)
        with pytest.raises(ValueError, match="concurrency"):
            await run_eval(cases, extract_fn=build_mock_extract_fn(cases), concurrency=-1)


class TestErrorIsolation:
    async def test_one_bad_doc_is_isolated(self):
        cases = load_cases()
        bad_doc_id = cases[10].doc_id
        mock_fn = build_mock_extract_fn(cases)

        async def flaky_extract_fn(path, mime_type):
            if Path(path).stem == bad_doc_id:
                raise ExtractionError("simulated failure for this doc")
            return await mock_fn(path, mime_type)

        result = await run_eval(cases, extract_fn=flaky_extract_fn)

        assert result.n_errors == 1
        assert result.n_scored == len(cases) - 1
        assert result.errors[0]["doc_id"] == bad_doc_id
        assert result.errors[0]["error_type"] == "ExtractionError"
        assert result.summary["n_docs"] == len(cases) - 1


class TestWriteResult:
    def test_writes_valid_json_artifact(self, tmp_path):
        result = RunResult(
            model="claude-sonnet-5",
            prompt_version="extract_v1",
            dataset_version="v1",
            review_threshold=0.8,
            started_at_utc="2026-07-30T12:34:56+00:00",
            concurrency=3,
            max_cost_usd=1.0,
            summary={"n_docs": 1, "overall_accuracy": 1.0},
            total_cost_usd=0.01,
            total_error_cost_usd=0.0,
            mean_cost_per_doc=0.01,
            total_input_tokens=100,
            total_output_tokens=50,
            latency_p50_ms=200.0,
            latency_p95_ms=250.0,
            n_scored=1,
            n_errors=0,
            n_skipped_cost_cap=0,
            errors=[],
            skipped_cost_cap=[],
            per_doc=[],
        )

        out_path = write_result(result, tmp_path)

        assert out_path.parent == tmp_path
        assert re.match(
            r"^\d{8}T\d{6}Z_claude-sonnet-5_extract_v1\.json$", out_path.name
        )
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["model"] == "claude-sonnet-5"
        assert loaded["prompt_version"] == "extract_v1"
        assert loaded["dataset_version"] == "v1"

    def test_default_out_dir_is_evals_results(self, tmp_path, monkeypatch):
        import app.evals.runner as runner_module

        monkeypatch.setattr(runner_module, "RESULTS_DIR", tmp_path / "results")

        result = RunResult(
            model="mock-model",
            prompt_version="extract_v1",
            dataset_version="v1",
            review_threshold=0.8,
            started_at_utc="2026-07-30T00:00:00+00:00",
            concurrency=3,
            max_cost_usd=1.0,
            summary=None,
            total_cost_usd=0.0,
            total_error_cost_usd=0.0,
            mean_cost_per_doc=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            latency_p50_ms=None,
            latency_p95_ms=None,
            n_scored=0,
            n_errors=0,
            n_skipped_cost_cap=0,
            errors=[],
            skipped_cost_cap=[],
            per_doc=[],
        )

        out_path = write_result(result)

        assert out_path.exists()
        assert out_path.parent == tmp_path / "results"

    def test_collision_gets_a_suffix_instead_of_clobbering(self, tmp_path):
        result = RunResult(
            model="claude-sonnet-5",
            prompt_version="extract_v1",
            dataset_version="v1",
            review_threshold=0.8,
            started_at_utc="2026-07-30T12:34:56+00:00",
            concurrency=3,
            max_cost_usd=1.0,
            summary=None,
            total_cost_usd=0.0,
            total_error_cost_usd=0.0,
            mean_cost_per_doc=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            latency_p50_ms=None,
            latency_p95_ms=None,
            n_scored=1,
            n_errors=0,
            n_skipped_cost_cap=0,
            errors=[],
            skipped_cost_cap=[],
            per_doc=[{"marker": "first"}],
        )

        first_path = write_result(result, tmp_path)

        result.per_doc = [{"marker": "second"}]
        second_path = write_result(result, tmp_path)

        assert first_path != second_path
        assert first_path.name == "20260730T123456Z_claude-sonnet-5_extract_v1.json"
        assert second_path.name == "20260730T123456Z_claude-sonnet-5_extract_v1-1.json"
        # Both files survive with their own content -- the second write
        # must not have clobbered the first.
        first_loaded = json.loads(first_path.read_text(encoding="utf-8"))
        second_loaded = json.loads(second_path.read_text(encoding="utf-8"))
        assert first_loaded["per_doc"] == [{"marker": "first"}]
        assert second_loaded["per_doc"] == [{"marker": "second"}]
