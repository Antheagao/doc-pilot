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

from app.evals.checks import check_metrics, doc_routing
from app.evals.dataset import load_cases
from app.evals.runner import (
    RunResult,
    build_mock_extract_fn,
    predicted_mock_accuracy,
    run_eval,
    write_result,
)
from app.extraction import ExtractionError, ExtractionResult
from app.extraction_fields import SCHEMA_V1, SCHEMA_V2


class TestMockRun:
    async def test_overall_accuracy_matches_predicted_constant(self):
        cases = load_cases()
        extract_fn = build_mock_extract_fn(cases)

        result = await run_eval(cases, extract_fn=extract_fn)

        predicted = predicted_mock_accuracy(cases)
        # 32 docs x 7 v1 fields = 224 slots, 2 wrong (tax perturbation on the
        # 3rd case, vendor->None on the 7th) -> 222/224. The mock and the
        # scoring still read only the seven v1 fields of each label.
        assert predicted == pytest.approx(222 / 224)
        assert result.summary["overall_accuracy"] == predicted
        assert result.total_cost_usd == 0.0
        assert result.mean_cost_per_doc == 0.0
        assert result.n_scored == 32
        assert result.n_errors == 0
        assert result.n_skipped_cost_cap == 0
        assert result.model == "mock-model"
        assert result.dataset_version == "v2"

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
        # The one perturbation inside the subset is the tax, which the
        # arithmetic catches: 1 of 1 incorrect fields.
        assert check_metrics(result.to_dict()).caught_by_checks == 1.0

    async def test_mock_caught_by_checks_is_pinned(self):
        cases = load_cases()
        result = await run_eval(cases, extract_fn=build_mock_extract_fn(cases))
        run = result.to_dict()

        metrics = check_metrics(run)

        # The two wrong mock fields are 003's tax (+0.10, which breaks
        # total_math) and 007's vendor (null, which no amount rule can see), so
        # the checks catch exactly one of two. This pins the replay path end to
        # end without a model.
        assert metrics.n_incorrect == 2
        assert metrics.n_caught_by_checks == 1
        assert metrics.caught_by_checks == 0.5
        tax_doc = next(d for d in run["per_doc"] if d["fields"]["tax"]["correct"] is False)
        routing = doc_routing(tax_doc, run["review_threshold"], SCHEMA_V1)
        assert routing["tax"].review_reason == "reconciliation:total_math"


class TestMockRunV2:
    """The mock under the v2 schema: a flawless reader, with gold evidence, who
    makes the same two fixed mistakes."""

    async def _run(self, cases=None):
        cases = cases if cases is not None else load_cases()
        return cases, await run_eval(
            cases, extract_fn=build_mock_extract_fn(cases, SCHEMA_V2), schema=SCHEMA_V2
        )

    async def test_overall_accuracy_matches_predicted_constant(self):
        cases, result = await self._run()

        predicted = predicted_mock_accuracy(cases, SCHEMA_V2)
        # 32 docs x 20 v2 fields = 640 slots, the same 2 wrong (003's tax nudged,
        # 007's vendor null) -> 638/640.
        assert predicted == pytest.approx(638 / 640)
        assert predicted == 0.996875
        assert result.summary["overall_accuracy"] == predicted
        assert result.prompt_version == "extract_v2"
        assert result.n_scored == 32
        assert result.summary["n_docs"] == 32
        assert len(result.per_doc[0]["fields"]) == 20

    async def test_v1_is_untouched_by_the_schema_parameter(self):
        cases = load_cases()
        v1 = await run_eval(cases, extract_fn=build_mock_extract_fn(cases))
        explicit = await run_eval(
            cases, extract_fn=build_mock_extract_fn(cases, SCHEMA_V1), schema=SCHEMA_V1
        )

        assert predicted_mock_accuracy(cases) == predicted_mock_accuracy(cases, SCHEMA_V1) == 222 / 224
        assert v1.summary["overall_accuracy"] == explicit.summary["overall_accuracy"] == 222 / 224
        assert v1.prompt_version == "extract_v1"
        # A v1 run carries no evidence anywhere in its artifact.
        assert "evidence_coverage" not in v1.summary
        assert "evidence" not in json.dumps(v1.to_dict())

    async def test_limit_changes_the_constant_the_same_way(self):
        cases, result = await self._run(load_cases(limit=5))

        assert predicted_mock_accuracy(cases, SCHEMA_V2) == pytest.approx((5 * 20 - 1) / (5 * 20))
        assert result.summary["overall_accuracy"] == predicted_mock_accuracy(cases, SCHEMA_V2)

    async def test_mock_caught_by_checks_and_by_quotes_are_pinned(self):
        _cases, result = await self._run()
        run = result.to_dict()

        metrics = check_metrics(run)

        # The same two wrong fields: 007's null vendor, which nothing can see, and
        # 003's tax, which the quote it kept (Tax: $2.69) contradicts. A quote
        # reason outranks the arithmetic's, which would also flag it.
        assert (metrics.n_incorrect, metrics.n_caught_by_checks, metrics.n_caught_by_quotes) == (2, 1, 1)
        assert metrics.caught_by_checks == 0.5
        assert metrics.caught_by_quotes == 0.5
        assert metrics.n_correct_flagged == 0
        assert metrics.quotes_checkable
        tax_doc = next(d for d in run["per_doc"] if d["fields"]["tax"]["correct"] is False)
        assert doc_routing(tax_doc, run["review_threshold"], SCHEMA_V2)["tax"].review_reason == "quote:tax"
        # Without the stored evidence the arithmetic still catches it: the tax no
        # longer equals the sum of its only tax line.
        del tax_doc["fields"]["tax"]["evidence"]
        assert (
            doc_routing(tax_doc, run["review_threshold"], SCHEMA_V2)["tax"].review_reason
            == "reconciliation:tax_lines_sum"
        )

    async def test_mock_evidence_coverage_is_pinned(self):
        _cases, result = await self._run()

        coverage = result.summary["evidence_coverage"]

        # 813 non-null values, each quoted, each citing the gold page and the gold
        # line (the perturbed tax keeps its gold quote; the null vendor has no leaf).
        assert (coverage["leaves"], coverage["with_quote"]) == (813, 813)
        assert (coverage["gold_judged"], coverage["page_agree"], coverage["quote_agree"]) == (813, 813, 813)
        assert coverage["per_field"]["line_items"]["leaves"] == 508
        assert coverage["per_field"]["tax_lines"]["leaves"] == 85
        assert coverage["per_field"]["vendor"]["leaves"] == 30  # 022 has no vendor, 007's is dropped

    async def test_the_mock_emits_gold_evidence_and_keeps_the_perturbed_taxs_quote(self):
        cases = load_cases()
        by_id = {case.doc_id: case for case in cases}
        fn = build_mock_extract_fn(cases, SCHEMA_V2)

        tax_doc = cases[2]
        tool = (await fn(tax_doc.image_path, tax_doc.mime_type)).tool_input
        vendor_doc = cases[6]
        vendor_tool = (await fn(vendor_doc.image_path, vendor_doc.mime_type)).tool_input

        assert tool["tax"]["value"] == round(tax_doc.fields["tax"] + 0.10, 2)
        assert tool["tax"]["evidence"] == by_id[tax_doc.doc_id].evidence["tax"]
        assert tool["total"]["evidence"] == tax_doc.evidence["total"]
        assert tool["line_items"]["value"][0]["total"]["evidence"] == tax_doc.evidence["line_items[0].total"]
        assert tool["line_items"]["value"][0]["total"]["evidence"] is not tax_doc.evidence["line_items[0].total"]
        assert tool["po_number"] == {"value": None, "confidence": 0.9, "evidence": None}
        assert vendor_tool["vendor"] == {"value": None, "confidence": 0.9, "evidence": None}

    async def test_a_v1_mock_answer_has_no_evidence_key(self):
        cases = load_cases(limit=1)
        fn = build_mock_extract_fn(cases)

        tool = (await fn(cases[0].image_path, cases[0].mime_type)).tool_input

        assert set(tool) == set(SCHEMA_V1.field_names)
        assert tool["total"] == {"value": cases[0].fields["total"], "confidence": 0.9}
        assert "evidence" not in json.dumps(tool)

    async def test_the_artifact_round_trips_through_json_with_its_evidence(self):
        _cases, result = await self._run(load_cases(limit=3))

        loaded = json.loads(json.dumps(result.to_dict()))

        assert loaded["per_doc"][0]["fields"]["total"]["evidence"]["quote"].startswith("Total")
        assert loaded["summary"]["evidence_coverage"]["leaves"] == result.summary["evidence_coverage"]["leaves"]


class TestRunEvalSchema:
    async def test_the_default_extract_fn_asks_for_the_schema(self, monkeypatch):
        import app.evals.runner as runner_module

        seen = []

        async def fake_extract(path, mime_type, *, schema=None):
            seen.append(schema)
            return ExtractionResult({}, "m", schema.prompt_version, 0, 0, 0.0, 1)

        monkeypatch.setattr(runner_module, "extract_document", fake_extract)
        cases = load_cases(limit=2)

        result = await run_eval(cases, schema=SCHEMA_V2)
        v1_result = await run_eval(cases)

        assert seen == [SCHEMA_V2, SCHEMA_V2, SCHEMA_V1, SCHEMA_V1]
        assert result.prompt_version == "extract_v2"
        assert v1_result.prompt_version == "extract_v1"


class TestRecordedForTheBar:
    """What the pre-registered bar needs from an artifact: residual schema
    violations and the stop reasons of every call."""

    async def test_the_residual_violations_are_the_extractions_not_the_scorers(self):
        cases = load_cases(limit=3)

        async def extract(path, mime_type):
            return ExtractionResult(
                tool_input={},
                model="m",
                prompt_version="extract_v1",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=1,
                schema_violations=2 if Path(path).stem == cases[1].doc_id else 0,
                stop_reasons=[{"first": "tool_use", "repair": "tool_use"}],
            )

        result = await run_eval(cases, extract_fn=extract)

        scored = [d for d in result.per_doc if d["status"] == "scored"]
        assert [d["residual_schema_violations"] for d in scored] == [0, 2, 0]
        assert result.total_residual_schema_violations == 2
        assert result.to_dict()["total_residual_schema_violations"] == 2
        assert scored[0]["stop_reasons"] == [{"first": "tool_use", "repair": "tool_use"}]

    async def test_an_errored_document_keeps_the_stop_reasons_of_its_calls(self):
        cases = load_cases(limit=2)

        async def extract(path, mime_type):
            raise ExtractionError(
                "cut off", stop_reasons=[{"first": "max_tokens", "repair": None}]
            )

        result = await run_eval(cases, extract_fn=extract)

        assert [e["stop_reasons"] for e in result.errors] == [[{"first": "max_tokens", "repair": None}]] * 2

    async def test_the_mock_records_a_clean_stop_and_no_residual_violation(self):
        cases = load_cases(limit=2)

        result = await run_eval(cases, extract_fn=build_mock_extract_fn(cases))

        assert result.total_residual_schema_violations == 0
        assert all(d["stop_reasons"] == [{"first": "tool_use", "repair": None}] for d in result.per_doc)


class TestPromptCacheTokens:
    """The artifact records the cache tokens beside the uncached input tokens."""

    async def test_each_document_and_the_run_record_cache_writes_and_reads(self):
        cases = load_cases(limit=2)

        async def extract(path, mime_type):
            first = Path(path).stem == cases[0].doc_id
            return ExtractionResult(
                tool_input={},
                model="m",
                prompt_version="extract_v2",
                input_tokens=500,
                output_tokens=100,
                cost_usd=0.0,
                latency_ms=1,
                cache_creation_input_tokens=8000 if first else 0,
                cache_read_input_tokens=0 if first else 8000,
            )

        result = await run_eval(cases, extract_fn=extract, schema=SCHEMA_V2)

        by_doc = {d["doc_id"]: d for d in result.per_doc}
        assert by_doc[cases[0].doc_id]["cache_creation_input_tokens"] == 8000
        assert by_doc[cases[0].doc_id]["cache_read_input_tokens"] == 0
        assert by_doc[cases[1].doc_id]["cache_creation_input_tokens"] == 0
        assert by_doc[cases[1].doc_id]["cache_read_input_tokens"] == 8000
        assert result.to_dict()["total_input_tokens"] == 1000
        assert result.to_dict()["total_cache_creation_input_tokens"] == 8000
        assert result.to_dict()["total_cache_read_input_tokens"] == 8000

    async def test_a_run_whose_schema_does_not_cache_keeps_the_v1_shape(self):
        cases = load_cases(limit=1)

        result = await run_eval(cases, extract_fn=build_mock_extract_fn(cases))

        assert "cache_creation_input_tokens" not in result.per_doc[0]
        assert "cache_read_input_tokens" not in result.per_doc[0]
        assert "total_cache_creation_input_tokens" not in result.to_dict()
        assert "total_cache_read_input_tokens" not in result.to_dict()

    async def test_a_billed_but_errored_call_records_its_cache_tokens(self):
        cases = load_cases(limit=1)

        async def extract(path, mime_type):
            raise ExtractionError(
                "refused",
                input_tokens=100,
                output_tokens=5,
                cost_usd=0.02,
                cache_creation_input_tokens=8000,
                cache_read_input_tokens=0,
            )

        result = await run_eval(cases, extract_fn=extract, schema=SCHEMA_V2)

        assert result.errors[0]["cache_creation_input_tokens"] == 8000
        assert result.errors[0]["cache_read_input_tokens"] == 0

    async def test_the_first_document_runs_alone_so_the_cache_is_written_once(self):
        import asyncio

        cases = load_cases(limit=4)
        events = []

        async def extract(path, mime_type):
            events.append(("start", Path(path).stem))
            await asyncio.sleep(0.01)
            events.append(("end", Path(path).stem))
            return ExtractionResult({}, "m", "extract_v2", 0, 0, 0.0, 1)

        await run_eval(cases, extract_fn=extract, concurrency=4, schema=SCHEMA_V2)

        first = cases[0].doc_id
        assert events[:2] == [("start", first), ("end", first)]
        assert {name for kind, name in events[2:] if kind == "start"} == {c.doc_id for c in cases[1:]}

    async def test_a_schema_that_does_not_cache_starts_its_documents_together(self):
        import asyncio

        cases = load_cases(limit=3)
        events = []

        async def extract(path, mime_type):
            events.append("start")
            await asyncio.sleep(0.01)
            events.append("end")
            return ExtractionResult({}, "m", "extract_v1", 0, 0, 0.0, 1)

        await run_eval(cases, extract_fn=extract, concurrency=3)

        assert events[:3] == ["start", "start", "start"]


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
