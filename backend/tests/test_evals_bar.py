"""Tests for app.evals.bar (the pre-registered bar for making schema v2 the
default) and its CLI, evals/check_bar.py.

Pure/offline: the artifacts are the committed Sonnet v1 run and a v2 mock run
made to look like a Sonnet run, then broken one term at a time.
"""

import importlib.util
import json
import sys
from copy import deepcopy

import pytest

from app.evals.bar import core_misses, evaluate_bar
from app.evals.dataset import EVALS_DIR, load_cases
from app.evals.report import core_counts, load_results
from app.evals.runner import build_mock_extract_fn, run_eval
from app.extraction_fields import SCHEMA_V2


def _spec():
    spec = importlib.util.spec_from_file_location("evals_check_bar_cli", EVALS_DIR / "check_bar.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_bar_cli = _spec()


@pytest.fixture(scope="module")
def v1_run():
    (run,) = [r for r in load_results() if r["model"] == "claude-sonnet-5" and r["n_scored"] == 25]
    return run


@pytest.fixture(scope="module")
async def mock_v2():
    cases = load_cases()
    result = await run_eval(cases, extract_fn=build_mock_extract_fn(cases, SCHEMA_V2), schema=SCHEMA_V2)
    return result.to_dict()


@pytest.fixture
def v2_run(mock_v2):
    """The v2 mock, dressed as a Sonnet run (the mock's model is 'mock-model')."""
    run = deepcopy(mock_v2)
    run["model"] = "claude-sonnet-5"
    return run


def _verdicts(v1, v2):
    return {term.name: term.passed for term in evaluate_bar(v1, v2)}


def _break_core_fields(run, count):
    """Make `count` more core-document fields wrong."""
    broken = 0
    for doc in run["per_doc"]:
        for field in doc["fields"].values():
            if doc["doc_id"][:3] <= "025" and field["correct"] and broken < count:
                field["correct"] = False
                broken += 1


def test_a_clean_v2_run_meets_every_term(v1_run, v2_run):
    verdicts = _verdicts(v1_run, v2_run)

    assert verdicts == {
        "runs to compare": True,
        "complete run": True,
        "core accuracy": True,
        "residual schema violations": True,
        "evidence coverage": True,
        "no max_tokens stop": True,
        "no unmasked card number": True,
    }


class TestRunsToCompare:
    def test_another_model_fails(self, v1_run, v2_run):
        v2_run["model"] = "claude-haiku-4-5"

        assert _verdicts(v1_run, v2_run)["runs to compare"] is False

    def test_two_v1_runs_fail(self, v1_run, v2_run):
        v2_run["prompt_version"] = "extract_v1"

        assert _verdicts(v1_run, v2_run)["runs to compare"] is False


class TestCompleteRun:
    def test_an_errored_document_fails(self, v1_run, v2_run):
        v2_run["n_errors"] = 1
        v2_run["n_scored"] = 31

        assert _verdicts(v1_run, v2_run)["complete run"] is False

    def test_fewer_than_32_scored_fails(self, v1_run, v2_run):
        v2_run["n_scored"] = 3

        assert _verdicts(v1_run, v2_run)["complete run"] is False

    def test_a_skipped_document_fails(self, v1_run, v2_run):
        v2_run["n_skipped_cost_cap"] = 2

        assert _verdicts(v1_run, v2_run)["complete run"] is False


class TestCoreAccuracy:
    def test_one_field_worse_is_allowed_and_two_is_not(self, v1_run, v2_run):
        v1_correct, scored = core_counts(v1_run)
        v2_correct = core_counts(v2_run)[0]
        assert v1_correct - v2_correct == 1

        assert _verdicts(v1_run, v2_run)["core accuracy"] is True
        _break_core_fields(v2_run, 1)
        assert core_counts(v2_run) == (v1_correct - 2, scored)
        assert _verdicts(v1_run, v2_run)["core accuracy"] is False

    def test_better_than_v1_passes(self, v1_run, v2_run):
        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "core accuracy"]

        assert "v2 is 1 field(s) worse than v1 (allowed 1)" in term.detail
        v2_run["per_doc"] = deepcopy(v1_run["per_doc"])
        assert _verdicts(v1_run, v2_run)["core accuracy"] is True

    def test_different_denominators_fail_even_when_the_share_is_better(self, v1_run, v2_run):
        v2_run["per_doc"] = [d for d in v2_run["per_doc"] if d["doc_id"][:3] != "025"]

        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "core accuracy"]

        assert term.passed is False
        assert "different core fields" in term.detail

    def test_the_misses_are_listed_by_document_and_field(self, v1_run, v2_run):
        assert core_misses(v1_run) == [("023-missing-field-order-no-currency", "currency")]
        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "core accuracy"]
        assert "003-clean-stationery-order tax" in term.detail
        assert "007-skewed-garden-center vendor" in term.detail
        assert "174/175" in term.detail
        assert "173/175" in term.detail


class TestResidualSchemaViolations:
    def test_one_residual_violation_fails(self, v1_run, v2_run):
        v2_run["total_residual_schema_violations"] = 1
        v2_run["per_doc"][0]["residual_schema_violations"] = 1

        assert _verdicts(v1_run, v2_run)["residual schema violations"] is False

    def test_a_document_with_one_fails_even_if_the_total_was_lost(self, v1_run, v2_run):
        v2_run["per_doc"][0]["residual_schema_violations"] = 1

        assert _verdicts(v1_run, v2_run)["residual schema violations"] is False

    def test_an_artifact_that_does_not_record_it_fails(self, v1_run, v2_run):
        del v2_run["total_residual_schema_violations"]

        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "residual schema violations"]

        assert term.passed is False
        assert "not recorded" in term.detail


class TestEvidenceCoverage:
    def _lower(self, run, coverage):
        run["summary"]["evidence_coverage"]["coverage"] = coverage
        run["summary"]["evidence_coverage"]["with_quote"] = int(coverage * 100)
        run["summary"]["evidence_coverage"]["leaves"] = 100

    def test_sonnet_needs_95_percent(self, v1_run, v2_run):
        self._lower(v2_run, 0.94)
        assert _verdicts(v1_run, v2_run)["evidence coverage"] is False
        self._lower(v2_run, 0.95)
        assert _verdicts(v1_run, v2_run)["evidence coverage"] is True

    def test_haiku_is_printed_but_not_required(self, v1_run, v2_run):
        v2_run["model"] = "claude-haiku-4-5"
        v1_run = {**v1_run, "model": "claude-haiku-4-5"}
        self._lower(v2_run, 0.5)

        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "evidence coverage"]

        assert term.passed is True
        assert "50.0%" in term.detail
        assert "Sonnet only" in term.detail

    def test_a_run_without_evidence_fails(self, v1_run, v2_run):
        del v2_run["summary"]["evidence_coverage"]

        assert _verdicts(v1_run, v2_run)["evidence coverage"] is False


class TestMaxTokensStops:
    def test_a_stop_in_a_scored_document_fails(self, v1_run, v2_run):
        v2_run["per_doc"][4]["stop_reasons"] = [{"first": "tool_use", "repair": "max_tokens"}]

        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "no max_tokens stop"]

        assert term.passed is False
        assert v2_run["per_doc"][4]["doc_id"] in term.detail

    def test_a_stop_in_an_errored_document_fails(self, v1_run, v2_run):
        v2_run["per_doc"].append(
            {
                "doc_id": "032-multipage-invoice-totals-page-two",
                "status": "error",
                "stop_reasons": [{"first": "tool_use", "repair": None}, {"first": "max_tokens", "repair": None}],
            }
        )

        assert _verdicts(v1_run, v2_run)["no max_tokens stop"] is False

    def test_an_old_error_message_that_names_the_stop_fails(self, v1_run, v2_run):
        v2_run["per_doc"].append(
            {
                "doc_id": "x",
                "status": "error",
                "error_message": "model did not return a usable tool call (stop_reason='max_tokens')",
            }
        )

        assert _verdicts(v1_run, v2_run)["no max_tokens stop"] is False

    def test_scored_documents_that_do_not_record_their_stops_fail(self, v1_run, v2_run):
        for doc in v2_run["per_doc"]:
            del doc["stop_reasons"]

        (term,) = [t for t in evaluate_bar(v1_run, v2_run) if t.name == "no max_tokens stop"]

        assert term.passed is False
        assert "not recorded" in term.detail


class TestCardNumbers:
    def test_a_card_number_in_a_quote_fails(self, v1_run, v2_run):
        v2_run["per_doc"][0]["fields"]["total"]["evidence"] = {
            "page": 1,
            "quote": "Paid by VISA 4111 1111 1111 1111 $88.57",
        }

        assert _verdicts(v1_run, v2_run)["no unmasked card number"] is False

    def test_a_card_number_in_a_value_fails(self, v1_run, v2_run):
        v2_run["per_doc"][0]["fields"]["vendor"]["actual"] = "4111111111111111"

        assert _verdicts(v1_run, v2_run)["no unmasked card number"] is False

    def test_the_masked_form_passes(self, v1_run, v2_run):
        v2_run["per_doc"][0]["fields"]["total"]["evidence"] = {"page": 1, "quote": "Paid by VISA ****1111 $88.57"}

        assert _verdicts(v1_run, v2_run)["no unmasked card number"] is True

    def test_sixteen_digit_floats_are_measurements_not_card_numbers(self, v1_run, v2_run):
        # 0.9634131736526946 is one of the committed Haiku artifact's means; its
        # digits pass the Luhn check by chance, as one in ten such floats do.
        v2_run["summary"]["mean_confidence_correct"] = 0.9634131736526946

        assert _verdicts(v1_run, v2_run)["no unmasked card number"] is True


class TestCli:
    def _write(self, tmp_path, v1, v2):
        v1_path, v2_path = tmp_path / "v1.json", tmp_path / "v2.json"
        v1_path.write_text(json.dumps(v1), encoding="utf-8")
        v2_path.write_text(json.dumps(v2), encoding="utf-8")
        return str(v1_path), str(v2_path)

    def _main(self, monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["check_bar.py", *argv])
        return check_bar_cli.main()

    def test_it_exits_0_and_prints_every_term_when_the_bar_is_met(self, monkeypatch, tmp_path, capsys, v1_run, v2_run):
        paths = self._write(tmp_path, v1_run, v2_run)

        code = self._main(monkeypatch, *paths)

        out = capsys.readouterr().out
        assert code == 0
        assert out.count("PASS") == 7
        assert "FAIL" not in out
        assert "bar: met" in out

    def test_it_exits_1_naming_the_failed_terms(self, monkeypatch, tmp_path, capsys, v1_run, v2_run):
        v2_run["total_residual_schema_violations"] = 3
        paths = self._write(tmp_path, v1_run, v2_run)

        code = self._main(monkeypatch, *paths)

        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL  residual schema violations: 3 in total" in out
        assert "bar: NOT met (residual schema violations)" in out

    def test_a_committed_v1_run_given_as_v2_fails_most_terms(self, monkeypatch, tmp_path, capsys, v1_run):
        paths = self._write(tmp_path, v1_run, v1_run)

        assert self._main(monkeypatch, *paths) == 1
        assert capsys.readouterr().out.count("FAIL") >= 4

    def test_an_unreadable_artifact_exits_2(self, monkeypatch, tmp_path, capsys, v1_run):
        (tmp_path / "bad.json").write_text("{", encoding="utf-8")
        good = tmp_path / "good.json"
        good.write_text(json.dumps(v1_run), encoding="utf-8")

        with pytest.raises(SystemExit) as exit_info:
            self._main(monkeypatch, str(good), str(tmp_path / "bad.json"))

        assert exit_info.value.code == 2
        assert "cannot read" in capsys.readouterr().err

    def test_a_file_that_is_not_an_artifact_exits_2(self, monkeypatch, tmp_path, capsys, v1_run):
        paths = self._write(tmp_path, v1_run, {"model": "x"})

        assert self._main(monkeypatch, *paths) == 2
