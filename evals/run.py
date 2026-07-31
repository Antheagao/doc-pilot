#!/usr/bin/env python
"""Thin CLI for running the doc-pilot extraction eval suite.

Stdlib argparse only -- the actual eval-running logic lives in
backend/app/evals/runner.py (run_eval, build_mock_extract_fn,
write_result). This script just wires argv to that module and prints a
human-readable summary.

Must be run with the backend venv's interpreter (or an activated
backend/.venv) since it imports `app`:

    backend\\.venv\\Scripts\\python.exe evals\\run.py --mock
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# app.evals.report's Ops table header uses ✓/✗, and Windows terminals
# default stdout to the system codepage (cp1252 etc.), which can't
# encode them -- reconfigure to utf-8 so `--report-only`/`--update-readme`
# don't crash on a plain `python evals/run.py` invocation from PowerShell.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the doc-pilot extraction eval suite.")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    parser.add_argument("--model", type=str, default=None, help="override EXTRACTION_MODEL")
    parser.add_argument(
        "--concurrency", type=int, default=3, help="max concurrent extract_fn calls"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.00,
        dest="max_cost",
        help="stop launching new docs once accumulated cost exceeds this (USD)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="use a deterministic zero-cost fake extract_fn"
    )
    parser.add_argument("--out", type=str, default=None, help="output directory for the result JSON")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="render tables from existing evals/results/*.json artifacts; run no extraction",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="write the eval tables into README.md and evals/results/latest.md "
        "(combinable with a normal run or with --report-only)",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


def _print_summary(result, cases, *, mock: bool) -> None:
    print(f"\nmodel={result.model}  prompt_version={result.prompt_version}  "
          f"dataset_version={result.dataset_version}")
    print(f"scored={result.n_scored}  errors={result.n_errors}  "
          f"skipped_cost_cap={result.n_skipped_cost_cap}")

    if mock:
        from app.evals.runner import predicted_mock_accuracy

        predicted = predicted_mock_accuracy(cases)
        actual = result.summary["overall_accuracy"] if result.summary else None
        print(f"\npredicted mock accuracy: {predicted:.6f}")
        print(f"actual mock accuracy:    {actual:.6f}" if actual is not None else "actual: n/a")

    summary = result.summary
    if summary is None:
        print("\nno docs were scored -- nothing to aggregate")
    else:
        print(f"\noverall_accuracy: {summary['overall_accuracy']:.6f}")
        print("per-field accuracy:")
        for name, acc in summary["per_field_accuracy"].items():
            print(f"  {name:<16} {acc:.4f}")
        print(f"caught_by_review: {summary['caught_by_review']}")
        print(f"total_hallucinations: {summary['total_hallucinations']}")
        print(f"total_schema_violations: {summary['total_schema_violations']}")

    print(f"\ntotal_cost_usd: ${result.total_cost_usd:.6f}")
    print(f"total_error_cost_usd: ${result.total_error_cost_usd:.6f}")
    print(f"mean_cost_per_doc: ${result.mean_cost_per_doc:.6f}")
    print(f"latency p50/p95 (ms): {result.latency_p50_ms} / {result.latency_p95_ms}")

    if result.errors:
        print(f"\nerrors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  {err['doc_id']}: {err['error_type']}: {err['error_message']}")

    if result.skipped_cost_cap:
        print(f"\nskipped_cost_cap ({len(result.skipped_cost_cap)}): {result.skipped_cost_cap}")


def _update_readme(results_dir: Path | None) -> None:
    from app.evals.report import (
        load_results,
        render_tables,
        update_readme,
        write_latest_md,
    )

    results = load_results(results_dir)
    readme_path = REPO_ROOT / "README.md"
    update_readme(render_tables(results), readme_path)
    latest_path = write_latest_md(results)
    print(f"\nupdated {readme_path}")
    print(f"wrote {latest_path}")


def main() -> None:
    args = _parse_args()

    try:
        import app.config  # noqa: F401
    except ImportError:
        print(
            "error: could not import `app` -- activate backend/.venv or run with "
            "backend\\.venv\\Scripts\\python.exe",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out) if args.out else None

    if args.report_only:
        from app.evals.report import load_results, render_tables

        results = load_results(out_dir)
        print(render_tables(results))
        if args.update_readme:
            _update_readme(out_dir)
        sys.exit(0)

    from app.config import get_settings
    from app.evals.dataset import load_cases
    from app.evals.runner import (
        build_mock_extract_fn,
        run_eval,
        write_result,
    )
    from app.extraction import _ensure_model_priced

    if args.model:
        os.environ["EXTRACTION_MODEL"] = args.model
        get_settings.cache_clear()
        _ensure_model_priced(args.model)

    cases = load_cases(limit=args.limit)

    extract_fn = build_mock_extract_fn(cases) if args.mock else None

    result = asyncio.run(
        run_eval(
            cases,
            extract_fn=extract_fn,
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost,
        )
    )

    _print_summary(result, cases, mock=args.mock)

    out_path = write_result(result, out_dir)
    print(f"\nwrote {out_path}")

    if args.update_readme:
        # Reload from ALL artifacts in out_dir, not just this run's
        # result -- the report is a history across every eval run, not
        # just the one just written.
        _update_readme(out_dir)

    if result.n_scored == 0:
        print(
            "\nerror: zero docs scored -- treating this as a failed run", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
