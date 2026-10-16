#!/usr/bin/env python
"""Thin CLI for the reconciliation eval (backend/app/evals/checks.py).

Breaks one printed amount at a time in every label, replays each broken
document through the arithmetic checks, and prints what they caught; then
replays the same broken documents with evidence quotes and prints what the
quote check adds. Also prints the checks' view of each committed extraction
run. Free and offline:
no API calls, no database, no key. Exits 1 if any label fails the checks
unmutated, since the table would then mean nothing.

Run with the backend venv's interpreter, from the repo root:

    backend\\.venv\\Scripts\\python.exe evals\\run_checks.py
    python evals/run_checks.py --update-readme
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the doc-pilot reconciliation eval.")
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="write the tables into README.md between the CHECKS_TABLE and QUOTES_TABLE markers",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from app.evals.checks import (
        CHECKS_END_MARKER,
        CHECKS_START_MARKER,
        QUOTES_END_MARKER,
        QUOTES_START_MARKER,
        check_metrics,
        failing_labels,
        render_checks_markdown,
        render_quotes_markdown,
        run_mutation_eval,
        run_quote_eval,
    )
    from app.evals.dataset import load_cases
    from app.evals.report import load_results, update_readme

    cases = load_cases()
    failing = failing_labels(cases)
    if failing:
        print("error: labels that fail the checks unmutated:", file=sys.stderr)
        for doc_id, report in failing:
            detail = " ".join(f"{c.id}: {c.detail}" for c in report.failing_checks)
            print(f"  {doc_id}: {detail}", file=sys.stderr)
        return 1

    result = run_mutation_eval(cases)
    markdown = render_checks_markdown(result)
    print(markdown)

    print(f"\nmisses ({len(result.misses)}):")
    for miss in result.misses:
        print(f"  {miss.doc_id} {miss.path} {miss.kind}: {miss.original} -> {miss.mutated}")

    quote_result = run_quote_eval(cases, result)
    quotes_markdown = render_quotes_markdown(quote_result)
    print(f"\n{quotes_markdown}")

    print(f"\ncaught_by_checks per committed run ({result.rules_version}):")
    for run in load_results():
        metrics = check_metrics(run)
        if metrics is None:
            print(f"  {run['model']} {run['started_at_utc']}: n/a")
            continue
        print(
            f"  {run['model']} {run['started_at_utc']}: "
            f"{metrics.n_caught_by_checks}/{metrics.n_incorrect} incorrect fields caught, "
            f"{metrics.n_docs_failing}/{metrics.n_docs} docs failing, "
            f"{metrics.n_correct_flagged} correct fields flagged"
        )

    if args.update_readme:
        readme_path = REPO_ROOT / "README.md"
        update_readme(
            markdown,
            readme_path,
            start_marker=CHECKS_START_MARKER,
            end_marker=CHECKS_END_MARKER,
        )
        update_readme(
            quotes_markdown,
            readme_path,
            start_marker=QUOTES_START_MARKER,
            end_marker=QUOTES_END_MARKER,
        )
        print(f"\nupdated {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
