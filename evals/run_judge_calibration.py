#!/usr/bin/env python
"""Score the agent-eval graders against human labels.

The deterministic rubric is scored offline, every run. With --judge, the
LLM judge (JUDGE_MODEL) grades the same hand-labeled answers against the
real API (needs ANTHROPIC_API_KEY; a few cents, capped by --max-cost).

    python evals/run_judge_calibration.py                      # rubric only, free
    python evals/run_judge_calibration.py --judge --update-readme
    python evals/run_judge_calibration.py --judge --grader groundedness --update-readme
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the agent-eval graders.")
    parser.add_argument("--judge", action="store_true", help="also run an LLM grader (live API)")
    parser.add_argument(
        "--grader", choices=("reference", "groundedness"), default="reference",
        help="which LLM grader --judge runs: the eval judge (reference facts; correct + grounded) "
        "or the reference-free grader used on live /ask answers (grounded only)",
    )
    parser.add_argument("--model", default=None, help="override JUDGE_MODEL")
    parser.add_argument("--effort", default=None, help="override JUDGE_EFFORT")
    parser.add_argument("--max-cost", type=float, default=0.50, dest="max_cost")
    parser.add_argument("--out", default=None, help="output directory for the result JSON")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    if args.model:
        os.environ["JUDGE_MODEL"] = args.model
    if args.effort:
        os.environ["JUDGE_EFFORT"] = args.effort

    from app.config import get_settings
    from app.evals.agent import load_questions
    from app.evals.dataset import load_cases
    from app.evals.judge import (
        JUDGE_END_MARKER,
        JUDGE_START_MARKER,
        load_calibration,
        load_calibration_results,
        render_calibration_table,
        run_calibration,
        write_calibration_result,
    )
    from app.evals.report import update_readme
    from app.evals.retrieval import load_gold_corpus

    out_dir = Path(args.out) if args.out else None
    settings = get_settings()

    if not args.report_only:
        if args.judge and not settings.anthropic_api_key:
            print("error: --judge needs ANTHROPIC_API_KEY", file=sys.stderr)
            return 1
        cases = load_cases()
        corpus_ids = {doc.doc_id for doc in load_gold_corpus(cases)}
        cases = [case for case in cases if case.doc_id in corpus_ids]
        _, questions = load_questions(cases)
        _, items = load_calibration(cases, questions)
        result = asyncio.run(
            run_calibration(
                items, settings, with_judge=args.judge, max_cost_usd=args.max_cost, grader=args.grader
            )
        )
        path = write_calibration_result(result, out_dir)
        print(f"wrote {path}\n")
        for row in result["disagreements"]:
            print(f"  disagreement: {row}")

    table = render_calibration_table(load_calibration_results(out_dir))
    print("\n" + table)
    if args.update_readme:
        readme_path = REPO_ROOT / "README.md"
        update_readme(table, readme_path, start_marker=JUDGE_START_MARKER, end_marker=JUDGE_END_MARKER)
        print(f"\nupdated {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
