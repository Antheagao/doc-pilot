#!/usr/bin/env python
"""Thin CLI for the /ask agent eval (backend/app/evals/agent.py).

Seeds the labeled corpus (perfect extraction records + gold page text) in
one rolled-back transaction, asks every question in
evals/agent/questions_v1.json through the real agent, scores the answers
with the deterministic rubric, and writes a JSON artifact to
evals/results/agent/. Calls the real API: needs ANTHROPIC_API_KEY, and stops
launching questions once --max-cost is reached.

    python evals/run_agent.py                                  # claude-opus-5-5, effort medium
    python evals/run_agent.py --model claude-sonnet-5-5 --effort low
    python evals/run_agent.py --report-only --update-readme
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the doc-pilot /ask agent eval.")
    parser.add_argument("--model", default=None, help="override AGENT_MODEL")
    parser.add_argument("--effort", default=None, help="override AGENT_EFFORT (low..max)")
    parser.add_argument(
        "--max-cost", type=float, default=2.00, dest="max_cost",
        help="stop launching questions once spend reaches this (USD)",
    )
    parser.add_argument("--only", default=None, help="comma-separated question ids to run")
    parser.add_argument("--out", default=None, help="output directory for the result JSON")
    parser.add_argument("--report-only", action="store_true", help="render the table; run nothing")
    parser.add_argument(
        "--update-readme", action="store_true",
        help="write the results table into README.md between the AGENT_TABLE markers",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.model:
        os.environ["AGENT_MODEL"] = args.model
    if args.effort:
        os.environ["AGENT_EFFORT"] = args.effort

    from app.config import get_settings
    from app.db import engine
    from app.evals.agent import (
        AGENT_END_MARKER,
        AGENT_START_MARKER,
        load_agent_results,
        load_questions,
        render_agent_table,
        run_agent_eval,
        write_agent_result,
    )
    from app.evals.dataset import load_cases
    from app.evals.report import update_readme
    from app.evals.retrieval import load_gold_corpus
    from app.extraction import _ensure_model_priced
    from app.retrieval.embeddings import build_embedder

    out_dir = Path(args.out) if args.out else None
    settings = get_settings()

    if not args.report_only:
        if not settings.anthropic_api_key:
            print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
            return 1
        _ensure_model_priced(settings.agent_model)
        cases = load_cases()
        corpus = load_gold_corpus(cases)
        corpus_ids = {doc.doc_id for doc in corpus}
        corpus_cases = [case for case in cases if case.doc_id in corpus_ids]
        version, questions = load_questions(corpus_cases)
        if args.only:
            wanted = set(args.only.split(","))
            questions = [q for q in questions if q.id in wanted]

        async def run():
            try:
                return await run_agent_eval(
                    engine,
                    corpus_cases,
                    corpus,
                    questions,
                    settings=settings,
                    embedder=build_embedder(settings),
                    max_cost_usd=args.max_cost,
                    question_set_version=version,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run())
        path = write_agent_result(result, out_dir)
        print(f"wrote {path}\n")
        for entry in result.per_question:
            if "scores" not in entry:
                print(f"  SKIP  {entry['id']} ({entry['skipped']})")
                continue
            mark = "PASS" if entry["scores"]["correct"] else "FAIL"
            tools = ",".join(call["name"] for call in entry["tool_calls"]) or "-"
            print(f"  {mark}  {entry['id']:<20} ${entry['cost_usd']:.4f}  tools={tools}")
            if not entry["scores"]["correct"]:
                print(f"        checks={entry['scores']['checks']}  answer={entry['answer'][:160]!r}")

    table = render_agent_table(load_agent_results(out_dir))
    print("\n" + table)
    if args.update_readme:
        readme_path = REPO_ROOT / "README.md"
        update_readme(table, readme_path, start_marker=AGENT_START_MARKER, end_marker=AGENT_END_MARKER)
        print(f"\nupdated {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
