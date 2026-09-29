#!/usr/bin/env python
"""Thin CLI for the retrieval eval (backend/app/evals/retrieval.py).

Indexes the gold page text of the labeled corpus, runs the query set in
every requested search mode and chunking config, prints a table, and
writes a JSON artifact to evals/results/retrieval/. Free and offline: no
API calls. Needs Postgres with pgvector (the same database the backend
uses -- everything happens in one rolled-back transaction).

Run with the backend venv's interpreter, from the repo root:

    backend\\.venv\\Scripts\\python.exe evals\\run_retrieval.py
    python evals/run_retrieval.py --chunk-sizes 0,400 --headers both --update-readme
"""

import argparse
import json
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the doc-pilot retrieval eval.")
    parser.add_argument(
        "--embedder",
        choices=("fastembed", "hashing"),
        default=None,
        help="override EMBEDDING_BACKEND (default: the configured backend)",
    )
    parser.add_argument(
        "--modes",
        default="lexical,dense,hybrid",
        help="comma-separated search modes to evaluate",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="0,400",
        help="comma-separated max chars per chunk; 0 = one chunk per page",
    )
    parser.add_argument(
        "--overlap", type=int, default=80, help="overlap chars for size-limited chunking"
    )
    parser.add_argument(
        "--headers",
        choices=("on", "off", "both"),
        default="on",
        help="contextual chunk headers (document title + page) on, off, or both",
    )
    parser.add_argument(
        "--lexical",
        default="idf",
        help="comma-separated full-text scorings to evaluate: idf, ts_rank",
    )
    parser.add_argument("--out", default=None, help="output directory for the result JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regression gate: compare this run to the newest committed artifact from the same "
        "embedder and exit 1 if any shared metric fell by more than --tolerance (the new artifact "
        "goes to a temp dir unless --out is given)",
    )
    parser.add_argument("--tolerance", type=float, default=0.005, help="for --check (default 0.005)")
    parser.add_argument(
        "--slack-queries", type=int, default=0, dest="slack_queries",
        help="for --check: also allow this many queries' worth of change per query group "
        "(cross-machine float noise; CI uses 1)",
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="write this run's metrics to evals/retrieval/snapshot_hashing_v1.json, the offline "
        "baseline the test suite checks exactly (use with --embedder hashing)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="render the table from the latest artifact; run nothing",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="write the latest run's table into README.md between the RETRIEVAL_TABLE markers",
    )
    return parser.parse_args()


def _print_failures(result: dict, limit: int = 12) -> None:
    """The hardest queries under the best config: where to look next."""
    best = max(result["configs"], key=lambda c: c["overall"]["mrr"])
    print(f"\nlowest-MRR queries under the best config ({best['mode']}):")
    for entry in best["hardest"][:limit]:
        print(
            f"  {entry['scores']['mrr']:.2f}  [{entry['type']}] {entry['query']!r} -> "
            f"top: {entry['ranked'][:3]}  relevant: {entry['relevant']}"
        )


def main() -> int:
    args = _parse_args()

    import os

    if args.embedder:
        os.environ["EMBEDDING_BACKEND"] = args.embedder

    from app.config import get_settings
    from app.db import engine
    from app.evals.dataset import load_cases
    from app.evals.report import update_readme
    from app.evals.retrieval import (
        RETRIEVAL_END_MARKER,
        RETRIEVAL_START_MARKER,
        SNAPSHOT_PATH,
        compare_retrieval_results,
        load_gold_corpus,
        load_latest_retrieval_result,
        load_latest_retrieval_result_for,
        load_queries,
        render_retrieval_table,
        run_retrieval_eval,
        snapshot_of,
        write_retrieval_result,
    )
    from app.retrieval.embeddings import build_embedder
    from app.retrieval.indexing import ChunkingConfig

    out_dir = Path(args.out) if args.out else None
    if args.check and out_dir is None:
        import tempfile

        out_dir = Path(tempfile.mkdtemp(prefix="retrieval-check-"))

    if not args.report_only:
        cases = load_cases()
        corpus = load_gold_corpus(cases)
        if not corpus:
            print(
                "error: no gold text found -- run `python backend/scripts/make_evals.py "
                "--text-only` first",
                file=sys.stderr,
            )
            return 1
        corpus_ids = {doc.doc_id for doc in corpus}
        corpus_cases = [case for case in cases if case.doc_id in corpus_ids]
        query_set_version, queries = load_queries(corpus_cases)

        headers = {"on": (True,), "off": (False,), "both": (False, True)}[args.headers]
        chunking_configs = [
            ChunkingConfig(max_chars=size, overlap_chars=args.overlap if size > 0 else 0,
                           context_headers=header)
            for size in (int(s) for s in args.chunk_sizes.split(","))
            for header in headers
        ]
        modes = tuple(mode.strip() for mode in args.modes.split(","))

        async def run():
            try:
                return await run_retrieval_eval(
                    engine,
                    corpus,
                    queries,
                    embedder=build_embedder(get_settings()),
                    chunking_configs=chunking_configs,
                    modes=modes,
                    lexical_scorings=tuple(s.strip() for s in args.lexical.split(",")),
                    query_set_version=query_set_version,
                    dataset_version=corpus_cases[0].dataset_version,
                )
            finally:
                await engine.dispose()

        result = asyncio.run(run())
        path = write_retrieval_result(result, out_dir)
        print(f"wrote {path}\n")
        fresh = result.to_dict()
        if args.write_snapshot:
            SNAPSHOT_PATH.write_text(json.dumps(snapshot_of(fresh), indent=2) + "\n", encoding="utf-8")
            print(f"wrote {SNAPSHOT_PATH}\n")
        if args.check:
            baseline = load_latest_retrieval_result_for(fresh["embedder"])
            if baseline is None:
                print(f"error: no committed artifact for {fresh['embedder']} to check against", file=sys.stderr)
                return 1
            regressions, improvements = compare_retrieval_results(
                fresh, baseline, tolerance=args.tolerance, slack_queries=args.slack_queries
            )
            any_change = sum(map(len, compare_retrieval_results(fresh, baseline, tolerance=0)))
            print(
                f"checked against run {baseline['started_at_utc']} (tolerance {args.tolerance}, "
                f"slack {args.slack_queries} quer{'y' if args.slack_queries == 1 else 'ies'}); "
                f"{any_change} metric(s) differ at all"
            )
            for line in improvements:
                print(f"  improved:  {line}")
            for line in regressions:
                print(f"  REGRESSED: {line}")
            if regressions:
                print(f"\n{len(regressions)} metric(s) regressed beyond tolerance", file=sys.stderr)
                return 1
            print("  no regressions\n")

    latest = load_latest_retrieval_result(out_dir)
    if latest is None:
        print("error: no retrieval results found", file=sys.stderr)
        return 1
    print(render_retrieval_table(latest))
    _print_failures(latest)

    if args.update_readme:
        readme_path = REPO_ROOT / "README.md"
        update_readme(
            render_retrieval_table(latest),
            readme_path,
            start_marker=RETRIEVAL_START_MARKER,
            end_marker=RETRIEVAL_END_MARKER,
        )
        print(f"\nupdated {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
