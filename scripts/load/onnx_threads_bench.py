"""Embedding throughput of P concurrent processes with T ONNX threads each.

Why it exists: every process's fastembed TextEmbedding (onnxruntime) defaults
its intra-op thread pool to all cores, so N worker containers on Docker's C CPUs
run C x N threads. This measures total chunks per second for every combination
of process count and thread count, so the `EMBEDDING_THREADS` rule and its
measurement (`embedding_threads` in phases.py) can be checked on any machine.

Each cell starts P fresh processes, loads the pinned bge-small model in each,
embeds one warm-up batch, waits for all of them at a barrier, then embeds
3-chunk batches of receipt-sized text for `--seconds` and adds up the rates.
A thread count of 0 leaves onnxruntime's default (every core). Cells whose
P x T passes MAX_TOTAL_THREADS are skipped and print "-": they only show more of
the collapse.

Run it in the load image, which holds the model volume of a load run (build the
image with `load.py stack up` first). It uses no model API and no database:

    docker run --rm -v doc-pilot-load_models:/data/models -v "$PWD/scripts/load:/bench:ro" \\
        -w /app doc-pilot-load-backend python /bench/onnx_threads_bench.py

    ... python /bench/onnx_threads_bench.py --processes 1,2,4,8 --threads 0,1,2,4,8 --seconds 20

It is a measuring tool, not a pytest test: fastembed is imported inside the
worker function only, so importing this module needs nothing installed.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time

MODEL = "BAAI/bge-small-en-v1.5"
MODEL_DIR = "/data/models"
MAX_TOTAL_THREADS = 32
# A batch the size the worker embeds: three ~110 character chunks of receipt text.
TEXTS = [
    "Receipt  Page 1  TRADER JOE'S #123  BANANAS ORGANIC 0.99  GREEK YOGURT PLAIN 5.49  SOURDOUGH LOAF 4.99  SUBTOTAL 11.47",
    "Invoice INV-2024-0042  Acme Supplies LLC  Net 30  Qty 12 Widget, blue, 3/8in  Unit 4.25  Line total 51.00  Tax 4.21",
    "TOTAL 63.42  VISA ****4421  AUTH 0912AB  Thank you for shopping with us  Returns within 30 days with receipt only.",
]


def embed_for(threads: int, seconds: float, results, barrier) -> None:
    """One process: load the model with `threads` (0 for the default), warm up,
    wait for the others, then put its chunks per second on `results`."""
    from fastembed import TextEmbedding

    from app.retrieval.model_fetch import ensure_pinned_model

    path = ensure_pinned_model(MODEL, MODEL_DIR)
    options = {"specific_model_path": str(path)} if path else {}
    if threads:
        options["threads"] = threads
    model = TextEmbedding(MODEL, cache_dir=MODEL_DIR, **options)
    list(model.embed(TEXTS))
    barrier.wait()
    chunks = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        list(model.embed(TEXTS))
        chunks += len(TEXTS)
    results.put(chunks / seconds)


def measure(processes: int, threads: int, seconds: float) -> float:
    """Total chunks per second of `processes` processes embedding at once."""
    results = multiprocessing.Queue()
    barrier = multiprocessing.Barrier(processes)
    workers = [
        multiprocessing.Process(target=embed_for, args=(threads, seconds, results, barrier))
        for _ in range(processes)
    ]
    for worker in workers:
        worker.start()
    rates = [results.get(timeout=seconds + 300) for _ in workers]
    for worker in workers:
        worker.join()
    return sum(rates)


def format_table(processes: list[int], threads: list[int], rates: dict[tuple[int, int], float]) -> str:
    """Rows of process counts, columns of thread counts, total chunks/s; "-"
    for a cell that was skipped."""
    header = ["P \\ T", *("default" if t == 0 else str(t) for t in threads)]
    rows = [header]
    for p in processes:
        rows.append([str(p), *(f"{rates[p, t]:.1f}" if (p, t) in rates else "-" for t in threads)])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join("  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in rows)


def parse_ints(text: str, *, minimum: int) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a comma-separated list of integers") from None
    if not values or any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(f"{text!r} must list integers of at least {minimum}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processes", type=lambda t: parse_ints(t, minimum=1), default=[1, 2, 4, 8])
    parser.add_argument(
        "--threads",
        type=lambda t: parse_ints(t, minimum=0),
        default=[0, 1, 2, 4, 8],
        help="ONNX threads per process; 0 is onnxruntime's default (every core)",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="measured seconds per cell")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    rates: dict[tuple[int, int], float] = {}
    for p in args.processes:
        for t in args.threads:
            if t and p * t > MAX_TOTAL_THREADS:
                continue
            rates[p, t] = measure(p, t, args.seconds)
            print(f"P={p} T={t or 'default'}: {rates[p, t]:.1f} chunks/s in total", file=sys.stderr, flush=True)
    print(f"Total chunks per second, {args.seconds:g} s per cell (P processes x T threads):")
    print(format_table(args.processes, args.threads, rates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
