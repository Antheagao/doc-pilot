"""Every committed eval run, as one series per suite: accuracy, cost and
latency per run over time, for the monitoring dashboard (GET
/monitoring/evals).

The three offline suites write differently shaped artifacts -- extraction
(app/evals/runner.py), the /ask agent (app/evals/agent.py) and retrieval
(app/evals/retrieval.py) -- so each is normalized to an EvalRunPoint: one
accuracy number (each suite's headline metric, named in
`accuracy_metric`), cost per item, and latency per item, plus a few
suite-specific extras. Like GET /stats, a corrupt or half-written
artifact is skipped with a warning, never a 500.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.evals.agent import RESULTS_DIR as AGENT_RESULTS_DIR
from app.evals.report import RESULTS_DIR as EXTRACTION_RESULTS_DIR
from app.evals.report import load_results
from app.evals.retrieval import RESULTS_DIR as RETRIEVAL_RESULTS_DIR
from app.retrieval.indexing import ChunkingConfig

logger = logging.getLogger(__name__)

Suite = Literal["extraction", "agent", "retrieval"]

# The retrieval eval measures a grid of configurations; the dashboard
# follows the one production search runs: hybrid, with this chunking.
PRODUCTION_RETRIEVAL_MODE = "hybrid"


@dataclass
class EvalRunPoint:
    suite: Suite
    started_at: datetime
    # What a line on the chart follows from run to run: the model (and
    # effort) for extraction and the agent, the embedder for retrieval.
    series: str
    # What else pins the run down: prompt, dataset or query-set version.
    detail: str
    # Documents, questions or queries scored.
    n: int
    accuracy: float | None
    accuracy_metric: str
    # Per document / question; None where the suite has no model cost
    # (retrieval runs a local embedding model).
    mean_cost_usd: float | None
    total_cost_usd: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    extra: dict[str, float | None] = field(default_factory=dict)


def parse_started_at(value: str) -> datetime:
    """Artifacts stamp runs either in ISO 8601's extended form (extraction)
    or its basic form (20260929T061630Z, the agent and retrieval evals);
    fromisoformat reads both. A stamp without an offset is UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _read_json_dir(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not directory.is_dir():
        return []
    loaded = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable eval artifact %s: %s", path, exc)
            continue
        if isinstance(data, dict):
            loaded.append((path, data))
        else:
            logger.warning("skipping eval artifact %s: not a JSON object", path)
    return loaded


def extraction_points(results_dir: Path | None = None) -> list[EvalRunPoint]:
    points = []
    for run in load_results(results_dir or EXTRACTION_RESULTS_DIR):
        summary = run["summary"]
        points.append(
            EvalRunPoint(
                suite="extraction",
                started_at=parse_started_at(run["started_at_utc"]),
                series=run["model"],
                detail=f"{run['prompt_version']} · dataset {run['dataset_version']}",
                n=run["n_scored"],
                accuracy=summary.get("overall_accuracy"),
                accuracy_metric="field accuracy",
                mean_cost_usd=run["mean_cost_per_doc"],
                total_cost_usd=run["total_cost_usd"],
                latency_p50_ms=run["latency_p50_ms"],
                latency_p95_ms=run["latency_p95_ms"],
                extra={
                    "caught_by_review": summary.get("caught_by_review"),
                    "hallucinations": summary.get("total_hallucinations"),
                },
            )
        )
    return points


def agent_points(results_dir: Path | None = None) -> list[EvalRunPoint]:
    points = []
    for path, run in _read_json_dir(results_dir or AGENT_RESULTS_DIR):
        try:
            summary = run["summary"]
            points.append(
                EvalRunPoint(
                    suite="agent",
                    started_at=parse_started_at(run["started_at_utc"]),
                    series=f"{run['model']} ({run['effort']})",
                    detail=f"{run['prompt_version']} · questions {run['question_set_version']}",
                    n=summary["n"],
                    accuracy=summary["accuracy"],
                    accuracy_metric="rubric correct",
                    mean_cost_usd=summary["mean_cost_usd"],
                    total_cost_usd=summary["total_cost_usd"],
                    latency_p50_ms=summary.get("latency_p50_ms"),
                    latency_p95_ms=summary.get("latency_p95_ms"),
                    extra={
                        "judge_grounded": summary.get("judge_grounded"),
                        "cites_relevant": summary.get("cites_relevant"),
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("skipping agent eval artifact %s: %r", path, exc)
    return points


def _production_config(configs: list[dict[str, Any]], chunking: ChunkingConfig) -> dict[str, Any] | None:
    wanted = {
        "max_chars": chunking.max_chars,
        "overlap_chars": chunking.overlap_chars,
        "context_headers": chunking.context_headers,
    }
    matches = [
        c for c in configs if c.get("mode") == PRODUCTION_RETRIEVAL_MODE and c.get("chunking") == wanted
    ]
    # Runs since lexical scoring became IDF-weighted record it; production uses it.
    for config in matches:
        if config.get("lexical_scoring") in (None, "idf"):
            return config
    return None


def retrieval_points(
    chunking: ChunkingConfig, results_dir: Path | None = None
) -> list[EvalRunPoint]:
    points = []
    for path, run in _read_json_dir(results_dir or RETRIEVAL_RESULTS_DIR):
        try:
            config = _production_config(run["configs"], chunking)
            if config is None:
                continue  # this run didn't measure the production configuration
            overall = config["overall"]
            points.append(
                EvalRunPoint(
                    suite="retrieval",
                    started_at=parse_started_at(run["started_at_utc"]),
                    series=run["embedder"],
                    detail=(
                        f"{PRODUCTION_RETRIEVAL_MODE} · queries {run['query_set_version']}"
                        f" · dataset {run['dataset_version']}"
                    ),
                    n=overall["n"],
                    accuracy=overall["recall@5"],
                    accuracy_metric="recall@5",
                    mean_cost_usd=None,
                    total_cost_usd=None,
                    latency_p50_ms=config.get("query_latency_p50_ms"),
                    latency_p95_ms=config.get("query_latency_p95_ms"),
                    extra={"mrr": overall.get("mrr"), "ndcg@10": overall.get("ndcg@10")},
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("skipping retrieval eval artifact %s: %r", path, exc)
    return points


def eval_history(chunking: ChunkingConfig) -> dict[Suite, list[EvalRunPoint]]:
    """Every suite's runs, oldest first."""
    history: dict[Suite, list[EvalRunPoint]] = {
        "extraction": extraction_points(),
        "agent": agent_points(),
        "retrieval": retrieval_points(chunking),
    }
    for points in history.values():
        points.sort(key=lambda p: p.started_at)
    return history
