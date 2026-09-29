"""Retrieval eval: how well does search find the right documents?

The extraction eval (app/evals/runner.py) scores fields against labels.
This one scores the retrieval layer (app/retrieval/) against a query set
whose relevance judgments are derived from those same labels, so -- like
the labels themselves -- they are correct by construction rather than
hand-assigned:

- Corpus: every labeled doc with gold page text in evals/text/ (written by
  scripts/make_evals.py --text-only from the same record the image was
  rendered from). Indexing gold text rather than a live transcription
  isolates retrieval quality from transcription quality: a miss here is a
  chunking/embedding/ranking miss, never an OCR one. It also makes the
  whole eval free and offline -- no API calls.
- Queries: evals/retrieval/queries_v1.json. Hand-written query strings,
  each with a relevance spec resolved against the labels at load time
  ("every doc whose line items include X", "every doc from vendor Y").
- Metrics, per query, over the ranked list of distinct documents (chunk
  hits collapsed to their document, best rank wins): recall@k, reciprocal
  rank, nDCG@10. Reported overall and per query type, for every
  (search mode x chunking config) combination in the run.
- Citation integrity: every returned hit's text must equal its page's
  stored text at [char_start:char_end]. Expected to be exactly 1.0 --
  it's an invariant check on the citation plumbing, run on real results.

Everything runs inside one transaction that is rolled back at the end, so
the eval leaves no rows behind in the database it runs against, and
search is scoped to the eval's own documents so pre-existing rows can't
leak into the rankings. Index scans are disabled for that transaction:
the eval measures the ranking function exactly, not an HNSW
approximation of it.
"""

import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.evals.dataset import EVALS_DIR, EvalCase
from app.models import Document, DocumentPage
from app.retrieval.embeddings import Embedder
from app.retrieval.indexing import ChunkingConfig, PageText, index_document_pages
from app.retrieval.search import SEARCH_MODES, SearchMode, search

TEXT_DIR = EVALS_DIR / "text"
QUERIES_PATH = EVALS_DIR / "retrieval" / "queries_v1.json"
RESULTS_DIR = EVALS_DIR / "results" / "retrieval"

RECALL_KS = (1, 3, 5, 10)
NDCG_K = 10
# Per-config query detail kept in the written artifact: only the lowest-MRR
# queries. Unlike an extraction eval run, this eval is free and takes
# seconds, so full per-query detail is one re-run away and isn't worth
# committing for every config of every run.
HARDEST_PER_CONFIG = 10
# Chunks fetched per query before collapsing to documents -- enough that
# the top-10 distinct documents are fully populated even when the best
# documents each contribute several chunks.
CHUNKS_PER_QUERY = 50


@dataclass(frozen=True)
class GoldDoc:
    doc_id: str
    mime_type: str
    text: str


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    type: str
    query: str
    relevant: frozenset[str]


def load_gold_corpus(cases: list[EvalCase], text_dir: Path = TEXT_DIR) -> list[GoldDoc]:
    """Every case that has gold page text. Cases without it (e.g. docs
    harvested from human review, which have an image but no generator
    record) are simply outside the retrieval corpus."""
    corpus = []
    for case in cases:
        path = text_dir / f"{case.doc_id}.txt"
        if path.is_file():
            corpus.append(GoldDoc(case.doc_id, case.mime_type, path.read_text(encoding="utf-8")))
    return corpus


def _resolve_relevant(spec: dict[str, Any], cases: list[EvalCase], query_id: str) -> frozenset[str]:
    if set(spec) == {"item"}:
        return frozenset(
            case.doc_id
            for case in cases
            if any(item["description"] == spec["item"] for item in case.fields["line_items"] or [])
        )
    if set(spec) == {"vendors"}:
        vendors = set(spec["vendors"])
        return frozenset(case.doc_id for case in cases if case.fields["vendor"] in vendors)
    if set(spec) == {"doc_ids"}:
        # Docs are named by their NNN prefix, which is unique in the corpus.
        by_prefix = {case.doc_id.split("-", 1)[0]: case.doc_id for case in cases}
        missing = [prefix for prefix in spec["doc_ids"] if prefix not in by_prefix]
        if missing:
            raise ValueError(f"{query_id}: unknown doc id prefix(es) {missing}")
        return frozenset(by_prefix[prefix] for prefix in spec["doc_ids"])
    raise ValueError(f"{query_id}: unrecognized relevance spec {spec!r}")


def load_queries(
    corpus_cases: list[EvalCase], path: Path = QUERIES_PATH
) -> tuple[str, list[RetrievalQuery]]:
    """Load the query set and resolve each query's relevant doc ids against
    the corpus labels. Loud on anything malformed, the same policy as
    app.evals.dataset: a query that resolves to zero relevant documents
    (a typo in an item name, a vendor that's no longer in the corpus)
    raises instead of silently scoring as an unanswerable query."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    seen: set[str] = set()
    for entry in raw["queries"]:
        query_id = entry["id"]
        if query_id in seen:
            raise ValueError(f"duplicate query id {query_id!r}")
        seen.add(query_id)
        relevant = _resolve_relevant(entry["relevant"], corpus_cases, query_id)
        if not relevant:
            raise ValueError(f"{query_id}: relevance spec matches no document in the corpus")
        queries.append(RetrievalQuery(query_id, entry["type"], entry["query"], relevant))
    return raw["query_set_version"], queries


# --- metrics ----------------------------------------------------------------


def recall_at(ranked: list[str], relevant: frozenset[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: frozenset[str]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked: list[str], relevant: frozenset[str], k: int) -> float:
    """Binary-relevance nDCG@k: DCG of the ranking over the DCG of a
    perfect one (every relevant doc first)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def _metric_keys() -> list[str]:
    return [f"recall@{k}" for k in RECALL_KS] + ["mrr", f"ndcg@{NDCG_K}"]


def score_ranking(ranked: list[str], relevant: frozenset[str]) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at(ranked, relevant, k) for k in RECALL_KS}
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    scores[f"ndcg@{NDCG_K}"] = ndcg_at(ranked, relevant, NDCG_K)
    return scores


def _mean_scores(per_query: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {"n": len(per_query)}
    for key in _metric_keys():
        means[key] = statistics.fmean(entry["scores"][key] for entry in per_query)
    return means


# --- run --------------------------------------------------------------------


@dataclass
class ConfigResult:
    mode: SearchMode
    chunking: dict[str, Any]
    n_chunks: int
    overall: dict[str, float]
    by_type: dict[str, dict[str, float]]
    citation_integrity: float
    query_latency_p50_ms: float
    per_query: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalRunResult:
    embedder: str
    query_set_version: str
    dataset_version: str | None
    started_at_utc: str
    n_docs: int
    n_queries: int
    configs: list[ConfigResult]

    def to_dict(self) -> dict[str, Any]:
        """The artifact form: every config's summary, plus its
        HARDEST_PER_CONFIG lowest-MRR queries in place of all of them."""
        data = asdict(self)
        for config in data["configs"]:
            per_query = config.pop("per_query")
            config["hardest"] = sorted(per_query, key=lambda e: (e["scores"]["mrr"], e["id"]))[
                :HARDEST_PER_CONFIG
            ]
        return data


async def _index_corpus(
    session: AsyncSession,
    doc_ids: dict[str, uuid.UUID],
    corpus: list[GoldDoc],
    embedder: Embedder,
    chunking: ChunkingConfig,
) -> int:
    n_chunks = 0
    for doc in corpus:
        page = PageText(page_number=1, text=doc.text, source="gold")
        n_chunks += await index_document_pages(
            session, doc_ids[doc.doc_id], [page], embedder, chunking
        )
    return n_chunks


async def _citation_integrity(session: AsyncSession, hits) -> float:
    """Fraction of hits whose cited span is exactly what the page says."""
    if not hits:
        return 1.0
    pages = {
        (page.document_id, page.page_number): page.text
        for page in (
            await session.execute(
                select(DocumentPage).where(
                    DocumentPage.document_id.in_({hit.document_id for hit in hits})
                )
            )
        ).scalars()
    }
    intact = sum(
        pages[(hit.document_id, hit.page_number)][hit.char_start : hit.char_end] == hit.text
        for hit in hits
    )
    return intact / len(hits)


async def run_retrieval_eval(
    engine: AsyncEngine,
    corpus: list[GoldDoc],
    queries: list[RetrievalQuery],
    *,
    embedder: Embedder,
    chunking_configs: list[ChunkingConfig],
    modes: tuple[SearchMode, ...] = SEARCH_MODES,
    query_set_version: str = "v1",
    dataset_version: str | None = None,
) -> RetrievalRunResult:
    started_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results: list[ConfigResult] = []

    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            await session.execute(text("SET LOCAL enable_indexscan = off"))

            documents = {
                doc.doc_id: Document(
                    filename=doc.doc_id,
                    mime_type=doc.mime_type,
                    storage_path=f"evals/text/{doc.doc_id}.txt",
                    status="extracted",
                )
                for doc in corpus
            }
            session.add_all(documents.values())
            await session.flush()
            doc_ids = {doc_id: document.id for doc_id, document in documents.items()}
            by_uuid = {document_id: doc_id for doc_id, document_id in doc_ids.items()}
            scope = list(doc_ids.values())

            for chunking in chunking_configs:
                n_chunks = await _index_corpus(session, doc_ids, corpus, embedder, chunking)
                for mode in modes:
                    per_query = []
                    latencies = []
                    all_hits = []
                    for query in queries:
                        start = time.perf_counter()
                        hits = await search(
                            session,
                            query.query,
                            embedder=embedder,
                            k=CHUNKS_PER_QUERY,
                            mode=mode,
                            candidates=CHUNKS_PER_QUERY,
                            document_ids=scope,
                        )
                        latencies.append((time.perf_counter() - start) * 1000)
                        all_hits.extend(hits)
                        ranked = list(dict.fromkeys(by_uuid[hit.document_id] for hit in hits))
                        per_query.append(
                            {
                                "id": query.id,
                                "type": query.type,
                                "query": query.query,
                                "relevant": sorted(query.relevant),
                                "ranked": ranked[:NDCG_K],
                                "scores": score_ranking(ranked, query.relevant),
                            }
                        )

                    types = sorted({entry["type"] for entry in per_query})
                    results.append(
                        ConfigResult(
                            mode=mode,
                            chunking=asdict(chunking),
                            n_chunks=n_chunks,
                            overall=_mean_scores(per_query),
                            by_type={
                                query_type: _mean_scores(
                                    [e for e in per_query if e["type"] == query_type]
                                )
                                for query_type in types
                            },
                            citation_integrity=await _citation_integrity(session, all_hits),
                            query_latency_p50_ms=statistics.median(latencies),
                            per_query=per_query,
                        )
                    )
        finally:
            await session.close()
            await transaction.rollback()

    return RetrievalRunResult(
        embedder=embedder.name,
        query_set_version=query_set_version,
        dataset_version=dataset_version,
        started_at_utc=started_at,
        n_docs=len(corpus),
        n_queries=len(queries),
        configs=results,
    )


def write_retrieval_result(result: RetrievalRunResult, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = result.embedder.replace("/", "_").replace(":", "_")
    path = out_dir / f"{result.started_at_utc}_{slug}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


# --- report -----------------------------------------------------------------

RETRIEVAL_START_MARKER = "<!-- RETRIEVAL_TABLE:START -->"
RETRIEVAL_END_MARKER = "<!-- RETRIEVAL_TABLE:END -->"

# Query types grouped into the report's columns: what kind of query is it?
_TYPE_COLUMNS = (
    ("Keyword", ("item_keyword", "vendor_keyword")),
    ("Paraphrase", ("item_paraphrase", "vendor_paraphrase")),
    ("Location", ("location",)),
    ("Amount", ("amount",)),
)


def _chunking_label(chunking: dict[str, Any]) -> str:
    size = "page" if chunking["max_chars"] <= 0 else f"{chunking['max_chars']}/{chunking['overlap_chars']}"
    return f"{size}{' +hdr' if chunking['context_headers'] else ''}"


def _type_group_recall(config: dict[str, Any], types: tuple[str, ...], k: int = 5) -> str:
    """Recall@k over a group of query types: the per-type means from
    by_type, weighted by each type's query count."""
    groups = [config["by_type"][t] for t in types if t in config["by_type"]]
    n = sum(group["n"] for group in groups)
    if not n:
        return "n/a"
    return f"{100 * sum(group[f'recall@{k}'] * group['n'] for group in groups) / n:.1f}%"


def render_retrieval_table(result: dict[str, Any]) -> str:
    """Markdown table for one run artifact: one row per (mode, chunking)
    config. Overall columns are means over every query; the per-type
    columns are Recall@5 within that group of query types."""
    header = (
        "| Mode | Chunking | Chunks | Recall@1 | Recall@5 | MRR | nDCG@10 | "
        + " | ".join(f"{name} R@5" for name, _ in _TYPE_COLUMNS)
        + " |"
    )
    divider = "|" + " --- |" * (7 + len(_TYPE_COLUMNS))
    rows = []
    for config in result["configs"]:
        overall = config["overall"]
        cells = [
            config["mode"],
            _chunking_label(config["chunking"]),
            str(config["n_chunks"]),
            f"{100 * overall['recall@1']:.1f}%",
            f"{100 * overall['recall@5']:.1f}%",
            f"{overall['mrr']:.3f}",
            f"{overall['ndcg@10']:.3f}",
            *(_type_group_recall(config, types) for _, types in _TYPE_COLUMNS),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    caption = (
        f"*{result['n_queries']} queries over {result['n_docs']} documents, embedder "
        f"`{result['embedder']}`, query set {result['query_set_version']}, run "
        f"{result['started_at_utc']}. Recall@k is the share of a query's relevant documents "
        "found in the top k (chunk hits collapsed to distinct documents), averaged over "
        "queries; chunking is max/overlap characters per chunk (\"page\" = one chunk per "
        "page), +hdr = contextual chunk headers.*"
    )
    return "\n".join([header, divider, *rows, "", caption])


def load_latest_retrieval_result(results_dir: Path | None = None) -> dict[str, Any] | None:
    paths = sorted((results_dir or RESULTS_DIR).glob("*.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))
