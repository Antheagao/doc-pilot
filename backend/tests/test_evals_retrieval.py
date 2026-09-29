"""The retrieval eval: metric math, query loading, and a full offline run
over the committed gold corpus with the hashing embedder."""

import json

import pytest

from app.db import engine
from app.evals.dataset import load_cases
from app.evals.report import update_readme
from app.evals.retrieval import (
    RETRIEVAL_END_MARKER,
    RETRIEVAL_START_MARKER,
    load_gold_corpus,
    load_queries,
    ndcg_at,
    recall_at,
    reciprocal_rank,
    render_retrieval_table,
    run_retrieval_eval,
    write_retrieval_result,
)
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.indexing import ChunkingConfig

# --- metrics ----------------------------------------------------------------


def test_recall_at_k() -> None:
    relevant = frozenset({"a", "b"})
    assert recall_at(["a", "x", "b"], relevant, 1) == 0.5
    assert recall_at(["a", "x", "b"], relevant, 3) == 1.0
    assert recall_at(["x", "y"], relevant, 5) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "a"], frozenset({"a"})) == 0.5
    assert reciprocal_rank(["x"], frozenset({"a"})) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_otherwise() -> None:
    relevant = frozenset({"a", "b"})
    assert ndcg_at(["a", "b", "x"], relevant, 10) == pytest.approx(1.0)
    assert ndcg_at(["b", "a"], relevant, 10) == pytest.approx(1.0)
    assert 0 < ndcg_at(["x", "a", "b"], relevant, 10) < 1
    assert ndcg_at(["x", "y"], relevant, 10) == 0.0


# --- loading ----------------------------------------------------------------


def _corpus_cases():
    cases = load_cases()
    corpus = load_gold_corpus(cases)
    ids = {doc.doc_id for doc in corpus}
    return corpus, [case for case in cases if case.doc_id in ids]


def test_gold_corpus_covers_every_generated_doc() -> None:
    corpus, _ = _corpus_cases()

    assert len(corpus) == 25
    by_id = {doc.doc_id: doc for doc in corpus}
    assert by_id["024-eur-bakery-berlin"].text.startswith("Lindenplatz Bakery\n")
    assert "27,82 EUR" in by_id["024-eur-bakery-berlin"].text


def test_committed_queries_resolve_against_the_labels() -> None:
    _, cases = _corpus_cases()

    version, queries = load_queries(cases)

    assert version == "v1"
    assert len(queries) == len({q.id for q in queries}) == 109
    by_id = {q.id: q for q in queries}
    assert by_id["item-kw-led-desk-lamp"].relevant == {
        "018-dense-office-outfitters",
        "022-missing-field-receipt-no-vendor",
    }
    assert by_id["vendor-para-hardware-store"].relevant == {
        "002-clean-hardware-invoice",
        "019-dense-hardware-warehouse",
    }


def _write_queries(tmp_path, queries) -> object:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"query_set_version": "t", "queries": queries}))
    return path


@pytest.mark.parametrize(
    "queries,match",
    [
        ([{"id": "q", "type": "t", "query": "x", "relevant": {"item": "Hovercraft"}}], "matches no document"),
        ([{"id": "q", "type": "t", "query": "x", "relevant": {"doc_ids": ["999"]}}], "unknown doc id"),
        ([{"id": "q", "type": "t", "query": "x", "relevant": {"color": "red"}}], "unrecognized relevance"),
        (
            [
                {"id": "q", "type": "t", "query": "x", "relevant": {"doc_ids": ["001"]}},
                {"id": "q", "type": "t", "query": "y", "relevant": {"doc_ids": ["002"]}},
            ],
            "duplicate query id",
        ),
    ],
)
def test_malformed_queries_fail_loudly(tmp_path, queries, match) -> None:
    _, cases = _corpus_cases()

    with pytest.raises(ValueError, match=match):
        load_queries(cases, _write_queries(tmp_path, queries))


# --- full run ---------------------------------------------------------------


async def test_full_eval_run_over_gold_corpus(tmp_path) -> None:
    """End to end with the offline hashing embedder: real gold corpus,
    real query set, real Postgres. Asserts properties that must hold for
    ANY embedder (exact-keyword lexical recall, citation integrity, the
    eval leaving no rows behind), not the hashing embedder's quality."""
    corpus, cases = _corpus_cases()
    version, queries = load_queries(cases)
    chunking = [ChunkingConfig(max_chars=200, overlap_chars=80, context_headers=True)]

    result = await run_retrieval_eval(
        engine,
        corpus,
        queries,
        embedder=HashingEmbedder(),
        chunking_configs=chunking,
        query_set_version=version,
    )

    assert [c.mode for c in result.configs] == ["dense", "lexical", "hybrid"]
    lexical = next(c for c in result.configs if c.mode == "lexical")
    # Every item/vendor keyword query names its target verbatim; full-text
    # search must put every relevant doc in the top 10.
    keyword = [e for e in lexical.per_query if e["type"] in ("item_keyword", "vendor_keyword")]
    assert all(e["scores"]["recall@10"] == 1.0 for e in keyword)
    for config in result.configs:
        assert config.citation_integrity == 1.0
        assert config.n_chunks > len(corpus)
        assert set(config.by_type) == {q.type for q in queries}

    table = render_retrieval_table(result.to_dict())
    assert table.count("\n| ") == 4  # divider + one row per mode, after the header
    path = write_retrieval_result(result, tmp_path)
    artifact = json.loads(path.read_text())
    assert artifact["n_queries"] == len(queries)
    for config in artifact["configs"]:
        assert "per_query" not in config
        assert len(config["hardest"]) == 10
        mrrs = [entry["scores"]["mrr"] for entry in config["hardest"]]
        assert mrrs == sorted(mrrs)
        assert sum(group["n"] for group in config["by_type"].values()) == len(queries)


async def test_eval_leaves_no_rows_behind() -> None:
    from sqlalchemy import func, select

    from app.db import async_session_maker
    from app.models import Document

    async with async_session_maker() as session:
        before = (await session.execute(select(func.count()).select_from(Document))).scalar_one()

    corpus, cases = _corpus_cases()
    _, queries = load_queries(cases)
    await run_retrieval_eval(
        engine,
        corpus[:3],
        [q for q in queries if q.relevant & {d.doc_id for d in corpus[:3]}][:5],
        embedder=HashingEmbedder(),
        chunking_configs=[ChunkingConfig(0, 0, False)],
        modes=("lexical",),
    )

    async with async_session_maker() as session:
        after = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
    assert after == before


def test_readme_update_uses_the_retrieval_markers(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"intro\n{RETRIEVAL_START_MARKER}\nold\n{RETRIEVAL_END_MARKER}\noutro\n"
        "<!-- EVAL_TABLE:START -->\nkeep\n<!-- EVAL_TABLE:END -->\n"
    )

    update_readme(
        "new table", readme, start_marker=RETRIEVAL_START_MARKER, end_marker=RETRIEVAL_END_MARKER
    )

    text = readme.read_text()
    assert "new table" in text and "old" not in text
    assert "keep" in text
