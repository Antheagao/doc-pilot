import math
import subprocess
import sys

import pytest

from app.config import BACKEND_DIR, Settings
from app.models import EMBEDDING_DIM
from app.retrieval.embeddings import (
    BGE_QUERY_INSTRUCTION,
    FastEmbedEmbedder,
    HashingEmbedder,
    build_embedder,
)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_normalized_and_full_width() -> None:
    embedder = HashingEmbedder()

    first = embedder.embed_query("LED Desk Lamp")
    again = HashingEmbedder().embed_documents(["LED Desk Lamp"])[0]

    assert first == again
    assert len(first) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


def test_hashing_embedder_scores_shared_words_above_unrelated_text() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("desk lamp")
    related, unrelated = embedder.embed_documents(["LED Desk Lamp  2  $19.70", "Cat Litter 10lb"])

    assert _cosine(query, related) > _cosine(query, unrelated)


def test_hashing_embedder_handles_empty_text() -> None:
    vector = HashingEmbedder().embed_query("")

    assert vector == [0.0] * EMBEDDING_DIM


def test_build_embedder_selects_backend() -> None:
    assert isinstance(build_embedder(Settings(embedding_backend="hashing")), HashingEmbedder)
    fastembed = build_embedder(Settings(embedding_backend="fastembed"))
    assert isinstance(fastembed, FastEmbedEmbedder)
    assert fastembed.name == "fastembed:BAAI/bge-small-en-v1.5"
    with pytest.raises(ValueError, match="unknown embedding_backend"):
        build_embedder(Settings(embedding_backend="word2vec"))


def test_fastembed_embedder_loads_nothing_until_first_use() -> None:
    """Constructing the embedder (which the API does at startup via its
    dependency) must not download or load the ONNX model."""
    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")

    assert embedder._model is None
    assert embedder.embed_documents([]) == []
    assert embedder._model is None


def test_bge_models_get_the_query_instruction_others_do_not() -> None:
    assert FastEmbedEmbedder("BAAI/bge-small-en-v1.5")._query_prefix == BGE_QUERY_INSTRUCTION
    assert FastEmbedEmbedder("sentence-transformers/all-MiniLM-L6-v2")._query_prefix == ""


def test_fastembed_embedder_applies_instruction_to_queries_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    class FakeVector(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        embedding_size = EMBEDDING_DIM

        def embed(self, texts):
            seen.append(list(texts))
            return iter(FakeVector([0.0] * EMBEDDING_DIM) for _ in texts)

    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(embedder, "_load", lambda: FakeModel())

    embedder.embed_documents(["LED Desk Lamp"])
    embedder.embed_query("desk lighting")

    assert seen == [["LED Desk Lamp"], [BGE_QUERY_INSTRUCTION + "desk lighting"]]


def test_importing_the_app_never_imports_fastembed() -> None:
    """fastembed (and onnxruntime under it) loads lazily, on first real
    embed. Checked in a fresh interpreter so no other test's imports can
    mask a regression."""
    code = (
        "import sys, app.main, app.worker, app.retrieval.search;"
        "assert 'fastembed' not in sys.modules, 'fastembed imported eagerly'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=BACKEND_DIR)


def test_fastembed_warm_up_runs_one_real_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(embedder, "embed_query", lambda text: calls.append(text) or [])

    embedder.warm_up()

    assert calls == ["warm up"]


def test_warm_up_embedder_never_raises(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    from app.retrieval import embeddings as embeddings_module

    class Broken(HashingEmbedder):
        def warm_up(self) -> None:
            raise OSError("no network for the model download")

    monkeypatch.setattr(embeddings_module, "get_embedder", lambda: Broken())

    embeddings_module.warm_up_embedder()  # must not raise

    assert "warm-up failed" in caplog.text
