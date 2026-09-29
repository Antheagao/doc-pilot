"""Text embedders for the retrieval index.

Two implementations behind one small interface:

- FastEmbedEmbedder: BAAI/bge-small-en-v1.5 run locally through fastembed
  (ONNX Runtime, CPU). A real semantic embedding model with no API key and
  no per-call cost -- the default, and what the committed retrieval eval
  numbers are measured with.
- HashingEmbedder: deterministic feature hashing of words and character
  trigrams. No model download and effectively instant, which is why the
  test suite and CI use it -- but it is lexical, not semantic ("lamp" and
  "lighting" share nothing), so it is a stand-in for plumbing tests, never
  a quality baseline to report.

Both return L2-normalized vectors of length EMBEDDING_DIM, the width of
document_chunks.embedding. Embedding is CPU-bound and synchronous; async
callers run it via starlette's run_in_threadpool.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.config import Settings, get_settings
from app.models import EMBEDDING_DIM


class Embedder(Protocol):
    # Recorded on every chunk (DocumentChunk.embedding_model) so vectors
    # from different models are never compared against each other.
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_WORD_RE = re.compile(r"\w+")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashingEmbedder:
    """Signed feature hashing (the "hashing trick") of lowercased word
    unigrams plus character trigrams of each word, into `dim` buckets.
    Trigrams give partial credit for morphology ("battery"/"batteries");
    the sign bit keeps colliding features from only ever adding up.
    blake2b rather than hash() because Python's str hash is salted per
    process, and the same text must map to the same vector in every run.
    """

    name = "hashing-v1"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        features = []
        for word in _WORD_RE.findall(text.lower()):
            features.append(f"w:{word}")
            padded = f"<{word}>"
            features.extend(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        return features

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            bucket = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# BGE's recommended instruction for the query side of short-query ->
# passage retrieval. Passages are embedded without it. See the model card
# for BAAI/bge-small-en-v1.5.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbedder:
    """A fastembed TextEmbedding, loaded once on first use (the ONNX
    session is ~70 MB and takes a second or two to initialize, so neither
    importing this module nor constructing the embedder pays for it)."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.name = f"fastembed:{model_name}"
        self.dim = EMBEDDING_DIM
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model_path = model_path
        self._model = None
        self._query_prefix = (
            BGE_QUERY_INSTRUCTION if model_name.lower().startswith("baai/bge-") else ""
        )

    def _load(self):
        if self._model is None:
            # Imported here, not at module top: fastembed pulls in
            # onnxruntime, which nothing on the hashing path (tests, CI)
            # should have to import.
            from fastembed import TextEmbedding

            kwargs = {}
            if self._model_path:
                kwargs["specific_model_path"] = self._model_path
            model = TextEmbedding(self._model_name, cache_dir=self._cache_dir, **kwargs)
            if model.embedding_size != self.dim:
                raise ValueError(
                    f"{self._model_name} produces {model.embedding_size}-d vectors but "
                    f"document_chunks.embedding is {self.dim}-d (app.models.EMBEDDING_DIM)"
                )
            self._model = model
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._load().embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().embed([self._query_prefix + text]))).tolist()


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    if settings.embedding_backend == "fastembed":
        return FastEmbedEmbedder(
            settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            model_path=settings.embedding_model_path,
        )
    raise ValueError(
        f"unknown embedding_backend {settings.embedding_backend!r}; "
        "expected 'fastembed' or 'hashing'"
    )


@lru_cache
def get_embedder() -> Embedder:
    """The process-wide embedder. Also a FastAPI dependency
    (routers/search.py), so tests can swap it via dependency_overrides."""
    return build_embedder(get_settings())
