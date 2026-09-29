"""Retrieval-quality regression gates: the offline eval checked exactly
against its committed snapshot, the comparison both gates use, and the
pinned embedding model the real-model gate (and production) loads."""

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from app.db import engine
from app.evals.dataset import load_cases
from app.evals.retrieval import (
    SNAPSHOT_PATH,
    compare_retrieval_results,
    load_gold_corpus,
    load_queries,
    run_retrieval_eval,
    snapshot_of,
)
from app.retrieval import model_fetch
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.indexing import ChunkingConfig
from app.retrieval.model_fetch import ModelFetchError, PinnedModel, ensure_pinned_model

REGENERATE = (
    "python evals/run_retrieval.py --embedder hashing --chunk-sizes 200 --headers on "
    "--out /tmp/retrieval --write-snapshot"
)


async def test_offline_eval_matches_its_committed_snapshot() -> None:
    """Any change to chunking, full-text scoring, fusion or the query set
    shows up here as a metric diff -- worse or better. If the change is
    intended, regenerate the snapshot so the diff is reviewed with it."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    cases = load_cases()
    corpus = load_gold_corpus(cases)
    ids = {doc.doc_id for doc in corpus}
    version, queries = load_queries([case for case in cases if case.doc_id in ids])

    result = await run_retrieval_eval(
        engine,
        corpus,
        queries,
        embedder=HashingEmbedder(),
        chunking_configs=[ChunkingConfig(200, 80, True)],
        query_set_version=version,
        dataset_version=cases[0].dataset_version,
    )

    regressions, improvements = compare_retrieval_results(
        snapshot_of(result.to_dict()), snapshot, tolerance=1e-9
    )
    changes = "\n".join([*regressions, *improvements])
    assert not changes, f"retrieval metrics changed:\n{changes}\n\nIf intended: {REGENERATE}"


# --- the comparison ---------------------------------------------------------------


def _result(mrr: float = 0.8, amount_recall: float = 0.6, integrity: float = 1.0, **fields) -> dict:
    scores = {"n": 10, "recall@1": 0.5, "recall@3": 0.6, "recall@5": 0.7, "recall@10": 0.8, "mrr": 0.8, "ndcg@10": 0.75}
    config = {
        "mode": "hybrid",
        "chunking": {"max_chars": 200, "overlap_chars": 80, "context_headers": True},
        "lexical_scoring": "idf",
        "overall": {**scores, "mrr": mrr},
        "by_type": {"amount": {**scores, "n": 5, "recall@5": amount_recall}},
        "citation_integrity": integrity,
    }
    return {"embedder": "e", "query_set_version": "v1", "dataset_version": "v1", "configs": [config], **fields}


def test_drops_beyond_tolerance_are_regressions_and_rises_are_reported() -> None:
    baseline = _result()

    regressions, improvements = compare_retrieval_results(
        _result(mrr=0.79, amount_recall=0.8), baseline, tolerance=0.005
    )

    assert regressions == ["hybrid (idf) 200/80 +hdr overall mrr: 0.8000 -> 0.7900 (-0.0100)"]
    assert improvements == ["hybrid (idf) 200/80 +hdr amount recall@5: 0.6000 -> 0.8000 (+0.2000)"]
    # Float noise inside the tolerance is neither.
    assert compare_retrieval_results(_result(mrr=0.797), baseline, tolerance=0.005) == ([], [])


def test_slack_allows_one_query_of_cross_machine_noise_per_group() -> None:
    # amount recall@5 over 5 queries: one query flipping is -0.2.
    baseline = _result()

    assert compare_retrieval_results(_result(amount_recall=0.4), baseline, tolerance=0.005, slack_queries=1) == ([], [])
    (two_queries,), _ = compare_retrieval_results(
        _result(amount_recall=0.2), baseline, tolerance=0.005, slack_queries=1
    )
    assert "amount recall@5" in two_queries


def test_incomparable_runs_fail_rather_than_pass() -> None:
    baseline = _result()

    (other_embedder,), _ = compare_retrieval_results(_result(embedder="x"), baseline, tolerance=0.01)
    assert other_embedder.startswith("not comparable: embedder")

    elsewhere = copy.deepcopy(baseline)
    elsewhere["configs"][0]["chunking"]["max_chars"] = 400
    (no_overlap,), _ = compare_retrieval_results(elsewhere, baseline, tolerance=0.01)
    assert "no search config in common" in no_overlap

    (integrity,), _ = compare_retrieval_results(_result(integrity=0.99), baseline, tolerance=0.01)
    assert "citation integrity" in integrity


# --- the pinned model ---------------------------------------------------------------


def _archive(tmp_path: Path, members: dict[str, bytes], name: str = "model.tar.gz") -> tuple[Path, str]:
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        for member, data in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


TOKENIZER = json.dumps({"model_max_length": 1000000000000000019884624838656, "do_lower_case": True}).encode()


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    archive, sha = _archive(
        tmp_path, {"m/tokenizer_config.json": TOKENIZER, "m/model_optimized.onnx": b"onnx"}
    )

    def pin(sha256: str = sha, url: str | None = None) -> None:
        monkeypatch.setitem(
            model_fetch.PINNED, "test/model", PinnedModel(url or archive.as_uri(), sha256, "m", 512)
        )

    pin()
    return pin


def test_the_pinned_archive_is_verified_extracted_and_patched(tmp_path, pinned, monkeypatch) -> None:
    dest = tmp_path / "models"

    path = ensure_pinned_model("test/model", dest)

    assert path == dest / "m" and (path / "model_optimized.onnx").read_bytes() == b"onnx"
    assert json.loads((path / "tokenizer_config.json").read_text())["model_max_length"] == 512
    # Only the model directory is left behind: no temp dirs, no archive.
    assert [p.name for p in dest.iterdir()] == ["m"]

    # Already verified: no second download.
    monkeypatch.setattr(model_fetch, "_download", lambda url, path: pytest.fail("downloaded twice"))
    assert ensure_pinned_model("test/model", dest) == path


def test_a_checksum_mismatch_loads_nothing(tmp_path, pinned) -> None:
    pinned(sha256="0" * 64)
    dest = tmp_path / "models"

    with pytest.raises(ModelFetchError, match="refusing to load it"):
        ensure_pinned_model("test/model", dest)

    assert list(dest.iterdir()) == []


def test_an_unverified_copy_in_the_way_is_replaced(tmp_path, pinned) -> None:
    stale = tmp_path / "models" / "m"
    stale.mkdir(parents=True)
    (stale / "model_optimized.onnx").write_bytes(b"who knows")

    path = ensure_pinned_model("test/model", tmp_path / "models")

    assert (path / "model_optimized.onnx").read_bytes() == b"onnx"


def test_archive_entries_cannot_escape_the_target(tmp_path, pinned) -> None:
    archive, sha = _archive(tmp_path, {"../escaped.txt": b"x", "m/tokenizer_config.json": TOKENIZER}, "evil.tar.gz")
    pinned(sha256=sha, url=archive.as_uri())

    with pytest.raises(tarfile.TarError):
        ensure_pinned_model("test/model", tmp_path / "models")

    assert not (tmp_path / "escaped.txt").exists()


def test_models_that_are_not_pinned_are_left_to_fastembed(tmp_path) -> None:
    assert ensure_pinned_model("some/other-model", tmp_path) is None
