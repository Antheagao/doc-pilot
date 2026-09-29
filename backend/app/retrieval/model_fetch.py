"""One pinned, checksummed copy of the embedding model, for the eval and
production alike.

Left to itself, fastembed downloads BAAI/bge-small-en-v1.5 from whatever
source its installed version prefers (currently a quantized ONNX export on
Hugging Face), so the model a deployment ran could differ from the one the
retrieval eval measured -- and the README's numbers would describe a model
nobody runs. Instead, a fastembed model listed in PINNED is fetched from a
fixed URL, its SHA-256 verified, and loaded from that directory. The
retrieval eval's committed artifacts were all measured with exactly this
archive.

One patch is applied after extraction: this archive's tokenizer_config.json
declares model_max_length = 1e30 (the tokenizers library's "unset"
sentinel), which overflows when fastembed sizes its truncation and the
model fails to load. It is set to 512, the model's real limit (its config
.json: max_position_embeddings = 512).

Stdlib only (urllib honours HTTPS_PROXY); safe to call from several
processes at once -- each extracts into its own temp directory and the
first to rename it into place wins.
"""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MARKER = ".docpilot-pinned"


@dataclass(frozen=True)
class PinnedModel:
    url: str
    sha256: str
    # The directory the archive extracts to.
    dirname: str
    max_length: int


PINNED: dict[str, PinnedModel] = {
    "BAAI/bge-small-en-v1.5": PinnedModel(
        url="https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
        sha256="3858004b3822f64f940280874b8f2d2dc25b34a4f3eb3cdf617bdceeb21ed9ed",
        dirname="fast-bge-small-en-v1.5",
        max_length=512,
    ),
}


class ModelFetchError(RuntimeError):
    pass


def default_models_dir() -> Path:
    return Path(tempfile.gettempdir()) / "doc-pilot-models"


def ensure_pinned_model(model_name: str, dest_dir: str | Path | None = None) -> Path | None:
    """The local directory of `model_name`'s pinned archive, downloading
    and verifying it first if needed. None if the model isn't pinned (the
    caller falls back to fastembed's own download)."""
    pinned = PINNED.get(model_name)
    if pinned is None:
        return None
    dest = Path(dest_dir) if dest_dir else default_models_dir()
    target = dest / pinned.dirname
    if _marker_ok(target, pinned):
        return target

    dest.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{pinned.dirname}-", dir=dest))
    try:
        archive = work / "model.tar.gz"
        logger.info("downloading pinned embedding model %s from %s", model_name, pinned.url)
        digest = _download(pinned.url, archive)
        if digest != pinned.sha256:
            raise ModelFetchError(
                f"{pinned.url} has SHA-256 {digest}, expected {pinned.sha256}; refusing to load it"
            )
        with tarfile.open(archive) as tar:
            # filter="data" rejects absolute paths, "..", links outside the
            # tree and device files.
            tar.extractall(work / "x", filter="data")
        extracted = work / "x" / pinned.dirname
        if not extracted.is_dir():
            raise ModelFetchError(f"{pinned.url} did not contain {pinned.dirname}/")
        _patch_max_length(extracted, pinned.max_length)
        (extracted / MARKER).write_text(pinned.sha256)
        staged = dest / f".{pinned.dirname}-{uuid.uuid4().hex}"
        extracted.rename(staged)
        try:
            os.rename(staged, target)
        except OSError:
            # Another process got there first (or a stale unmarked copy is
            # in the way). A verified copy wins; anything else is replaced.
            if not _marker_ok(target, pinned):
                shutil.rmtree(target, ignore_errors=True)
                os.rename(staged, target)
            else:
                shutil.rmtree(staged, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target


def _marker_ok(target: Path, pinned: PinnedModel) -> bool:
    marker = target / MARKER
    return marker.is_file() and marker.read_text().strip() == pinned.sha256


def _download(url: str, path: Path) -> str:
    sha = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as out:
        while chunk := response.read(1 << 20):
            sha.update(chunk)
            out.write(chunk)
    return sha.hexdigest()


def _patch_max_length(model_dir: Path, max_length: int) -> None:
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_max_length", 0) > max_length:
        config["model_max_length"] = max_length
        config_path.write_text(json.dumps(config, indent=2) + "\n")
