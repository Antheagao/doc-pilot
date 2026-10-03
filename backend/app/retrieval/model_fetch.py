"""One pinned, checksummed copy of the embedding model, for the eval and
production alike.

Left to itself, fastembed downloads BAAI/bge-small-en-v1.5 from whatever
source its installed version prefers (currently a quantized ONNX export on
Hugging Face), so the model a deployment ran could differ from the one the
retrieval eval measured -- and the README's numbers would describe a model
nobody runs. Instead, a fastembed model listed in PINNED is fetched file by
file from a fixed revision of a fixed repository, each file's SHA-256
verified, and loaded from that directory.

The source is the model's own repository, BAAI/bge-small-en-v1.5 (MIT), at
a pinned commit: its fp32 ONNX export, saved under the name fastembed
loads (model_optimized.onnx), and its tokenizer files. (Until October 2026
the pin was qdrant's archive of an fp32 export on Cloud Storage; that
bucket went private, so the pin moved to the authors' export of the same
weights. Its first retrieval-eval run in CI reproduced every committed
metric at the reported precision.)

One patch is applied after download: a tokenizer_config.json that declares
model_max_length larger than the model's real limit (the tokenizers
library's "unset" sentinel, 1e30, overflows when fastembed sizes its
truncation) is set to 512 -- config.json: max_position_embeddings = 512.

Stdlib only (urllib honours HTTPS_PROXY); safe to call from several
processes at once -- each downloads into its own temp directory and the
first to rename it into place wins.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MARKER = ".docpilot-pinned"


@dataclass(frozen=True)
class PinnedFile:
    # Where the file lives under PinnedModel.base_url ...
    source: str
    # ... and the name it is saved (and loaded) under.
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if "/" in self.name or "\\" in self.name or self.name in ("", ".", ".."):
            raise ValueError(f"pinned file name {self.name!r} must be a plain file name")


@dataclass(frozen=True)
class PinnedModel:
    base_url: str
    files: tuple[PinnedFile, ...]
    # The directory the files are saved in, under the cache directory.
    dirname: str
    max_length: int

    @property
    def pin_id(self) -> str:
        """What a verified copy records in its marker file: changing any
        pinned file invalidates every cached copy."""
        listing = "\n".join(f"{f.name} {f.sha256}" for f in sorted(self.files, key=lambda f: f.name))
        return hashlib.sha256(f"{self.base_url}\n{listing}".encode()).hexdigest()


BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

PINNED: dict[str, PinnedModel] = {
    "BAAI/bge-small-en-v1.5": PinnedModel(
        base_url=f"https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/{BGE_SMALL_REVISION}",
        files=(
            PinnedFile("onnx/model.onnx", "model_optimized.onnx", "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"),
            PinnedFile("config.json", "config.json", "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"),
            PinnedFile("tokenizer.json", "tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
            PinnedFile("tokenizer_config.json", "tokenizer_config.json", "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3"),
            PinnedFile("special_tokens_map.json", "special_tokens_map.json", "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
            PinnedFile("vocab.txt", "vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
        ),
        dirname="bge-small-en-v1.5",
        max_length=512,
    ),
}


class ModelFetchError(RuntimeError):
    pass


def default_models_dir() -> Path:
    return Path(tempfile.gettempdir()) / "doc-pilot-models"


def ensure_pinned_model(model_name: str, dest_dir: str | Path | None = None) -> Path | None:
    """The local directory of `model_name`'s pinned files, downloading and
    verifying them first if needed. None if the model isn't pinned (the
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
        staged = work / pinned.dirname
        staged.mkdir()
        mismatches = []
        for file in pinned.files:
            url = f"{pinned.base_url}/{file.source}"
            logger.info("downloading pinned embedding model file %s", url)
            try:
                digest = _download(url, staged / file.name)
            except OSError as exc:  # URLError, timeouts, disk errors
                raise ModelFetchError(
                    f"could not download the pinned {model_name} from {url}: {exc}. "
                    "Allow outbound HTTPS to that host, or copy the model over from a machine that "
                    "can (`python scripts/fetch_embedding_model.py` prints where it lands) and set "
                    "EMBEDDING_MODEL_PATH to it. There is deliberately no fallback to another "
                    "export: it would serve a model the retrieval eval never measured."
                ) from exc
            if digest != file.sha256:
                mismatches.append(f"{file.source}: SHA-256 {digest}, expected {file.sha256}")
        if mismatches:
            # Every file is checked before refusing, so one run shows all of them.
            raise ModelFetchError(
                f"{pinned.base_url} does not match the pin; refusing to load it:\n  "
                + "\n  ".join(mismatches)
            )
        _patch_max_length(staged, pinned.max_length)
        (staged / MARKER).write_text(pinned.pin_id)
        moved = dest / f".{pinned.dirname}-{uuid.uuid4().hex}"
        staged.rename(moved)
        try:
            os.rename(moved, target)
        except OSError:
            # Another process got there first (or a stale unmarked copy is
            # in the way). A verified copy wins; anything else is replaced.
            if not _marker_ok(target, pinned):
                shutil.rmtree(target, ignore_errors=True)
                os.rename(moved, target)
            else:
                shutil.rmtree(moved, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target


def _marker_ok(target: Path, pinned: PinnedModel) -> bool:
    marker = target / MARKER
    return marker.is_file() and marker.read_text().strip() == pinned.pin_id


def _download(url: str, path: Path) -> str:
    sha = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as out:
        commit = response.headers.get("X-Repo-Commit")
        if commit:
            logger.info("%s is at revision %s", url, commit)
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
