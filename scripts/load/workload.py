"""The load workload as pure functions: what a virtual user sends and how a
reply is judged.

locustfile.py (run by Locust inside the loadgen container) and the host-side
harness both import this module, and it needs nothing outside the standard
library, so its tests run in the backend venv without Locust or Docker. It
holds:

- the corpus uploads and the nonce that makes each one byte-unique,
- the pools the reads draw from (search queries, agent questions),
- the SSE parser that finds the first event and the answer in a stream,
- the rule that turns an answer into a success or an error,
- the scenario description the artifact digests.

Nonce. The scripted model recognizes an upload by the SHA-256 of its pixel
content (backend/app/scripted_model.py content_digest: a PNG's IHDR, PLTE, tRNS,
IDAT and IEND chunks; a JPEG without its COM segments). A PNG tEXt chunk or a
JPEG COM segment therefore makes every upload byte-unique, so storage and
indexing do real work for each, while extraction still replays the committed
answer for that corpus document.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import re
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

TENANT_SLUGS = ("load-large", "load-medium", "load-small")
AUTH_KINDS = ("key", "cookie")

# The documents the harness uploads: 001 to 025, the ones with committed
# labels, gold page text and Sonnet answers (the PDF, 032, has no gold text).
CORPUS_DOC_NUMBERS = range(1, 26)
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# What a user does in the read phases, in tasks per hundred: the mix of the
# plan. Tests and the smoke may override it (a smoke needs every endpoint seen).
DEFAULT_READ_MIX = {"search": 70, "ask": 15, "ask_stream": 10, "documents": 5}
READ_ENDPOINTS = tuple(DEFAULT_READ_MIX)
SEARCH_K = 5
SEARCH_MODE = "hybrid"

# Most virtual users hold an API key; one in five signs in with a password and
# keeps the session cookie (the real argon2 and Origin paths).
KEY_USER_WEIGHT = 4
COOKIE_USER_WEIGHT = 1

# The agent questions that expect no answer. The scripted agent cites whatever
# it finds, and the harness counts an answer without a citation as an error, so
# a question built to be refused would be counted against the server.
EXCLUDED_QUESTION_TYPES = frozenset({"abstain"})

NONCE_KEY = "load-nonce"
SAMPLE_ERROR_MAX_CHARS = 200


# --- user classes -----------------------------------------------------------


def user_class_name(tenant: str, auth: str) -> str:
    """The key of one virtual-user class: `load-large:key`."""
    return f"{tenant}:{auth}"


def default_user_weights() -> dict[str, int]:
    """Every tenant with both kinds of user, keys outnumbering sign-ins."""
    return {
        user_class_name(tenant, auth): KEY_USER_WEIGHT if auth == "key" else COOKIE_USER_WEIGHT
        for tenant in TENANT_SLUGS
        for auth in AUTH_KINDS
    }


def parse_user_weights(text: str | None) -> dict[str, int]:
    """The user classes a run uses from its JSON (`{"load-large:key": 1}`), or
    the defaults when absent. A class with weight 0 is left out. ValueError
    names a class or weight that is not valid."""
    if not text:
        return default_user_weights()
    raw = json.loads(text)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("user weights must be a non-empty JSON object")
    known = set(default_user_weights())
    weights = {}
    for name, weight in raw.items():
        if name not in known:
            raise ValueError(f"unknown user class {name!r}")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
            raise ValueError(f"weight of {name!r} must be a non-negative integer")
        if weight:
            weights[name] = weight
    if not weights:
        raise ValueError("every user class has weight 0")
    return weights


def parse_read_mix(text: str | None) -> dict[str, int]:
    """The read mix from its JSON, or the default; endpoints with weight 0 are
    left out."""
    if not text:
        return dict(DEFAULT_READ_MIX)
    raw = json.loads(text)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("the read mix must be a non-empty JSON object")
    mix = {}
    for name, weight in raw.items():
        if name not in DEFAULT_READ_MIX:
            raise ValueError(f"unknown read endpoint {name!r}")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
            raise ValueError(f"weight of {name!r} must be a non-negative integer")
        if weight:
            mix[name] = weight
    if not mix:
        raise ValueError("every read endpoint has weight 0")
    return mix


# --- the corpus and the nonce -----------------------------------------------


@dataclass(frozen=True)
class CorpusFile:
    name: str
    media_type: str
    data: bytes


def corpus_documents(evals_dir: Path) -> list[CorpusFile]:
    """The 25 corpus documents under evals/docs, in file name order."""
    found = []
    for path in sorted((evals_dir / "docs").iterdir()):
        match = re.match(r"(\d{3})-", path.name)
        media_type = MEDIA_TYPES.get(path.suffix.lower())
        if match and int(match[1]) in CORPUS_DOC_NUMBERS and media_type:
            found.append(CorpusFile(path.name, media_type, path.read_bytes()))
    if len(found) != len(CORPUS_DOC_NUMBERS):
        raise ValueError(f"expected {len(CORPUS_DOC_NUMBERS)} corpus documents under {evals_dir}/docs")
    return found


def _png_text_chunk(nonce: str) -> bytes:
    data = NONCE_KEY.encode("latin-1") + b"\0" + nonce.encode("latin-1")
    body = b"tEXt" + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def inject_png_nonce(raw: bytes, nonce: str) -> bytes:
    """The PNG with a tEXt chunk holding `nonce` placed before IEND. The nonce
    must be Latin-1 (tEXt is). ValueError when the bytes are not a PNG."""
    signature = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(signature):
        raise ValueError("not a PNG")
    position = len(signature)
    while position + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[position : position + 4])
        if raw[position + 4 : position + 8] == b"IEND":
            return raw[:position] + _png_text_chunk(nonce) + raw[position:]
        position += 8 + length + 4
    raise ValueError("a PNG without an IEND chunk")


def inject_jpeg_nonce(raw: bytes, nonce: str) -> bytes:
    """The JPEG with a COM segment holding `nonce` right after the SOI marker.
    ValueError when the bytes are not a JPEG."""
    if not raw.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG")
    data = nonce.encode("latin-1")
    if len(data) > 0xFFFF - 2:
        raise ValueError("the nonce does not fit one COM segment")
    segment = b"\xff\xfe" + struct.pack(">H", len(data) + 2) + data
    return raw[:2] + segment + raw[2:]


def with_nonce(raw: bytes, media_type: str, nonce: str) -> bytes:
    """`raw` made byte-unique by `nonce` without touching its pixels."""
    if media_type == "image/png":
        return inject_png_nonce(raw, nonce)
    if media_type == "image/jpeg":
        return inject_jpeg_nonce(raw, nonce)
    raise ValueError(f"no nonce scheme for {media_type}")


# --- the pools the reads draw from ------------------------------------------


def search_queries(evals_dir: Path) -> list[str]:
    """The retrieval eval's 109 queries (item and vendor names, paraphrases,
    locations, amounts): text derived from the corpus, in file order."""
    data = json.loads((evals_dir / "retrieval" / "queries_v1.json").read_text(encoding="utf-8"))
    return [entry["query"] for entry in data["queries"]]


def ask_questions(evals_dir: Path, exclude_types: Iterable[str] = ()) -> list[str]:
    """The agent eval's questions, less the ones that expect no answer and any
    further question types a run leaves out (the smoke leaves out the reverse
    lookups, whose documents its ten uploads do not include)."""
    skipped = EXCLUDED_QUESTION_TYPES | set(exclude_types)
    data = json.loads((evals_dir / "agent" / "questions_v1.json").read_text(encoding="utf-8"))
    return [entry["question"] for entry in data["questions"] if entry["type"] not in skipped]


def parse_exclude_types(text: str | None) -> tuple[str, ...]:
    """Question types from a comma-separated setting."""
    return tuple(sorted(part for part in (text or "").split(",") if part))


# --- server-sent events -----------------------------------------------------


class SseFrame(NamedTuple):
    event: str
    data: str


_LINE_END = re.compile(r"\r\n|\r|\n")


class SseParser:
    """Frames of a server-sent-events stream from arbitrary byte chunks.

    A chunk may hold part of a frame or several; a multi-byte character or a
    CRLF may be cut between chunks. Feed chunks as they arrive: `feed` returns
    the frames each chunk completed, so the caller can stamp the moment a frame
    became whole (the point of the harness: time to first event and to the
    answer). A frame is dispatched by its blank line; comment lines (`:`) and
    the `id` and `retry` fields are ignored."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial = ""
        self._after_cr = False
        self._event = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[SseFrame]:
        text = self._decoder.decode(chunk)
        if not text:
            return []
        if self._after_cr and text.startswith("\n"):
            text = text[1:]  # the second half of a CRLF cut across chunks
        self._after_cr = text.endswith("\r")
        frames = []
        lines = _LINE_END.split(self._partial + text)
        self._partial = lines.pop()  # no line end yet: wait for the rest
        for line in lines:
            frame = self._line(line)
            if frame is not None:
                frames.append(frame)
        return frames

    def _line(self, line: str) -> SseFrame | None:
        if not line:
            if not self._event and not self._data:
                return None
            frame = SseFrame(self._event or "message", "\n".join(self._data))
            self._event, self._data = "", []
            return frame
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None


# --- what counts as an answer -----------------------------------------------


def answer_problem(run: object) -> str | None:
    """Why a stored /ask run is an error, or None when it is an answer: its
    status must be `answered` and it must cite at least one source. A reply
    that cites nothing is counted against the server, whatever its status."""
    if not isinstance(run, dict):
        return "bad_answer"
    if run.get("status") != "answered":
        return f"status_{run.get('status')}"
    citations = run.get("citations")
    if not isinstance(citations, list) or not citations:
        return "no_citation"
    return None


def stream_frame_problem(frame: SseFrame) -> str | None:
    """Why a frame ends a stream badly: an `error` event, or an `answer` event
    whose run is not an answer. None for any other frame, and for a good
    answer."""
    if frame.event == "error":
        return "stream_error"
    if frame.event != "answer":
        return None
    try:
        run = json.loads(frame.data).get("run")
    except (ValueError, AttributeError):
        return "bad_answer"
    return answer_problem(run)


def short_error(text: object) -> str:
    """An error as it is stored in a sample: one short line."""
    line = " ".join(str(text).split())
    return line[:SAMPLE_ERROR_MAX_CHARS]


# --- the raw samples file ---------------------------------------------------


def make_sample(
    name: str,
    *,
    tenant: str,
    auth: str,
    start: float,
    ms: float,
    status: int,
    error: str | None = None,
    **extra: object,
) -> dict:
    """One request as it is recorded: its own UTC wall-clock start (seconds
    since the epoch), its duration, the HTTP status, and an error code when it
    failed. `extra` carries what only some requests have (the time to first
    event and to the answer of a stream, the id of an uploaded document)."""
    return {
        "name": name,
        "tenant": tenant,
        "auth": auth,
        "start": round(start, 6),
        "ms": round(ms, 3),
        "status": status,
        "ok": error is None,
        "error": None if error is None else short_error(error),
        **extra,
    }


class SampleWriter:
    """Appends samples to a JSON-lines file, one flushed line each, so a run
    that is killed keeps what it measured."""

    def __init__(self, path: Path | str) -> None:
        # Held open for the life of the process; closed by close().
        self._handle = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def write(self, sample: dict) -> None:
        self._handle.write(json.dumps(sample, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._handle.close()


# --- the scenario description -----------------------------------------------


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scenario_description(
    evals_dir: Path,
    *,
    read_mix: dict[str, int] | None = None,
    user_weights: dict[str, int] | None = None,
    exclude_question_types: Iterable[str] = (),
) -> dict:
    """Everything the generator sends, as data the artifact digests: the read
    mix and the routes it hits, the pools by digest, each corpus document by the
    SHA-256 of its bytes, how uploads are made unique, and the users. Two runs
    with equal descriptions drove the same traffic."""
    questions = ask_questions(evals_dir, exclude_question_types)
    queries = search_queries(evals_dir)
    return {
        "version": 1,
        "read_mix": dict(read_mix or DEFAULT_READ_MIX),
        "routes": {
            "search": f"GET /search?q=<query>&k={SEARCH_K}&mode={SEARCH_MODE}",
            "ask": "POST /ask {question}",
            "ask_stream": "POST /ask/stream {question}",
            "documents": "GET /documents",
            "upload": "POST /documents multipart file",
        },
        "search_queries": {"count": len(queries), "sha256": _digest(queries)},
        "ask_questions": {
            "count": len(questions),
            "sha256": _digest(questions),
            "excluded_types": sorted(EXCLUDED_QUESTION_TYPES | set(exclude_question_types)),
        },
        "corpus": [
            {"name": doc.name, "sha256": hashlib.sha256(doc.data).hexdigest()}
            for doc in corpus_documents(evals_dir)
        ],
        "upload_nonce": f"png tEXt / jpeg COM chunk {NONCE_KEY!r}, unique per upload",
        "users": dict(user_weights or default_user_weights()),
        "think_time_ms": 0,
        "answer_rule": "status answered and at least one citation",
    }
