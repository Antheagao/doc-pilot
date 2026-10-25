"""The scripted model: a stand-in for the Anthropic client that costs nothing.

MODEL_PROVIDER=scripted makes extraction._build_client return a
ScriptedModelClient instead of anthropic.AsyncAnthropic. It exposes the two
entry points the app calls, `messages.create` and `beta.messages.create`, and
returns the SDK's own response types (Message, BetaMessage, their text and
tool_use blocks, Usage, BetaCitationSearchResultLocation). So the code under
test is the real code: card masking, the agent loop's citation resolution and
refusal fallback echo, the telemetry spans, the schema repair path (which can
run, but no answer from the default corpus triggers it: every replayed and
label answer has 0 schema violations, so the load mix measures the single-call
extraction path only), and the pricing, which prices the scripted model at 0
(extraction.PRICING_PER_MTOK) and stores that on every row. The load test
exists to measure the worker, the queue, the database and the API, and a
stand-in that skipped any of them would measure nothing.

It exists in app/ and not in the tests because the load stack runs the
production image, which holds app/ and no tests/. Like HashingEmbedder it is an
explicit, opt-in stand-in: Settings refuses it beside an API key, and refuses a
real model name beside it, so a process cannot both spend and be scripted.

Four kinds of request, told apart by shape, the way the app makes them:

- extraction: `messages.create` with the forced `record_extraction` tool. The
  document is recognized by the SHA-256 of its pixel content (content_digest),
  so an upload made byte-unique with a metadata chunk still reads as its corpus
  document; the answer is what the committed Sonnet run returned for that
  document and schema (app/evals/replay.py), else the label at confidence 0.9
  with its gold evidence, and for bytes no corpus document owns a fixed answer
  at confidence 0.5, which routes it to review.
- transcription: `messages.create` with the transcription prompt and no tools.
  The document's gold page text plus one `Ref` line from the digest of the full
  bytes, so every upload's text, chunks and vectors differ.
- the agent: `beta.messages.create` with TOOL_DEFINITIONS. A fixed policy: one
  tool call (query_extractions for a question with an amount or a total, else
  search_documents), get_page on the top hit, then an answer citing up to two
  of the search results it was given, by their real source and block indices.
  The app does not stream model calls, so nothing here streams; a request that
  asks to is refused.
- the judge: `beta.messages.create` with a JSON-schema `output_config.format`.
  A verdict built from the schema it was given.

A request the real API would reject, or that matches no kind, raises
anthropic.BadRequestError, so a stand-in fails the way the real thing does and a
change that sends a malformed request is caught offline instead of after a
billed call. Everything is deterministic: ids, tokens, answers. Time is the one
configurable cost (Settings.scripted_model_profile), spent through an injected
`sleep` so tests assert exact delays.

Tokens are counted, not guessed: usage is estimate_tokens over the request's
text (four characters a token, rounded up) plus DOCUMENT_BLOCK_TOKENS for each
image or PDF, and the output is the same estimate over what the reply holds. The
"replay" profile instead reports the tokens the committed run recorded for the
document, and extraction takes its latency from there too.
"""

import asyncio
import base64
import binascii
import functools
import hashlib
import json
import re
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Literal, NamedTuple

import anthropic
import httpx2
from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from app.agent.requests import InvalidRequestError, validate_request
from app.agent.tools import SOURCE_SCHEME, SUMMARY_SOURCE, TOOL_DEFINITIONS
from app.config import SCRIPTED_MODEL, Settings
from app.evals.checks import label_tool_input
from app.evals.dataset import EVALS_DIR, LABELS_DIR, EvalCase, load_cases
from app.evals.replay import (
    REPLAY_ARTIFACT,
    RESULTS_DIR,
    TRANSCRIPTION_PROXY_ARTIFACT,
    Replay,
    _replayed_tool_input,
    scored_replay,
)
from app.evals.retrieval import load_gold_corpus
from app.extraction import IMAGE_MIME_TYPES, PDF_MIME_TYPE
from app.extraction_fields import SCHEMAS, ExtractionSchema, record_extraction_tool
from app.transcription import TRANSCRIBE_PROMPT_TEXT

RequestKind = Literal["extraction", "transcription", "agent", "judge"]

# What an image or a PDF page counts for in estimate_tokens: about a megapixel
# image at the API's width-times-height-over-750 rule, a round figure for a
# scripted request, not a measurement of any one document.
DOCUMENT_BLOCK_TOKENS = 1_600

# The confidence of a label replayed for a document no committed run scored: above
# the default review threshold (0.8), so those fields are routed like a model's
# confident reads and the review queue is not flooded.
LABEL_CONFIDENCE = 0.9
# The confidence of the fixed answer for bytes no corpus document owns. Under
# the default review threshold (0.8), so every field of it goes to review.
GENERIC_CONFIDENCE = 0.5

# What an unrecognized document "says". A real, valid record, so the repair
# path, reconciliation and the review queue see a document that is merely
# doubtful, not one that is broken.
GENERIC_FIELDS: dict[str, Any] = {
    "vendor": "Scripted Vendor",
    "document_date": "2026-01-01",
    "currency": "USD",
    "line_items": [{"description": "Scripted item", "quantity": 1, "unit_price": 10.0, "total": 10.0}],
    "tax_lines": [],
    "subtotal": 10.0,
    "tax": 0.0,
    "total": 10.0,
}
GENERIC_PAGE_TEXT = "Scripted Vendor\nScripted item 1 x 10.00\nTotal 10.00 USD"

# The explanation of every verdict the scripted judge returns.
JUDGE_EXPLANATION = "Scripted verdict: the scripted judge accepts every answer."

# The parameters each kind of request carries, and the ones it may also carry.
# The real SDK raises TypeError for a parameter it does not know and the API
# answers 400 for one a model rejects, so a request with anything else fails here
# instead of being quietly ignored.
_REQUIRED: dict[RequestKind, frozenset[str]] = {
    "extraction": frozenset({"model", "max_tokens", "system", "tools", "tool_choice", "messages"}),
    "transcription": frozenset({"model", "max_tokens", "system", "messages"}),
    "agent": frozenset({"model", "max_tokens", "system", "tools", "messages"}),
    "judge": frozenset({"model", "max_tokens", "system", "messages", "output_config"}),
}
_OPTIONAL: dict[RequestKind, frozenset[str]] = {
    "extraction": frozenset(),
    "transcription": frozenset(),
    "agent": frozenset({"cache_control"}),
    "judge": frozenset(),
}
_BETA_KINDS: frozenset[RequestKind] = frozenset({"agent", "judge"})

_ENDPOINT_URL = "https://scripted.invalid/v1/messages"
_AMOUNT = re.compile(r"[$€£]\s?(\d+(?:\.\d+)?)|\b(\d+\.\d{2})\b")
_AGGREGATE = re.compile(r"\b(?:total|spent|spend|spending|how much)\b", re.IGNORECASE)
_DOCUMENT_SOURCE = re.compile(
    re.escape(SOURCE_SCHEME) + r"/(?P<document_id>[0-9a-f-]{36})(?:/pages/(?P<page>\d+))?"
)


def estimate_tokens(text: str) -> int:
    """The scripted token count of some text: four characters a token, rounded
    up."""
    return ceil(len(text) / 4)


def bad_request(message: str) -> anthropic.BadRequestError:
    """The 400 the real API would send, as the SDK would raise it."""
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    request = httpx2.Request("POST", _ENDPOINT_URL)
    response = httpx2.Response(400, request=request, json=body)
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


# --- recognizing a document ---------------------------------------------------


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The PNG chunks that decide the pixels; the rest (tEXt, iTXt, tIME, ...) are
# metadata a load generator may vary freely.
_PNG_PIXEL_CHUNKS = frozenset({b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"})
_JPEG_COMMENT = 0xFE


def _png_pixels(raw: bytes) -> bytes | None:
    """The type and data of each pixel chunk of a PNG, or None when the bytes
    are not a well-formed PNG."""
    if not raw.startswith(_PNG_SIGNATURE):
        return None
    pixels = bytearray()
    position = len(_PNG_SIGNATURE)
    while position + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[position : position + 4])
        kind = raw[position + 4 : position + 8]
        end = position + 8 + length + 4  # the data, then its CRC
        if end > len(raw):
            return None
        if kind in _PNG_PIXEL_CHUNKS:
            pixels += kind + raw[position + 8 : position + 8 + length]
        if kind == b"IEND":
            return bytes(pixels)
        position = end
    return None


def _jpeg_without_comments(raw: bytes) -> bytes | None:
    """A JPEG with its COM segments cut out, or None when the bytes are not a
    well-formed JPEG header and scan."""
    if not raw.startswith(b"\xff\xd8"):
        return None
    kept = bytearray(raw[:2])
    position = 2
    while position + 4 <= len(raw):
        if raw[position] != 0xFF:
            return None
        marker = raw[position + 1]
        if marker == 0xDA:  # start of scan: the rest is entropy-coded data
            return bytes(kept + raw[position:])
        (length,) = struct.unpack(">H", raw[position + 2 : position + 4])
        end = position + 2 + length
        if end > len(raw):
            return None
        if marker != _JPEG_COMMENT:
            kept += raw[position:end]
        position = end
    return None


def content_digest(raw: bytes, media_type: str) -> str:
    """The SHA-256 of a document's pixel content: a PNG's pixel chunks, a JPEG
    without its comments, else every byte. A load generator can add a metadata
    chunk (PNG tEXt, JPEG COM) to make each upload byte-unique, and the scripted
    model still recognizes the picture. Bytes that do not parse as their media
    type fall back to every byte, so a corrupt file is simply not recognized."""
    if media_type == "image/png":
        content, scheme = _png_pixels(raw), "png"
    elif media_type == "image/jpeg":
        content, scheme = _jpeg_without_comments(raw), "jpeg"
    else:
        content, scheme = None, "raw"
    if content is None:
        content, scheme = raw, "raw"
    return hashlib.sha256(scheme.encode() + b"\0" + content).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    """A deterministic id: the same request always gets the same one."""
    return prefix + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


# --- what the stand-in knows --------------------------------------------------


@dataclass(frozen=True)
class CorpusDocument:
    """One document of the committed eval corpus."""

    doc_id: str
    case: EvalCase
    gold_text: str | None


class Numbers(NamedTuple):
    """What one call costs under the replay profile: milliseconds and tokens."""

    latency_ms: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _numbers(entry: dict[str, Any]) -> Numbers:
    return Numbers(
        entry["latency_ms"],
        entry["input_tokens"],
        entry["output_tokens"],
        entry.get("cache_creation_input_tokens", 0),
        entry.get("cache_read_input_tokens", 0),
    )


def _median_numbers(artifact: dict[str, Any]) -> Numbers:
    """A committed run's median latency and mean tokens: what a document the run
    did not score takes under the replay profile."""
    scored = artifact["n_scored"]
    return Numbers(
        artifact["latency_p50_ms"],
        round(artifact["total_input_tokens"] / scored),
        round(artifact["total_output_tokens"] / scored),
    )


class ScriptedCorpus:
    """The committed eval corpus and runs the scripted model answers from: which
    document a set of pixels is, what a committed run said about it, and what
    that call took. Built once per process (default_corpus); the work is reading
    the labels, hashing the images, and loading two artifacts."""

    def __init__(
        self,
        cases: list[EvalCase],
        gold_text: dict[str, str],
        primary: dict[str, Any],
        transcription_proxy: dict[str, Any],
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        self._documents: dict[str, CorpusDocument] = {}
        for case in cases:
            digest = content_digest(case.image_path.read_bytes(), case.mime_type)
            self._documents[digest] = CorpusDocument(case.doc_id, case, gold_text.get(case.doc_id))
        self._primary = primary
        self._results_dir = results_dir
        self._proxy_entries = {
            e["doc_id"]: e for e in transcription_proxy["per_doc"] if e["status"] == "scored"
        }
        self._proxy_median = _median_numbers(transcription_proxy)
        self._primary_median = _median_numbers(primary)
        self._replays: dict[tuple[str, str], Replay | None] = {}

    @classmethod
    def load(cls) -> "ScriptedCorpus":
        cases = load_cases()
        if not cases:
            raise FileNotFoundError(f"no eval labels under {LABELS_DIR}")
        gold = {doc.doc_id: doc.text for doc in load_gold_corpus(cases)}
        return cls(
            cases,
            gold,
            json.loads(REPLAY_ARTIFACT.read_text(encoding="utf-8")),
            json.loads(TRANSCRIPTION_PROXY_ARTIFACT.read_text(encoding="utf-8")),
        )

    def recognize(self, raw: bytes, media_type: str) -> CorpusDocument | None:
        return self._documents.get(content_digest(raw, media_type))

    def replay(self, document: CorpusDocument, schema: ExtractionSchema) -> Replay | None:
        """What the committed Sonnet run returned for this document under this
        schema, or None (the label stands in)."""
        key = (document.doc_id, schema.prompt_version)
        if key not in self._replays:
            self._replays[key] = scored_replay(
                document.doc_id, schema, self._primary, self._results_dir
            )
        return self._replays[key]

    def extraction_numbers(self, replay: Replay | None) -> Numbers:
        """The replay profile's numbers for an extraction call: the committed
        entry's, else the primary run's median."""
        return _numbers(replay.entry) if replay and replay.entry else self._primary_median

    def transcription_numbers(self, document: CorpusDocument | None) -> Numbers:
        """The replay profile's numbers for a transcription call. No committed
        run transcribed anything, so the Haiku run's extraction of the same
        document stands in: a vision model reading the same picture. A proxy, and
        labeled as one wherever the numbers are published."""
        entry = self._proxy_entries.get(document.doc_id) if document else None
        return _numbers(entry) if entry else self._proxy_median


@functools.cache
def default_corpus() -> ScriptedCorpus:
    return ScriptedCorpus.load()


def require_corpus() -> ScriptedCorpus:
    """Load the default corpus now, for a scripted process's startup: the
    production image holds app/ and no evals/ (the load stack mounts it), so a
    missing corpus must stop the API and the worker with this message, not fail
    the first job."""
    try:
        return default_corpus()
    except OSError as exc:
        raise RuntimeError(
            f"MODEL_PROVIDER=scripted needs the committed eval corpus ({EVALS_DIR}: labels, "
            f"images, gold text and results), which could not be read: {exc}"
        ) from exc


# --- the replies --------------------------------------------------------------


@dataclass
class _Reply:
    """A reply before it becomes an SDK message: its content blocks as the API's
    JSON, why it stopped, and what it cost in tokens."""

    content: list[dict[str, Any]]
    stop_reason: Literal["end_turn", "tool_use"]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _text_block(text: str, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"type": "text", "text": text, "citations": citations}


def _tool_use_block(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def _output_tokens(content: list[dict[str, Any]]) -> int:
    """The scripted output tokens of a reply: its text and its tool inputs."""
    text = "".join(
        block["text"] if block["type"] == "text" else json.dumps(block["input"], sort_keys=True)
        for block in content
    )
    return max(1, estimate_tokens(text))


def _field(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _as_dict(block: Any) -> Any:
    """A content block as plain JSON, whether it is a dict or an SDK object."""
    return block.model_dump() if hasattr(block, "model_dump") else block


def _text_of(value: Any) -> str:
    """Every string a request value carries that the model would read: the text
    of text blocks, the lines and titles of search results, the input of a
    tool_use block. Documents count separately (DOCUMENT_BLOCK_TOKENS)."""
    value = _as_dict(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text_of(item) for item in value)
    if not isinstance(value, dict):
        return ""
    match value.get("type"):
        case "text":
            return str(value.get("text") or "")
        case "search_result":
            return f"{value.get('title', '')}\n{_text_of(value.get('content'))}"
        case "tool_use":
            return json.dumps(value.get("input"), sort_keys=True)
        case "tool_result":
            return _text_of(value.get("content"))
        case _:
            return ""


def _document_blocks(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        content = _as_dict(message["content"])
        if isinstance(content, list):
            count += sum(1 for block in content if _field(block, "type") in ("image", "document"))
    return count


def request_input_tokens(request: dict[str, Any]) -> int:
    """The scripted input tokens of a request: its system prompt, tool
    definitions and messages through estimate_tokens, and each document at
    DOCUMENT_BLOCK_TOKENS."""
    messages = request["messages"]
    text = "\n".join(
        [
            _text_of(request.get("system")),
            json.dumps(request.get("tools") or [], sort_keys=True),
            *(_text_of(message["content"]) for message in messages),
        ]
    )
    return estimate_tokens(text) + DOCUMENT_BLOCK_TOKENS * _document_blocks(messages)


# --- the client ---------------------------------------------------------------


class _Endpoint:
    """`messages` or `beta.messages` of the client: the one method the app
    calls."""

    def __init__(self, client: "ScriptedModelClient", *, beta: bool) -> None:
        self._client = client
        self._beta = beta

    async def create(self, **request: Any) -> Message | BetaMessage:
        return await self._client.respond(request, beta=self._beta)


class _Beta:
    def __init__(self, client: "ScriptedModelClient") -> None:
        self.messages = _Endpoint(client, beta=True)


class ScriptedModelClient:
    """The drop-in for anthropic.AsyncAnthropic described in the module
    docstring. `sleep` is awaited with each simulated latency, in seconds, and
    is injectable so a test can assert the exact delays; `corpus` defaults to the
    committed one."""

    def __init__(
        self,
        settings: Settings,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        corpus: ScriptedCorpus | None = None,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._corpus = corpus
        self.messages = _Endpoint(self, beta=False)
        self.beta = _Beta(self)

    @property
    def corpus(self) -> ScriptedCorpus:
        if self._corpus is None:
            self._corpus = default_corpus()
        return self._corpus

    async def _wait(self, milliseconds: float) -> None:
        """Spend the simulated time of one call: nothing under the zero profile,
        else the milliseconds scaled by SCRIPTED_MODEL_LATENCY_SCALE."""
        if self._settings.scripted_model_profile != "replay":
            return
        seconds = milliseconds * self._settings.scripted_model_latency_scale / 1000
        if seconds > 0:
            await self._sleep(seconds)

    async def respond(self, request: dict[str, Any], *, beta: bool) -> Message | BetaMessage:
        """Answer one request, or raise the BadRequestError the API would."""
        try:
            kind = self._kind(request, beta=beta)
            self._check_common(kind, request)
            handler = {
                "extraction": self._extraction,
                "transcription": self._transcription,
                "agent": self._agent,
                "judge": self._judge,
            }[kind]
            reply = await handler(request)
        except InvalidRequestError as exc:
            raise bad_request(str(exc)) from exc
        payload = {
            "id": _stable_id("msg_", kind, json.dumps(reply.content, sort_keys=True)),
            "type": "message",
            "role": "assistant",
            "model": SCRIPTED_MODEL,
            "content": reply.content,
            "stop_reason": reply.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cache_creation_input_tokens": reply.cache_creation_input_tokens,
                "cache_read_input_tokens": reply.cache_read_input_tokens,
            },
        }
        return BetaMessage.model_validate(payload) if beta else Message.model_validate(payload)

    # --- telling the kinds apart and refusing the malformed ---------------------

    @staticmethod
    def _kind(request: dict[str, Any], *, beta: bool) -> RequestKind:
        tools, choice = request.get("tools"), request.get("tool_choice")
        kind: RequestKind
        if choice is not None:
            if not (
                isinstance(tools, list)
                and len(tools) == 1
                and isinstance(tools[0], dict)
                and tools[0].get("name") == "record_extraction"
                and choice == {"type": "tool", "name": "record_extraction"}
            ):
                raise InvalidRequestError("tool_choice must force the record_extraction tool")
            kind = "extraction"
        elif tools is not None:
            if tools != TOOL_DEFINITIONS:
                raise InvalidRequestError("tools match neither the extraction nor the agent request")
            kind = "agent"
        elif "output_config" in request:
            kind = "judge"
        elif request.get("system") == TRANSCRIBE_PROMPT_TEXT:
            kind = "transcription"
        else:
            raise InvalidRequestError("the request matches no kind the scripted model answers")
        if (kind in _BETA_KINDS) != beta:
            raise InvalidRequestError(
                f"a {kind} request goes to {'beta.messages' if kind in _BETA_KINDS else 'messages'}"
                ".create"
            )
        return kind

    @staticmethod
    def _check_common(kind: RequestKind, request: dict[str, Any]) -> None:
        required = _REQUIRED[kind]
        allowed = required | _OPTIONAL[kind]
        if request.get("stream"):
            raise InvalidRequestError("the scripted model does not stream; the app never asks it to")
        if missing := sorted(required - request.keys()):
            raise InvalidRequestError(f"missing required parameters: {', '.join(missing)}")
        if unexpected := sorted(request.keys() - allowed):
            raise InvalidRequestError(f"unexpected parameters: {', '.join(unexpected)}")
        if request["model"] != SCRIPTED_MODEL:
            raise InvalidRequestError(f"model: unknown model {request['model']!r}")
        max_tokens = request["max_tokens"]
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise InvalidRequestError("max_tokens must be a positive integer")
        messages = request["messages"]
        if not isinstance(messages, list) or not messages:
            raise InvalidRequestError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                raise InvalidRequestError("each message needs a role of user or assistant")
            if "content" not in message:
                raise InvalidRequestError("each message needs content")

    # --- extraction -------------------------------------------------------------

    @staticmethod
    def _schema_of(request: dict[str, Any]) -> ExtractionSchema:
        """The extraction schema a request asks for: the one whose tool and
        prompt it carries. The tool alone cannot tell v2 from v3, which differ
        in their prompt."""
        system = request.get("system")
        prompt = (
            system
            if isinstance(system, str)
            else "".join(b.get("text", "") for b in system or [] if isinstance(b, dict))
        )
        tool = request["tools"][0]
        for schema in SCHEMAS.values():
            if prompt == schema.prompt_text and tool == record_extraction_tool(schema):
                return schema
        raise InvalidRequestError(
            "the record_extraction tool and system prompt match no known extraction schema"
        )

    @staticmethod
    def _document_of(messages: list[dict[str, Any]]) -> tuple[bytes, str]:
        """The bytes and media type of the one document block of the first user
        message, checked the way the API checks a base64 source."""
        content = messages[0]["content"]
        if messages[0]["role"] != "user" or not isinstance(content, list):
            raise InvalidRequestError("the first message must be a user message of blocks")
        blocks = [b for b in content if _field(b, "type") in ("image", "document")]
        if len(blocks) != 1:
            raise InvalidRequestError("an extraction request holds exactly one image or document")
        block = blocks[0]
        source = block.get("source") or {}
        media_type = source.get("media_type")
        allowed = {"image": IMAGE_MIME_TYPES, "document": {PDF_MIME_TYPE}}[block["type"]]
        if source.get("type") != "base64" or media_type not in allowed:
            raise InvalidRequestError(f"unsupported {block['type']} source")
        try:
            return base64.b64decode(source.get("data", ""), validate=True), media_type
        except (binascii.Error, ValueError, TypeError):
            raise InvalidRequestError(f"the {block['type']} data is not valid base64") from None

    async def _extraction(self, request: dict[str, Any]) -> _Reply:
        schema = self._schema_of(request)
        messages = request["messages"]
        # One call, or the schema-repair reprompt: the same conversation plus the
        # model's tool_use and a tool_result saying what to fix.
        if len(messages) not in (1, 3):
            raise InvalidRequestError("an extraction request has one message, or three when repairing")
        raw, media_type = self._document_of(messages)
        document = self.corpus.recognize(raw, media_type)
        replay = self.corpus.replay(document, schema) if document else None

        if replay and replay.entry:
            tool_input, _ = _replayed_tool_input(replay.entry, document.case, schema)
        else:
            fields = document.case.fields if document else GENERIC_FIELDS
            confidence = LABEL_CONFIDENCE if document else GENERIC_CONFIDENCE
            evidence = (document.case.evidence if document else {}) if schema.evidence else None
            tool_input = label_tool_input(
                {name: fields.get(name) for name in schema.field_names}, confidence, evidence
            )

        digest = hashlib.sha256(raw).hexdigest()
        tool_id = _stable_id("toolu_", digest, str(len(messages)))
        content = [_tool_use_block(tool_id, "record_extraction", tool_input)]
        return await self._reply(request, content, "tool_use", self.corpus.extraction_numbers(replay))

    # --- transcription ----------------------------------------------------------

    async def _transcription(self, request: dict[str, Any]) -> _Reply:
        messages = request["messages"]
        if len(messages) != 1:
            raise InvalidRequestError("a transcription request has one message")
        raw, media_type = self._document_of(messages)
        document = self.corpus.recognize(raw, media_type)
        text = (document.gold_text if document else None) or GENERIC_PAGE_TEXT
        # The line that makes every upload's page text, and so its chunks and
        # vectors, its own, however many times the same picture is uploaded.
        text = f"{text.rstrip()}\nRef {hashlib.sha256(raw).hexdigest()[:12]}"
        return await self._reply(
            request, [_text_block(text)], "end_turn", self.corpus.transcription_numbers(document)
        )

    # --- the agent --------------------------------------------------------------

    async def _agent(self, request: dict[str, Any]) -> _Reply:
        validate_request(request)
        messages = request["messages"]
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user" and isinstance(m["content"], str)),
            None,
        )
        if question is None:
            raise InvalidRequestError("an agent request needs the user's question as a text message")
        rounds = [
            m["content"]
            for m in messages
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(_field(b, "type") == "tool_result" for b in m["content"])
        ]
        results = _search_results(messages)
        documents = [(index, block) for index, block in results if block["source"] != SUMMARY_SOURCE]

        top = _DOCUMENT_SOURCE.match(documents[0][1]["source"]) if documents else None
        step = len(rounds)
        stop_reason: Literal["end_turn", "tool_use"] = "tool_use"
        if step == 0:
            name, tool_input = _first_tool_call(question)
            content = [_tool_use_block(_stable_id("toolu_", question, "0"), name, tool_input)]
        elif step == 1 and top:
            tool_input = {"document_id": top["document_id"], "page_number": int(top["page"] or 1)}
            content = [_tool_use_block(_stable_id("toolu_", question, "1"), "get_page", tool_input)]
        else:
            content = _answer_blocks(documents)
            stop_reason = "end_turn"
        await self._wait(self._settings.scripted_agent_step_ms)
        return _Reply(content, stop_reason, request_input_tokens(request), _output_tokens(content))

    # --- the judge --------------------------------------------------------------

    async def _judge(self, request: dict[str, Any]) -> _Reply:
        output = request["output_config"]
        schema = output.get("format", {}).get("schema") if isinstance(output, dict) else None
        if (
            not isinstance(output, dict)
            or set(output) != {"format"}
            or output["format"].get("type") != "json_schema"
            or not isinstance(schema, dict)
            or not isinstance(schema.get("properties"), dict)
        ):
            raise InvalidRequestError("output_config.format must be a json_schema with properties")
        messages = request["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise InvalidRequestError("a judge request is one user message of text")
        verdict = {name: _verdict_value(name, spec) for name, spec in schema["properties"].items()}
        content = [_text_block(json.dumps(verdict))]
        await self._wait(self._settings.scripted_judge_ms)
        return _Reply(content, "end_turn", request_input_tokens(request), _output_tokens(content))

    # --- shared -----------------------------------------------------------------

    async def _reply(
        self,
        request: dict[str, Any],
        content: list[dict[str, Any]],
        stop_reason: Literal["end_turn", "tool_use"],
        replay: Numbers,
    ) -> _Reply:
        """A reply for an extraction or a transcription: the replay profile's
        recorded tokens and latency for the document, else the formula's tokens
        and no wait."""
        if self._settings.scripted_model_profile == "replay":
            await self._wait(replay.latency_ms)
            return _Reply(
                content,
                stop_reason,
                replay.input_tokens,
                replay.output_tokens,
                replay.cache_creation_input_tokens,
                replay.cache_read_input_tokens,
            )
        return _Reply(content, stop_reason, request_input_tokens(request), _output_tokens(content))


def _search_results(messages: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Every search_result block of the conversation with its index among them,
    in order across messages and tool results: the `search_result_index` a
    citation of it carries."""
    found: list[tuple[int, dict[str, Any]]] = []
    for message in messages:
        if message["role"] != "user" or not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if _field(block, "type") != "tool_result":
                continue
            content = _field(block, "content")
            for item in content if isinstance(content, list) else []:
                if _field(item, "type") == "search_result":
                    found.append((len(found), item))
    return found


def _first_tool_call(question: str) -> tuple[str, dict[str, Any]]:
    """The agent's first move: query_extractions for a question about an amount
    or a total (filtered to the amount when it names one), else
    search_documents with the question itself."""
    amount = _AMOUNT.search(question)
    if amount or _AGGREGATE.search(question):
        if amount:
            value = float(amount[1] or amount[2])
            return "query_extractions", {"min_total": value, "max_total": value}
        return "query_extractions", {}
    return "search_documents", {"query": question}


def _cited_line(block: dict[str, Any]) -> int:
    """Which line of a search result the scripted answer cites: a record's total
    when it has one, else the first line."""
    lines = [line["text"] for line in block["content"]]
    return next((i for i, text in enumerate(lines) if text.startswith("total:")), 0)


def _answer_blocks(documents: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """The final answer: a sentence for up to two of the search results the
    agent was given (the top hit, and the page it then opened), each citing its
    result by the real source and block index."""
    if not documents:
        return [_text_block("I could not find a document that answers that question.")]
    chosen = [documents[0]]
    if documents[-1][1]["source"] != documents[0][1]["source"]:
        chosen.append(documents[-1])
    blocks = [_text_block("According to your documents: ")]
    for number, (index, result) in enumerate(chosen):
        line = _cited_line(result)
        text = result["content"][line]["text"]
        citation = {
            "type": "search_result_location",
            "source": result["source"],
            "title": result["title"],
            "cited_text": text,
            "search_result_index": index,
            "start_block_index": line,
            "end_block_index": line + 1,
        }
        blocks.append(_text_block(f"{text}.", [citation]))
        if number < len(chosen) - 1:
            blocks.append(_text_block(" Also: "))
    return blocks


def _verdict_value(name: str, spec: dict[str, Any]) -> Any:
    """A value valid for one property of the judge's JSON schema, always the
    favorable one: true, the highest enum member, an empty list."""
    if "enum" in spec:
        members = spec["enum"]
        return max(members) if all(isinstance(m, int | float) for m in members) else members[0]
    kind = spec.get("type")
    kind = next((k for k in kind if k != "null"), None) if isinstance(kind, list) else kind
    match kind:
        case "array":
            return []
        case "boolean":
            return True
        case "integer":
            return spec.get("minimum", 1)
        case "number":
            return spec.get("minimum", 0)
        case "string":
            return JUDGE_EXPLANATION
    raise InvalidRequestError(f"the scripted judge cannot fill {name!r}: unsupported type {kind!r}")
