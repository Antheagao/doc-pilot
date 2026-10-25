"""Tests for the workload: the nonce, the SSE parser, the answer rule, the pools
and the scenario description (run: python -m pytest ../scripts -q).

locustfile.py itself needs Locust, which only the loadgen image has, so these
test workload.py, which holds everything the scenarios decide; no Docker needed.
"""

from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path

import load
import pytest
import workload
from PIL import Image

from app.scripted_model import content_digest, default_corpus

EVALS = Path(__file__).resolve().parents[2] / "evals"


def png_chunks(raw: bytes) -> list[tuple[bytes, bytes, int]]:
    """(type, data, stored crc) of every chunk of a PNG."""
    chunks = []
    position = 8
    while position < len(raw):
        (length,) = struct.unpack(">I", raw[position : position + 4])
        kind = raw[position + 4 : position + 8]
        data = raw[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", raw[position + 8 + length : position + 12 + length])
        chunks.append((kind, data, crc))
        position += 12 + length
    return chunks


@pytest.fixture(scope="module")
def corpus() -> list[workload.CorpusFile]:
    return workload.corpus_documents(EVALS)


# --- the nonce keeps the pixels --------------------------------------------------


def test_the_corpus_is_the_25_documents_with_gold_text(corpus: list[workload.CorpusFile]) -> None:
    assert [doc.name[:3] for doc in corpus] == [f"{n:03d}" for n in range(1, 26)]
    assert {doc.media_type for doc in corpus} == {"image/png", "image/jpeg"}
    assert [doc.name for doc in corpus if doc.media_type == "image/jpeg"] == ["014-noisy-home-goods.jpg"]


def test_a_png_nonce_is_a_valid_text_chunk_before_iend(corpus: list[workload.CorpusFile]) -> None:
    doc = corpus[0]
    marked = workload.inject_png_nonce(doc.data, "abc123")
    chunks = png_chunks(marked)
    assert chunks[-1][0] == b"IEND"
    kind, data, crc = chunks[-2]
    assert kind == b"tEXt"
    assert data == b"load-nonce\0abc123"
    assert crc == zlib.crc32(kind + data)
    Image.open(io.BytesIO(marked)).load()  # a decoder accepts it, pixels and all


@pytest.mark.parametrize("number", range(25))
def test_a_nonce_never_changes_the_digest_the_scripted_model_reads(
    corpus: list[workload.CorpusFile], number: int
) -> None:
    """The reason for the nonce: every upload is byte-unique and still 'read'
    as its corpus document. Proved with the scripted model's own digest, and by
    its corpus recognizing the marked bytes."""
    doc = corpus[number]
    first = workload.with_nonce(doc.data, doc.media_type, "nonce-one")
    second = workload.with_nonce(doc.data, doc.media_type, "nonce-two")
    assert first != doc.data and second != doc.data and first != second
    expected = content_digest(doc.data, doc.media_type)
    assert content_digest(first, doc.media_type) == expected
    assert content_digest(second, doc.media_type) == expected
    recognized = default_corpus().recognize(first, doc.media_type)
    assert recognized is not None and recognized.doc_id == Path(doc.name).stem


def test_a_jpeg_nonce_is_a_comment_segment_and_still_a_jpeg(corpus: list[workload.CorpusFile]) -> None:
    doc = next(d for d in corpus if d.media_type == "image/jpeg")
    marked = workload.inject_jpeg_nonce(doc.data, "abc123")
    assert marked[:2] == b"\xff\xd8"
    assert marked[2:4] == b"\xff\xfe"
    (length,) = struct.unpack(">H", marked[4:6])
    assert marked[6 : 4 + length] == b"abc123"
    Image.open(io.BytesIO(marked)).load()


def test_a_changed_pixel_does_change_the_digest(corpus: list[workload.CorpusFile]) -> None:
    """The control for the proofs above: the digest is not blind."""
    doc = corpus[0]
    chunks = png_chunks(doc.data)
    idat = next(data for kind, data, _ in chunks if kind == b"IDAT")
    other = doc.data.replace(idat, idat[:-1] + bytes([idat[-1] ^ 1]), 1)
    assert content_digest(other, "image/png") != content_digest(doc.data, "image/png")


def test_the_wrong_kind_of_file_is_refused() -> None:
    with pytest.raises(ValueError, match="not a PNG"):
        workload.inject_png_nonce(b"JFIF", "x")
    with pytest.raises(ValueError, match="not a JPEG"):
        workload.inject_jpeg_nonce(b"\x89PNG", "x")
    with pytest.raises(ValueError, match="application/pdf"):
        workload.with_nonce(b"%PDF", "application/pdf", "x")
    with pytest.raises(ValueError, match="IEND"):
        workload.inject_png_nonce(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 0) + b"IHDR" + b"\0" * 4, "x")


# --- server-sent events ------------------------------------------------------------

STREAM = (
    b"event: model_call\ndata: {\"step\": 1}\n\n"
    b"event: tool_call\ndata: {\"name\": \"search_documents\"}\n\n"
    b"event: answer\ndata: {\"type\": \"answer\"}\n\n"
)
EXPECTED = [
    workload.SseFrame("model_call", '{"step": 1}'),
    workload.SseFrame("tool_call", '{"name": "search_documents"}'),
    workload.SseFrame("answer", '{"type": "answer"}'),
]


def feed_all(chunks: list[bytes]) -> list[workload.SseFrame]:
    parser = workload.SseParser()
    return [frame for chunk in chunks for frame in parser.feed(chunk)]


def test_joined_frames_in_one_chunk_all_arrive() -> None:
    assert feed_all([STREAM]) == EXPECTED


def test_a_frame_is_whole_only_when_its_blank_line_arrives() -> None:
    parser = workload.SseParser()
    assert parser.feed(b"event: model_call\ndata: {\"step\": 1}\n") == []
    assert parser.feed(b"\n") == [EXPECTED[0]]


def test_every_split_of_the_stream_gives_the_same_frames() -> None:
    for cut in range(1, len(STREAM)):
        assert feed_all([STREAM[:cut], STREAM[cut:]]) == EXPECTED, cut
    assert feed_all([bytes([b]) for b in STREAM]) == EXPECTED


def test_crlf_cut_between_chunks_is_one_line_end() -> None:
    frames = feed_all([b"event: a\r", b"\ndata: x\r\n\r", b"\nevent: b\r\ndata: y\r\n\r\n"])
    assert frames == [workload.SseFrame("a", "x"), workload.SseFrame("b", "y")]


def test_a_multibyte_character_cut_between_chunks_survives() -> None:
    raw = "event: answer\ndata: café €\n\n".encode()
    for cut in range(1, len(raw)):
        assert feed_all([raw[:cut], raw[cut:]]) == [workload.SseFrame("answer", "café €")], cut


def test_comments_other_fields_and_multiline_data() -> None:
    frames = feed_all([b": keep-alive\n\nid: 7\nretry: 100\nevent: e\ndata: one\ndata: two\n\ndata: bare\n\n"])
    assert frames == [workload.SseFrame("e", "one\ntwo"), workload.SseFrame("message", "bare")]


def test_an_unfinished_frame_is_never_returned() -> None:
    assert feed_all([b"event: answer\ndata: {}"]) == []


def test_the_apps_own_stream_format_parses() -> None:
    """The route writes `event: <type>\\ndata: <json>\\n\\n` (routers/ask.py)."""
    event = {"type": "answer", "run": {"status": "answered", "citations": [{"number": 1}]}}
    raw = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
    [frame] = feed_all([raw])
    assert frame.event == "answer"
    assert workload.stream_frame_problem(frame) is None


# --- what counts as an answer -----------------------------------------------------


def test_an_answer_needs_the_answered_status_and_a_citation() -> None:
    assert workload.answer_problem({"status": "answered", "citations": [{"number": 1}]}) is None
    assert workload.answer_problem({"status": "answered", "citations": []}) == "no_citation"
    assert workload.answer_problem({"status": "answered"}) == "no_citation"
    assert workload.answer_problem({"status": "refused", "citations": [{"number": 1}]}) == "status_refused"
    assert workload.answer_problem("not a run") == "bad_answer"


def test_a_stream_ends_badly_on_an_error_event_or_an_uncited_answer() -> None:
    cited = json.dumps({"run": {"status": "answered", "citations": [{"number": 1}]}})
    uncited = json.dumps({"run": {"status": "answered", "citations": []}})
    assert workload.stream_frame_problem(workload.SseFrame("answer", cited)) is None
    assert workload.stream_frame_problem(workload.SseFrame("answer", uncited)) == "no_citation"
    assert workload.stream_frame_problem(workload.SseFrame("answer", "{")) == "bad_answer"
    assert workload.stream_frame_problem(workload.SseFrame("answer", "[]")) == "bad_answer"
    assert workload.stream_frame_problem(workload.SseFrame("error", "{}")) == "stream_error"
    assert workload.stream_frame_problem(workload.SseFrame("model_call", "{}")) is None


# --- the pools ----------------------------------------------------------------------


def test_the_search_queries_are_the_retrieval_evals() -> None:
    queries = workload.search_queries(EVALS)
    assert len(queries) == 109
    assert "Cedar Bird Feeder" in queries


def test_the_questions_are_the_agent_evals_less_the_ones_built_to_be_refused() -> None:
    questions = workload.ask_questions(EVALS)
    assert len(questions) == 16
    assert "What was the total on my receipt from Cascade Coffee Roasters?" in questions
    assert not any("Target" in q or "car wash" in q for q in questions)
    fewer = workload.ask_questions(EVALS, ["reverse_lookup"])
    assert len(fewer) == 14
    assert not any("425.58" in q for q in fewer)


def test_exclusions_come_from_a_comma_separated_setting() -> None:
    assert workload.parse_exclude_types("reverse_lookup,safety") == ("reverse_lookup", "safety")
    assert workload.parse_exclude_types(None) == ()
    assert workload.parse_exclude_types("") == ()


def test_the_default_users_are_both_kinds_for_every_tenant() -> None:
    weights = workload.default_user_weights()
    assert set(weights) == {f"{t}:{a}" for t in workload.TENANT_SLUGS for a in ("key", "cookie")}
    assert weights["load-large:key"] == 4 and weights["load-large:cookie"] == 1


def test_the_tenants_are_the_ones_load_py_seeds() -> None:
    assert workload.TENANT_SLUGS == tuple(tenant.slug for tenant in load.TENANTS)


def test_user_weights_parse_and_refuse_nonsense() -> None:
    assert workload.parse_user_weights(None) == workload.default_user_weights()
    assert workload.parse_user_weights('{"load-large:key": 2, "load-small:key": 0}') == {"load-large:key": 2}
    for bad in ('{"nobody:key": 1}', '{"load-large:key": -1}', '{"load-large:key": 1.5}', "[]", "{}", '{"load-large:key": 0}'):
        with pytest.raises(ValueError):
            workload.parse_user_weights(bad)


def test_read_mix_parses_and_refuses_nonsense() -> None:
    assert workload.parse_read_mix(None) == {"search": 70, "ask": 15, "ask_stream": 10, "documents": 5}
    assert workload.parse_read_mix('{"search": 1, "ask": 0}') == {"search": 1}
    for bad in ('{"nothing": 1}', '{"search": true}', "{}", '{"search": 0}'):
        with pytest.raises(ValueError):
            workload.parse_read_mix(bad)


# --- samples ------------------------------------------------------------------------


def test_a_sample_carries_its_own_stamp_and_an_error_makes_it_not_ok() -> None:
    good = workload.make_sample("search", tenant="load-small", auth="key", start=1.5, ms=12.3456, status=200)
    assert good == {
        "name": "search",
        "tenant": "load-small",
        "auth": "key",
        "start": 1.5,
        "ms": 12.346,
        "status": 200,
        "ok": True,
        "error": None,
    }
    bad = workload.make_sample(
        "ask_stream", tenant="t", auth="cookie", start=2, ms=3, status=200, error="x" * 500, answer_ms=None
    )
    assert bad["ok"] is False and len(bad["error"]) == workload.SAMPLE_ERROR_MAX_CHARS
    assert "answer_ms" in bad


def test_the_writer_appends_one_flushed_line_per_sample(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    writer = workload.SampleWriter(path)
    writer.write({"name": "a", "ms": 1})
    assert json.loads(path.read_text().splitlines()[0]) == {"name": "a", "ms": 1}  # before close
    writer.write({"name": "b", "ms": 2})
    writer.close()
    assert [json.loads(line)["name"] for line in path.read_text().splitlines()] == ["a", "b"]


# --- the scenario -------------------------------------------------------------------


def test_the_scenario_description_is_deterministic_and_names_what_it_sends() -> None:
    first = workload.scenario_description(EVALS)
    assert first == workload.scenario_description(EVALS)
    assert first["read_mix"] == workload.DEFAULT_READ_MIX
    assert first["search_queries"]["count"] == 109
    assert first["ask_questions"]["count"] == 16
    assert len(first["corpus"]) == 25
    assert all(len(doc["sha256"]) == 64 for doc in first["corpus"])
    json.dumps(first)  # plain data


def test_anything_that_changes_the_traffic_changes_the_description() -> None:
    base = workload.scenario_description(EVALS)
    assert workload.scenario_description(EVALS, read_mix={"search": 1}) != base
    assert workload.scenario_description(EVALS, user_weights={"load-large:key": 1}) != base
    assert workload.scenario_description(EVALS, exclude_question_types=["reverse_lookup"]) != base
