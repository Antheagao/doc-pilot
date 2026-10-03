"""Langfuse (app/langfuse_link.py): what doc-pilot sends it -- span types,
cost, sessions and scores -- checked against fakes, and end to end with
the real SDK against a local stand-in for the Langfuse API (no network)."""

import gzip
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider

from app import langfuse_link, telemetry
from app.langfuse_link import (
    COST_DETAILS,
    OBSERVATION_TYPE,
    SESSION_ID,
    USAGE_DETAILS,
    observation_attributes,
    record_generation,
    score_eval_run,
    score_feedback,
    score_judgment,
)


class FakeClient:
    def __init__(self, fail: bool = False):
        self.scores: list[dict] = []
        self.fail = fail

    def create_score(self, **kwargs):
        if self.fail:
            raise RuntimeError("ingestion queue full")
        self.scores.append(kwargs)


@pytest.fixture
def fake(monkeypatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(langfuse_link, "_client", client)
    return client


def _run(**overrides) -> SimpleNamespace:
    fields = {
        "id": uuid.UUID(int=7),
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "judge_grounded": True,
        "judge_answers_question": False,
        "judge_model": "claude-sonnet-5-5",
        "judge_prompt_version": "groundedness_v1",
        "feedback": None,
    }
    return SimpleNamespace(**{**fields, **overrides})


# --- off by default ----------------------------------------------------------------


def test_nothing_is_added_while_langfuse_is_off(monkeypatch) -> None:
    monkeypatch.setattr(langfuse_link, "_client", None)
    span = SimpleNamespace(set_attribute=lambda *a: pytest.fail("attribute set"))

    assert observation_attributes("agent", session_id="c-1") == {}
    record_generation(span, cost_usd=0.01, usage={"input": 1})
    score_feedback(_run(feedback="up"))  # no client: nothing to do, no error


def test_either_exporter_turns_tracing_on(monkeypatch) -> None:
    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert not telemetry.tracing_enabled()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    assert not telemetry.tracing_enabled()  # both keys, or nothing
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-1")
    assert telemetry.tracing_enabled() and not telemetry.otlp_enabled()


# --- span attributes -------------------------------------------------------------------


def test_spans_are_typed_for_langfuse_and_chats_are_sessions(fake) -> None:
    assert observation_attributes("tool") == {OBSERVATION_TYPE: "tool"}
    assert observation_attributes("agent", session_id="c-1") == {
        OBSERVATION_TYPE: "agent",
        SESSION_ID: "c-1",
    }


def test_a_model_call_is_a_generation_with_doc_pilots_cost(fake) -> None:
    attributes = {}
    span = SimpleNamespace(set_attribute=attributes.__setitem__)
    usage = SimpleNamespace(
        input_tokens=900, output_tokens=120, cache_read_input_tokens=300, cache_creation_input_tokens=None
    )
    response = SimpleNamespace(id="m", model="claude-opus-5-5", stop_reason="end_turn", usage=usage)

    telemetry.record_model_response(span, response, 0.0123)

    assert attributes[OBSERVATION_TYPE] == "generation"
    assert json.loads(attributes[COST_DETAILS]) == {"total": 0.0123}
    assert json.loads(attributes[USAGE_DETAILS]) == {
        "input": 900, "output": 120, "cache_read_input_tokens": 300,
    }


# --- scores ------------------------------------------------------------------------------


def test_the_graders_verdict_is_scored_on_the_answers_trace(fake) -> None:
    score_judgment(_run())

    grounded, answers = fake.scores
    assert grounded == {
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "data_type": "BOOLEAN",
        "name": "grounded",
        "value": 1.0,
        "score_id": f"{uuid.UUID(int=7)}-grounded",
        "metadata": {
            "ask_run_id": str(uuid.UUID(int=7)),
            "grader_model": "claude-sonnet-5-5",
            "grader_prompt_version": "groundedness_v1",
        },
    }
    assert (answers["name"], answers["value"]) == ("answers_question", 0.0)
    # Verdicts only: the grader's explanation can quote the receipt.
    assert all("comment" not in score for score in fake.scores)


def test_an_unreadable_verdict_or_untraced_run_scores_nothing(fake) -> None:
    score_judgment(_run(judge_grounded=None))
    score_judgment(_run(trace_id=None))

    assert fake.scores == []


def test_feedback_is_one_score_per_answer(fake) -> None:
    score_feedback(_run(feedback="up"))
    score_feedback(_run(feedback="down"))

    first, second = fake.scores
    assert (first["value"], second["value"]) == (1.0, 0.0)
    # Same id: Langfuse upserts, so the rating is replaced, not doubled.
    assert first["score_id"] == second["score_id"] == f"{uuid.UUID(int=7)}-feedback"


def test_eval_rubric_results_land_on_each_questions_trace(fake) -> None:
    result = {
        "started_at_utc": "20261003T120000Z",
        "question_set_version": "v1",
        "prompt_version": "agent_v1",
        "per_question": [
            {"id": "q1", "type": "lookup", "trace_id": "a" * 32,
             "scores": {"correct": True, "cites_relevant": False}},
            {"id": "q2", "type": "abstain", "trace_id": "b" * 32, "scores": {"correct": False}},
            {"id": "q3", "type": "lookup", "skipped": "cost_cap"},
            {"id": "q4", "type": "lookup", "trace_id": None, "scores": {"correct": True}},
        ],
    }

    assert score_eval_run(result) == 3
    assert [(s["trace_id"][0], s["name"], s["value"]) for s in fake.scores] == [
        ("a", "eval_correct", 1.0), ("a", "eval_cites_relevant", 0.0), ("b", "eval_correct", 0.0),
    ]
    assert fake.scores[0]["metadata"]["question_id"] == "q1"


def test_a_failing_score_never_fails_the_caller(monkeypatch) -> None:
    monkeypatch.setattr(langfuse_link, "_client", FakeClient(fail=True))

    score_feedback(_run(feedback="up"))  # logged, not raised


# --- end to end, real SDK --------------------------------------------------------------------


class _LangfuseStandIn(BaseHTTPRequestHandler):
    received: ClassVar[list[tuple[str, bytes]]] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        self.received.append((self.path, body))
        payload = b'{"successes": [], "errors": []}'
        self.send_response(207 if "ingestion" in self.path else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.do_POST()

    def log_message(self, *args):
        pass


@pytest.fixture
def langfuse_api(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LangfuseStandIn)
    _LangfuseStandIn.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # A fresh key per test: the SDK keeps one client per public key.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-lf-{uuid.uuid4()}")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    yield _LangfuseStandIn.received
    server.shutdown()


def test_the_sdk_exports_genai_spans_and_scores_and_drops_the_rest(langfuse_api) -> None:
    provider = TracerProvider()
    langfuse_link.attach(provider)
    try:
        tracer = provider.get_tracer("doc-pilot")
        with tracer.start_as_current_span(
            "invoke_agent doc-pilot-ask",
            attributes={"gen_ai.operation.name": "invoke_agent", **observation_attributes("agent", "c-9")},
        ) as agent:
            with tracer.start_as_current_span("chat claude-opus-5-5", attributes={"gen_ai.operation.name": "chat"}) as chat:
                record_generation(chat, cost_usd=0.02, usage={"input": 10, "output": 5})
            with tracer.start_as_current_span("SELECT docpilot", attributes={"db.system.name": "postgresql"}):
                pass
            with tracer.start_as_current_span("evaluate agent_answer", attributes=observation_attributes("evaluator")):
                pass
        trace_id = format(agent.get_span_context().trace_id, "032x")
        score_feedback(_run(feedback="up", trace_id=trace_id))
        provider.force_flush()
    finally:
        langfuse_link.shutdown()
        provider.shutdown()

    spans = {}
    for path, body in langfuse_api:
        if path.endswith("/api/public/otel/v1/traces"):
            request = ExportTraceServiceRequest()
            request.ParseFromString(body)
            for resource_spans in request.resource_spans:
                for scope_spans in resource_spans.scope_spans:
                    for span in scope_spans.spans:
                        spans[span.name] = {a.key: a.value for a in span.attributes}
    assert set(spans) == {"invoke_agent doc-pilot-ask", "chat claude-opus-5-5", "evaluate agent_answer"}
    chat = spans["chat claude-opus-5-5"]
    assert chat[OBSERVATION_TYPE].string_value == "generation"
    assert json.loads(chat[COST_DETAILS].string_value) == {"total": 0.02}
    assert spans["invoke_agent doc-pilot-ask"][SESSION_ID].string_value == "c-9"

    batches = [
        json.loads(body)["batch"] for path, body in langfuse_api if path.endswith("/api/public/ingestion")
    ]
    (score,) = [event["body"] for batch in batches for event in batch if event["type"] == "score-create"]
    assert (score["traceId"], score["name"], score["value"]) == (trace_id, "user_feedback", 1.0)
    assert score["id"] == f"{uuid.UUID(int=7)}-feedback"
