"""OpenTelemetry tracing: one trace per document, model calls described with
the GenAI semantic conventions.

What a trace looks like (one uploaded receipt, end to end):

    POST /documents                         (FastAPI auto-instrumentation)
    └─ job extract                          (worker; parent = the upload)
       ├─ chat claude-sonnet-5              (gen_ai.* tokens, finish reason, cost)
       └─ job index                         (enqueued by the extract job)
          ├─ chat claude-haiku-4-5          (one per page transcribed)
          └─ embeddings fastembed:BAAI/...  (chunk count)

Jobs run minutes apart in a different process, so the parent link crosses
the queue: every job row stores the W3C `traceparent` of whatever enqueued
it (the upload request, or the extract job that queued an index job), and
the worker starts the job's span as a child of that context.

Off by default. Nothing is installed or exported unless the standard
OTEL_EXPORTER_OTLP_ENDPOINT environment variable is set, so tests, CI and a
plain `docker compose up` pay nothing. With it set (see docker-compose.yml's
`tracing` profile, which runs Jaeger), spans go out over OTLP/HTTP.

Content capture: prompts, document images, transcriptions and search
queries are never recorded on spans -- receipts carry personal data, and
the GenAI conventions make message content opt-in for the same reason.
Spans carry models, token counts, finish reasons, cost, and ids.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import Span, SpanKind

TRACER_NAME = "doc-pilot"

# GenAI semantic-convention attribute names
# (https://github.com/open-telemetry/semantic-conventions-genai). Spelled
# out here rather than imported: opentelemetry-semantic-conventions still
# ships these constants, but marks them deprecated in place since the GenAI
# conventions moved to their own repository -- the names are unchanged.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"

# doc-pilot's own attributes, namespaced so they can't collide with a
# future convention.
DOCPILOT_COST_USD = "docpilot.cost_usd"
DOCPILOT_PROMPT_VERSION = "docpilot.prompt_version"
DOCPILOT_DOCUMENT_ID = "docpilot.document.id"
DOCPILOT_ASK_RUN_ID = "docpilot.ask_run.id"
DOCPILOT_PAGE_NUMBER = "docpilot.page_number"
DOCPILOT_JOB_ID = "docpilot.job.id"
DOCPILOT_JOB_KIND = "docpilot.job.kind"
DOCPILOT_JOB_ATTEMPT = "docpilot.job.attempt"
DOCPILOT_JOB_OUTCOME = "docpilot.job.outcome"


def tracer() -> trace.Tracer:
    # Looked up per call rather than cached at import: before a provider
    # is installed this returns a proxy that starts delegating as soon as
    # one is, so modules imported before configure_tracing() still trace.
    return trace.get_tracer(TRACER_NAME)


def tracing_enabled() -> bool:
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


def configure_tracing(service_name: str) -> bool:
    """Install an OTLP-exporting TracerProvider when
    OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise leave OpenTelemetry's
    no-op default in place. Returns whether tracing is on.

    OTEL_SERVICE_NAME, if set, overrides `service_name` (the SDK's
    standard resource detection), so one image can run as several
    services.
    """
    if not tracing_enabled():
        return False

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME") or service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return True


def instrument_api(app: Any, engine: Any) -> None:
    """HTTP server spans for every request plus a span per SQL statement.
    Called only when configure_tracing() turned tracing on."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz")
    instrument_engine(engine)


def instrument_engine(engine: Any) -> None:
    """SQL spans for a process with no HTTP server (the worker)."""
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    # skip_dep_check: the instrumentor declares support for SQLAlchemy
    # < 2.1 and silently instruments nothing on 2.1.x, but the engine
    # event hooks it attaches (before/after_cursor_execute) are unchanged
    # in 2.1. tests/test_telemetry.py asserts SQL spans really appear, so
    # an actual incompatibility fails CI instead of going dark.
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine, skip_dep_check=True)


@contextmanager
def untraced() -> Iterator[None]:
    """Suppress instrumentation (SQL spans, mainly) for work that belongs
    to no trace. The worker's claim poll runs every second whether or not
    there's work; traced, each poll would be its own one-span trace and
    bury the real ones."""
    from opentelemetry.instrumentation.utils import suppress_instrumentation

    with suppress_instrumentation():
        yield


def shutdown_tracing() -> None:
    """Flush buffered spans on process exit (the batch processor would
    otherwise drop whatever it hadn't exported yet)."""
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if shutdown is not None:
        shutdown()


def current_traceparent() -> str | None:
    """The W3C traceparent of the active span, for storing on a job row so
    the worker can continue this trace. None when there is no recording
    span (tracing off), which the worker reads as "start a new trace"."""
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


def context_from_traceparent(traceparent: str | None) -> otel_context.Context:
    """The parent context a job's span should start under: the enqueuer's
    span if the job recorded one, otherwise an empty context (a new root
    span -- never whatever span happens to be active in the worker)."""
    if not traceparent:
        return otel_context.Context()
    return extract({"traceparent": traceparent}, context=otel_context.Context())


@contextmanager
def model_call_span(
    model: str,
    *,
    max_tokens: int,
    prompt_version: str,
    **attributes: Any,
) -> Iterator[Span]:
    """A GenAI `chat` client span around one messages.create() call. An
    exception raised inside is recorded on the span and marks it as an
    error (the SDK default), then re-raised unchanged."""
    with tracer().start_as_current_span(
        f"chat {model}",
        kind=SpanKind.CLIENT,
        attributes={
            GEN_AI_OPERATION_NAME: "chat",
            GEN_AI_PROVIDER_NAME: "anthropic",
            GEN_AI_REQUEST_MODEL: model,
            GEN_AI_REQUEST_MAX_TOKENS: max_tokens,
            DOCPILOT_PROMPT_VERSION: prompt_version,
            **attributes,
        },
    ) as span:
        yield span


def record_model_response(span: Span, response: Any, cost_usd: float) -> None:
    """Response-side GenAI attributes: id, model, finish reason, usage,
    and doc-pilot's own cost figure. Reads defensively -- cache token
    counts (and, on test doubles, id/model) may be absent."""
    for key, value in (
        (GEN_AI_RESPONSE_ID, getattr(response, "id", None)),
        (GEN_AI_RESPONSE_MODEL, getattr(response, "model", None)),
    ):
        if isinstance(value, str):
            span.set_attribute(key, value)
    if response.stop_reason:
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [response.stop_reason])
    usage = response.usage
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
    for key, name in (
        (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, "cache_read_input_tokens"),
        (GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, "cache_creation_input_tokens"),
    ):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            span.set_attribute(key, value)
    span.set_attribute(DOCPILOT_COST_USD, cost_usd)
