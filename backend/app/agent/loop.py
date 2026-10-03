"""The /ask agent: a manual tool-use loop over app.agent.tools.

Why a manual loop rather than the SDK's (beta) tool runner: every model
call here gets its own GenAI `chat` span with real timing, the step and
dollar budgets are checked between calls, and a tool failure becomes an
is_error tool_result the model can recover from -- all of which need a
hand on each iteration. The loop is append-only (every assistant turn is
sent back exactly as returned, thinking blocks included), which is what
adaptive-thinking models require of a conversation history.

One run is one `invoke_agent` span; under it, each model call is a `chat`
span and each tool call an `execute_tool` span, so a question's trace
reads as the agent's plan: which tools, in what order, what each cost.

The same loop answers a follow-up in a per-document chat
(app/routers/chat.py): the earlier turns go in as plain text, the tools
are scoped to the one document (ToolContext.document_ids), and a second
system block (DOCUMENT_CHAT_PROMPT) tells the model so.
"""

import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import anthropic
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.agent.tools import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    Source,
    ToolContext,
    ToolError,
)
from app.config import Settings
from app.extraction import (
    PRICING_PER_MTOK,
    PROMPTS_DIR,
    _build_client,
    _classify_api_error,
)
from app.langfuse_link import observation_attributes
from app.telemetry import (
    DOCPILOT_COST_USD,
    GEN_AI_AGENT_NAME,
    GEN_AI_CONVERSATION_ID,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    current_traceparent,
    model_call_span,
    record_model_response,
    tracer,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "doc-pilot-ask"
AGENT_PROMPT_PATH = PROMPTS_DIR / "agent_v1.md"
AGENT_PROMPT_VERSION = AGENT_PROMPT_PATH.stem
AGENT_PROMPT_TEXT = AGENT_PROMPT_PATH.read_text(encoding="utf-8")
DOCUMENT_CHAT_PROMPT_PATH = PROMPTS_DIR / "document_chat_v1.md"


@dataclass(frozen=True)
class AgentPrompt:
    """The system prompt a run is given, and the version it is stored and
    traced under."""

    version: str
    system: str | list[dict[str, Any]]


# POST /ask: questions across every document.
ASK_PROMPT = AgentPrompt(AGENT_PROMPT_VERSION, AGENT_PROMPT_TEXT)
# A chat about one document: the same instructions, plus a block saying
# the tools only see that document and earlier turns aren't evidence.
DOCUMENT_CHAT_PROMPT = AgentPrompt(
    f"{AGENT_PROMPT_VERSION}+{DOCUMENT_CHAT_PROMPT_PATH.stem}",
    [
        {"type": "text", "text": AGENT_PROMPT_TEXT},
        {"type": "text", "text": DOCUMENT_CHAT_PROMPT_PATH.read_text(encoding="utf-8")},
    ],
)

# Models that take `output_config.effort` and adaptive thinking, and the
# subset that accept server-side refusal fallbacks (fallbacks="default").
# Anything else (e.g. claude-haiku-4-5) is called without them.
_EFFORT_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"}
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Prompt-cache pricing relative to base input: 5-minute cache writes bill
# at 1.25x; reads at 0.1x, except Claude Opus 5.5 at 0.05x (claude-api
# skill, 2026-09). usage.input_tokens counts only the uncached remainder,
# so leaving these out would under-report every cached call.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = {"claude-opus-5-5": 0.05}
CACHE_READ_MULTIPLIER_DEFAULT = 0.10

Status_ = Literal["answered", "refused", "truncated", "step_limit", "budget_exceeded"]


@dataclass
class Citation:
    number: int
    document_id: str
    filename: str
    page_number: int | None
    kind: str
    cited_text: str
    # Page text span, for chunk/page sources; None for extraction records.
    char_start: int | None
    char_end: int | None
    # Record fields cited, for extraction-record sources.
    fields: list[str]


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    is_error: bool
    result_summary: str


@dataclass
class AgentResult:
    status: Status_
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCall]
    steps: int
    model: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    refusal_category: str | None = None
    trace_id: str | None = None
    # W3C traceparent of the run's invoke_agent span, so later work about
    # this answer (the eval's LLM judge) can attach to the same trace.
    traceparent: str | None = None
    # Citations whose source this run never registered -- should never
    # happen with API-generated citations; counted rather than trusted.
    unresolved_citations: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class Turn:
    """One earlier question and its answer, in a conversation."""

    question: str
    answer: str


_CITATION_MARKER = re.compile(r"\s?\[\d+\]")


def history_messages(turns: Sequence[Turn]) -> list[dict[str, Any]]:
    """Earlier turns as plain user / assistant text, oldest first. The [n]
    markers are stripped: the sources they number aren't in this request,
    so the model couldn't resolve them -- and shouldn't cite them. A turn
    with no answer (a refusal, an API error) is left out; the
    conversation still alternates."""
    messages: list[dict[str, Any]] = []
    for turn in turns:
        answer = _CITATION_MARKER.sub("", turn.answer).strip()
        if answer:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": answer})
    return messages


def call_cost_usd(requested_model: str, response: Any) -> float:
    """Cost of one call, priced at the model that actually served it (a
    refusal fallback may have), including prompt-cache reads and writes."""
    model = response.model if response.model in PRICING_PER_MTOK else requested_model
    price_in, price_out = PRICING_PER_MTOK[model]
    usage = response.usage
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    read_multiplier = CACHE_READ_MULTIPLIER.get(model, CACHE_READ_MULTIPLIER_DEFAULT)
    input_cost = price_in * (
        usage.input_tokens + cache_write * CACHE_WRITE_MULTIPLIER + cache_read * read_multiplier
    )
    return (input_cost + price_out * usage.output_tokens) / 1_000_000


def model_request_options(model: str, effort: str, refusal_fallback: bool) -> dict[str, Any]:
    """Per-model request options: effort + adaptive thinking where the
    model takes them, server-side refusal fallback where it's offered.
    Shared with the LLM judge (app/evals/judge.py)."""
    options: dict[str, Any] = {}
    if model in _EFFORT_MODELS:
        options["output_config"] = {"effort": effort}
        options["thinking"] = {"type": "adaptive"}
    if refusal_fallback and model in _FALLBACK_MODELS:
        options["betas"] = [FALLBACK_BETA]
        options["fallbacks"] = "default"
    return options


def _request_options(settings: Settings) -> dict[str, Any]:
    return model_request_options(
        settings.agent_model, settings.agent_effort, settings.agent_refusal_fallback
    )


def _resolve_citation(raw: Any, sources: dict[str, Source]) -> dict[str, Any] | None:
    source = sources.get(getattr(raw, "source", None))
    if source is None:
        return None
    blocks = source.blocks[raw.start_block_index : raw.end_block_index]
    spans = [b for b in blocks if isinstance(b, tuple)]
    return {
        "key": (source.source, raw.start_block_index, raw.end_block_index),
        "document_id": str(source.document_id),
        "filename": source.filename,
        "page_number": source.page_number,
        "kind": source.kind,
        "cited_text": raw.cited_text,
        "char_start": min(s for s, _ in spans) if spans else None,
        "char_end": max(e for _, e in spans) if spans else None,
        "fields": [b for b in blocks if isinstance(b, str)],
    }


def _assemble_answer(
    content: list[Any], sources: dict[str, Source]
) -> tuple[str, list[Citation], int]:
    """The final turn's text with [n] markers after each cited passage,
    and the numbered citations. Identical citations share a number."""
    text_parts: list[str] = []
    citations: list[Citation] = []
    numbers: dict[tuple, int] = {}
    unresolved = 0
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text_parts.append(block.text)
        markers = []
        for raw in getattr(block, "citations", None) or []:
            resolved = _resolve_citation(raw, sources)
            if resolved is None:
                unresolved += 1
                continue
            key = resolved.pop("key")
            if key not in numbers:
                numbers[key] = len(numbers) + 1
                citations.append(Citation(number=numbers[key], **resolved))
            if numbers[key] not in markers:
                markers.append(numbers[key])
        text_parts.extend(f" [{n}]" for n in markers)
    return "".join(text_parts).strip(), citations, unresolved


def _echoable(content: list[Any]) -> list[Any]:
    """An assistant turn as it must be sent back. Normally that's exactly
    as returned. After a refusal fallback took over mid-output, the API's
    rule is to drop thinking and tool_use blocks that precede the last
    `fallback` block (they belong to the declined attempt); everything
    after it echoes normally."""
    types = [getattr(block, "type", None) for block in content]
    if "fallback" not in types:
        return content
    boundary = len(types) - 1 - types[::-1].index("fallback")
    return [
        block
        for index, block in enumerate(content)
        if index >= boundary
        or getattr(block, "type", None) not in ("thinking", "redacted_thinking", "tool_use")
    ]


def _summarize_result(content: list[dict[str, Any]]) -> str:
    """A short, content-free description of a tool result for the API
    response and logs (the result itself is document text)."""
    results = sum(1 for block in content if block.get("type") == "search_result")
    texts = [block["text"] for block in content if block.get("type") == "text"]
    summary = f"{results} source(s)"
    return f"{summary}; {texts[0]}" if texts else summary


async def _run_tool(ctx: ToolContext, block: Any) -> tuple[dict[str, Any], ToolCall]:
    with tracer().start_as_current_span(
        f"execute_tool {block.name}",
        kind=SpanKind.INTERNAL,
        attributes={
            GEN_AI_OPERATION_NAME: "execute_tool",
            GEN_AI_TOOL_NAME: block.name,
            GEN_AI_TOOL_CALL_ID: block.id,
            **observation_attributes("tool"),
        },
    ) as span:
        handler = TOOL_HANDLERS.get(block.name)
        try:
            if handler is None:
                raise ToolError(f"unknown tool {block.name!r}")
            # A savepoint per call: a tool whose SQL fails (a timeout, a bad
            # cast) rolls back to here instead of leaving the transaction
            # aborted -- which would fail every later tool call and then the
            # INSERT that stores the (already paid-for) answer.
            async with ctx.session.begin_nested():
                content = await handler(ctx, dict(block.input or {}))
            is_error = False
        except ToolError as exc:
            content = [{"type": "text", "text": f"Error: {exc}"}]
            is_error = True
            span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception as exc:  # a tool bug must not end the run
            logger.exception("tool %s failed", block.name)
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            content = [{"type": "text", "text": "Error: the tool failed unexpectedly."}]
            is_error = True
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        return result, ToolCall(block.name, dict(block.input or {}), is_error, _summarize_result(content))


EventSink = Callable[[dict[str, Any]], Awaitable[None]]


async def answer_question(
    ctx: ToolContext,
    question: str,
    settings: Settings,
    client: anthropic.AsyncAnthropic | None = None,
    on_event: EventSink | None = None,
    *,
    prompt: AgentPrompt = ASK_PROMPT,
    history: Sequence[Turn] = (),
    conversation_id: str | None = None,
) -> AgentResult:
    """Run the agent on one question. Never raises for model behavior
    (refusal, truncation, budgets) -- those come back as the result's
    status. API errors raise ExtractionError subclasses, classified the
    same way as the rest of the app's model calls.

    on_event, when given, is awaited with the run's progress as it
    happens -- {"type": "model_call", step, cost_usd, stop_reason} after
    each model call, {"type": "tool_start", name, input} before each tool
    runs and {"type": "tool_call", name, input, is_error, result_summary}
    after -- so POST /ask/stream can show the agent working instead of
    ten silent seconds.

    history is the conversation so far (oldest first) when this question
    is a follow-up; conversation_id, when given, is recorded on the run's
    span (gen_ai.conversation.id) so a chat's turns can be found together."""

    async def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            await on_event(event)

    client = client or _build_client(settings)
    model = settings.agent_model
    options = _request_options(settings)
    messages: list[dict[str, Any]] = [
        *history_messages(history),
        {"role": "user", "content": question},
    ]
    result = AgentResult(
        status="step_limit",
        answer="",
        citations=[],
        tool_calls=[],
        steps=0,
        model=model,
        prompt_version=prompt.version,
        messages=messages,
    )
    start = time.perf_counter()

    with tracer().start_as_current_span(
        f"invoke_agent {AGENT_NAME}",
        kind=SpanKind.INTERNAL,
        attributes={
            GEN_AI_OPERATION_NAME: "invoke_agent",
            GEN_AI_PROVIDER_NAME: "anthropic",
            GEN_AI_AGENT_NAME: AGENT_NAME,
            GEN_AI_REQUEST_MODEL: model,
            # A document chat is a Langfuse session: its turns, together.
            **observation_attributes("agent", session_id=conversation_id),
        },
    ) as agent_span:
        if conversation_id is not None:
            agent_span.set_attribute(GEN_AI_CONVERSATION_ID, conversation_id)
        span_context = agent_span.get_span_context()
        if span_context.is_valid:
            result.trace_id = format(span_context.trace_id, "032x")
            result.traceparent = current_traceparent()

        for _ in range(settings.agent_max_steps):
            with model_call_span(
                model, max_tokens=settings.agent_max_tokens, prompt_version=prompt.version
            ) as span:
                try:
                    response = await client.beta.messages.create(
                        model=model,
                        max_tokens=settings.agent_max_tokens,
                        system=prompt.system,
                        tools=TOOL_DEFINITIONS,
                        messages=messages,
                        # Automatic prompt caching: each step re-sends the
                        # whole conversation, so everything before the
                        # newest turn is a cache read.
                        cache_control={"type": "ephemeral"},
                        **options,
                    )
                except anthropic.AnthropicError as exc:
                    raise _classify_api_error(exc) from exc
                cost = call_cost_usd(model, response)
                record_model_response(span, response, cost)

            result.steps += 1
            result.model = response.model or model
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            result.cache_read_input_tokens += getattr(response.usage, "cache_read_input_tokens", None) or 0
            result.cost_usd += cost
            await emit(
                {
                    "type": "model_call",
                    "step": result.steps,
                    "cost_usd": result.cost_usd,
                    "stop_reason": response.stop_reason,
                }
            )

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                result.status = "refused"
                result.refusal_category = getattr(details, "category", None)
                break

            messages.append({"role": "assistant", "content": _echoable(response.content)})

            if response.stop_reason == "max_tokens":
                result.status = "truncated"
                result.answer, result.citations, result.unresolved_citations = _assemble_answer(
                    response.content, ctx.sources
                )
                break

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                result.status = "answered"
                result.answer, result.citations, result.unresolved_citations = _assemble_answer(
                    response.content, ctx.sources
                )
                break

            if result.cost_usd >= settings.agent_max_cost_usd:
                result.status = "budget_exceeded"
                break

            # Sequential, not gathered: the tools share one DB session.
            # All results go back in ONE user message, as the API expects.
            tool_results = []
            for block in tool_uses:
                await emit({"type": "tool_start", "name": block.name, "input": dict(block.input or {})})
                tool_result, call = await _run_tool(ctx, block)
                tool_results.append(tool_result)
                result.tool_calls.append(call)
                await emit({"type": "tool_call", **asdict(call)})
            messages.append({"role": "user", "content": tool_results})

        result.latency_ms = int((time.perf_counter() - start) * 1000)
        agent_span.set_attribute("docpilot.agent.status", result.status)
        agent_span.set_attribute("docpilot.agent.steps", result.steps)
        agent_span.set_attribute("docpilot.agent.tool_calls", len(result.tool_calls))
        agent_span.set_attribute(DOCPILOT_COST_USD, result.cost_usd)
    return result
