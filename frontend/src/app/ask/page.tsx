"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  askQuestionStream,
  getAskRun,
  StreamInterruptedError,
  listAskRuns,
  sendAskFeedback,
  ApiError,
  type AskEvent,
  type AskResponse,
  type AskRunSummary,
  type Citation,
  type Feedback,
} from "@/lib/api";

const EXAMPLES = [
  "How much have I spent at Northgate Office Outfitters?",
  "Which store's receipt came to $425.58?",
  "Where did I buy a light for my workspace?",
  "How much did I spend in May 2026?",
];

const STATUS_NOTES: Record<string, string> = {
  refused: "The model declined this question.",
  truncated: "The answer was cut off by the output limit.",
  step_limit: "The agent hit its step limit before finishing.",
  budget_exceeded: "The agent hit its per-question cost cap before finishing.",
};

/** The answer text with each [n] marker turned into a link to its source. */
function AnswerText({ answer, citations }: { answer: string; citations: Citation[] }) {
  const known = new Set(citations.map((c) => c.number));
  const parts = answer.split(/\s?\[(\d+)\]/);
  return (
    <p className="answer-text">
      {parts.map((part, i) => {
        if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
        const n = Number(part);
        return known.has(n) ? (
          <a key={i} href={`#citation-${n}`} className="cite-marker">
            {n}
          </a>
        ) : (
          <Fragment key={i}>[{part}]</Fragment>
        );
      })}
    </p>
  );
}

function sourceLabel(c: Citation): string {
  if (c.kind === "record") {
    return `extracted fields${c.fields.length ? `: ${c.fields.join(", ")}` : ""}`;
  }
  return `page ${c.page_number}${c.kind === "page" ? " (full page)" : ""}`;
}

/** A person's verdict on the answer: stored with the run, and counted
 * beside the automatic grader's in /stats. */
function FeedbackButtons({
  response,
  onChange,
}: {
  response: AskResponse;
  onChange: (updated: AskResponse) => void;
}) {
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);

  async function rate(rating: Feedback) {
    setSaving(true);
    setFailed(false);
    try {
      onChange(await sendAskFeedback(response.id, rating));
    } catch {
      setFailed(true);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="feedback-row" role="group" aria-label="Was this answer right?">
      <span className="example-label">Was this right?</span>
      {(["up", "down"] as const).map((rating) => (
        <button
          key={rating}
          type="button"
          className={`mode-option feedback-option${response.feedback === rating ? " active" : ""}`}
          aria-pressed={response.feedback === rating}
          disabled={saving}
          onClick={() => rate(rating)}
        >
          {rating === "up" ? "Yes" : "No"}
        </button>
      ))}
      {failed && <span className="feedback-note">Couldn&apos;t save that. Try again?</span>}
    </div>
  );
}

/** The background groundedness grader's verdict, when this answer was
 * sampled for one. */
function Judgment({ response }: { response: AskResponse }) {
  const judgment = response.judgment;
  if (!judgment) {
    return response.judge_sampled ? (
      <p className="answer-meta">Queued for an automatic groundedness check.</p>
    ) : null;
  }
  if (judgment.error || judgment.grounded === null) {
    return <p className="answer-meta">The automatic check couldn&apos;t grade this answer.</p>;
  }
  return (
    <div className="judgment">
      <span className={`review-badge ${judgment.grounded ? "" : "judgment-fail"}`}>
        {judgment.grounded ? "grounded" : "not grounded"}
      </span>
      {judgment.answers_question === false && (
        <span className="review-badge judgment-fail">doesn&apos;t answer the question</span>
      )}
      <span className="muted-inline">
        automatic check ({judgment.model}, {judgment.prompt_version})
      </span>
      {judgment.unsupported_claims.length > 0 && (
        <ul className="judgment-claims">
          {judgment.unsupported_claims.map((claim, i) => (
            <li key={i}>{claim}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface LiveStep {
  name: string;
  input: Record<string, unknown>;
  summary: string | null;
  isError: boolean;
}

/** What the agent is doing right now, from the stream's events. */
function LiveProgress({ steps, modelCalls, cost }: { steps: LiveStep[]; modelCalls: number; cost: number }) {
  return (
    <div className="review-row answer-card live-progress" aria-live="polite">
      <p className="answer-meta">
        Working: {modelCalls === 0 ? "reading the question" : `step ${modelCalls}`}
        {cost > 0 && <> · ${cost.toFixed(4)} so far</>}
      </p>
      {steps.length > 0 && (
        <ol className="tool-trail">
          {steps.map((step, i) => (
            <li key={i} className={step.isError ? "tool-error" : undefined}>
              <code className="tool-name">{step.name}</code>
              <code className="tool-input">{JSON.stringify(step.input)}</code>
              <span className="tool-summary">{step.summary ?? "running…"}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function RunChips({ run }: { run: AskRunSummary }) {
  return (
    <span className="rank-chips">
      {run.status !== "answered" && <span className="rank-chip">{run.status.replace("_", " ")}</span>}
      {run.feedback && <span className="rank-chip">{run.feedback === "up" ? "marked right" : "marked wrong"}</span>}
      {run.judge_grounded !== null && (
        <span className="rank-chip">{run.judge_grounded ? "grounded" : "not grounded"}</span>
      )}
    </span>
  );
}

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<AskRunSummary[]>([]);
  const [liveSteps, setLiveSteps] = useState<LiveStep[]>([]);
  const [modelCalls, setModelCalls] = useState(0);
  const [liveCost, setLiveCost] = useState(0);

  const refreshHistory = useCallback(() => {
    listAskRuns(10)
      .then(setHistory)
      .catch(() => setHistory([]));
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  /** Reopen a stored answer: read back from the database, not asked again. */
  async function showRun(id: string) {
    setError(null);
    try {
      const run = await getAskRun(id);
      setResponse(run);
      setQuestion(run.question);
    } catch (err) {
      setError(err instanceof ApiError ? `API error: ${err.message}` : "Could not reach the API.");
    }
  }

  function onEvent(event: AskEvent) {
    switch (event.type) {
      case "model_call":
        setModelCalls(event.step);
        setLiveCost(event.cost_usd);
        break;
      case "tool_start":
        setLiveSteps((steps) => [
          ...steps,
          { name: event.name, input: event.input, summary: null, isError: false },
        ]);
        break;
      case "tool_call":
        // Tools run one at a time, so the finished one is the last started.
        setLiveSteps((steps) =>
          steps.map((step, i) =>
            i === steps.length - 1
              ? { ...step, summary: event.result_summary, isError: event.is_error }
              : step
          )
        );
        break;
      case "answer":
        setResponse(event.run);
        break;
      case "error":
        setError(
          event.status === 503
            ? `The model is unavailable right now: ${event.detail}`
            : `The agent failed: ${event.detail}`
        );
        break;
    }
  }

  async function run(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    setLiveSteps([]);
    setModelCalls(0);
    setLiveCost(0);
    try {
      await askQuestionStream(q.trim(), onEvent);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.status === 503
            ? `The agent is unavailable: ${err.message}`
            : err.status === 429
              ? `Not right now: ${err.message}.`
              : `API error: ${err.message}`
          : err instanceof StreamInterruptedError
            ? `${err.message} The agent keeps working on the server; the answer will appear under recent questions.`
            : "Could not reach the API. Is the backend running?"
      );
    } finally {
      setLoading(false);
      refreshHistory();
    }
  }

  return (
    <div>
      <h1 className="page-title">Ask</h1>
      <p className="review-intro">
        Ask about your documents. The agent chooses between searching their text and querying
        the extracted (human-verified) fields, and every citation points at the exact line or
        field it used.
      </p>

      <form
        className="query-form"
        onSubmit={(e) => {
          e.preventDefault();
          run(question);
        }}
      >
        <label htmlFor="ask-question" className="visually-hidden">
          Question
        </label>
        <input
          id="ask-question"
          className="review-input query-input"
          placeholder="e.g. How much did I spend in euros?"
          value={question}
          maxLength={1000}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit" className="btn btn-primary" disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      <div className="example-row">
        <span className="example-label">Try:</span>
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="example-chip"
            disabled={loading}
            onClick={() => {
              setQuestion(example);
              run(example);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && <LiveProgress steps={liveSteps} modelCalls={modelCalls} cost={liveCost} />}

      {response && (
        <section aria-live="polite">
          {STATUS_NOTES[response.status] && (
            <div className="error-banner">
              {STATUS_NOTES[response.status]}
              {response.refusal_category ? ` (${response.refusal_category})` : ""}
            </div>
          )}
          {response.answer && (
            <div className="review-row answer-card">
              <AnswerText answer={response.answer} citations={response.citations} />
              <FeedbackButtons
                response={response}
                onChange={(updated) => {
                  setResponse(updated);
                  refreshHistory();
                }}
              />
              <Judgment response={response} />
            </div>
          )}

          {response.citations.length > 0 && (
            <>
              <h2 className="section-heading">Sources</h2>
              <ol className="hit-list">
                {response.citations.map((c) => (
                  <li key={c.number} id={`citation-${c.number}`} className="review-row hit-row">
                    <div className="hit-head">
                      <span className="cite-marker static">{c.number}</span>
                      <Link href={`/documents/${c.document_id}`} className="hit-doc">
                        {c.filename}
                      </Link>
                      <span className="hit-page">{sourceLabel(c)}</span>
                    </div>
                    <pre className="passage">{c.cited_text}</pre>
                  </li>
                ))}
              </ol>
            </>
          )}

          {response.tool_calls.length > 0 && (
            <>
              <h2 className="section-heading">How it got there</h2>
              <ol className="tool-trail">
                {response.tool_calls.map((call, i) => (
                  <li key={i} className={call.is_error ? "tool-error" : undefined}>
                    <code className="tool-name">{call.name}</code>
                    <code className="tool-input">{JSON.stringify(call.input)}</code>
                    <span className="tool-summary">{call.result_summary}</span>
                  </li>
                ))}
              </ol>
            </>
          )}

          <p className="answer-meta">
            {response.model} · {response.steps} step{response.steps === 1 ? "" : "s"} · $
            {response.cost_usd.toFixed(4)} · {(response.latency_ms / 1000).toFixed(1)}s ·{" "}
            {response.input_tokens + response.cache_read_input_tokens} in /{" "}
            {response.output_tokens} out tokens
            {response.trace_id && <> · trace {response.trace_id.slice(0, 12)}</>}
          </p>
        </section>
      )}

      {history.length > 0 && (
        <section>
          <h2 className="section-heading">Recent questions</h2>
          <ol className="hit-list">
            {history.map((item) => (
              <li key={item.id} className="review-row hit-row history-row">
                <button type="button" className="history-question" onClick={() => showRun(item.id)}>
                  {item.question}
                </button>
                <RunChips run={item} />
                <span className="hit-page">
                  {new Date(item.created_at).toLocaleString()} · ${item.cost_usd.toFixed(4)}
                </span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
