"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  askQuestionStream,
  getAskRun,
  StreamInterruptedError,
  listAskRuns,
  ApiError,
  type AskEvent,
  type AskResponse,
  type AskRunSummary,
} from "@/lib/api";
import {
  AnswerText,
  FeedbackButtons,
  Judgment,
  LiveProgress,
  STATUS_NOTES,
  sourceLabel,
  useLiveProgress,
} from "@/components/AgentRun";

const EXAMPLES = [
  "How much have I spent at Northgate Office Outfitters?",
  "Which store's receipt came to $425.58?",
  "Where did I buy a light for my workspace?",
  "How much did I spend in May 2026?",
];

function RunChips({ run }: { run: AskRunSummary }) {
  return (
    <span className="rank-chips">
      {run.document_id && (
        <Link href={`/documents/${run.document_id}`} className="rank-chip">
          document chat
        </Link>
      )}
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
  const live = useLiveProgress();

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
    live.onEvent(event);
    switch (event.type) {
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
    live.reset();
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
      {loading && <LiveProgress steps={live.steps} modelCalls={live.modelCalls} cost={live.cost} />}

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
