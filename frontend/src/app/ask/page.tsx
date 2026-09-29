"use client";

import { Fragment, useState } from "react";
import Link from "next/link";
import { askQuestion, ApiError, type AskResponse, type Citation } from "@/lib/api";

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

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      setResponse(await askQuestion(q.trim()));
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.status === 503
            ? `The agent is unavailable: ${err.message}`
            : `API error: ${err.message}`
          : "Could not reach the API. Is the backend running?"
      );
    } finally {
      setLoading(false);
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
      {loading && <div className="empty-state">Working: searching and reading your documents…</div>}

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
    </div>
  );
}
