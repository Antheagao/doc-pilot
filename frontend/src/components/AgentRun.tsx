"use client";

// The pieces of an agent answer shared by the Ask page and the
// per-document chat: the cited answer text, the person's verdict, the
// automatic grader's, and the live progress of a run in flight.

import { Fragment, useCallback, useState } from "react";
import {
  sendAskFeedback,
  type AskEvent,
  type AskResponse,
  type Citation,
  type Feedback,
} from "@/lib/api";

export const STATUS_NOTES: Record<string, string> = {
  refused: "The model declined this question.",
  truncated: "The answer was cut off by the output limit.",
  step_limit: "The agent hit its step limit before finishing.",
  budget_exceeded: "The agent hit its per-question cost cap before finishing.",
};

/** The answer text with each [n] marker turned into a link to its source
 * -- or, with onCite, a button that selects it. */
export function AnswerText({
  answer,
  citations,
  onCite,
  activeNumber = null,
}: {
  answer: string;
  citations: Citation[];
  onCite?: (citation: Citation) => void;
  activeNumber?: number | null;
}) {
  const byNumber = new Map(citations.map((c) => [c.number, c]));
  const parts = answer.split(/\s?\[(\d+)\]/);
  return (
    <p className="answer-text">
      {parts.map((part, i) => {
        if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
        const citation = byNumber.get(Number(part));
        if (!citation) return <Fragment key={i}>[{part}]</Fragment>;
        return onCite ? (
          <button
            key={i}
            type="button"
            className={`cite-marker${activeNumber === citation.number ? " active" : ""}`}
            aria-label={`Show source ${citation.number}: ${sourceLabel(citation)}`}
            aria-pressed={activeNumber === citation.number}
            onClick={() => onCite(citation)}
          >
            {citation.number}
          </button>
        ) : (
          <a key={i} href={`#citation-${citation.number}`} className="cite-marker">
            {citation.number}
          </a>
        );
      })}
    </p>
  );
}

/** A record citation's field names, as a person would say them. */
export function fieldLabel(field: string): string {
  const item = /^line_items\[(\d+)\]$/.exec(field);
  if (item) return `line item ${Number(item[1]) + 1}`;
  if (field === "review_status") return "review note";
  return field.replace(/_/g, " ");
}

export function sourceLabel(c: Citation): string {
  if (c.kind === "record") {
    return `extracted field${c.fields.length === 1 ? "" : "s"}${
      c.fields.length ? `: ${c.fields.map(fieldLabel).join(", ")}` : ""
    }`;
  }
  return `page ${c.page_number}${c.kind === "page" ? " (full page)" : ""}`;
}

/** A person's verdict on the answer: stored with the run, and counted
 * beside the automatic grader's in /stats. */
export function FeedbackButtons({
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
export function Judgment({ response }: { response: AskResponse }) {
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

export interface LiveStep {
  name: string;
  input: Record<string, unknown>;
  summary: string | null;
  isError: boolean;
}

/** A run's progress from its stream events: reset() before a run, and
 * pass every event to onEvent (it ignores the answer and error events). */
export function useLiveProgress() {
  const [steps, setSteps] = useState<LiveStep[]>([]);
  const [modelCalls, setModelCalls] = useState(0);
  const [cost, setCost] = useState(0);

  const reset = useCallback(() => {
    setSteps([]);
    setModelCalls(0);
    setCost(0);
  }, []);

  const onEvent = useCallback((event: AskEvent) => {
    switch (event.type) {
      case "model_call":
        setModelCalls(event.step);
        setCost(event.cost_usd);
        break;
      case "tool_start":
        setSteps((current) => [
          ...current,
          { name: event.name, input: event.input, summary: null, isError: false },
        ]);
        break;
      case "tool_call":
        // Tools run one at a time, so the finished one is the last started.
        setSteps((current) =>
          current.map((step, i) =>
            i === current.length - 1
              ? { ...step, summary: event.result_summary, isError: event.is_error }
              : step
          )
        );
        break;
    }
  }, []);

  return { steps, modelCalls, cost, reset, onEvent };
}

/** What the agent is doing right now, from the stream's events. */
export function LiveProgress({
  steps,
  modelCalls,
  cost,
  className = "review-row answer-card",
}: {
  steps: LiveStep[];
  modelCalls: number;
  cost: number;
  className?: string;
}) {
  return (
    <div className={`${className} live-progress`} aria-live="polite">
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
