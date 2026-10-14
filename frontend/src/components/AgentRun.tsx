"use client";

// The pieces of an agent answer shared by the Ask page and the
// per-document chat: the cited answer text, the person's verdict, the
// automatic grader's, and the live progress of a run in flight.

import { Fragment, useCallback, useState, type ReactNode } from "react";
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

type AnswerBlock = { kind: "p"; lines: string[] } | { kind: "ul" | "ol"; items: string[] };

/** The Markdown the model writes, as far as answers use it: paragraphs,
 * bullet and numbered lists, headings (shown bold), and **bold**. Anything
 * else stays literal text, and nothing is ever parsed as HTML. */
export function answerBlocks(answer: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  // Whether the next line may join the last block (a blank line closes it).
  let open = false;
  for (const raw of answer.split("\n")) {
    const line = raw.trim();
    const last = open ? blocks[blocks.length - 1] : undefined;
    open = line !== "";
    if (!line) continue;
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(line);
    const heading = /^#{1,6}\s+(.*)$/.exec(line);
    if (bullet || numbered) {
      const kind = bullet ? "ul" : "ol";
      const item = (bullet ?? numbered)![1];
      if (last?.kind === kind) last.items.push(item);
      else blocks.push({ kind, items: [item] });
    } else if (heading) {
      blocks.push({ kind: "p", lines: [`**${heading[1]}**`] });
      open = false;
    } else if (last && last.kind !== "p" && raw !== line) {
      // An indented line continues the list item above it.
      last.items[last.items.length - 1] += ` ${line}`;
    } else if (last?.kind === "p") {
      last.lines.push(line);
    } else {
      blocks.push({ kind: "p", lines: [line] });
    }
  }
  return blocks;
}

/** The answer text, rendered from its Markdown, with each [n] marker
 * turned into a link to its source -- or, with onCite, a button that
 * selects it. */
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

  const marker = (number: string, key: string) => {
    const citation = byNumber.get(Number(number));
    if (!citation) return <Fragment key={key}>[{number}]</Fragment>;
    return onCite ? (
      <button
        key={key}
        type="button"
        className={`cite-marker${activeNumber === citation.number ? " active" : ""}`}
        aria-label={`Show source ${citation.number}: ${sourceLabel(citation)}`}
        aria-pressed={activeNumber === citation.number}
        onClick={() => onCite(citation)}
      >
        {citation.number}
      </button>
    ) : (
      <a key={key} href={`#citation-${citation.number}`} className="cite-marker">
        {citation.number}
      </a>
    );
  };

  // Each run of markers stays on one line with the word before it, so a
  // narrow column never starts a line with a lone marker.
  const markers = (text: string, key: string) => {
    const parts: ReactNode[] = [];
    let last = 0;
    for (const match of text.matchAll(/(\S*)((?:\s?\[\d+\])+)/g)) {
      parts.push(text.slice(last, match.index));
      const numbers = [...match[2].matchAll(/\[(\d+)\]/g)].map((m) => m[1]);
      parts.push(
        <span key={`${key}.${match.index}`} className="cite-anchor">
          {match[1]}
          {numbers.map((n, i) => marker(n, `${key}.${match.index}.${i}`))}
        </span>,
      );
      last = match.index + match[0].length;
    }
    parts.push(text.slice(last));
    return parts;
  };

  const inline = (text: string, key: string) =>
    text.split(/\*\*(.+?)\*\*/).map((part, i) =>
      i % 2 === 0 ? (
        <Fragment key={`${key}.${i}`}>{markers(part, `${key}.${i}`)}</Fragment>
      ) : (
        <strong key={`${key}.${i}`}>{markers(part, `${key}.${i}`)}</strong>
      ),
    );

  return (
    <div className="answer-text">
      {answerBlocks(answer).map((block, b) => {
        if (block.kind === "p") return <p key={b}>{inline(block.lines.join(" "), `${b}`)}</p>;
        const List = block.kind;
        return (
          <List key={b}>
            {block.items.map((item, i) => (
              <li key={i}>{inline(item, `${b}.${i}`)}</li>
            ))}
          </List>
        );
      })}
    </div>
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
