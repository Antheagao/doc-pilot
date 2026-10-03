"use client";

import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  chatStream,
  getConversation,
  listConversations,
  StreamInterruptedError,
  type AskEvent,
  type AskResponse,
  type Citation,
} from "@/lib/api";
import {
  AnswerText,
  FeedbackButtons,
  Judgment,
  LiveProgress,
  STATUS_NOTES,
  fieldLabel,
  useLiveProgress,
} from "@/components/AgentRun";

const SUGGESTIONS = [
  "What's the total?",
  "When was this, and from which store?",
  "What was the most expensive item?",
];

/** The citation a person picked: which turn's, and its number there. */
export interface SelectedCitation {
  turnId: string;
  citation: Citation;
}

function SourceChip({
  citation,
  active,
  onSelect,
}: {
  citation: Citation;
  active: boolean;
  onSelect: () => void;
}) {
  const where =
    citation.kind === "record"
      ? citation.fields.map(fieldLabel).join(", ") || "extracted fields"
      : `page ${citation.page_number}`;
  return (
    <li>
      <button
        type="button"
        className={`chat-source${active ? " active" : ""}`}
        aria-pressed={active}
        onClick={onSelect}
      >
        <span className="cite-marker static">{citation.number}</span>
        <span className="chat-source-kind">
          {citation.kind === "record" ? "extracted field" : "document text"}
        </span>
        <span className="chat-source-where">{where}</span>
      </button>
      {active && <pre className="passage chat-passage">{citation.cited_text}</pre>}
    </li>
  );
}

function Turn({
  turn,
  selected,
  onSelect,
  onUpdate,
}: {
  turn: AskResponse;
  selected: SelectedCitation | null;
  onSelect: (selection: SelectedCitation | null) => void;
  onUpdate: (updated: AskResponse) => void;
}) {
  const activeNumber = selected?.turnId === turn.id ? selected.citation.number : null;
  const toggle = (citation: Citation) =>
    onSelect(activeNumber === citation.number ? null : { turnId: turn.id, citation });
  return (
    <li className="chat-turn">
      <div className="chat-bubble chat-user">{turn.question}</div>
      <div className="chat-bubble chat-assistant">
        {STATUS_NOTES[turn.status] && (
          <p className="chat-status">
            {STATUS_NOTES[turn.status]}
            {turn.refusal_category ? ` (${turn.refusal_category})` : ""}
          </p>
        )}
        {turn.answer && (
          <AnswerText
            answer={turn.answer}
            citations={turn.citations}
            onCite={toggle}
            activeNumber={activeNumber}
          />
        )}
        {turn.citations.length > 0 && (
          <ol className="chat-sources" aria-label="Sources">
            {turn.citations.map((c) => (
              <SourceChip
                key={c.number}
                citation={c}
                active={activeNumber === c.number}
                onSelect={() => toggle(c)}
              />
            ))}
          </ol>
        )}
        {turn.answer && <FeedbackButtons response={turn} onChange={onUpdate} />}
        <Judgment response={turn} />
        <p className="chat-meta">
          {turn.steps} step{turn.steps === 1 ? "" : "s"} · ${turn.cost_usd.toFixed(4)} ·{" "}
          {(turn.latency_ms / 1000).toFixed(1)}s
        </p>
      </div>
    </li>
  );
}

/** A conversation about one document. Every answer cites the extracted
 * field or the line of text it came from; selecting a field citation
 * points at that row of the extraction (onSelect). */
export default function ChatPanel({
  documentId,
  selected,
  onSelect,
}: {
  documentId: string;
  selected: SelectedCitation | null;
  onSelect: (selection: SelectedCitation | null) => void;
}) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<AskResponse[]>([]);
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useLiveProgress();
  const threadEnd = useRef<HTMLDivElement | null>(null);
  // Set once the person sends or starts over: from then on the thread
  // follows new turns, and a slow restore must not replace their chat.
  const engaged = useRef(false);

  // Pick up the most recent conversation about this document, if any.
  useEffect(() => {
    let cancelled = false;
    listConversations(documentId, 1)
      .then(async ([latest]) => {
        if (!latest || cancelled || engaged.current) return;
        const stored = await getConversation(documentId, latest.conversation_id);
        if (cancelled || engaged.current) return;
        setConversationId(latest.conversation_id);
        setTurns(stored);
      })
      .catch(() => {
        // No history to show is not an error worth a banner.
      });
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  useEffect(() => {
    if (engaged.current) threadEnd.current?.scrollIntoView({ block: "nearest" });
  }, [turns.length, pending]);

  function onEvent(event: AskEvent) {
    live.onEvent(event);
    if (event.type === "answer") {
      setTurns((current) => [...current, event.run]);
      setConversationId(event.run.conversation_id);
    } else if (event.type === "error") {
      setError(
        event.status === 503
          ? `The model is unavailable right now: ${event.detail}`
          : `The agent failed: ${event.detail}`
      );
    }
  }

  async function send(text: string) {
    const q = text.trim();
    if (!q || pending !== null) return;
    engaged.current = true;
    setPending(q);
    setQuestion("");
    setError(null);
    live.reset();
    try {
      await chatStream(documentId, q, conversationId, onEvent);
    } catch (err) {
      setQuestion(q);
      setError(
        err instanceof ApiError
          ? err.status === 503
            ? `The agent is unavailable: ${err.message}`
            : err.status === 429
              ? `Not right now: ${err.message}.`
              : `API error: ${err.message}`
          : err instanceof StreamInterruptedError
            ? `${err.message} The answer is still being worked on; reload the page to see it.`
            : "Could not reach the API. Is the backend running?"
      );
    } finally {
      setPending(null);
    }
  }

  function startOver() {
    engaged.current = true;
    setConversationId(null);
    setTurns([]);
    setError(null);
    onSelect(null);
  }

  return (
    <section className="results-panel chat-panel" aria-labelledby="chat-heading">
      <div className="chat-head">
        <h2 id="chat-heading" className="chat-title">
          Ask about this document
        </h2>
        {turns.length > 0 && (
          <button
            type="button"
            className="btn"
            onClick={startOver}
            disabled={pending !== null}
          >
            New conversation
          </button>
        )}
      </div>

      {turns.length === 0 && pending === null && (
        <>
          <p className="chat-intro">
            Questions here only see this document. Each answer cites the extracted field or the
            line of text it came from: select a citation to find it.
          </p>
          <div className="example-row">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className="example-chip"
                onClick={() => send(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </>
      )}

      {(turns.length > 0 || pending !== null) && (
        <ol className="chat-thread">
          {turns.map((turn) => (
            <Turn
              key={turn.id}
              turn={turn}
              selected={selected}
              onSelect={onSelect}
              onUpdate={(updated) =>
                setTurns((current) => current.map((t) => (t.id === updated.id ? updated : t)))
              }
            />
          ))}
          {pending !== null && (
            <li className="chat-turn">
              <div className="chat-bubble chat-user">{pending}</div>
              <LiveProgress
                steps={live.steps}
                modelCalls={live.modelCalls}
                cost={live.cost}
                className="chat-bubble chat-assistant"
              />
            </li>
          )}
        </ol>
      )}
      <div ref={threadEnd} />

      {error && <div className="error-banner">{error}</div>}

      <form
        className="chat-form"
        onSubmit={(e) => {
          e.preventDefault();
          void send(question);
        }}
      >
        <label htmlFor="chat-question" className="visually-hidden">
          Your question about this document
        </label>
        <input
          id="chat-question"
          className="review-input query-input"
          placeholder={turns.length ? "Ask a follow-up…" : "e.g. How much was the tax?"}
          value={question}
          maxLength={1000}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={pending !== null || !question.trim()}
        >
          {pending !== null ? "Thinking…" : "Send"}
        </button>
      </form>
    </section>
  );
}
