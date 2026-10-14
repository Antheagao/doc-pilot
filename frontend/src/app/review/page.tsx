"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  getReviewQueue,
  resolveReviewField,
  ApiError,
  type FieldLeaf,
  type ReviewQueueItem,
} from "@/lib/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { EMPTY_VALUE, formatAmount } from "@/lib/format";
import { checkText, humanize, pathLabel } from "@/lib/reconciliation";

// Fields that hold an amount of money. Mirrors MONEY_FIELDS in the backend
// (app/extraction.py).
const NUMERIC_FIELDS = new Set(["subtotal", "tax", "total"]);

// A plain decimal exactly as the API takes it: optional minus, digits,
// optional fraction. Commas and spaces are refused, never stripped: "45,25"
// would silently become 4525. Whether it fits the currency's digits is the
// backend's call (a 422 with a sentence, shown in the banner).
const PLAIN_DECIMAL = /^-?\d+(\.\d+)?$/;

function isLeaf(value: unknown): value is FieldLeaf {
  return (
    typeof value === "object" &&
    value !== null &&
    "value" in value &&
    "confidence" in value
  );
}

function leafValue(value: unknown): unknown {
  return isLeaf(value) ? value.value : value;
}

// The stored line_items array holds {value, confidence} leaves per cell;
// a human correction shouldn't have to hand-write confidence scores, so
// the editor prefills (and submits) the plain semantic rows instead.
function lineItemsToSemantic(items: unknown[]): unknown[] {
  return items.map((item) => {
    if (typeof item !== "object" || item === null) return item;
    const row = item as Record<string, unknown>;
    return {
      // `?? null` keeps all four keys, so an untouched draft is submittable.
      description: leafValue(row.description) ?? null,
      quantity: leafValue(row.quantity) ?? null,
      unit_price: leafValue(row.unit_price) ?? null,
      total: leafValue(row.total) ?? null,
    };
  });
}

function initialDraft(item: ReviewQueueItem): string {
  if (item.field_name === "line_items") {
    const items = Array.isArray(item.value) ? item.value : [];
    return JSON.stringify(lineItemsToSemantic(items), null, 2);
  }
  const value = leafValue(item.value);
  return value === null || value === undefined ? "" : String(value);
}

/** Parse the correction input into the semantic value the backend stores.
 * An amount stays a string, never a Number, so no digit is rounded on the
 * way.
 * Throws with a user-facing message when the input can't be parsed. */
function parseDraft(fieldName: string, raw: string): unknown {
  if (fieldName === "line_items") {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw new Error("Line items must be valid JSON.");
    }
    if (!Array.isArray(parsed)) {
      throw new Error("Line items must be a JSON array.");
    }
    return parsed;
  }
  const trimmed = raw.trim();
  if (trimmed === "") return null; // empty input asserts "field is absent"
  if (NUMERIC_FIELDS.has(fieldName)) {
    if (!PLAIN_DECIMAL.test(trimmed)) {
      throw new Error(
        "Enter a plain number such as 45.25, with a dot for decimals and no " +
          "thousands separators (or leave it empty for none)."
      );
    }
    return trimmed;
  }
  return trimmed;
}

// A line total suggestion: "line_items[2].total" names the row to change.
const LINE_TOTAL_PATH = /^line_items\[(\d+)\]\.total$/;

/** The correction draft with a suggestion applied, or null when the
 * suggestion does not target this field (or the draft cannot take it). A
 * scalar is replaced by the suggested value; for line items only that row's
 * total changes in the JSON draft, and anything typed so far is kept. */
function draftWithSuggestion(
  item: ReviewQueueItem,
  suggestion: NonNullable<ReviewQueueItem["suggestion"]>,
  currentDraft: string | null
): string | null {
  if (item.field_name !== "line_items") {
    return suggestion.path === item.field_name ? suggestion.value : null;
  }
  const match = LINE_TOTAL_PATH.exec(suggestion.path);
  if (!match) return null;
  let rows: unknown;
  try {
    rows = JSON.parse(currentDraft ?? initialDraft(item));
  } catch {
    rows = JSON.parse(initialDraft(item));
  }
  if (!Array.isArray(rows)) return null;
  const row = rows[Number(match[1])];
  if (typeof row !== "object" || row === null) return null;
  (row as Record<string, unknown>).total = suggestion.value;
  return JSON.stringify(rows, null, 2);
}

function formatDisplayValue(item: ReviewQueueItem): string {
  if (item.field_name === "line_items") {
    const items = Array.isArray(item.value) ? item.value : [];
    return `${items.length} line item${items.length === 1 ? "" : "s"}`;
  }
  const value = leafValue(item.value);
  if (value === null || value === undefined || value === "") return EMPTY_VALUE;
  // The API pads amounts to the document currency's digits; the currency is
  // shown beside the amount so a misread one is as visible as a misread digit.
  if (NUMERIC_FIELDS.has(item.field_name)) {
    return item.currency
      ? `${formatAmount(value)} ${item.currency}`
      : formatAmount(value);
  }
  return String(value);
}

function ReviewRow({
  item,
  onResolved,
  onError,
}: {
  item: ReviewQueueItem;
  onResolved: (fieldId: string) => void;
  onError: (message: string | null) => void;
}) {
  const [correcting, setCorrecting] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  // What the last "Use ..." click filled in, said once in a live region.
  const [prefilled, setPrefilled] = useState("");
  const hintId = `review-hint-${item.field_id}`;
  const lowConfidence = `Low confidence (${Math.round(item.confidence * 100)}%)`;
  const reasonText =
    item.review_reason !== "confidence"
      ? item.checks.length > 0
        ? "These amounts do not add up:"
        : "Flagged by the arithmetic check."
      : item.checks.length > 0
        ? `${lowConfidence}, and these amounts do not add up:`
        : lowConfidence;
  const suggestion = item.suggestion;
  const suggestedDraft = suggestion
    ? draftWithSuggestion(item, suggestion, correcting ? draft : null)
    : null;

  const usePrefill = () => {
    if (suggestion === null || suggestedDraft === null) return;
    setDraft(suggestedDraft);
    setCorrecting(true);
    onError(null);
    setPrefilled(
      `Filled in ${formatAmount(suggestion.value)} for ${pathLabel(suggestion.path)}. Nothing is saved until you choose Save correction.`
    );
  };

  const resolve = async (
    resolution:
      | { action: "approve" }
      | { action: "correct"; corrected_value: unknown }
  ) => {
    setBusy(true);
    onError(null);
    try {
      await resolveReviewField(item.field_id, resolution);
      onResolved(item.field_id);
    } catch (err) {
      onError(
        err instanceof ApiError
          ? `API error: ${err.message}`
          : "Could not reach the API. Is the backend running?"
      );
      setBusy(false);
    }
  };

  const submitCorrection = async () => {
    let corrected: unknown;
    try {
      corrected = parseDraft(item.field_name, draft);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Invalid correction.");
      return;
    }
    await resolve({ action: "correct", corrected_value: corrected });
  };

  return (
    <div className="review-row">
      <div className="review-row-main">
        <div className="review-row-context">
          <Link
            href={`/documents/${item.document_id}`}
            className="review-doc-link"
          >
            {item.filename}
          </Link>
          <span className="review-field-name">
            {item.field_name.replace(/_/g, " ")}
          </span>
        </div>
        <div className="review-row-value">
          <span className="field-value">{formatDisplayValue(item)}</span>
          <ConfidenceBadge confidence={item.confidence} needsReview />
        </div>
      </div>

      <div className="review-reason">
        <p>
          <span className="review-reason-label">{reasonText}</span>
        </p>
        {item.checks.length > 0 && (
          <ul className="review-checks">
            {item.checks.map((check) => (
              <li key={check.id}>
                <span className="review-check-formula">{humanize(check.formula)}</span>
                <span className="review-check-equation">{checkText(check)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {suggestion && suggestedDraft !== null && (
        <div className="review-suggestion">
          <span>
            Suggested: {pathLabel(suggestion.path)} {formatAmount(suggestion.value)} - one
            change that makes every check pass
          </span>
          <button className="btn" disabled={busy} onClick={usePrefill}>
            Use {formatAmount(suggestion.value)}
          </button>
        </div>
      )}

      {correcting && (
        <div className="review-correct-form">
          {item.field_name === "line_items" ? (
            <textarea
              className="review-input review-textarea"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={8}
              spellCheck={false}
              disabled={busy}
            />
          ) : (
            <input
              className="review-input"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Corrected value (leave empty for none)"
              disabled={busy}
              onKeyDown={(e) => {
                if (e.key === "Enter") void submitCorrection();
              }}
            />
          )}
        </div>
      )}
      {/* Mounted before it has text, so a screen reader announces the change. */}
      <div className="review-prefill-note" aria-live="polite">
        {correcting ? prefilled : ""}
      </div>

      <div className="review-actions">
        {!correcting && (
          <>
            <button
              className="btn btn-primary"
              disabled={busy}
              aria-describedby={item.checks.length > 0 ? hintId : undefined}
              onClick={() => void resolve({ action: "approve" })}
            >
              Approve
            </button>
            <button
              className="btn"
              disabled={busy}
              onClick={() => {
                setDraft(initialDraft(item));
                setPrefilled("");
                setCorrecting(true);
              }}
            >
              Correct…
            </button>
          </>
        )}
        {correcting && (
          <>
            <button
              className="btn btn-primary"
              disabled={busy}
              onClick={() => void submitCorrection()}
            >
              Save correction
            </button>
            <button
              className="btn"
              disabled={busy}
              onClick={() => {
                setCorrecting(false);
                setPrefilled("");
                onError(null);
              }}
            >
              Cancel
            </button>
          </>
        )}
      </div>
      {!correcting && item.checks.length > 0 && (
        <p className="review-hint" id={hintId}>
          Approving keeps {item.checks.length === 1 ? "this check" : "these checks"} failing and
          sends the other amounts in {item.checks.length === 1 ? "it" : "them"} to review.
          If the value is wrong, correct it instead.
        </p>
      )}
    </div>
  );
}

export default function ReviewPage() {
  const [items, setItems] = useState<ReviewQueueItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const fetched = await getReviewQueue();
      setItems(fetched);
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `API error: ${err.message}`
          : "Could not reach the API. Is the backend running?"
      );
    }
  }, []);

  useEffect(() => {
    // Initial load only, no polling here: the queue only changes when an
    // extraction lands or a reviewer acts, and a stale entry resolves to a
    // clean 409 + refresh rather than corrupting anything.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const handleResolved = (fieldId: string) => {
    setItems((prev) =>
      prev === null ? prev : prev.filter((i) => i.field_id !== fieldId)
    );
    // Resolving a field can clear other fields of the same document or flag
    // new ones, so the queue is read again.
    void refresh();
  };

  return (
    <div>
      <div className="section-heading">Review queue</div>
      <p className="review-intro">
        Fields the model was unsure of, or whose amounts do not add up.
        Approve the value if it matches the document, or correct it. The
        original extraction is kept alongside your correction either way.
      </p>

      {error && <div className="error-banner">{error}</div>}

      {items === null && !error && <div className="empty-state">Loading…</div>}

      {items !== null && items.length === 0 && (
        <div className="empty-state">
          Nothing to review: every field cleared the confidence threshold and
          the amounts add up.
        </div>
      )}

      {items !== null && items.length > 0 && (
        <div className="review-list">
          {items.map((item) => (
            <ReviewRow
              key={item.field_id}
              item={item}
              onResolved={handleResolved}
              onError={setError}
            />
          ))}
        </div>
      )}
    </div>
  );
}
