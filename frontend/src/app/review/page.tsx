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

// Fields whose corrections should be parsed as numbers rather than kept
// as strings. Mirrors the backend extraction schema (app/extraction.py).
const NUMERIC_FIELDS = new Set(["subtotal", "tax", "total"]);

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
      description: leafValue(row.description),
      quantity: leafValue(row.quantity),
      unit_price: leafValue(row.unit_price),
      total: leafValue(row.total),
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
    const n = Number(trimmed);
    if (Number.isNaN(n)) {
      throw new Error(`"${fieldName}" must be a number (or empty for none).`);
    }
    return n;
  }
  return trimmed;
}

function formatDisplayValue(item: ReviewQueueItem): string {
  if (item.field_name === "line_items") {
    const items = Array.isArray(item.value) ? item.value : [];
    return `${items.length} line item${items.length === 1 ? "" : "s"}`;
  }
  const value = leafValue(item.value);
  if (value === null || value === undefined || value === "") return "—";
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

      <div className="review-actions">
        {!correcting && (
          <>
            <button
              className="btn btn-primary"
              disabled={busy}
              onClick={() => void resolve({ action: "approve" })}
            >
              Approve
            </button>
            <button
              className="btn"
              disabled={busy}
              onClick={() => {
                setDraft(initialDraft(item));
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
                onError(null);
              }}
            >
              Cancel
            </button>
          </>
        )}
      </div>
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
    // Initial load only — no polling here: the queue only changes when an
    // extraction lands or a reviewer acts, and a stale entry resolves to a
    // clean 409 + refresh rather than corrupting anything.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const handleResolved = (fieldId: string) => {
    setItems((prev) =>
      prev === null ? prev : prev.filter((i) => i.field_id !== fieldId)
    );
  };

  return (
    <div>
      <div className="section-heading">Review queue</div>
      <p className="review-intro">
        Fields the model extracted with low confidence. Approve the value if
        it matches the document, or correct it — the original extraction is
        kept alongside your correction either way.
      </p>

      {error && <div className="error-banner">{error}</div>}

      {items === null && !error && <div className="empty-state">Loading…</div>}

      {items !== null && items.length === 0 && (
        <div className="empty-state">
          Nothing to review — every extracted field cleared the confidence
          threshold.
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
