"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import Link from "next/link";
import {
  getReviewQueue,
  resolveReviewField,
  ApiError,
  type FieldLeaf,
  type ReviewQueueItem,
} from "@/lib/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { evidenceAt } from "@/lib/evidence";
import {
  EMPTY_VALUE,
  MONEY_FIELDS,
  fieldLabel,
  formatAmount,
  formatEvidence,
  formatRate,
  sentenceCase,
} from "@/lib/format";
import {
  checkText,
  humanize,
  pathLabel,
  quoteReasonPath,
  suggestionWhy,
} from "@/lib/reconciliation";

// A plain decimal exactly as the API takes it: optional minus, digits,
// optional fraction. Commas and spaces are refused, never stripped: "45,25"
// would silently become 4525. Whether it fits the currency's digits is the
// backend's call (a 422 with a sentence, shown in the banner).
const PLAIN_DECIMAL = /^-?\d+(\.\d+)?$/;

// A tax rate is the printed percent: unsigned, at most this many decimal
// places and no more than 100. Mirrors parse_rate_input in the backend
// (app/money.py).
const PLAIN_RATE = /^(-?)(\d+)(?:\.(\d+))?$/;
const MAX_RATE_DECIMALS = 6;
const MAX_RATE_PERCENT = 100;

// The last four digits of a card, exactly (app/pii.py CARD_LAST4_RE).
const CARD_LAST4 = /^[0-9]{4}$/;

const PLAIN_AMOUNT_HELP =
  "Enter a plain number such as 45.25, with a dot for decimals and no " +
  "thousands separators (or leave it empty for none).";

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

/** The tax lines editor's row: every cell is what the reviewer typed. */
interface TaxLineRow {
  label: string;
  rate: string;
  base: string;
  amount: string;
}

const TAX_LINE_CELLS = ["label", "rate", "base", "amount"] as const;

function cellText(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

/** The editor's rows for a queue item's stored tax lines. */
function initialTaxRows(item: ReviewQueueItem): TaxLineRow[] {
  const items: unknown[] = Array.isArray(item.value) ? item.value : [];
  return items.map((entry) => {
    const row = typeof entry === "object" && entry !== null ? (entry as Record<string, unknown>) : {};
    return {
      label: cellText(leafValue(row.label)),
      rate: cellText(leafValue(row.rate)),
      base: cellText(leafValue(row.base)),
      amount: cellText(leafValue(row.amount)),
    };
  });
}

/** The tax lines the editor holds as the plain rows the backend stores
 * (label, rate, base, amount), checked the way normalize_correction checks
 * them. An empty cell is none; a row with every cell empty is no row. Amounts
 * stay strings; whether they fit the currency's digits is the backend's call.
 * Throws with a user-facing message naming the line and cell. */
function parseTaxRows(rows: TaxLineRow[]): unknown[] | null {
  const parsed = rows.flatMap((row, index) => {
    if (TAX_LINE_CELLS.every((cell) => row[cell].trim() === "")) return [];
    const line = `Tax line ${index + 1}`;
    const cell = (key: keyof TaxLineRow) => (row[key].trim() === "" ? null : row[key].trim());
    const rate = cell("rate");
    if (rate !== null) {
      const match = PLAIN_RATE.exec(rate);
      if (!match) {
        throw new Error(
          `${line} rate: write the printed percent as a plain number, such as 8.875 for 8.875%.`
        );
      }
      // Like parse_rate_input: zeros past the sixth place and leading zeros
      // count for nothing ("0100" is 100), and "-0" is 0.
      const negative = match[1] === "-";
      const whole = match[2].replace(/^0+(?=\d)/, "");
      const fraction = (match[3] ?? "").replace(/0+$/, "");
      if (fraction.length > MAX_RATE_DECIMALS) {
        throw new Error(`${line} rate: at most ${MAX_RATE_DECIMALS} decimal places.`);
      }
      const isZero = whole === "0" && fraction === "";
      // Only the whole part is ever a Number, and a few digits at that.
      const over =
        (negative && !isZero) ||
        whole.length > 3 ||
        Number(whole) > MAX_RATE_PERCENT ||
        (Number(whole) === MAX_RATE_PERCENT && fraction !== "");
      if (over) {
        throw new Error(`${line} rate: between 0 and ${MAX_RATE_PERCENT}, the printed percent.`);
      }
    }
    for (const key of ["base", "amount"] as const) {
      const value = cell(key);
      if (value !== null && !PLAIN_DECIMAL.test(value)) {
        throw new Error(`${line} ${key}: ${PLAIN_AMOUNT_HELP}`);
      }
    }
    return [{ label: cell("label"), rate, base: cell("base"), amount: cell("amount") }];
  });
  return parsed.length === 0 ? null : parsed;
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
  if (MONEY_FIELDS.has(fieldName)) {
    if (!PLAIN_DECIMAL.test(trimmed)) throw new Error(PLAIN_AMOUNT_HELP);
    return trimmed;
  }
  if (fieldName === "payment_card_last4" && !CARD_LAST4.test(trimmed)) {
    throw new Error("Enter exactly the last four digits of the card, such as 4242 (or leave it empty).");
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

// A tax line amount suggestion: "tax_lines[0].amount" names the row to change.
const TAX_LINE_AMOUNT_PATH = /^tax_lines\[(\d+)\]\.amount$/;

/** The tax lines editor's rows with a suggestion applied, or null when the
 * suggestion does not target this field's amounts (or names a row that is not
 * there). Only that row's amount changes; anything typed so far is kept. */
function taxRowsWithSuggestion(
  item: ReviewQueueItem,
  suggestion: NonNullable<ReviewQueueItem["suggestion"]>,
  currentRows: TaxLineRow[] | null
): TaxLineRow[] | null {
  const match = TAX_LINE_AMOUNT_PATH.exec(suggestion.path);
  if (item.field_name !== "tax_lines" || !match) return null;
  const rows = currentRows ?? initialTaxRows(item);
  const index = Number(match[1]);
  if (index >= rows.length) return null;
  return rows.map((row, i) => (i === index ? { ...row, amount: suggestion.value } : row));
}

function formatDisplayValue(item: ReviewQueueItem): string {
  if (item.field_name === "line_items") {
    const items = Array.isArray(item.value) ? item.value : [];
    return `${items.length} line item${items.length === 1 ? "" : "s"}`;
  }
  if (item.field_name === "tax_lines") {
    const items = Array.isArray(item.value) ? item.value : [];
    return `${items.length} tax line${items.length === 1 ? "" : "s"}`;
  }
  const value = leafValue(item.value);
  if (value === null || value === undefined || value === "") return EMPTY_VALUE;
  // The API pads amounts to the document currency's digits; the currency is
  // shown beside the amount so a misread one is as visible as a misread digit.
  if (MONEY_FIELDS.has(item.field_name)) {
    return item.currency
      ? `${formatAmount(value)} ${item.currency}`
      : formatAmount(value);
  }
  return String(value);
}

function TaxLineSummary({ item }: { item: ReviewQueueItem }) {
  const rows = initialTaxRows(item);
  if (rows.length === 0) return null;
  return (
    <ul className="review-evidence">
      {rows.map((row, i) => (
        <li key={i}>
          Tax line {i + 1}: {row.label || EMPTY_VALUE}
          {row.rate && row.base
            ? `, ${formatRate(row.rate)} of ${formatAmount(row.base)} =`
            : ","}{" "}
          {formatAmount(row.amount || null)}
        </li>
      ))}
    </ul>
  );
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
  // The tax lines editor's rows, and the row just added (its label gets the
  // focus when it mounts).
  const [taxRows, setTaxRows] = useState<TaxLineRow[]>([]);
  const [focusRow, setFocusRow] = useState<number | null>(null);
  const addRowRef = useRef<HTMLButtonElement>(null);
  // Focus follows the editor: opening it (Correct..., or Use) moves focus into
  // its first input with the text selected, because the button that was
  // pressed may unmount; Cancel gives focus back to Correct....
  const [focusRequest, setFocusRequest] = useState(0);
  const editorRef = useRef<HTMLDivElement>(null);
  const correctRef = useRef<HTMLButtonElement>(null);
  const returnFocus = useRef(false);
  // A selector for the input Use should focus, when it is not the first.
  const focusTarget = useRef<string | null>(null);
  const [busy, setBusy] = useState(false);
  // What the last "Use ..." click filled in, said once in a live region.
  const [prefilled, setPrefilled] = useState("");
  const hintId = `review-hint-${item.field_id}`;
  const isTaxLines = item.field_name === "tax_lines";
  const lowConfidence = `Low confidence (${Math.round(item.confidence * 100)}%)`;
  // The path of the value whose quote disagrees with it, when that is why the
  // field is here, and the page and quote it was read from.
  const quotedPath = quoteReasonPath(item.review_reason);
  const quoted = quotedPath === null ? undefined : evidenceAt(item.value, quotedPath);
  const onlyQuoteChecks = item.checks.length > 0 && item.checks.every((c) => c.kind === "quote");
  const reasonText =
    quotedPath !== null
      ? item.checks.length > 0
        ? "The value disagrees with the number printed in its quote:"
        : "The value disagrees with the number printed in its quote."
      : item.review_reason !== "confidence"
        ? item.checks.length > 0
          ? "These amounts do not add up:"
          : "Flagged by the arithmetic check."
        : item.checks.length > 0
          ? `${lowConfidence}, and these amounts do not add up:`
          : lowConfidence;
  const suggestion = item.suggestion;
  const suggestedDraft =
    suggestion && !isTaxLines
      ? draftWithSuggestion(item, suggestion, correcting ? draft : null)
      : null;
  const suggestedRows =
    suggestion && isTaxLines
      ? taxRowsWithSuggestion(item, suggestion, correcting ? taxRows : null)
      : null;
  const canUseSuggestion = suggestedDraft !== null || suggestedRows !== null;
  // A tax read once and held twice: the suggestion names `tax`, and the tax
  // line holding the same number follows in its own review afterwards.
  const pairedTax = suggestion?.path === "tax" && suggestion.detail.includes("tax line");

  useEffect(() => {
    if (focusRequest === 0) return;
    // The row a suggestion changed (its amount), else the first input; a tax
    // lines editor with no rows has only its Add button to take the focus.
    const editor = editorRef.current;
    const field =
      (focusTarget.current && editor?.querySelector<HTMLInputElement>(focusTarget.current)) ||
      editor?.querySelector<HTMLInputElement | HTMLTextAreaElement>("input, textarea");
    if (field) {
      field.focus();
      field.select();
    } else {
      addRowRef.current?.focus();
    }
  }, [focusRequest]);

  useEffect(() => {
    if (correcting || !returnFocus.current) return;
    returnFocus.current = false;
    correctRef.current?.focus();
  }, [correcting]);

  const usePrefill = () => {
    if (suggestion === null || !canUseSuggestion) return;
    if (suggestedRows !== null) setTaxRows(suggestedRows);
    else if (suggestedDraft !== null) setDraft(suggestedDraft);
    const row = TAX_LINE_AMOUNT_PATH.exec(suggestion.path);
    focusTarget.current = row ? `[aria-label="Tax line ${Number(row[1]) + 1} amount"]` : null;
    setCorrecting(true);
    setFocusRequest((n) => n + 1);
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
      corrected = isTaxLines ? parseTaxRows(taxRows) : parseDraft(item.field_name, draft);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Invalid correction.");
      return;
    }
    await resolve({ action: "correct", corrected_value: corrected });
  };

  const setTaxCell = (index: number, key: keyof TaxLineRow, value: string) =>
    setTaxRows((rows) => rows.map((row, i) => (i === index ? { ...row, [key]: value } : row)));

  const submitOnEnter = (e: KeyboardEvent) => {
    if (e.key === "Enter") void submitCorrection();
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
          <span className="review-field-name">{fieldLabel(item.field_name)}</span>
        </div>
        <div className="review-row-value">
          <span className="field-value">{formatDisplayValue(item)}</span>
          <ConfidenceBadge confidence={item.confidence} needsReview />
        </div>
        {quotedPath !== null && (
          <p className="review-evidence">
            {quoted
              ? `${sentenceCase(pathLabel(quotedPath))} read from ${formatEvidence(quoted)}`
              : `${sentenceCase(pathLabel(quotedPath))} has no quote to read from.`}
          </p>
        )}
        {isTaxLines && !correcting && <TaxLineSummary item={item} />}
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

      {suggestion && canUseSuggestion && (
        <div className="review-suggestion">
          <span>
            Suggested: {pathLabel(suggestion.path)} {formatAmount(suggestion.value)} -{" "}
            {suggestionWhy(suggestion)}
          </span>
          <button className="btn" disabled={busy} onClick={usePrefill}>
            Use {formatAmount(suggestion.value)}
          </button>
        </div>
      )}
      {pairedTax && !correcting && (
        <p className="review-hint">
          A tax line holds the same number. Applying this corrects the tax only; the next check
          then suggests the tax line.
        </p>
      )}

      {correcting && (
        <div className="review-correct-form" ref={editorRef}>
          {isTaxLines ? (
            <>
              <table className="review-tax-lines">
                <thead>
                  <tr>
                    <th scope="col">Label</th>
                    <th scope="col">Rate (%)</th>
                    <th scope="col">Base</th>
                    <th scope="col">Amount</th>
                    <th scope="col">
                      <span className="visually-hidden">Remove</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {taxRows.map((row, i) => (
                    <tr key={i}>
                      {TAX_LINE_CELLS.map((key) => (
                        <td key={key} data-label={key === "rate" ? "Rate (%)" : fieldLabel(key)}>
                          <input
                            className="review-input"
                            value={row[key]}
                            onChange={(e) => setTaxCell(i, key, e.target.value)}
                            aria-label={`Tax line ${i + 1} ${key}`}
                            inputMode={key === "label" ? "text" : "decimal"}
                            autoComplete="off"
                            spellCheck={false}
                            autoFocus={key === "label" && focusRow === i}
                            disabled={busy}
                            onKeyDown={submitOnEnter}
                          />
                        </td>
                      ))}
                      <td>
                        <button
                          type="button"
                          className="btn"
                          disabled={busy}
                          aria-label={`Remove tax line ${i + 1}`}
                          onClick={() => {
                            setTaxRows((rows) => rows.filter((_, j) => j !== i));
                            setFocusRow(null);
                            addRowRef.current?.focus();
                          }}
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="review-tax-lines-actions">
                <button
                  type="button"
                  ref={addRowRef}
                  className="btn"
                  disabled={busy}
                  onClick={() => {
                    setFocusRow(taxRows.length);
                    setTaxRows((rows) => [...rows, { label: "", rate: "", base: "", amount: "" }]);
                  }}
                >
                  Add tax line
                </button>
              </div>
            </>
          ) : item.field_name === "line_items" ? (
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
              onKeyDown={submitOnEnter}
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
              ref={correctRef}
              className="btn"
              disabled={busy}
              onClick={() => {
                if (isTaxLines) setTaxRows(initialTaxRows(item));
                else setDraft(initialDraft(item));
                setFocusRow(null);
                setPrefilled("");
                focusTarget.current = null;
                setCorrecting(true);
                setFocusRequest((n) => n + 1);
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
                returnFocus.current = true;
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
          {onlyQuoteChecks
            ? "Approving keeps this value as it is and stops checking it against its quote. If the value is wrong, correct it instead."
            : `Approving keeps ${item.checks.length === 1 ? "this check" : "these checks"} failing and sends the other amounts in ${item.checks.length === 1 ? "it" : "them"} to review. If the value is wrong, correct it instead.`}
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
