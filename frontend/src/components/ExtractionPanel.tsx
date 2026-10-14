import type {
  Extraction,
  ExtractedField,
  FieldLeaf,
  ReviewAction,
} from "@/lib/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import ReconciliationPanel from "@/components/ReconciliationPanel";
import { EMPTY_VALUE, formatAmount, formatDuration } from "@/lib/format";

// Display order for the scalar fields; line_items is rendered separately
// as a table. Any field not listed here (there shouldn't be any today, but
// a future prompt version could add one) still renders, just after these
// (see the extraFields computation below).
const SCALAR_FIELD_ORDER = [
  "vendor",
  "document_date",
  "subtotal",
  "tax",
  "total",
  "currency",
];

// Scalar fields that hold an amount of money in the document currency.
const MONEY_FIELDS: ReadonlySet<string> = new Set(["subtotal", "tax", "total"]);

const NO_HIGHLIGHT: ReadonlySet<string> = new Set();

function noop() {}

// Values the document did not print that the arithmetic check worked out
// (an absent subtotal, an absent line total), by path: "subtotal",
// "line_items[2].total". They show only where the printed value is missing,
// always tagged, so a computed number never reads as an extracted one.
type ComputedValues = ReadonlyMap<string, string>;

const NO_COMPUTED: ComputedValues = new Map();

function ComputedTag() {
  return <span className="computed-tag">computed</span>;
}

/** The DOM id of a field's row, by the name a chat citation uses for it
 * ("total", "line_items[2]"), so the chat can scroll to it. */
export function fieldElementId(field: string): string {
  return `field-${field.replace(/\[(\d+)\]/, "-$1")}`;
}

function isLeaf(value: unknown): value is FieldLeaf {
  return (
    typeof value === "object" &&
    value !== null &&
    "value" in value &&
    "confidence" in value &&
    typeof (value as { confidence: unknown }).confidence === "number"
  );
}

function formatScalar(value: unknown): string {
  if (value === null || value === undefined || value === "") return EMPTY_VALUE;
  return String(value);
}

function formatNumber(value: unknown): string {
  return typeof value === "number" ? String(value) : EMPTY_VALUE;
}

function formatCost(value: unknown): string {
  return typeof value === "number" ? `$${value.toFixed(4)}` : EMPTY_VALUE;
}

function formatLatency(value: unknown): string {
  return typeof value === "number" ? formatDuration(value) : EMPTY_VALUE;
}

/** The value a field currently shows: the human correction when there is
 * one, else the model's extraction. */
function displayedValue(field: ExtractedField): unknown {
  if (field.review_action === "corrected") return field.corrected_value;
  return isLeaf(field.value) ? field.value.value : null;
}

// This project explicitly expects partially-malformed extraction payloads
// (bad tool-call output, a future schema change, ...) to render rather
// than crash the viewer, so every line-item entry and sub-field is treated
// as `unknown` and validated at render time instead of trusted via the
// static LineItem type.
function LineItemCell({
  leaf,
  money = false,
  numeric = false,
  computed,
}: {
  leaf: unknown;
  /** Set for money columns, which group the integer part (formatAmount). */
  money?: boolean;
  /** Right-aligns the cell, for the quantity and money columns. */
  numeric?: boolean;
  /** The value the arithmetic check worked out for a cell with no printed
   * value; shown tagged "computed". */
  computed?: string;
}) {
  const format = money ? formatAmount : formatScalar;
  // Corrected line-item rows (review_action === "corrected") hold plain
  // scalar values, not {value, confidence} leaves (a human correction
  // has no confidence score). Render those directly, with no low styling.
  if (!isLeaf(leaf)) {
    const scalar =
      leaf === null || typeof leaf === "string" || typeof leaf === "number";
    const shown = scalar ? format(leaf) : EMPTY_VALUE;
    return (
      <td className={numeric ? "num" : undefined}>
        {shown === EMPTY_VALUE && computed !== undefined ? <Computed value={computed} /> : shown}
      </td>
    );
  }
  const low = leaf.confidence < 0.8;
  const shown = format(leaf.value);
  return (
    <td className={[numeric && "num", low && "cell-low"].filter(Boolean).join(" ") || undefined}>
      {shown === EMPTY_VALUE && computed !== undefined ? <Computed value={computed} /> : shown}
    </td>
  );
}

/** A computed amount in a cell or row: the tag, then the number. */
function Computed({ value }: { value: string }) {
  return (
    <>
      <ComputedTag /> {formatAmount(value)}
    </>
  );
}

function LineItemsTable({
  items,
  highlight,
  computed,
}: {
  items: unknown[];
  highlight: ReadonlySet<string>;
  computed: ComputedValues;
}) {
  if (items.length === 0) {
    return <div className="field-value null-value">No line items extracted</div>;
  }
  return (
    <table className="line-items-table">
      <thead>
        <tr>
          <th>Description</th>
          <th className="num">Qty</th>
          <th className="num">Unit price</th>
          <th className="num">Total</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, i) => {
          const row =
            typeof item === "object" && item !== null
              ? (item as Record<string, unknown>)
              : {};
          return (
            <tr
              key={i}
              id={fieldElementId(`line_items[${i}]`)}
              className={highlight.has(`line_items[${i}]`) ? "cited" : undefined}
            >
              <LineItemCell leaf={row.description} />
              <LineItemCell leaf={row.quantity} numeric />
              <LineItemCell leaf={row.unit_price} money numeric />
              <LineItemCell
                leaf={row.total}
                money
                numeric
                computed={computed.get(`line_items[${i}].total`)}
              />
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ReviewBadge({ action }: { action: ReviewAction }) {
  return <span className={`review-badge ${action}`}>{action}</span>;
}

function ScalarFieldRow({
  field,
  cited,
  computed,
}: {
  field: ExtractedField;
  cited: boolean;
  /** The arithmetic check's value for this field when it printed none. */
  computed?: string;
}) {
  const leaf = isLeaf(field.value) ? field.value : null;
  const corrected = field.review_action === "corrected";
  const format = MONEY_FIELDS.has(field.field_name) ? formatAmount : formatScalar;
  const display = format(displayedValue(field));
  return (
    <div id={fieldElementId(field.field_name)} className={`field-row${cited ? " cited" : ""}`}>
      <span className="field-name">{field.field_name.replace(/_/g, " ")}</span>
      <span className="field-value-wrap">
        <span className={`field-value${display === EMPTY_VALUE ? " null-value" : ""}`}>
          {display}
        </span>
        {display === EMPTY_VALUE && computed !== undefined && (
          <span className="field-computed">
            <Computed value={computed} />
          </span>
        )}
        {corrected && (
          <span className="field-original">was {format(leaf?.value)}</span>
        )}
        <span className="badge-row">
          <ConfidenceBadge
            confidence={field.confidence}
            // Once a human has resolved the field, stop styling it as a
            // pending low-confidence warning.
            needsReview={field.needs_review && field.review_action === null}
          />
          {field.review_action && <ReviewBadge action={field.review_action} />}
        </span>
      </span>
    </div>
  );
}

/** `highlight` names the fields a selected chat citation or reconciliation
 * operand points at, as the citation names them ("total", "line_items[2]").
 * `picked` is the one operand field the reconciliation panel has selected,
 * and `onPick` is called when an operand there is chosen. */
export default function ExtractionPanel({
  extraction,
  highlight = NO_HIGHLIGHT,
  picked = null,
  onPick = noop,
}: {
  extraction: Extraction;
  highlight?: ReadonlySet<string>;
  picked?: string | null;
  onPick?: (path: string) => void;
}) {
  const computed: ComputedValues = extraction.reconciliation
    ? new Map(
        extraction.reconciliation.derived.flatMap((d) =>
          d.value === null ? [] : [[d.path, d.value] as const]
        )
      )
    : NO_COMPUTED;
  const byName = new Map(extraction.fields.map((f) => [f.field_name, f]));
  const orderedFields = SCALAR_FIELD_ORDER.map((name) => byName.get(name)).filter(
    (f): f is ExtractedField => f !== undefined
  );
  // Anything not in the known scalar order and not line_items, e.g. a
  // field a future prompt version adds, still gets shown, after the
  // ordered ones, so a needs_review field never silently disappears.
  const knownNames = new Set([...SCALAR_FIELD_ORDER, "line_items"]);
  const extraFields = extraction.fields.filter((f) => !knownNames.has(f.field_name));
  const scalarFields = [...orderedFields, ...extraFields];

  const lineItemsField = byName.get("line_items");
  // A corrected line_items field displays the human-supplied rows (plain
  // values, see LineItemCell) instead of the model's extraction.
  const lineItemsCorrected =
    lineItemsField?.review_action === "corrected" &&
    Array.isArray(lineItemsField.corrected_value);
  const lineItems = lineItemsCorrected
    ? (lineItemsField.corrected_value as unknown[])
    : Array.isArray(lineItemsField?.value)
      ? lineItemsField.value
      : [];

  return (
    <div className="results-panel">
      <div className="field-list">
        {scalarFields.map((field) => (
          <ScalarFieldRow
            key={field.field_name}
            field={field}
            cited={highlight.has(field.field_name)}
            computed={computed.get(field.field_name)}
          />
        ))}
      </div>

      {lineItemsField && (
        <div>
          <div
            id={fieldElementId("line_items")}
            className={`field-row line-items-header${highlight.has("line_items") ? " cited" : ""}`}
          >
            <span className="field-name">line items</span>
            <span className="badge-row">
              <ConfidenceBadge
                confidence={lineItemsField.confidence}
                needsReview={
                  lineItemsField.needs_review &&
                  lineItemsField.review_action === null
                }
              />
              {lineItemsField.review_action && (
                <ReviewBadge action={lineItemsField.review_action} />
              )}
            </span>
          </div>
          <LineItemsTable items={lineItems} highlight={highlight} computed={computed} />
        </div>
      )}

      <ReconciliationPanel
        reconciliation={extraction.reconciliation}
        picked={picked}
        onPick={onPick}
      />

      <div className="footer-strip">
        <span>
          model <strong>{formatScalar(extraction.model)}</strong>
        </span>
        <span>
          prompt <strong>{formatScalar(extraction.prompt_version)}</strong>
        </span>
        <span>
          cost <strong>{formatCost(extraction.cost_usd)}</strong>
        </span>
        <span>
          latency <strong>{formatLatency(extraction.latency_ms)}</strong>
        </span>
        <span>
          tokens{" "}
          <strong>
            {formatNumber(extraction.input_tokens)} in /{" "}
            {formatNumber(extraction.output_tokens)} out
          </strong>
        </span>
      </div>
    </div>
  );
}
