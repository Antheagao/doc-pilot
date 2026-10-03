import type {
  Extraction,
  ExtractedField,
  FieldLeaf,
  ReviewAction,
} from "@/lib/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";

// Display order for the scalar fields; line_items is rendered separately
// as a table. Any field not listed here (there shouldn't be any today, but
// a future prompt version could add one) still renders, just after these
// — see the extraFields computation below.
const SCALAR_FIELD_ORDER = [
  "vendor",
  "document_date",
  "subtotal",
  "tax",
  "total",
  "currency",
];

const NO_HIGHLIGHT: ReadonlySet<string> = new Set();

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
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function formatNumber(value: unknown): string {
  return typeof value === "number" ? String(value) : "—";
}

function formatCost(value: unknown): string {
  return typeof value === "number" ? `$${value.toFixed(4)}` : "—";
}

function formatLatency(value: unknown): string {
  return typeof value === "number" ? `${value}ms` : "—";
}

// This project explicitly expects partially-malformed extraction payloads
// (bad tool-call output, a future schema change, ...) to render rather
// than crash the viewer, so every line-item entry and sub-field is treated
// as `unknown` and validated at render time instead of trusted via the
// static LineItem type.
function LineItemCell({ leaf }: { leaf: unknown }) {
  // Corrected line-item rows (review_action === "corrected") hold plain
  // scalar values, not {value, confidence} leaves — a human correction
  // has no confidence score. Render those directly, with no low styling.
  if (!isLeaf(leaf)) {
    const scalar =
      leaf === null || typeof leaf === "string" || typeof leaf === "number";
    return <td>{scalar ? formatScalar(leaf) : "—"}</td>;
  }
  const low = leaf.confidence < 0.8;
  return (
    <td className={low ? "cell-low" : undefined}>{formatScalar(leaf.value)}</td>
  );
}

function LineItemsTable({
  items,
  highlight,
}: {
  items: unknown[];
  highlight: ReadonlySet<string>;
}) {
  if (items.length === 0) {
    return <div className="field-value null-value">No line items extracted</div>;
  }
  return (
    <table className="line-items-table">
      <thead>
        <tr>
          <th>Description</th>
          <th>Qty</th>
          <th>Unit price</th>
          <th>Total</th>
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
              <LineItemCell leaf={row.quantity} />
              <LineItemCell leaf={row.unit_price} />
              <LineItemCell leaf={row.total} />
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

function ScalarFieldRow({ field, cited }: { field: ExtractedField; cited: boolean }) {
  const leaf = isLeaf(field.value) ? field.value : null;
  const corrected = field.review_action === "corrected";
  const display = formatScalar(corrected ? field.corrected_value : leaf?.value);
  return (
    <div id={fieldElementId(field.field_name)} className={`field-row${cited ? " cited" : ""}`}>
      <span className="field-name">{field.field_name.replace(/_/g, " ")}</span>
      <span className="field-value-wrap">
        <span className={`field-value${display === "—" ? " null-value" : ""}`}>
          {display}
        </span>
        {corrected && (
          <span className="field-original">was {formatScalar(leaf?.value)}</span>
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

/** `highlight` names the fields a selected chat citation points at, as
 * the citation names them ("total", "line_items[2]"). */
export default function ExtractionPanel({
  extraction,
  highlight = NO_HIGHLIGHT,
}: {
  extraction: Extraction;
  highlight?: ReadonlySet<string>;
}) {
  const byName = new Map(extraction.fields.map((f) => [f.field_name, f]));
  const orderedFields = SCALAR_FIELD_ORDER.map((name) => byName.get(name)).filter(
    (f): f is ExtractedField => f !== undefined
  );
  // Anything not in the known scalar order and not line_items — e.g. a
  // field a future prompt version adds — still gets shown, after the
  // ordered ones, so a needs_review field never silently disappears.
  const knownNames = new Set([...SCALAR_FIELD_ORDER, "line_items"]);
  const extraFields = extraction.fields.filter((f) => !knownNames.has(f.field_name));
  const scalarFields = [...orderedFields, ...extraFields];

  const lineItemsField = byName.get("line_items");
  // A corrected line_items field displays the human-supplied rows (plain
  // values — see LineItemCell) instead of the model's extraction.
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
          />
        ))}
      </div>

      {lineItemsField && (
        <div>
          <div className="field-row line-items-header">
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
          <LineItemsTable items={lineItems} highlight={highlight} />
        </div>
      )}

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
