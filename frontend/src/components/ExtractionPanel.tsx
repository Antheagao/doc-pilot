import type { Extraction, ExtractedField, FieldLeaf } from "@/lib/api";
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
  const valid = isLeaf(leaf);
  const low = !valid || leaf.confidence < 0.8;
  const display = valid ? formatScalar(leaf.value) : "—";
  return <td className={valid && low ? "cell-low" : undefined}>{display}</td>;
}

function LineItemsTable({ items }: { items: unknown[] }) {
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
            <tr key={i}>
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

function ScalarFieldRow({ field }: { field: ExtractedField }) {
  const leaf = isLeaf(field.value) ? field.value : null;
  const display = formatScalar(leaf?.value);
  return (
    <div className="field-row">
      <span className="field-name">{field.field_name.replace(/_/g, " ")}</span>
      <span className="field-value-wrap">
        <span className={`field-value${display === "—" ? " null-value" : ""}`}>
          {display}
        </span>
        <ConfidenceBadge confidence={field.confidence} needsReview={field.needs_review} />
      </span>
    </div>
  );
}

export default function ExtractionPanel({ extraction }: { extraction: Extraction }) {
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
  const lineItems = Array.isArray(lineItemsField?.value) ? lineItemsField.value : [];

  return (
    <div className="results-panel">
      <div className="field-list">
        {scalarFields.map((field) => (
          <ScalarFieldRow key={field.field_name} field={field} />
        ))}
      </div>

      {lineItemsField && (
        <div>
          <div className="field-row line-items-header">
            <span className="field-name">line items</span>
            <ConfidenceBadge
              confidence={lineItemsField.confidence}
              needsReview={lineItemsField.needs_review}
            />
          </div>
          <LineItemsTable items={lineItems} />
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
