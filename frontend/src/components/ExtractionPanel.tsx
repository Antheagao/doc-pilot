"use client";

import { useState, type ReactNode } from "react";
import type {
  Evidence,
  Extraction,
  ExtractedField,
  FieldLeaf,
  ReviewAction,
} from "@/lib/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import ReconciliationPanel from "@/components/ReconciliationPanel";
import { evidenceAt, hasEvidence, leafEvidence } from "@/lib/evidence";
import {
  EMPTY_VALUE,
  MONEY_FIELDS,
  fieldLabel,
  formatAmount,
  formatDuration,
  formatEvidence,
  formatRate,
  sentenceCase,
} from "@/lib/format";
import { fieldNameOfPath, quotePath } from "@/lib/reconciliation";

// Display order for the scalar fields of a schema v1 extraction; line_items is
// rendered separately as a table. Any field not listed here (there shouldn't
// be any today, but a future prompt version could add one) still renders,
// just after these (see the extraFields computation below).
const SCALAR_FIELD_ORDER = [
  "vendor",
  "document_date",
  "subtotal",
  "tax",
  "total",
  "currency",
];

// A schema v2 extraction reads like the document: who and what it is, the
// lines, the amounts down to what is due, how it was paid. The tax lines are a
// table inside Amounts; the line items sit between the first two groups.
const FIELD_GROUPS = [
  {
    title: "Document",
    fields: [
      "vendor",
      "document_date",
      "invoice_number",
      "po_number",
      "vendor_tax_id",
      "due_date",
      "payment_terms",
    ],
  },
  {
    title: "Amounts",
    fields: [
      "subtotal",
      "discount",
      "shipping",
      "tax",
      "tax_lines",
      "tip",
      "total",
      "amount_paid",
      "amount_due",
      "currency",
    ],
  },
  { title: "Payment", fields: ["payment_method", "payment_card_last4"] },
] as const;

// Only a v2 extraction has any of these, so their presence picks the grouped
// layout and a v1 document keeps its flat list exactly as it was.
const V2_FIELDS: ReadonlySet<string> = new Set(
  FIELD_GROUPS.flatMap((group) => group.fields).filter((name) => !SCALAR_FIELD_ORDER.includes(name))
);

// What the quote check says about a value that disagrees with its quote.
const QUOTE_CHIP_TEXT = "Value disagrees with the printed number";

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

// The prompt's size, which prompt caching splits three ways (input_tokens is
// only the uncached remainder): "9,329 in (8,000 cached) / 2,964 out". An
// extraction with no cache tokens reads as it always did.
function formatPromptTokens(extraction: Extraction): string {
  const written = extraction.cache_creation_input_tokens;
  const read = extraction.cache_read_input_tokens;
  if (!written && !read) return `${formatNumber(extraction.input_tokens)} in`;
  const total = extraction.input_tokens + written + read;
  const parts = [
    read ? `${read.toLocaleString("en-US")} cached` : null,
    written ? `${written.toLocaleString("en-US")} cache write` : null,
  ].filter((part) => part !== null);
  return `${total.toLocaleString("en-US")} in (${parts.join(", ")})`;
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

/** What the panel needs to know besides the field itself: the values the
 * arithmetic computed, the values whose quote disagrees with them (by report
 * path), and whether the quotes are shown. */
interface PanelView {
  computed: ComputedValues;
  quoteFails: ReadonlySet<string>;
  showEvidence: boolean;
}

/** Marks a value whose evidence quote prints a different number. */
function QuoteChip() {
  return (
    <span className="quote-chip">
      <span aria-hidden="true">✕</span> {QUOTE_CHIP_TEXT}
    </span>
  );
}

/** Where a value was read, shown under it while the toggle is on. A leaf that
 * cites nothing says so. */
function EvidenceNote({ evidence }: { evidence: Evidence | null }) {
  return evidence ? (
    <span className="field-evidence">{formatEvidence(evidence)}</span>
  ) : (
    <span className="field-evidence none">no quote cited</span>
  );
}

// This project explicitly expects partially-malformed extraction payloads
// (bad tool-call output, a future schema change, ...) to render rather
// than crash the viewer, so every row of an array field and each of its cells
// is treated as `unknown` and validated at render time instead of trusted via
// the static LineItem and TaxLine types.
function ArrayCell({
  leaf,
  format = formatScalar,
  numeric = false,
  computed,
  flagged = false,
}: {
  leaf: unknown;
  /** How the value reads: formatAmount for money columns (grouped integer
   * part), formatRate for a tax rate. */
  format?: (value: unknown) => string;
  /** Right-aligns the cell, for the quantity, rate and money columns. */
  numeric?: boolean;
  /** The value the arithmetic check worked out for a cell with no printed
   * value; shown tagged "computed". */
  computed?: string;
  /** The cell's quote prints a different number than its value. */
  flagged?: boolean;
}) {
  // Corrected rows (review_action === "corrected") hold plain scalar values,
  // not {value, confidence} leaves (a human correction has no confidence
  // score). Render those directly, with no low styling.
  const plain = !isLeaf(leaf);
  const raw = plain ? leaf : leaf.value;
  const scalar = raw === null || typeof raw === "string" || typeof raw === "number";
  const shown = scalar || !plain ? format(raw) : EMPTY_VALUE;
  const low = !plain && leaf.confidence < 0.8;
  const className = [numeric && "num", low && "cell-low"].filter(Boolean).join(" ") || undefined;
  return (
    <td className={className}>
      {shown === EMPTY_VALUE && computed !== undefined ? <Computed value={computed} /> : shown}
      {flagged && <QuoteChip />}
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

interface ArrayColumn {
  key: string;
  title: string;
  numeric?: boolean;
  format?: (value: unknown) => string;
}

const LINE_ITEM_COLUMNS: ArrayColumn[] = [
  { key: "description", title: "Description" },
  { key: "quantity", title: "Qty", numeric: true },
  { key: "unit_price", title: "Unit price", numeric: true, format: formatAmount },
  { key: "total", title: "Total", numeric: true, format: formatAmount },
];

const TAX_LINE_COLUMNS: ArrayColumn[] = [
  { key: "label", title: "Label" },
  { key: "rate", title: "Rate", numeric: true, format: formatRate },
  { key: "base", title: "Base", numeric: true, format: formatAmount },
  { key: "amount", title: "Amount", numeric: true, format: formatAmount },
];

/** The cells of one row that cite the same text, so a line item whose four
 * cells were all read from one printed line shows that line once. */
function rowEvidence(row: Record<string, unknown>, columns: ArrayColumn[]) {
  const groups: { evidence: Evidence; cells: string[] }[] = [];
  for (const column of columns) {
    const evidence = leafEvidence(row[column.key]);
    if (!evidence) continue;
    const same = groups.find(
      (g) => g.evidence.page === evidence.page && g.evidence.quote === evidence.quote
    );
    if (same) same.cells.push(column.title.toLowerCase());
    else groups.push({ evidence, cells: [column.title.toLowerCase()] });
  }
  return groups;
}

/** A table of the rows of an array field (line items, tax lines). `name` is
 * the field's name, which row ids and quote paths are built from. */
function ArrayTable({
  name,
  columns,
  items,
  highlight,
  view,
  className,
}: {
  name: "line_items" | "tax_lines";
  columns: ArrayColumn[];
  items: unknown[];
  highlight: ReadonlySet<string>;
  view: PanelView;
  className?: string;
}) {
  return (
    <table className={className ? `line-items-table ${className}` : "line-items-table"}>
      <thead>
        <tr>
          {columns.map((column) => (
            <th key={column.key} className={column.numeric ? "num" : undefined}>
              {column.title}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {items.map((item, i) => {
          const row =
            typeof item === "object" && item !== null ? (item as Record<string, unknown>) : {};
          const evidence = view.showEvidence ? rowEvidence(row, columns) : [];
          return (
            <ArrayRow key={i} columns={columns} evidence={evidence}>
              <tr
                id={fieldElementId(`${name}[${i}]`)}
                className={highlight.has(`${name}[${i}]`) ? "cited" : undefined}
              >
                {columns.map((column) => (
                  <ArrayCell
                    key={column.key}
                    leaf={row[column.key]}
                    format={column.format}
                    numeric={column.numeric}
                    computed={view.computed.get(`${name}[${i}].${column.key}`)}
                    flagged={view.quoteFails.has(`${name}[${i}].${column.key}`)}
                  />
                ))}
              </tr>
            </ArrayRow>
          );
        })}
      </tbody>
    </table>
  );
}

/** A row and, while the quotes are shown, the line that says where its cells
 * were read. */
function ArrayRow({
  columns,
  evidence,
  children,
}: {
  columns: ArrayColumn[];
  evidence: ReturnType<typeof rowEvidence>;
  children: ReactNode;
}) {
  return (
    <>
      {children}
      {evidence.length > 0 && (
        <tr className="evidence-row">
          <td colSpan={columns.length}>
            {evidence.map((group) => (
              <span key={`${group.evidence.page}-${group.evidence.quote}`} className="field-evidence">
                {sentenceCase(group.cells.join(", "))}: {formatEvidence(group.evidence)}
              </span>
            ))}
          </td>
        </tr>
      )}
    </>
  );
}

function LineItemsTable({
  items,
  highlight,
  view,
  corrected,
}: {
  items: unknown[];
  highlight: ReadonlySet<string>;
  view: PanelView;
  /** The rows are a person's answer, not the model's extraction. */
  corrected: boolean;
}) {
  if (items.length === 0) {
    return (
      <div className="field-value null-value">
        {corrected ? "No line items" : "No line items extracted"}
      </div>
    );
  }
  return (
    <ArrayTable
      name="line_items"
      columns={LINE_ITEM_COLUMNS}
      items={items}
      highlight={highlight}
      view={view}
    />
  );
}

function ReviewBadge({ action }: { action: ReviewAction }) {
  return <span className={`review-badge ${action}`}>{action}</span>;
}

/** The rows an array field currently shows: a human correction's plain rows
 * when it was corrected, else the model's. */
function arrayRows(field: ExtractedField | undefined): unknown[] {
  if (!field) return [];
  const rows = field.review_action === "corrected" ? field.corrected_value : field.value;
  return Array.isArray(rows) ? rows : [];
}

/** A header for an array field's table: its name and review state. */
function ArrayHeader({
  field,
  label,
  highlight,
}: {
  field: ExtractedField;
  label: string;
  highlight: ReadonlySet<string>;
}) {
  return (
    <div
      id={fieldElementId(field.field_name)}
      className={`field-row line-items-header${highlight.has(field.field_name) ? " cited" : ""}`}
    >
      <span className="field-name">{label}</span>
      <span className="badge-row">
        <ConfidenceBadge
          confidence={field.confidence}
          needsReview={field.needs_review && field.review_action === null}
        />
        {field.review_action && <ReviewBadge action={field.review_action} />}
      </span>
    </div>
  );
}

function LineItemsBlock({
  field,
  highlight,
  view,
}: {
  field: ExtractedField;
  highlight: ReadonlySet<string>;
  view: PanelView;
}) {
  return (
    <div>
      <ArrayHeader field={field} label="line items" highlight={highlight} />
      <LineItemsTable
        items={arrayRows(field)}
        highlight={highlight}
        view={view}
        corrected={field.review_action === "corrected"}
      />
    </div>
  );
}

function TaxLinesBlock({
  field,
  highlight,
  view,
}: {
  field: ExtractedField;
  highlight: ReadonlySet<string>;
  view: PanelView;
}) {
  const rows = arrayRows(field);
  return (
    <div className="field-array">
      <ArrayHeader field={field} label="tax lines" highlight={highlight} />
      {rows.length === 0 ? (
        <div className="field-value null-value">No tax lines</div>
      ) : (
        <ArrayTable
          name="tax_lines"
          columns={TAX_LINE_COLUMNS}
          items={rows}
          highlight={highlight}
          view={view}
          className="tax-lines-table"
        />
      )}
    </div>
  );
}

function ScalarFieldRow({
  field,
  cited,
  view,
}: {
  field: ExtractedField;
  cited: boolean;
  view: PanelView;
}) {
  const leaf = isLeaf(field.value) ? field.value : null;
  const corrected = field.review_action === "corrected";
  const format = MONEY_FIELDS.has(field.field_name) ? formatAmount : formatScalar;
  const display = format(displayedValue(field));
  const computed = view.computed.get(field.field_name);
  // A person's value has no quote; the model's belongs to the value it read.
  const evidence = view.showEvidence && !corrected ? leafEvidence(field.value) : undefined;
  return (
    <div id={fieldElementId(field.field_name)} className={`field-row${cited ? " cited" : ""}`}>
      <span className="field-name">{fieldLabel(field.field_name)}</span>
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
        {evidence !== undefined && display !== EMPTY_VALUE && <EvidenceNote evidence={evidence} />}
        {view.quoteFails.has(field.field_name) && <QuoteChip />}
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

/** A scalar field the document does not carry: no value, nothing computed for
 * it, nobody waiting on it. Such a field folds into one muted line instead of
 * a row of dashes. */
function isAbsent(
  field: ExtractedField,
  computed: ComputedValues,
  highlight: ReadonlySet<string>
): boolean {
  // A person's answer stays visible, with its badge and what it replaced.
  if (field.review_action !== null) return false;
  if (field.needs_review) return false;
  if (computed.has(field.field_name) || highlight.has(field.field_name)) return false;
  return field.field_name === "tax_lines"
    ? arrayRows(field).length === 0
    : formatScalar(displayedValue(field)) === EMPTY_VALUE;
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
  const [showEvidence, setShowEvidence] = useState(false);
  const { reconciliation } = extraction;
  const computed: ComputedValues = reconciliation
    ? new Map(
        reconciliation.derived.flatMap((d) =>
          d.value === null ? [] : [[d.path, d.value] as const]
        )
      )
    : NO_COMPUTED;
  const quoteFails = new Set(
    (reconciliation?.checks ?? [])
      .filter((c) => c.kind === "quote" && c.status === "fail")
      .map(quotePath)
  );
  const view: PanelView = { computed, quoteFails, showEvidence };
  const byName = new Map(extraction.fields.map((f) => [f.field_name, f]));
  const grouped = extraction.fields.some((f) => V2_FIELDS.has(f.field_name));

  const row = (field: ExtractedField) => (
    <ScalarFieldRow
      key={field.field_name}
      field={field}
      cited={highlight.has(field.field_name)}
      view={view}
    />
  );

  // Anything not in the known scalar order and not an array field, e.g. a
  // field a future prompt version adds, still gets shown, after the ordered
  // ones, so a needs_review field never silently disappears.
  const knownNames = new Set([
    ...SCALAR_FIELD_ORDER,
    ...FIELD_GROUPS.flatMap((group) => group.fields),
    "line_items",
  ]);
  const extraFields = extraction.fields.filter((f) => !knownNames.has(f.field_name));

  const lineItemsField = byName.get("line_items");
  const lineItemsBlock = lineItemsField && (
    <LineItemsBlock field={lineItemsField} highlight={highlight} view={view} />
  );

  // The v2 layout: groups of rows, the null ones folded into a single line.
  const absent: string[] = [];
  const groups = grouped
    ? FIELD_GROUPS.map((group) => {
        const fields = group.fields.flatMap((name) => {
          const field = byName.get(name);
          if (!field) return [];
          if (isAbsent(field, computed, highlight)) {
            absent.push(fieldLabel(name));
            return [];
          }
          return [field];
        });
        return { title: group.title, fields };
      }).filter((group) => group.fields.length > 0)
    : [];
  const groupBlock = (title: string) => {
    const group = groups.find((g) => g.title === title);
    return (
      group && (
        <section className="field-group" aria-labelledby={`group-${title.toLowerCase()}`}>
          <h2 className="field-group-title" id={`group-${title.toLowerCase()}`}>
            {title}
          </h2>
          <div className="field-list">
            {group.fields.map((field) =>
              field.field_name === "tax_lines" ? (
                <TaxLinesBlock key="tax_lines" field={field} highlight={highlight} view={view} />
              ) : (
                row(field)
              )
            )}
          </div>
        </section>
      )
    );
  };

  const v1Fields = [
    ...SCALAR_FIELD_ORDER.flatMap((name) => byName.get(name) ?? []),
    ...extraFields,
  ];

  return (
    <div className="results-panel">
      {hasEvidence(extraction.fields.map((f) => f.value)) && (
        <div className="evidence-toolbar">
          <button
            type="button"
            className="btn evidence-toggle"
            aria-expanded={showEvidence}
            onClick={() => setShowEvidence((shown) => !shown)}
          >
            {showEvidence ? "Hide" : "Show"} where each value was read
          </button>
        </div>
      )}

      {grouped ? (
        <>
          {groupBlock("Document")}
          {lineItemsBlock && <div className="field-block">{lineItemsBlock}</div>}
          {groupBlock("Amounts")}
          {groupBlock("Payment")}
          {absent.length > 0 && (
            <p className="field-absent">Not on this document: {absent.join(", ")}</p>
          )}
          {extraFields.length > 0 && <div className="field-list">{extraFields.map(row)}</div>}
        </>
      ) : (
        <>
          <div className="field-list">{v1Fields.map(row)}</div>
          {lineItemsBlock}
        </>
      )}

      <ReconciliationPanel
        reconciliation={reconciliation}
        picked={picked}
        onPick={onPick}
        evidenceOf={(path) => {
          const field = byName.get(fieldNameOfPath(path));
          return field ? evidenceAt(field.value, path) : undefined;
        }}
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
            {formatPromptTokens(extraction)} /{" "}
            {formatNumber(extraction.output_tokens)} out
          </strong>
        </span>
      </div>
    </div>
  );
}
