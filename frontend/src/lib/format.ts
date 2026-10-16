/** Shared display formatters, so one value reads the same on every page. */

/** What a missing value renders as. */
export const EMPTY_VALUE = "-";

/** The extracted fields that hold an amount of money in the document currency
 * (the money fields of the backend's extraction schema, SCHEMA_V2 in
 * app/extraction_fields.py): the panel groups their digits and the review form
 * takes them as plain decimals. */
export const MONEY_FIELDS: ReadonlySet<string> = new Set([
  "subtotal",
  "discount",
  "shipping",
  "tax",
  "tip",
  "total",
  "amount_paid",
  "amount_due",
]);

// How a person says a field; anything not listed is its name with spaces.
const FIELD_LABELS: Readonly<Record<string, string>> = {
  po_number: "PO number",
  vendor_tax_id: "vendor tax ID",
  payment_terms: "terms",
  payment_card_last4: "card last 4",
};

/** A field's name as a person reads it: "document date", "PO number". */
export function fieldLabel(fieldName: string): string {
  return FIELD_LABELS[fieldName] ?? fieldName.replace(/_/g, " ");
}

const integerFormatter = new Intl.NumberFormat("en-US");

// What the API serves for a document amount or quantity: an optional minus
// sign, digits, an optional fraction (backend/app/money.py to_api_string).
const DECIMAL_STRING = /^(-?)(\d+)(\.\d+)?$/;

/** A document amount with grouping on the integer part ("1,234.50") and no
 * currency symbol: the currency is shown as its own field. The API already
 * pads the fraction to the currency's minor unit and keeps extra digits, so
 * the fraction is shown exactly as given and nothing is rounded; the integer
 * part goes through BigInt so a huge amount keeps every digit.
 *
 * A missing value (null, undefined, "") renders as EMPTY_VALUE. A value that
 * is not a decimal string (garbage the model stored, which the API passes
 * through) renders as stored, so a bad read stays visible instead of
 * turning into a plausible number; anything that is not text or a number
 * renders as EMPTY_VALUE. */
export function formatAmount(value: unknown): string {
  if (value === null || value === undefined || value === "") return EMPTY_VALUE;
  if (typeof value === "number") return String(value);
  if (typeof value !== "string") return EMPTY_VALUE;
  const match = DECIMAL_STRING.exec(value);
  if (!match) return value;
  const [, sign, integer, fraction = ""] = match;
  return `${sign}${integerFormatter.format(BigInt(integer))}${fraction}`;
}

/** A tax rate: the API serves the printed percent as a plain decimal string
 * ("8.875"), shown with its sign ("8.875%"). A missing value renders as
 * EMPTY_VALUE and a non-numeric stored value as stored, like formatAmount. */
export function formatRate(value: unknown): string {
  if (value === null || value === undefined || value === "") return EMPTY_VALUE;
  if (typeof value === "number") return `${value}%`;
  if (typeof value !== "string") return EMPTY_VALUE;
  return DECIMAL_STRING.test(value) ? `${value}%` : value;
}

/** Where a value was read, for people: "p. 2", or "page unknown" when the
 * model cited text on a page the document does not have (the API serves that
 * page as null). */
export function formatPage(page: number | null | undefined): string {
  return typeof page === "number" ? `p. ${page}` : "page unknown";
}

/** A quote with its column padding collapsed to single spaces. */
export function formatQuote(quote: string): string {
  return `“${quote.trim().replace(/\s+/g, " ")}”`;
}

/** A duration given in milliseconds: "850 ms" below a second, "5.2 s" from
 * a second up. */
export function formatDuration(ms: number): string {
  const rounded = Math.round(ms);
  if (rounded < 1000) return `${rounded} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

const TIMESTAMP_FORMAT: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
};

/** A timestamp in the viewer's locale and time zone, to the minute. */
export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, TIMESTAMP_FORMAT);
}

/** Where a value was read, as one line: "p. 1 · “Total: $508.20”". */
export function formatEvidence(evidence: { page: number | null; quote: string }): string {
  return `${formatPage(evidence.page)} · ${formatQuote(evidence.quote)}`;
}

/** A phrase that opens a sentence, with its first letter in capitals:
 * "amount due" becomes "Amount due". */
export function sentenceCase(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
