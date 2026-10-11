/** Shared display formatters, so one value reads the same on every page. */

/** What a missing value renders as. */
export const EMPTY_VALUE = "-";

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
