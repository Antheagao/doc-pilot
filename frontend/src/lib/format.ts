/** Shared display formatters, so one value reads the same on every page. */

/** What a missing value renders as. */
export const EMPTY_VALUE = "-";

const DEFAULT_FRACTION_DIGITS = 2;

const amountFormatters = new Map<number, Intl.NumberFormat>();

/** The minor-unit digits of an ISO 4217 code (2 for USD, 0 for JPY, 3 for
 * KWD); 2 when the code is missing or not a valid currency code. */
function fractionDigits(currency: string | null | undefined): number {
  if (!currency) return DEFAULT_FRACTION_DIGITS;
  try {
    const resolved = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
    }).resolvedOptions();
    return resolved.maximumFractionDigits ?? DEFAULT_FRACTION_DIGITS;
  } catch (err) {
    if (err instanceof RangeError) return DEFAULT_FRACTION_DIGITS;
    throw err;
  }
}

/** A document amount with grouping and exactly the currency's minor-unit
 * digits ("1,234.50"), and no currency symbol: the currency is shown as its
 * own field. Anything that is not a finite number renders as EMPTY_VALUE. */
export function formatAmount(
  value: unknown,
  currency: string | null | undefined
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return EMPTY_VALUE;
  const digits = fractionDigits(currency);
  let formatter = amountFormatters.get(digits);
  if (!formatter) {
    formatter = new Intl.NumberFormat("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
    amountFormatters.set(digits, formatter);
  }
  return formatter.format(value);
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
