/** Display helpers for the reconciliation report (backend/app/reconcile.py),
 * shared by the document page's panel and the review queue. Amounts arrive as
 * decimal strings and are only ever formatted here, never computed with. */

import type {
  ReconciliationCheck,
  ReconciliationOperand,
  ReconciliationSuggestion,
} from "@/lib/api";
import { formatAmount } from "@/lib/format";

// "line_items[2].total" -> row index 2 and column "total"; a bare
// "line_items[2]" has no column. "tax_lines[1].amount" reads the same way.
const ROW_PATH = /^(line_items|tax_lines)\[(\d+)\](?:\.(\w+))?$/;
const ROW_PATH_ANYWHERE = /(?:line_items|tax_lines)\[\d+\](?:\.\w+)?/g;

// What a row of each array field is called.
const ROW_NOUN = { line_items: "line", tax_lines: "tax line" } as const;

// The review reason and the id of a quote check both start with this, and what
// follows is the report path of the value.
const QUOTE_PREFIX = "quote:";

/** An equation with more terms than this collapses to "sum of 12 line
 * totals". */
export const MAX_SHOWN_TERMS = 4;

/** What a person calls a path of the report: "subtotal", "line 3 total",
 * "tax line 2 amount". */
export function pathLabel(path: string): string {
  const match = ROW_PATH.exec(path);
  if (!match) return path.replace(/_/g, " ");
  const row = `${ROW_NOUN[match[1] as keyof typeof ROW_NOUN]} ${Number(match[2]) + 1}`;
  return match[3] ? `${row} ${match[3].replace(/_/g, " ")}` : row;
}

/** The field the extraction panel shows for a report path, named the way a
 * chat citation names it ("line_items[2]", "tax_lines[0]"), so fieldElementId
 * and the highlight set find its row. */
export function fieldOfPath(path: string): string {
  const match = ROW_PATH.exec(path);
  return match ? `${match[1]}[${match[2]}]` : path;
}

/** The stored field a report path belongs to: "tax_lines[1].amount" is the
 * "tax_lines" field, "total" is "total". */
export function fieldNameOfPath(path: string): string {
  return /^[^.[]+/.exec(path)?.[0] ?? path;
}

/** A backend sentence or formula in the words the panel uses: report paths
 * become "line 1 total" or "tax line 1 amount", snake_case names lose their
 * underscores, and the multiplication sign replaces "x". */
export function humanize(text: string): string {
  return text
    .replace(ROW_PATH_ANYWHERE, (path) => pathLabel(path))
    .replace(/([a-z])_([a-z])/g, "$1 $2")
    .replace(/ x /g, " × ");
}

// The operands that are dates (ISO strings), not amounts.
const DATE_PATHS: ReadonlySet<string> = new Set(["document_date", "due_date"]);

/** An operand's number as text: an amount grouped by formatAmount, a date as
 * the ISO date the extraction panel shows. A missing value is EMPTY_VALUE. */
export function amountText(operand: ReconciliationOperand): string {
  if (DATE_PATHS.has(operand.path) && operand.value !== null) return operand.value;
  return formatAmount(operand.value);
}

// A tax line's rate is a fraction in the report (rate / 100, "0.08875") and a
// percent everywhere else a person reads it.
const TAX_LINE_CHECK = /^tax_line_math\[\d+\]$/;
const DECIMAL_COEF = /^(-?)(\d+)(?:\.(\d+))?$/;

/** A coefficient that is a fraction, as the percent it is: "0.08875" is
 * "8.875", "0.065" is "6.5", "1" is "100". Decimal strings only, no Number. */
export function fractionAsPercent(fraction: string): string {
  const match = DECIMAL_COEF.exec(fraction);
  if (!match) return fraction;
  const [, sign, integer, rest = ""] = match;
  const digits = rest.padEnd(2, "0");
  const whole = `${integer}${digits.slice(0, 2)}`.replace(/^0+(?=\d)/, "");
  const decimals = digits.slice(2).replace(/0+$/, "");
  return `${sign}${whole}${decimals ? `.${decimals}` : ""}`;
}

/** Whether an operand is a base the engine worked out for a tax line that
 * printed none (the subtotal less the discount, say). It has no cell in the
 * tax lines table, so there is nothing for it to point at. */
export function isComputedBase(operand: ReconciliationOperand): boolean {
  return operand.source === "computed" && /^tax_lines\[\d+\]\.base$/.test(operand.path);
}

/** How an operand sits in an equation without being added: a discount the
 * subtotal is already after reads "already taken off"; a tax already inside the
 * amounts reads "included". */
export function includedNote(operand: ReconciliationOperand): string {
  return operand.path === "discount" ? "already taken off" : "included";
}

/** What a suggestion is for, after "Suggested: <path> <value> - ": the one
 * change that makes every check pass, and when a quote check was among those
 * it fixes, that the new value is the number its quote prints. */
export function suggestionWhy(suggestion: ReconciliationSuggestion): string {
  const quoted = suggestion.check_ids.some((id) => id.startsWith(QUOTE_PREFIX));
  return quoted
    ? "the number printed in its quote, and the one change that makes every check pass"
    : "one change that makes every check pass";
}

/** The report path a quote check judges: the check id is "quote:<path>". */
export function quotePath(check: ReconciliationCheck): string {
  return check.id.startsWith(QUOTE_PREFIX) ? check.id.slice(QUOTE_PREFIX.length) : check.id;
}

/** Whether a review reason is the quote disagreeing with the value, and the
 * path of that value ("quote:tax_lines[0].amount" gives "tax_lines[0].amount"). */
export function quoteReasonPath(reason: string | null): string | null {
  return reason !== null && reason.startsWith(QUOTE_PREFIX) ? reason.slice(QUOTE_PREFIX.length) : null;
}

export interface EquationTerm {
  sign: "+" | "-";
  /** The coefficient without its sign; null when it is 1. For a tax line it
   * is the printed percent ("8.875%"), with `rate` set. */
  coef: string | null;
  /** The term reads "8.875% of 90.00" instead of "0.08875 × 90.00". */
  rate: boolean;
  operand: ReconciliationOperand;
}

export interface EquationView {
  /** The terms that count; an operand whose source is "included" is not one. */
  terms: EquationTerm[];
  /** Operands already contained in the other amounts (a tax-inclusive
   * reading): shown beside the sum, not added to it. */
  included: ReconciliationOperand[];
  /** "sum of 12 line totals" when the sum is too long to spell out. */
  collapsed: string | null;
  expected: string;
  actual: string;
  rhs: ReconciliationOperand;
  /** "=" when the sides agree to the digit, "≈" when they agree within the
   * rounding the rule allows, "≠" when they do not. */
  relation: "=" | "≈" | "≠";
  /** How far apart the sides are, unsigned. */
  off: string | null;
}

// Decimal strings are never parsed to Number; a zero is spotted by its text.
function isZero(value: string): boolean {
  return /^-?0+(\.0+)?$/.test(value);
}

function unsigned(value: string): string {
  return value.startsWith("-") ? value.slice(1) : value;
}

/** The numbers of an equation check, or null for a check that is not an
 * evaluated equation (a bound, a presence check, a skipped one). */
export function equationView(check: ReconciliationCheck): EquationView | null {
  const { rhs, expected, actual } = check;
  if (check.kind !== "equation" || check.status === "skipped") return null;
  if (rhs === null || expected === null || actual === null) return null;
  const terms: EquationTerm[] = [];
  const included: ReconciliationOperand[] = [];
  const taxLine = TAX_LINE_CHECK.test(check.id);
  for (const { coef, operand } of check.terms) {
    if (coef === null) return null;
    // A tax-inclusive reading marks its operand; a coefficient of 0 alone
    // is an ordinary term (a line whose quantity reads 0).
    if (operand.source === "included") {
      included.push(operand);
      continue;
    }
    const abs = unsigned(coef);
    terms.push({
      sign: coef.startsWith("-") ? "-" : "+",
      coef: taxLine ? `${fractionAsPercent(abs)}%` : abs === "1" ? null : abs,
      rate: taxLine,
      operand,
    });
  }
  let collapsed: string | null = null;
  if (terms.length > MAX_SHOWN_TERMS) {
    const lineTotals = terms.every(
      (t) => /^line_items\[\d+\]\.total$/.test(t.operand.path) && t.coef === null
    );
    collapsed = `sum of ${terms.length} ${lineTotals ? "line totals" : "amounts"}`;
  }
  const relation = check.status === "fail" ? "≠" : expected === actual ? "=" : "≈";
  return {
    terms,
    included,
    collapsed,
    expected: formatAmount(expected),
    actual: formatAmount(actual),
    rhs,
    relation,
    off: check.delta === null || isZero(check.delta) ? null : formatAmount(unsigned(check.delta)),
  };
}

/** One term as text: "9.15", "4 × 9.15", "8.875% of 90.00", "(-5.00)". */
function termText(term: EquationTerm): string {
  const value = amountText(term.operand);
  const shown = value.startsWith("-") ? `(${value})` : value;
  if (term.coef === null) return shown;
  return `${term.coef} ${term.rate ? "of" : "×"} ${shown}`;
}

/** The left side of an equation as text: the terms joined by their signs, or
 * the collapsed sum. */
export function leftSideText(view: EquationView): string {
  if (view.collapsed !== null) return view.collapsed;
  return view.terms
    .map((term, i) => {
      const lead = i === 0 ? (term.sign === "-" ? "-" : "") : ` ${term.sign} `;
      return `${lead}${termText(term)}`;
    })
    .join("");
}

/** An evaluated equation on one line, e.g. "36.60 + 8.65 = 45.25 ≠ 46.25
 * subtotal". Anything else falls back to the check's own sentence; a quote
 * check's is the value beside the number its quote prints. */
export function checkText(check: ReconciliationCheck): string {
  const view = equationView(check);
  if (view === null) return humanize(check.detail ?? check.formula);
  const included = view.included
    .map((o) => ` (${pathLabel(o.path)} ${amountText(o)} ${includedNote(o)})`)
    .join("");
  const left = `${leftSideText(view)}${included}`;
  const rhs = `${view.actual} ${pathLabel(view.rhs.path)}`;
  return view.relation === "="
    ? `${left} = ${rhs}`
    : `${left} = ${view.expected} ${view.relation} ${rhs}`;
}
