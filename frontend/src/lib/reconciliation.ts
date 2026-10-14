/** Display helpers for the reconciliation report (backend/app/reconcile.py),
 * shared by the document page's panel and the review queue. Amounts arrive as
 * decimal strings and are only ever formatted here, never computed with. */

import type { ReconciliationCheck, ReconciliationOperand } from "@/lib/api";
import { formatAmount } from "@/lib/format";

// "line_items[2].total" -> line index 2 and column "total"; a bare
// "line_items[2]" has no column.
const LINE_PATH = /^line_items\[(\d+)\](?:\.(\w+))?$/;

/** An equation with more terms than this collapses to "sum of 12 line
 * totals". */
export const MAX_SHOWN_TERMS = 4;

/** What a person calls a path of the report: "subtotal", "line 3 total". */
export function pathLabel(path: string): string {
  const match = LINE_PATH.exec(path);
  if (!match) return path.replace(/_/g, " ");
  const line = Number(match[1]) + 1;
  return match[2] ? `line ${line} ${match[2].replace(/_/g, " ")}` : `line ${line}`;
}

/** The field the extraction panel shows for a report path, named the way a
 * chat citation names it ("line_items[2]"), so fieldElementId and the
 * highlight set find its row. */
export function fieldOfPath(path: string): string {
  const match = LINE_PATH.exec(path);
  return match ? `line_items[${match[1]}]` : path;
}

/** A backend sentence or formula in the words the panel uses: report paths
 * become "line 1 total", snake_case names lose their underscores, and the
 * multiplication sign replaces "x". */
export function humanize(text: string): string {
  return text
    .replace(/line_items\[\d+\](?:\.\w+)?/g, (path) => pathLabel(path))
    .replace(/([a-z])_([a-z])/g, "$1 $2")
    .replace(/ x /g, " × ");
}

export function amountText(operand: ReconciliationOperand): string {
  return formatAmount(operand.value);
}

export interface EquationTerm {
  sign: "+" | "-";
  /** The coefficient without its sign; null when it is 1. */
  coef: string | null;
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
      coef: abs === "1" ? null : abs,
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

/** One term as text: "9.15", "4 × 9.15", "(-5.00)". */
function termText(term: EquationTerm): string {
  const value = amountText(term.operand);
  const shown = value.startsWith("-") ? `(${value})` : value;
  return term.coef === null ? shown : `${term.coef} × ${shown}`;
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
 * subtotal". Anything else falls back to the check's own sentence. */
export function checkText(check: ReconciliationCheck): string {
  const view = equationView(check);
  if (view === null) return humanize(check.detail ?? check.formula);
  const included = view.included.map((o) => ` (${pathLabel(o.path)} ${amountText(o)} included)`).join("");
  const left = `${leftSideText(view)}${included}`;
  const rhs = `${view.actual} ${pathLabel(view.rhs.path)}`;
  return view.relation === "="
    ? `${left} = ${rhs}`
    : `${left} = ${view.expected} ${view.relation} ${rhs}`;
}
