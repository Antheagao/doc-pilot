/** Reading the evidence (page and quote) a schema v2 leaf carries. Stored
 * payloads can be partly malformed, so everything here is validated at read
 * time instead of trusted through the static types. */

import type { Evidence } from "@/lib/api";

// "total" is a header field; "tax_lines[1].amount" is one cell of an array.
const CELL_PATH = /^(\w+)\[(\d+)\]\.(\w+)$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The evidence of one stored leaf: the {page, quote} it cites, null when the
 * leaf cites none, and undefined when it has no `evidence` key at all (a v1
 * extraction, which never carried any). */
export function leafEvidence(leaf: unknown): Evidence | null | undefined {
  if (!isRecord(leaf) || !("evidence" in leaf)) return undefined;
  const { evidence } = leaf;
  if (!isRecord(evidence) || typeof evidence.quote !== "string") return null;
  return {
    page: typeof evidence.page === "number" ? evidence.page : null,
    quote: evidence.quote,
  };
}

/** The evidence of the value at a report path, read from the stored value of
 * its field: the leaf itself for "total", the cell of the row for
 * "tax_lines[1].amount". Undefined and null mean the same as in leafEvidence. */
export function evidenceAt(fieldValue: unknown, path: string): Evidence | null | undefined {
  const cell = CELL_PATH.exec(path);
  if (!cell) return leafEvidence(fieldValue);
  if (!Array.isArray(fieldValue)) return undefined;
  const row: unknown = fieldValue[Number(cell[2])];
  return isRecord(row) ? leafEvidence(row[cell[3]]) : undefined;
}

/** Whether any leaf of an extraction cites evidence (or cites none on
 * purpose): true for a schema v2 extraction, false for v1. */
export function hasEvidence(fieldValues: readonly unknown[]): boolean {
  return fieldValues.some((value) =>
    Array.isArray(value)
      ? value.some((row) => isRecord(row) && Object.values(row).some((cell) => leafEvidence(cell) !== undefined))
      : leafEvidence(value) !== undefined
  );
}
