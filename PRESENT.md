# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## Batch B - Extraction schema v2: richer fields plus evidence (one prompt bump, one paid eval re-run) (promoted 2026-10-14)

Not yet expanded: the next `/batch-loop` run plans it with the planner against current master, then works it.
B5's live eval re-run needs Anthony's approval of its spend before it runs.

Goal: capture what real invoices carry, and make every leaf say where it came from.
Fields and evidence change in the same prompt version so the live eval runs only once.

- **B1 Schema v2** (`prompts/extract_v2.md`, `RECORD_EXTRACTION_TOOL` v2; v1 stays for the eval comparison).
  New fields: `invoice_number`, `po_number`, `due_date`, `payment_terms`, `discount`, `shipping`, `tip`, `amount_paid`, `amount_due`, `tax_lines` (label, rate, amount), `payment_method` (last 4 only, never full numbers), `vendor_tax_id`.
  Every leaf gains `evidence: {page, quote}`, where `quote` is the verbatim printed text the value was read from.
- **B2 Reconciliation rules v2:** `subtotal - discount + shipping + tax + tip = total`, `sum(tax_lines) = tax`, `tax_line.rate x base = tax_line.amount`, `total - amount_paid = amount_due`, `due_date >= document_date`.
- **B3 Value-quote agreement**, a deterministic anti-hallucination check.
  Parse the number out of each money leaf's quote with the existing `retrieval/normalize.py` rules (comma decimals, thousands separators, currency symbols) and require it to equal the extracted value.
  A mismatch means the model wrote a number it did not read, which becomes a review reason.
- **B4 PDF chunk merge** carries evidence pages through `merge_page_tool_inputs`, so single-call and chunked PDFs both record a page per field.
- **B5 Eval:** extend labels and `make_evals.py` with the new fields (record-first generation keeps labels correct by construction), add a few documents with discounts, shipping, multiple tax lines, and partial payments, then run Sonnet and Haiku v1 against v2 under the usual cost cap.
- **B6 Demo GIF:** an invoice with a discount, shipping, and two tax lines extracts and reconciles end to end; a second clip shows the value-quote check catching a number that is not on the page.

## Follow-ups from Batch 0 (not yet scheduled)

- The top `screenshots/demo.gif` predates the Search, Ask, and Dashboard navigation; re-recording it with the harness costs one live extraction (about $0.02).
- The Screenshots section still shows stills that the Features GIFs now cover (`search-hybrid.png`, `search-amount.png`, `review-correct.png`).
- The macOS session's four commits carry `Co-Authored-By` trailers, against the global no-attribution rule; that machine's claude-brain setup needs bootstrapping.

## Carry-over (accepted, not blocking)

- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
