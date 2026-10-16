# FUTURE - Upcoming Batches

> Planned-but-not-active work, in build order. When PRESENT.md is empty, `/batch-loop` promotes the next batch from here, expanding it into concrete tasks with *Verify* lines first (planner agent for anything non-trivial).

Rewritten 2026-10-09 after the 18-PR stack (retrieval, /ask, chat, monitoring, Terraform, Langfuse) was merged into local master at `98c4ae9`.
The old Deploy and Stretch sections are gone: deploy was cancelled 2026-08-03, pgvector search shipped in the stack, and CSV export is folded into Batch C.

**Scope (Anthony, 2026-10-09):** local-only resume demonstration, never hosted or published to any server.
Everything must run on one machine through `docker compose up`, and every feature must be showable in a short GIF.
The Terraform Cloud Run code stays as an infrastructure-as-code sample but is never applied.

Batch 0 (demo GIF pipeline) was promoted to PRESENT.md on 2026-10-09; Batch A on 2026-10-11; Batch B on 2026-10-14.

**Every batch below ends with a demo task:** record that batch's GIF(s) with the harness and add them to the gallery.
The batch is not done until its GIF exists.

## Theme: from "extracts receipts" to "a back office can trust and act on it"

Two principles run through every batch below.

1. **The model reads, code computes.**
   The model's only job is to transcribe what is printed and say where it is.
   Every sum, difference, rate, roll-up, comparison, and currency conversion is done by deterministic code over `Decimal`, versioned, unit-tested, and re-runnable on old data without a model call.
2. **Every value points at its evidence.**
   A field is not trusted because the model is confident; it is trusted because its quote is found on the page, its box can be drawn, and its numbers reconcile.
   Grounding failures and arithmetic failures become review signals alongside model confidence.

This matters for the headline eval result: Haiku missed 4.6% of fields and flagged none of them (caught_by_review 0.0).
Model confidence alone cannot catch its own errors; deterministic checks and grounding can.

Batch A (deterministic money and reconciliation) was promoted to PRESENT.md on 2026-10-11.

## Batch C - Highlighting: field boxes over the rendered document

Goal: hover a field and its source lights up on the page; hover a box and its field lights up; search hits and chat citations highlight too.

- **C1 Geometry layer.**
  New `page_words` table (page_id, word index, text, normalized box x0/y0/x1/y1 in 0-1 page space, OCR confidence, source).
  Behind an `OcrProvider` interface selected by `OCR_PROVIDER`, three sources:
  - digital PDFs: the embedded text layer via PyMuPDF, always on (exact, free, no model);
  - images and scanned PDFs, default: the vision model returns word or line boxes for the page, stored with `source = model` and drawn as "approximate";
  - images and scanned PDFs, optional local OCR (`OCR_PROVIDER=tesseract`): Tesseract word boxes, installed through an opt-in compose profile so the default image stays small; exact boxes, free, offline.
  The eval in C6 reports localization accuracy for each provider side by side, which is itself a resume talking point.
  The Haiku transcription stays as the search text; geometry is a separate layer.
- **C2 Evidence grounding** (`backend/app/grounding.py`).
  Match each leaf's `evidence.quote` against that page's words with normalized fuzzy token-sequence alignment (rapidfuzz), and produce one or more boxes (a quote can wrap lines) plus a match score.
  Persist as `field_regions` (extracted_field_id, page, boxes JSONB, score, method).
  An unmatched quote marks the field ungrounded, which is a review reason.
  Line-item rows get the union of their cells' boxes.
  Considered and rejected as the primary source: model-predicted bounding boxes (imprecise, not verifiable) and API Citations on document blocks (they do not combine with forced tool use and give text ranges, not boxes).
- **C3 Viewer.**
  Replace the `<iframe>` PDF preview with `pdfjs-dist` canvas rendering, and the image preview with an `<img>` in the same frame, both under an SVG overlay drawn in normalized coordinates (zoom and rotation safe).
  Two-way linking: hover or focus a field row to highlight and scroll to its box; hover a box to highlight its row.
  Box colors encode state (low confidence, failed check, ungrounded, reviewed) using the existing WCAG-checked palette; keyboard navigable, works in light and dark, respects reduced motion.
- **C4 Citations and search.**
  Chat and /ask citations, and `/search` hits, already carry `char_start`/`char_end` into page text that the UI ignores.
  Map them to boxes by running the same matcher on the cited text, and open the document scrolled to the highlighted region.
- **C5 Click to correct.**
  In review, click or lasso words on the page to fill the corrected value, and the correction stores its new region, so the audit trail shows where a human read the value from.
- **C6 Eval.**
  `make_evals.py` draws every line at a known position, so it can emit gold word and field boxes (rotated for skewed docs).
  New metrics: grounded rate, localization accuracy (IoU >= 0.5), and false-grounding rate on the adversarial document, per OCR provider.
- **C7 Demo GIFs:** hovering fields lights up their boxes on a skewed receipt and the reverse; a chat citation jumps to its highlighted line; click-to-correct in review.

## Batch D - Multi-document totals, reports, and export

Goal: the answer to "what did we spend with Acme in Q3" is computed by code from reviewed records, never by the model.

- **D1 Aggregation API** (`/reports/totals`): filter by date range, vendor, status, and reconciliation status; group by vendor, month, category, or currency.
  Never sums across currencies; each currency is its own total.
  Every number links to the documents it came from.
  Documents with pending review or failed checks are counted separately ("12 documents, 2 unreviewed") rather than silently included.
- **D2 Currency conversion**, optional and explicit: a dated `fx_rates` table (imported, not fetched by the model), conversion at the document date, with the rate and its source shown beside every converted figure.
- **D3 Collections / expense reports:** group documents into a named report with deterministic totals, a reconciliation summary, and a printable view.
- **D4 Export:** CSV and XLSX of headers and line items, plus QuickBooks- and Xero-compatible bill import CSVs.
  Money as decimal strings, and reviewed values win over raw ones (`records.py` `field_value`).
- **D5 Agent alignment:** the /ask agent's `extraction_summary` tool calls the same aggregation code, so the agent and the reports page can never disagree.
- **D6 Demo GIF:** filter Q3 spend by vendor, totals split by currency with unreviewed documents called out, click a total to see its documents, export to XLSX.

## Batch E - Duplicates and vendor master

- **E1 Duplicate detection:** exact (file hash) and near (perceptual image hash) duplicates on upload; semantic duplicates on (normalized vendor, invoice_number) or (vendor, date, total).
  Flag and link, never delete; the "paid twice" catch is a concrete dollar saving companies understand.
- **E2 Vendor master:** a `vendors` table with aliases, populated from reviewer confirmations; extraction maps free-text vendor names to a canonical vendor with a match score, and low-score matches go to review.
- **E3 Per-vendor defaults and anomaly checks:** expected currency, usual tax rate, typical amount range; a deterministic "outside this vendor's normal range" flag.
- **E4 Demo GIF:** uploading a re-photographed copy of an existing invoice raises a "possible duplicate" banner linking the original.

## Batch F - Document types beyond receipts and invoices

- **F1 Classifier step** (cheap model, own eval) that sets `doc_type` before extraction and picks the schema and prompt.
- **F2 New schemas, each with its own deterministic checks:** purchase orders, bank and card statements (opening balance + credits - debits = closing balance, per-row running balance), W-9 / vendor onboarding forms (TIN format checks), and utility bills.
- **F3 Three-way match:** purchase order vs invoice vs receipt, line by line, with quantity and price variances computed by code and a tolerance policy.
- **F4 Demo GIF:** a bank statement whose running balance breaks on one row, highlighted; a three-way match showing a quantity variance.

## Batch G - Back-office workflow (local, single machine)

No multi-tenant auth, organizations, or hosting: those were dropped with the local-only scope.
What remains is the workflow a company would recognize, all demoable on one machine.

- **G1 Real audit trail:** a `field_events` history table with before, after, reason, and a local display name for the actor, replacing the single correction slot; allow re-review instead of 409.
- **G2 Policy rules and approvals:** a deterministic rules engine ("over $5,000 needs approval", "missing receipt over $75", "weekend spend") with an approval queue.
- **G3 Webhooks:** outbound, signed, retried through the existing job queue, demonstrated against a small local receiver container in compose.
- **G4 Data handling:** PII redaction in exports, and per-document deletion that also removes chunks, words, regions, and files.
- **G5 Demo GIF:** a large invoice trips an approval rule, gets approved, and the history panel shows every change with its reason.

## Engine gaps found in Batch B reviews (not scheduled)

- An amount paid larger than the total is checkable even when no amount due is printed; the mutation eval misses 029's `amount_paid` misreads for lack of that check (found in the B7 review, 2026-10-15).

## Decisions needed from Anthony

- **Live eval budget** for Batch B (v2 schema re-run, roughly $0.50 to $1.00 at past rates), Batch C (model-box provider eval), and Batch F (classifier eval).

Resolved 2026-10-09: no hosting ever (local resume demo only); OCR is model boxes by default with optional local Tesseract.

## Carry-over (accepted, not blocking)

- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
- H3 (`line_items` non-list stored as `[]`) is closed by A3's `lines_present` check (PRESENT.md).
