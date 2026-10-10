# FUTURE - Upcoming Batches

> Planned-but-not-active work, in build order. When PRESENT.md is empty, `/batch-loop` promotes the next batch from here, expanding it into concrete tasks with *Verify* lines first (planner agent for anything non-trivial).

Rewritten 2026-10-09 after the 18-PR stack (retrieval, /ask, chat, monitoring, Terraform, Langfuse) was merged into local master at `98c4ae9`.
The old Deploy and Stretch sections are gone: deploy was cancelled 2026-08-03, pgvector search shipped in the stack, and CSV export is folded into Batch C.

**Scope (Anthony, 2026-10-09):** local-only resume demonstration, never hosted or published to any server.
Everything must run on one machine through `docker compose up`, and every feature must be showable in a short GIF.
The Terraform Cloud Run code stays as an infrastructure-as-code sample but is never applied.

## Batch 0 - Demo GIF pipeline and the backlog of merged features

Goal: a repeatable way to record one short GIF per feature, then GIFs for everything the merged stack added.

- **0.1 Recording harness** (`scripts/demos/`): one Playwright script per scenario against the compose stack, then ffmpeg palette conversion to GIF, plus a WebM/MP4 pair for the portfolio site, the same pipeline used for `screenshots/demo.gif`.
  Fixed viewport, light theme, a scripted cursor and pauses so every GIF has the same look and pace, and a target of 15 seconds or less and under 1 MB each.
- **0.2 Deterministic demo data:** a seed script that loads chosen eval documents and their label-perfect records (`evals/corpus.py` already does this for the agent eval) so recordings need no live model calls unless the scenario is the extraction itself.
  Staged review cases (a plausible misread, a failed check) are seeded explicitly, as session 5 did by hand.
- **0.3 GIFs for merged features:** hybrid search, /ask with live agent steps, per-document chat with field citations, the review queue correction flow, and the monitoring dashboard.
- **0.4 README gallery:** a "Features" section with one GIF and one sentence per feature, the main demo GIF staying at the top.

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

## Batch A - Deterministic money and reconciliation (no model change)

Goal: every extraction gets a reconciliation report computed by code, and failed checks route to review.
Works on the current 7-field schema, so it ships with zero live spend.

- **A1 Money module** (`backend/app/money.py`).
  Parse a leaf value into `Decimal` via `Decimal(str(v))`, quantize to the currency's ISO 4217 minor unit (USD 2, JPY 0, BHD 3; unknown or null currency defaults to 2 and is reported as an assumption).
  Reject NaN, infinity, and values that need more precision than the minor unit allows.
  Serialize money in API responses as canonical decimal strings, never floats.
  Fix the review UI's `Number()` parsing of corrections to send strings.
- **A2 Reconciliation engine** (`backend/app/reconcile.py`), a pure function: reviewed record in, list of `Check` out.
  Each check carries `id`, `formula`, operands with source field paths, `expected`, `actual`, `delta`, `tolerance`, and `status` (`pass`, `fail`, `skipped` when an operand is missing).
  Initial rules:
  - per line: `quantity x unit_price = line total` (tolerance one minor unit);
  - `sum(line totals) = subtotal` (tolerance: half a minor unit per line, for per-line rounding);
  - `subtotal + tax = total`;
  - implied tax rate `tax / subtotal` within 0-30%, reported as a number either way;
  - non-negative totals, and line items present when a subtotal is present.
  Rules are versioned (`RULES_VERSION`), so a stored report says which rule set produced it.
  This also closes the H3 carry-over gap: a malformed `line_items` that silently stores `[]` now fails the sum check.
- **A3 Persist and route.**
  New `reconciliations` table (extraction_id, rules_version, checks JSONB, status, computed_at).
  Compute after extraction and again after every review resolve, so corrections re-reconcile.
  A failed check sets `needs_review` on the fields it involves, with a new `review_reason` column (`confidence`, `reconciliation:<check_id>`, later `grounding`).
  When exactly one operand of a failed equation is the odd one out, attach a *suggested* value (for example "total computes to 13.53"), shown to the reviewer and never auto-applied.
- **A4 UI.**
  A reconciliation panel on the document page that renders each check as its equation, `12.50 + 1.03 = 13.53` with a pass or fail mark, and links its operands to their field rows.
  Values the code derived (such as a subtotal computed when none was printed) are labeled "computed", visually distinct from "extracted".
  Review queue rows show the reason and the suggestion.
- **A5 Evals and tests.**
  Property tests (hypothesis) that perturb one amount of a consistent record and assert a check catches it.
  New eval metric `caught_by_checks` beside `caught_by_review`, measured with mutated mock extractions, plus a re-score of the committed Haiku artifact to show how many of its silent misses arithmetic would have caught.
  All 25 eval labels are already internally consistent, so they serve as the passing baseline.
- **A6 Demo GIF:** a receipt with a misread subtotal fails `sum(line totals) = subtotal`, the reconciliation panel shows the red equation and the suggested value, the reviewer accepts it, and the check turns green.

## Batch B - Extraction schema v2: richer fields plus evidence (one prompt bump, one paid eval re-run)

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

## Decisions needed from Anthony

- **Live eval budget** for Batch B (v2 schema re-run, roughly $0.50 to $1.00 at past rates), Batch C (model-box provider eval), and Batch F (classifier eval).

Resolved 2026-10-09: no hosting ever (local resume demo only); OCR is model boxes by default with optional local Tesseract.

## Carry-over (accepted, not blocking)

- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
- H3 (`line_items` non-list stored as `[]`) is resolved by Batch A2.
