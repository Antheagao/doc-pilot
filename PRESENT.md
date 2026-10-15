# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## Batch B - Extraction schema v2: richer fields plus evidence (promoted 2026-10-14, expanded 2026-10-15)

Expanded by the planner against master `3d25754`.
Goal: capture what real invoices carry, and make every leaf say where it came from.
Zero-spend work comes first; B10, B11, and B11b spend money and each needs Anthony's approval before it runs.

Key design decisions (planner defaults; Anthony can override):
- Storage: JSONB leaves, no new columns, no backfill. Every v2 leaf is `{value, confidence, evidence: {page, quote} | null}`, cells of `line_items` and `tax_lines` included; v1 extractions keep their 7 rows untouched.
- v1 and v2 coexist through a schema registry keyed by `prompt_version` (`app/extraction_fields.py`: `ExtractionSchema`, `SCHEMAS`, `schema_for`); the worker extracts with `Settings.extraction_prompt_version`, default `extract_v1` until the paid eval passes a pre-registered bar. v1 stays byte-identical.
- API and UI: no new endpoint; evidence rides inside the leaves. Tax-line `rate` is the printed percent as a plain decimal string; code divides by 100.
- `rules_v2` on a v1 record gives exactly the `rules_v1` verdicts (a v2-only check is emitted only when one of its v2 operands is present), pinned by a golden fixture, so the published numbers do not move.
- A quote mismatch gets its own review reason, `quote:<path>`, as a `kind: "quote"` check in the same stored report; one migration widens the review-reason CHECK.
- PII in code, not just the prompt: `payment_method` (brand only) and `payment_card_last4` (exactly four digits); `mask_payment_fields` runs before anything is stored or logged, and corrections get the same rule as 422s.
- Tax lines are `(label, rate, base, amount)`; every printed tax line goes in `tax_lines`.
- The demo seed keeps replaying the committed v1 Sonnet run for 001-025; new documents 026+ replay the newest committed v2 artifact, else label plus gold evidence, tagged and counted.
- Pre-registered acceptance bar for making v2 the default: per model, core accuracy (the 7 v1 fields on 001-025) no worse than its v1 run by more than one field; 0 residual schema violations; Sonnet evidence coverage >= 95%; no `max_tokens` stops; no unmasked card number in the artifact.

### Tasks

- [x] **B1 Versioned extraction schemas (refactor, v1 byte-identical)** - `edbd2b5` - `app/extraction_fields.py` (`FieldSpec`, `ExtractionSchema`, `SCHEMA_V1`, `SCHEMAS`, `schema_for`, `record_extraction_tool`); `schema` threaded through `extract_document`, `_extract_one_block`, `_extract_chunked_pdf`, `merge_page_tool_inputs`, `_schema_violations`, `build_extracted_fields`; module constants removed from every importer; `Settings.extraction_prompt_version` (validated) in config, compose, `.env.example`; a captured v1 tool fixture.
  *Verify:* `ruff check .`; `pytest -q`; `python evals/run.py --mock` predicted = actual 0.988571; the README sync test passes; `git diff --stat backend/prompts` empty; no API key.
  *Result (2026-10-15):* 906 tests pass; the v1 tool, prompts, max tokens, field names, and tie-breaks equal a fixture captured from master before any edit (the reviewer re-derived it from `3d25754`); mock 0.988571 predicted and actual; README and prompts unchanged. Review fixes: readers of committed artifacts degrade to n/a on an unknown or missing prompt version instead of taking down every report and `/monitoring/evals`, and prompt files are read when the schema is built, so a missing one fails at startup rather than after a billed call.
- [x] **B2 Reconciliation rules_v2 (pure engine)** - golden `tests/fixtures/rules_v1_reports.json` frozen before any edit; `ReconcileInput` gains discount, shipping, tip, amount_paid, amount_due, dates, `tax_lines`; `total_math` generalized; `tax_lines_sum`, `tax_line_math[i]`, `balance_math`, `due_date_order`, new `non_negative:*`; suggestions over the new money paths.
  *Verify:* reconcile, property, and routing tests pass twice identically; the golden test; regenerating the README tables changes only `rules_v1` to `rules_v2` (Haiku still 50.0%, mutation eval still 1604 of 1628); `reconcile_all.py --dry-run` on the dev stack shows 0 routing changes, after a `pg_dump` backup.
  *Result (2026-10-15):* 987 tests pass. A v1-shaped record keeps its rules_v1 report except the version string: a golden fixture written by the old engine (151 reports: 25 labels, 53 committed-run replays, the staged 016 case, 72 edge shapes) and a 3000-case old-versus-new differential with 0 mismatches. README regeneration changes only `rules_v1` to `rules_v2` (Haiku still 50.0%, mutation eval still 1604 of 1628, 0 wrong fixes). The Mac dev database holds no extractions and predates A4 (backed up to `~/doc-pilot-backups/` anyway), so `reconcile_all.py` ran on the demo database instead: 25 extractions, 0 routing changes in the dry run and after the real run. Review fixes: a rate-only tax line under the tax-included reading is checked on the net amount (it had false-failed real VAT receipts with a wrong fix and false-passed a misread total), suggestion re-runs use only the first tax base (a later base had turned one misread into a wrong fix on a correct field), a negative `amount_due` (overpayment) can be suggested, and the widened tax-included blind spot (a total read as the subtotal with its adjustments) is documented and pinned.
- [ ] **B3 Value-quote agreement check and the `quote:` review reason** - printed-amount patterns moved into `app/money.py` (`printed_amounts`); `app/quotes.py` (`quote_agreement`); `quote:<path>` checks in `reconcile.py`; routing, a migration widening the review-reason CHECK, `caught_by_quotes` in `evals/checks.py`.
  *Verify:* `pytest -q`; migration downgrade/upgrade on `docpilot_test` with one head; `run_retrieval.py --check` passes; README regeneration changes nothing.
- [ ] **B4 Schema v2 contract and persistence (prompt shipped, not the default)** - v2 `FieldSpec`s; `prompts/extract_v2.md` and `repair_v2.md`; `_coerce_evidence`; `mask_payment_fields`; v2 persistence in `build_extracted_fields`; `records.py` corrections for the new fields and the PII 422s; `record_lines` for present v2 fields only.
  *Verify:* `ruff check .`; `pytest -q`; a fake-client v2 worker test stores 20 fields; the default is still `extract_v1`; no 13+ digit run survives in `raw_response`.
- [ ] **B5 PDF evidence pages through single-call and chunked paths** - image evidence page 1, out-of-range pages null, `merge_page_tool_inputs` carries each winner's page; v1 merge byte-identical.
  *Verify:* `pytest tests/test_extraction.py -q`; `pytest -q`.
- [ ] **B6 Eval dataset v2: labels, 7 new documents, gold evidence** - `make_evals.py` v2 fields (001-025 images never regenerated: `--labels-only`), documents 026-032 (discount and shipping invoice, tip, partial payment, PII card, EUR VAT lines, tax lines only, two-page PDF), `dataset_version` "v2", validators, harvest.
  *Verify:* `git status --porcelain evals/docs evals/text` lists only the new documents; old labels' v1 keys unchanged; every label passes rules_v2 with 0 gold-quote failures; the retrieval corpus stays 25; `pytest -q`.
- [ ] **B7 v2 eval machinery (offline)** - scoring matchers for the new fields, evidence in artifacts, mock v2, `--prompt` and `--doc`, replay by artifact schema, the quote eval with its README table, core-accuracy and v2 report tables.
  *Verify:* with no API key: `run.py --mock` (v1 and v2), `--report-only --update-readme`, `run_checks.py --update-readme`; README diff only in marker regions; `pytest -q`.
- [ ] **B8 UI for the new fields and evidence, demo seed for 026+** - grouped fields, tax-lines table, folded empty fields, an evidence toggle, quote chips, the review page's `quote:` reason and tax-line corrections; seed replays for 026+ and a staged 028 quote case. From the B2 review: `lib/reconciliation.ts` (`pathLabel`, `fieldOfPath`, `humanize`) must learn `tax_lines[i].*`; `tax_line_math` coefficients (rate/100, e.g. 0.08875) must render as the printed percent (8.875%); a computed fallback base (`tax_lines[i].base`) must not be a button aimed at a missing element; the discount-applied reading's "(discount 10.00 included)" needs wording that says the discount is already taken off.
  *Verify:* eslint, build, backend gates; seed reports 32 documents; in a browser 026 all green, 028 two quote chips, 016 and 018 unchanged; light and dark at 360, 768, 1440; keyboard only; demo spend $0.00.
- [ ] **B9 Verify model prices before any paid run (zero spend)** - confirm published prices for `claude-sonnet-5` and `claude-haiku-4-5` and that both are served; update `PRICING_PER_MTOK`; recompute the B10 and B11 caps.
  *Verify:* `pytest -q`; `_compute_cost_usd` equals the published price.
- [ ] **B10 PAID: v2 smoke on 3 documents (about $0.10, cap $0.15)** - Sonnet v2 on 026, 029, 032 into `evals/results/smoke`. ⚠ Needs Anthony's approval.
  *Verify:* 3 scored, 0 errors, no `max_tokens` stop, 0 residual schema violations, evidence coverage >= 90%, 032's totals cite page 2, no card number in the artifact, spend under the cap.
- [ ] **B11 PAID: full v2 runs, Sonnet and Haiku (about $1.20-1.65)** - both artifacts committed with the regenerated tables; flip the default to `extract_v2` only if the pre-registered bar holds. ⚠ Needs Anthony's approval.
  *Verify:* 32 scored each; the core column compares v1 and v2 on the same 25 documents; the bar checked line by line; `pytest -q`.
- [ ] **B11b PAID, optional: v1 baselines on dataset v2 (about $0.57-0.77)** - only adds v1 on the new documents. ⚠ Needs Anthony's approval.
- [ ] **B12 Demo GIFs: v2 invoice and quote check (zero spend)** - replay of the committed v2 Sonnet artifact for 026-032; `invoice-v2` and `quote-check` scenarios; gallery captions; boards rotated.
  *Verify:* `seed --reset`, both records, `check`; demo spend $0.00; demo gates.

### Open questions for Anthony (from planning)

- Paid scope: B10 plus B11 (default), and B11b?
- Flip the worker to `extract_v2` automatically when the bar passes (default), or only after you review the numbers?
- B12 from replay plus a staged case (default, $0), or a live upload take (about $0.04 per take)?
- A non-null amount with no evidence: report only (default), or route it to review once B11 shows real coverage?

Risks: v2 output length on dense documents (`max_tokens` 8192), stale model prices, quote parsing false flags (measured, pinned at 0 on gold), a fabricated quote agreeing with a fabricated value (Batch C2), real-world tax and discount layouts, card numbers in transcribed page text (G4), and a 27-file refactor in B1.

## Follow-ups from Batch 0 (not yet scheduled)

- The top `screenshots/demo.gif` predates the Search, Ask, and Dashboard navigation; re-recording it with the harness costs one live extraction (about $0.02).
- The Screenshots section still shows stills that the Features GIFs now cover (`search-hybrid.png`, `search-amount.png`, `review-correct.png`).
- The macOS session's four commits carry `Co-Authored-By` trailers, against the global no-attribution rule; that machine's claude-brain setup needs bootstrapping.

## Carry-over (accepted, not blocking)

- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
