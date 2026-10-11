# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## Batch A - Deterministic money and reconciliation (promoted 2026-10-11)

Expanded by the planner against master `91db9c7`; zero model spend.
Prototyped against the committed runs: the rules catch 4 of Haiku's 8 silent misses (`caught_by_checks` 50.0% vs `caught_by_review` 0.0%), and every label and every Sonnet record passes.

Design decisions (planner defaults, taken 2026-10-11; Anthony can override):
- Money is `Decimal` in Python, parsed at every edge by `money.to_decimal`; API JSON carries every number read from a document (amounts and quantities) as a canonical decimal string padded to the currency's minor unit, keeping extra digits so precision failures stay visible. Machine metrics (confidence, cost, tokens) stay numbers. No JSONB data migration: stored model leaves are the audit record; new corrections are stored as canonical strings.
- ISO 4217 minor units come from an in-repo table generated once from the SIX list (not `pycountry`, which lacks them, nor `babel`, whose CLDR digits differ from ISO for about a dozen codes); the frontend stops deriving digits from Intl.
- Reconciliation is computed on write (worker, seed, corpus, every review resolve, and `scripts/reconcile_all.py` for old rows) and stored one row per extraction with `rules_version`, because `needs_review` must be persisted for the queue.
- Routing: reviewed fields are frozen; a failing check flags its unique suggestion's field if there is one, else every unverified operand; correcting the culprit auto-clears the others; approving a still-failing value flags the remaining operands. A DB CHECK ties `needs_review` to a non-null `review_reason`.
- Rounding tolerance only where rounding happened; the header equation is exact; a null tax counts as 0 with a recorded assumption; suggestions only when exactly one change fixes every failing check without breaking a passing one.
- `reconciliation.gif` replaces `review-correction.gif`; the live ask and chat GIFs are not re-recorded (the panel sits below 018's line items, off-screen at 1280x800).

### Tasks

- [x] **A1 Money module and internal consumers** (no API change) - `backend/app/money.py` (`MoneyError`, `ISO_4217_MINOR_UNITS`, `currency_precision`, `to_decimal`, `fits_minor_unit`, `to_api_string`, `to_plain_string`, `parse_amount_input`, `to_label_number`), `MONEY_FIELDS` / `LINE_ITEM_MONEY_KEYS` beside `TOP_LEVEL_FIELDS`, agent tools (`_money`, `_matches`, `extraction_summary`) and `evals/harvest.py` through it.
  *Verify:* `ruff check .`; `pytest tests/test_money.py tests/test_agent.py tests/test_evals_harvest.py tests/test_evals_judge.py -q`; `pytest -q`; JPY 0, BHD/KWD/IQD 3, CLF 4, more than 150 codes, `to_api_string(Decimal("46.2"), 2) == "46.20"`, `to_decimal("1e3")` raises; a string-corrected total is counted in "Sum of totals".
  *Result:* 165 ISO 4217 codes from the SIX list published 2026-09-17; 626 tests pass. Review fixes: a malformed line-items correction no longer aborts the whole harvest, sums trap inexact rounding, the parser accepts ASCII digits only, and harvest converts quantities for A2.
- [ ] **A2 Decimal strings at the API, validated corrections, frontend in step** - `records.api_field_value` / `normalize_correction` (transform `leaf["value"]` only), documents and review routers, 422 with a sentence on invalid corrections, `ReviewQueueItem.currency`, frontend `DecimalString` types, `formatAmount` groups the integer part and keeps the given fraction digits, the review page sends validated strings.
  *Verify:* backend gates, `npx eslint src`, `npm run build`; on the demo stack 016 serves `"value": "46.25"` and `"unit_price": {"value": "9.15"}`; 016 and 018 pixel-identical to before; a correction sends `"45.25"`; `45.255` shows the 422 message; light and dark at 360 and 1440 px.
- [ ] **A3 Reconciliation engine (pure)** - `backend/app/reconcile.py` (`RULES_VERSION = "rules_v1"`, `ReconcileInput.from_tool_input` / `from_fields`, `reconcile`, `report_to_json`; checks `line_math[i]`, `lines_sum`, `total_math`, `tax_rate`, `non_negative:*`, `lines_present`, `format:*`; linear `terms` so Batch B's discount, shipping, and tip generalize), `hypothesis` in `dev` with a derandomized profile.
  *Verify:* the example and property suites pass twice with identical output; all 25 labels and every Sonnet record pass; Haiku 018/019/020 fail with the expected check ids; staged 016 suggests subtotal 45.25; H3 non-list line_items fails `lines_present`; `pytest -q`.
- [ ] **A4 Persist and route** - Alembic revision adding `extracted_fields.review_reason` (backfilled, two CHECKs) and `reconciliations` (unique per extraction, cascade); `review_routing.route` (pure) and `refresh_reconciliation` (row lock, never commits) called by the worker, seed, corpus, and `resolve_field`; queue items carry reason, failing checks, and suggestion; `ExtractionOut.reconciliation`; `scripts/reconcile_all.py --dry-run`.
  *Verify:* `pytest -q` with the routing scenario table; migration downgrade/upgrade round-trips with one head; seed still reports 4 pending with 016 subtotal `reconciliation:lines_sum`; on the demo stack correcting 016 to `45.25` turns the report to `pass`; `reconcile_all.py --dry-run` then a real run on the dev stack.
- [ ] **A5 UI: reconciliation panel, computed labels, queue reasons and suggestions** - `ReconciliationPanel.tsx` below the line items (equations with a mark and the words Fails/Passes, long sums collapsed, per-line checks summarized, assumptions as notes, rules version), operands highlight their field, "computed" tags, review page shows reason, equation, and a "Use 45.25" button that prefills but never submits; `review_correction` scenario locator scoped to `#field-subtotal`.
  *Verify:* eslint and build; on the demo stack 016 shows two failing equations and the suggestion, the queue flow turns both green with "was 46.25"; 018 at 1280x800 matches the live GIFs' last frames; light and dark at 360, 768, 1440 px; keyboard only.
- [ ] **A6 Evals: caught_by_checks and the mutation eval** - `app/evals/checks.py` (`check_metrics` derived at render time from committed artifacts, deterministic `run_mutation_eval`), report and history columns, offline `evals/run_checks.py` with its own README marker table, scorer through `money.to_decimal`, harvest refuses labels that fail reconciliation.
  *Verify:* with no API key, `python evals/run.py --report-only --update-readme` and `python evals/run_checks.py --update-readme` change only marker regions and prose; the Haiku row reads 0.0% by review and 50.0% by checks; mock `caught_by_checks == 0.5` pinned; `pytest -q`.
- [ ] **A7 Demo GIF and gallery** - `reconciliation` scenario (016's failing equations and suggestion, operand highlight, queue reason, "Use 45.25", save, both checks green), `screenshots/features/reconciliation.gif` replacing `review-correction.gif`, README caption, boards rotated.
  *Verify:* `demo.py stack up`, `seed --reset`, `record reconciliation`, `check`; today's demo spend still $0.00; demo gates pass; the README renders the GIF.


## Follow-ups from Batch 0 (not yet scheduled)

- Answers can contain Markdown (`**Date:**`) that the UI renders as raw asterisks: render a safe Markdown subset or tell the prompts to answer in plain text.
- The top `screenshots/demo.gif` predates the Search, Ask, and Dashboard navigation; re-recording it with the harness costs one live extraction (about $0.02).
- The Screenshots section still shows stills that the Features GIFs now cover (`search-hybrid.png`, `search-amount.png`, `review-correct.png`).
- The macOS session's four commits carry `Co-Authored-By` trailers, against the global no-attribution rule; that machine's claude-brain setup needs bootstrapping.

## Carry-over (accepted, not blocking)

- H3: a well-formed `line_items` leaf with a non-list value silently stores `[]` (closed by A3's `lines_present` check).
- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
