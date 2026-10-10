# PREVIOUS — Completed Work

> Append-only log of finished batches/tasks, newest first. `/batch-loop` moves completed tasks here from [PRESENT.md](PRESENT.md) with the date and commit hash. Keep entries short — detail lives in git history.

## Batch 5 — Ship signals, deploy deferred (COMPLETE 2026-08-01)

### 2026-08-01 — batch-loop invocation 7 (S1–S4) — BATCH 5 COMPLETE (deploy deferred to FUTURE.md by Anthony)
- [x] S1 Container images — python:3.12-slim editable-install (BACKEND_DIR stays /app so prompts resolve), non-root, COPY allowlist + .dockerignore double-lock on .env; Next standalone 3-stage; review caught the fresh-clone break (untracked empty frontend/public/ → .gitkeep) — `acc7bd0`
- [x] S2 Full-stack compose — db/migrate/api/worker/frontend, shared uploads volume at identical mounts, key on worker only, migrate success gates api+worker; proven live (upload → extracted @ $0.0094 through the compose network); review caught the worker inheriting the API healthcheck (permanently unhealthy) — disabled for worker+migrate; CI gains compose config job — `ecd531f`
- [x] S3 Live smoke suite — RUN_LIVE_SMOKE=1 opt-in, $0.10 hard cap, 2 round-trips against the committed eval corpus (one direct + one through the queue with document_id scoping); live run 2 passed @ $0.019 — `3334294`
- [x] S4 README final pass — compose-first quickstart, "Why these tech choices" section, cost figures reconciled (smoke sample vs eval mean), invisible LIVE_DEMO slot, eval table byte-untouched — `f142563`
- Batch totals: 212 mocked tests + 2 opt-in live; live spend this batch ≈ $0.03 (one compose-path upload + smoke run), all under caps.

## Batch 4 — Cost tracking + failure hardening (COMPLETE 2026-08-01)

### 2026-08-01 — batch-loop invocation 6 (H4–H6) — BATCH 4 COMPLETE
- [x] H4 Model-refusal path — ModelRefusalError subclass → terminal "refused" document status, WCAG-AA status chip (9.37:1/11.65:1), distinct detail-page copy; clean SHIP review — `56d9def`
- [x] H5 Rate-limit-aware backoff — 4xx client errors fail fast (413 points at chunking), 429/529/5xx carry retry-after into the queue as an equal-jitter backoff floor (capped 900s post-review); SDK exception hierarchy verified against installed anthropic 0.120.2 — `28ebdc4`
- [x] H6 PDF chunking — pypdf split above 5 pages, sequential per-page calls, merge with earliest/latest tie-breaks + 0.5 conflict cap, _chunks provenance in raw_response, zero-spend refusal over 20 pages/zero pages; single-call canary unchanged; README failure-handling table — `c999c24`
- 212 backend tests green; eval tests proven green with Postgres stopped; mock accuracy 0.988571 stable all batch; $0 live API spend (optional ≤$0.15 live-confirmation gate never needed).

### 2026-07-31/08-01 — batch-loop invocation 5 (H1–H3)
- [x] H1 /stats endpoint — SQL rollups + percentile_cont, review counts reuse _PENDING, eval artifact projection; Fable caught a malformed-artifact 500 — `a432573`
- [x] H2 Stats strip — fetch-once client component, degrades to nothing on backend-down; review caught fractional-ms rendering — `e9e0d86`
- [x] H3 Schema-repair reprompt — is_error tool_result retry, strictly-fewer-violations acceptance (review fixed truncation-corrupted counting), repair_v1.md versioned separately, schema_repair off-switch — `8a5a999`

## Batch 3 — Human-in-the-loop review queue (COMPLETE 2026-07-31, external session — pulled from origin)

Completed in another session and discovered via `git pull` on 2026-07-31; boards synced after the fact.

- [x] Review queue — field-level approve/correct endpoints (`routers/review.py`), review columns migration `b7a91c4de2f0` (review_action/reviewed_at/corrected_value as the audit trail — columns on `extracted_fields`, not a separate corrections table), full review UI (`frontend/src/app/review/page.tsx`), queue badge, 230-line test file — `c98df60`
- [x] Data flywheel — `app/evals/harvest.py` + `scripts/harvest_corrections.py`: fully-reviewed documents export as new eval labels with `source: "human-review"` into the main `evals/docs|labels` dirs (deviation from the planned `evals/from_corrections/` — labels validated by the eval loader's own validator, idempotent via source_document_id); **plus CI (GitHub Actions, lint+migrate+test both stacks) and 4 README screenshots pulled forward from Batch 5** — `a78ed5d`
- [x] Harvest tests made robust to a shared dev database — `8f63cf5`
- [x] Security hardening (unplanned) — upload content sniffing, `X-Content-Type-Options: nosniff`, tightened CORS + DB binding — `046ea79`
- [x] Design system modernization (unplanned) — WCAG-verified palette in both themes, focus rings, reduced-motion support — `5d9a82b`

## Batch 2 — Evals (COMPLETE 2026-07-31)

### 2026-07-31 — batch-loop invocation 4 (E3–E5) — BATCH 2 COMPLETE
- [x] E3 Runner + CLI — semaphore concurrency, cost cap that counts billed-but-errored calls, --mock with analytically predicted accuracy asserted exactly — `05edaa9`
- [x] E4 Report + README injection — generated tables traceable to committed artifacts, marker-region-only rewrites — `359601e`
- [x] E5 Live runs — Sonnet 99.4% ($0.0125/doc, caught_by_review 1.0) vs Haiku 95.4% ($0.0052/doc, caught_by_review 0.0); README table live; $0.51 total spend — `ff1a1d1`

### 2026-07-30 — batch-loop invocation 3, continued (E1–E2)
- [x] E1 Synthetic eval dataset — 25 docs/labels, 8 difficulty classes, record-first generation (labels correct by construction), deterministic — `5bb357d`
- [x] E2 Loader + scorer — persistence-parity scoring via _coerce_leaf, integer-cent tolerance (Fable caught the float bug), loud label validation, caught_by_review metric; 99 tests total — `ad7d725`

## Batch 1 — Stage 1 core loop (COMPLETE 2026-07-30)

### 2026-07-30 — batch-loop invocation 3 (T7 + Batch 2 planning)
- [x] T7 Wire-up + docs — README quickstart (verified from a literal clean-venv follow-through: 26 tests + one live extraction $0.0114/5.2s), mermaid architecture diagram, 3 PIL-generated PII-free samples — `5445496`
- Batch 1 done: the full core loop works end to end. Batch 2 (evals) promoted from FUTURE.md, expanded into E1–E5 by the Opus planner against the real code.

### 2026-07-30 — batch-loop invocation 2 (T4–T6)
- [x] T4 SKIP LOCKED worker — claim/commit split, run_after backoff, orphan reclaim; Fable proved 2 high-sev bugs live (poisoned session, stranded jobs), fixed + regression-tested — `edcb524`
- [x] T5 VLM extraction — forced tool-use, {value, confidence} leaves, prompt as versioned file, cost/latency persisted; live smoke $0.0086 / 3.7s; 7 review findings fixed (malformed-leaf coercion, non-retryable classification, prompt-injection guard) — `6f23b9a`
- [x] T6 Next.js viewer — upload, split view, confidence badges, cost strip; 6 review findings fixed (self-healing polling, crash guards, keyboard a11y) — `09eb4d9`
- GitHub remote wired up mid-invocation (Antheagao/doc-pilot); pushing is now part of the loop protocol
- API key relocated from tracked .env.example → gitignored backend/.env before it could be committed

### 2026-07-30 — batch-loop invocation 1 (T1–T3)
- [x] T1 Backend scaffold — FastAPI + healthz, compose Postgres (host port **5434**; 5432/5433 taken by workoutDB containers), .env.example, .gitignore — `a0a4099`
- [x] T2 DB schema — documents/jobs/extractions/extracted_fields, async Alembic, migration applied + verified in psql — `d06ea87`
- [x] T3 Upload + document APIs — POST/list/detail/file, ASGI body-size middleware, mime allowlist, single-transaction insert; deep-reviewer found 7 defects (all fixed), 7 tests green — `a3f85c3`

## Pre-build (before batch loop existed)

### 2026-07-30
- [x] Repo created, git initialized, README written (pitch + architecture + principles) — commit `1fc1ad6`
- [x] CLAUDE.md + UPDATES.md session tooling created
- [x] Multi-model agent workflow verified (planner=Opus, fast-reader=Haiku, coder=Sonnet, deep-reviewer=Fable; `tiered-task` skill)
- [x] Batch loop system created: PRESENT/PREVIOUS/FUTURE task files + `/batch-loop` command
