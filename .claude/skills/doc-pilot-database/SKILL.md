---
name: doc-pilot-database
description: doc-pilot's PostgreSQL 16 plus pgvector schema, Alembic revision chain, async SQLAlchemy session rules, money storage, and known gaps. Use when writing an Alembic migration, changing a model, column, or index, or writing a query or transaction in doc-pilot.
---

# doc-pilot database overlay

Load the global `database` guide for the rules; this file holds what exists here.
The sqlx subsections of the global guide map to SQLAlchemy and Alembic here.

## Stack and migrations

- PostgreSQL 16 with pgvector (`pgvector/pgvector:pg16-trixie`, pinned to match the glibc 2.41 the dev volume was initialized under), SQLAlchemy 2 async ORM over asyncpg; models in `backend/app/models.py`, engine and `async_session_maker` (`expire_on_commit=False`) in `app/db.py`.
- Migrations: `backend/alembic/versions/<revision>_<slug>.py`; async `alembic/env.py` sets `target_metadata = Base.metadata`; the module docstring's first line is the header.
- Chain: `0eb534561f35` initial, `c538cc803902` `jobs.run_after`, `b7a91c4de2f0` review columns, `e3f19a7c2d40` retrieval tables and `CREATE EXTENSION vector`, `f7c2d9e4a1b3` `jobs.traceparent`, `a4d8e2f6b901` `search_aliases`, `b9e4c1d7f302` `ask_runs`, `c6f1a2b3d4e5` spend time indexes, `d2a7b5c9e013` chat columns, `e8c3a6f1b254` reconciliations and `extracted_fields.review_reason`, `a7d2f4c9b108` the `quote:` review reason (head; its downgrade is lossy: `quote:%` rows become `confidence`).
- Compose's one-shot `migrate` service runs `alembic upgrade head`; the test session does the same to its own `docpilot_test` database (created when missing) before the first test.

## Schema facts

- Tables: `documents`, `jobs`, `extractions`, `extracted_fields`, `reconciliations`, `document_pages`, `document_chunks`, `ask_runs`.
- Local exception: primary keys are `UUID` with a Python-side `uuid.uuid4` default, because ids appear in URLs and upload file names.
- Timestamps are `DateTime(timezone=True)` with `server_default=func.now()`; no table has `updated_at`.
- `document_chunks.embedding` is `Vector(384)` (`EMBEDDING_DIM`) with an HNSW `vector_cosine_ops` index; `tsv` is a persisted generated column with a GIN index, and `a4d8e2f6b901` shows how to change one (drop the index and column, add both back).
- A new embedding width is a migration plus `backend/scripts/reindex.py`; migrations copy constants as literals (`EMBEDDING_DIM = 384` in `e3f19a7c2d40`), never import them.
- `CHECK` constraints today: `ck_jobs_has_target`, `ck_ask_runs_feedback`, `ck_ask_runs_chat_scope`, `ck_extracted_fields_review_reason` (`review_reason ~ '^(confidence|reconciliation:.+|quote:.+)$'`), `ck_extracted_fields_review_reason_matches` (`needs_review = (review_reason IS NOT NULL)`), `ck_reconciliations_status` (`pass`, `fail`, `skipped`).
- Review flags: `extracted_fields.review_reason` says why a field is flagged (`REVIEW_REASON_CONFIDENCE`, `reconciliation_reason(check_id)` in `models.py`, or a quote check's id `quote:<path>`, prefix `REVIEW_REASON_QUOTE_PREFIX`).
  A resolved field keeps its `needs_review` and reason (the queue filters on `reviewed_at`), which is why the backfill in `e8c3a6f1b254` set `'confidence'` on every row with `needs_review`, reviewed or not; every writer that sets `needs_review` must set the reason in the same statement.
- `reconciliations`: one row per extraction (`uq_reconciliations_extraction_id`, `ON DELETE CASCADE` from `extractions`), `report` is `reconcile.report_to_json` verbatim, `status` and `rules_version` repeat its keys.
  There is no index on `status` (three values, one row per extraction); add one when a Batch D query measures the need.
  Written only by `review_routing.refresh_reconciliation`, which locks the extraction row (`FOR UPDATE`), never commits, and is called by the worker (inside a savepoint), the seed, the eval corpus, `resolve_field`, and `scripts/reconcile_all.py`.
- Money: `cost_usd` columns are `Numeric(10, 6)` typed `Mapped[float]`; extracted amounts are JSON numbers inside `extracted_fields.value` leaves (`{"value", "confidence"}`), and corrections sit in `corrected_value`.
  Stored leaves stay the model's JSON numbers (no data migration); new corrections are stored as canonical decimal strings (`records.normalize_correction`), and older numeric corrections are read through the same path.
  The API serves every document amount as a decimal string, parsed with `Decimal` in `app/money.py`; the eval scorer compares integer cents.
- Day boundaries are UTC in code: `budget.day_start`, and `routers/monitoring.py` groups with `func.timezone("UTC", ...)`.

## Session rules

- Job handlers never commit: the worker commits in `complete_job` or rolls back in `fail_job`, and each handler rolls back before its model call so no transaction spans it.
- After `session.rollback()` every loaded ORM object is expired, and `AsyncSession` cannot lazy-load, so reading `job.id` raises `MissingGreenlet`.
  Copy ids into locals before any rollback, carry the ids, and re-fetch with `session.get(Model, id)` (as `process_document_job` does for the document).
- Queries are ORM or Core expressions; raw `text()` appears only for `SET LOCAL enable_indexscan = off` in `app/evals/`.

## Known gaps

- Closed sets with no `CHECK`: `documents.status`, `jobs.state`, `jobs.kind`, `extracted_fields.review_action`, `document_pages.source`, `ask_runs.status`.
- No `ON DELETE` except `jobs.ask_run_id` (`CASCADE`), and no index on `jobs.document_id`, `extractions.document_id`, or `document_chunks.page_id`; no document deletion path exists (Batch G4).
- No migration states a retention period for receipt text (`document_pages`, `document_chunks`) or stored questions and answers (`ask_runs`).
