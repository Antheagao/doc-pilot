---
name: doc-pilot-database
description: doc-pilot's PostgreSQL 16 plus pgvector schema, Alembic revision chain, async SQLAlchemy session rules, money storage, and known gaps. Use when writing an Alembic migration, changing a model, column, or index, or writing a query or transaction in doc-pilot.
---

# doc-pilot database overlay

Load the global `database` guide for the rules; this file holds what exists here.
The sqlx subsections of the global guide map to SQLAlchemy and Alembic here.

## Stack and migrations

- PostgreSQL 16 with pgvector (`pgvector/pgvector:pg16-trixie`, pinned to match the glibc 2.41 the dev volume was initialized under), SQLAlchemy 2 async ORM over asyncpg; models in `backend/app/models.py`, engine and `async_session_maker` (`expire_on_commit=False`) in `app/db.py`.
- Two database URLs: `DATABASE_URL` is the runtime login role `docpilot_runtime` (the suite uses `docpilot_test_runtime`), which row-level security binds, and `MIGRATION_DATABASE_URL` is the owner `docpilot`, used only by `alembic/env.py` (`migration_database_url or database_url`) and `scripts/provision_db_role.py`.
- Migrations: `backend/alembic/versions/<revision>_<slug>.py`; async `alembic/env.py` sets `target_metadata = Base.metadata`; the module docstring's first line is the header.
- Chain: `0eb534561f35` initial, `c538cc803902` `jobs.run_after`, `b7a91c4de2f0` review columns, `e3f19a7c2d40` retrieval tables and `CREATE EXTENSION vector`, `f7c2d9e4a1b3` `jobs.traceparent`, `a4d8e2f6b901` `search_aliases`, `b9e4c1d7f302` `ask_runs`, `c6f1a2b3d4e5` spend time indexes, `d2a7b5c9e013` chat columns, `e8c3a6f1b254` reconciliations and `extracted_fields.review_reason`, `a7d2f4c9b108` the `quote:` review reason (its downgrade is lossy: `quote:%` rows become `confidence`), `b4e7c1a9d625` `extractions.cache_creation_input_tokens` and `cache_read_input_tokens`, `aa652338a4ae` the five identity tables and `CREATE EXTENSION citext`, `127aac687508` `organization_id` on the eight content tables with the composite foreign keys (its downgrade is lossy: every organization's rows land in the default one on the next upgrade), `73a7d8691932` the grants, row-level security, and `docpilot_spend_since` (head).
- PostgreSQL 16 is a floor, not just the pinned image: `app/agent/extraction_query.py` calls `pg_input_is_valid`, which exists from 16 on (the compose image, the CI service, and the test server are 16.15).
- `pg_stat_statements` exists only in the load stack (`docker-compose.load.yml`): the overlay preloads the library with `shared_preload_libraries`, and `load.py preflight` runs `CREATE EXTENSION IF NOT EXISTS pg_stat_statements` as the owner in `docpilot_load` only (it refuses any other database name), so no migration creates it and the dev, demo, and test databases never have it; the harness resets it before each phase and snapshots the top statements into the artifact.
- Compose's one-shot `migrate` service runs `alembic upgrade head` as the owner and then `scripts/provision_db_role.py`; the test session migrates its own `docpilot_test` database (created when missing) as the owner and provisions `docpilot_test_runtime` before the first test.
- A login role and its password do not belong in a migration: `app/db_roles.py` creates or alters the runtime role from the user and password of `DATABASE_URL` over `MIGRATION_DATABASE_URL`, sends a SCRAM verifier, and refuses an owner or a different server or database.
- Roles are cluster-wide and the dev, demo, and test databases share a cluster, so the group role `docpilot_app` (NOLOGIN, holds every grant) is dropped by a downgrade only when no login role is a member and no other database grants to it (SQLSTATE `2BP01` otherwise).

## Schema facts

- Content tables (`TenantMixin`): `documents`, `jobs`, `extractions`, `extracted_fields`, `reconciliations`, `document_pages`, `document_chunks`, `ask_runs`.
  Identity tables (global, no RLS, `IDENTITY_TABLES`): `organizations`, `users` (`email citext`), `memberships` (PK `(organization_id, user_id)`, role `owner` or `member`), `auth_sessions`, `api_keys`.
- `organization_id uuid NOT NULL DEFAULT docpilot_current_organization_id()` is on every content table; code never sets it.
  The function is `STABLE` and reads `nullif(current_setting('docpilot.organization_id', true), '')::uuid`, so with no tenant it is NULL and an insert is refused.
  The default organization has the literal id `270ad65a-150a-498e-bbb1-584f6d21d86e` and slug `default` (copied into migrations, not imported).
- Composite foreign keys tie a child to its parent's organization, each backed by `UNIQUE (id, organization_id)` on the parent (`documents`, `extractions`, `document_pages`, `ask_runs`): `(document_id, organization_id)` on jobs, extractions, document_pages, document_chunks, and ask_runs, `(extraction_id, organization_id)` on extracted_fields and reconciliations, `(page_id, organization_id)` on document_chunks, and `(ask_run_id, organization_id)` on jobs.
  A nullable `document_id` skips the check (MATCH SIMPLE).
- The two settings: `docpilot.organization_id` (the tenant) and `docpilot.queue_scope` (`on` only in `tenancy.queue_session`), both transaction-local and set by the `after_begin` listener on every new transaction.
- Policies: `tenant_isolation` (`FOR ALL`, `USING` and `WITH CHECK` the tenant) on each content table with ENABLE and FORCE, plus `jobs_queue_claim` (`FOR SELECT`) and `jobs_queue_update` (`FOR UPDATE`) on `jobs` alone.
  The role holds `UPDATE` on `jobs` only for `state`, `attempts`, `last_error`, `started_at`, `finished_at`, and `run_after`; `alembic_version` has no grant.
  A new table needs its own `GRANT` to `docpilot_app` and, for a content table, its policies, or `tests/test_rls.py` fails.
- `docpilot_spend_since(timestamptz) RETURNS numeric` is `SECURITY DEFINER` with a fixed `search_path`, executable only by `docpilot_app`; it is the one way the application sums across tenants.
- Measured search plans under RLS (PostgreSQL 16.15, pgvector 0.8.7):
  - The GIN index on `tsv` cannot serve `tsv @@ query`, because the match function is not leakproof, so lexical search is the tenant's rows through `ix_document_chunks_organization_id` plus a filter; `_term_idf` counts the tenant's chunks.
  - The HNSW index still orders dense search with the policy as a filter, a tiny tenant is planned as an exact sort over its own rows, and `_dense_ranking` sets `hnsw.iterative_scan = relaxed_order` so a small tenant still fills `k`; the retrieval eval pins `enable_indexscan = off`, so its numbers do not move.
- `query_extractions` is answered in SQL (`app/agent/extraction_query.py`, `find_matches`): the filters, the count, and the exact per-currency sums run as one ORM select over each document's latest extraction (`DISTINCT ON (document_id)`), amounts read as `numeric`, and user text is only ever a bound parameter compared with `strpos()`, never a `LIKE` pattern.
  Each filter is three-valued: where SQL and the Python rule `ExtractionFilters.matches` are not certain to agree (case-folding edge cases, non-ASCII text, rare ISO date forms, oddly shaped corrections, a value that is not a JSON string), the result is NULL and that document is loaded and judged by the unchanged Python matcher, so the answers equal the old all-Python ones; `tests/test_extraction_query.py` holds a property against a verbatim copy of the old implementation.
  Being ORM selects, the statements get the organization filter and row-level security like any other.
- Run `ANALYZE` after a bulk load: autovacuum's analyze notices thousands of new rows up to a minute late, and a query planned on stale estimates can be ten times slower; the load harness runs it as the owner before every measured window (`phases.ANALYZE_SQL`).
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
- Settling a job is a compare-and-set: `complete_job`, `fail_job`, and `requeue_job` update `jobs` `WHERE id, state = 'processing' AND attempts = <the attempt this worker claimed>` (`worker._still_claimed`) in the handler's own transaction, and zero rows updated means the job was reclaimed, so the handler's writes roll back and the outcome is `stale`.
  `reclaim_orphaned_jobs` runs on the database clock (`now() - lease`), fails a job that has used `MAX_ATTEMPTS` instead of requeueing it, and keeps a requeued job's `attempts`, so the next claim's number differs from the first worker's.
  The column-level `UPDATE` grant on `jobs` is unchanged by this.
- Each process names its connections (`db.name_process`: `doc-pilot-api`, `doc-pilot-worker`, `doc-pilot-mcp` in `pg_stat_activity.application_name`), and its pool is `DB_POOL_SIZE` plus `DB_MAX_OVERFLOW` with a `DB_POOL_TIMEOUT_SECONDS` wait; Postgres allows 100 connections across every process, so size them together when you scale workers.
- After `session.rollback()` every loaded ORM object is expired, and `AsyncSession` cannot lazy-load, so reading `job.id` raises `MissingGreenlet`.
  Copy ids into locals before any rollback, carry the ids, and re-fetch with `session.get(Model, id)` (as `process_document_job` does for the document).
- Queries are ORM expressions on a session bound to an organization (`get_session`, `tenant_session`, `queue_session`); a content table is never reached through `text()`, a `Table` statement, or a write that names `organization_id`, which `app/tenancy.py` refuses or row-level security catches.
  Raw `text()` appears for `SET LOCAL enable_indexscan = off` in `app/evals/` and for the `set_config` calls of `app/tenancy.py`, both with bound parameters.
- Identity rows are read and written on a session with no organization (`get_identity_session_factory`), only by `app/auth/`, the auth routes, `scripts/accounts.py`, and the demo seed.
- A new session that touches content needs `bind_tenant` first; `TenantScopeError` is the sign it was missed.

## Known gaps

- Closed sets with no `CHECK`: `documents.status`, `jobs.state`, `jobs.kind`, `extracted_fields.review_action`, `document_pages.source`, `ask_runs.status`.
- No `ON DELETE` on the content keys except `jobs.ask_run_id` and `reconciliations.extraction_id` (`CASCADE`), and `organizations.id` is `RESTRICT`; no document, user, or organization deletion path exists (Batch G4).
- No migration states a retention period for receipt text (`document_pages`, `document_chunks`) or stored questions and answers (`ask_runs`).
