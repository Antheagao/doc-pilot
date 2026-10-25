# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## Batch N - Auth and multi-tenant isolation (planned 2026-10-25)

Planned by the planner against master `57a5fe1`, from item 4 of the portfolio queue (`project-dossier/PROJECT-QUEUE.md`).
Goal: users and organizations, session sign-in for the browser, API keys for MCP and programmatic access, and every query scoped to one organization twice over: by the ORM layer in the app and by Postgres row-level security under a non-owner role.
Done when: the README Security model section is rewritten for multi-tenant use and the isolation tests run in CI.
doc-pilot stays local-only and is never hosted; the design is built as if it would be, and nothing is deployed.
Model spend is $0.00 for the whole batch: no task calls a model, and every agent test uses the scripted fake client.

### Spike results the design rests on (2026-10-25, throwaway database and role on the dev server, both dropped)

- Server: PostgreSQL 16.15, pgvector 0.8.7 (iterative HNSW scans exist), citext 1.6 available.
  The compose and CI role `docpilot` is a superuser with BYPASSRLS that owns every table, so today RLS could not apply to the app at all.
- A `STABLE` SQL function `docpilot_current_organization_id()` reading `nullif(current_setting('docpilot.organization_id', true), '')::uuid` works as both the column default and the policy predicate.
  The planner can use it as an index condition (`Index Cond: organization_id = ...`).
- Under the non-owner role with FORCE RLS:
  - With no tenant set, reads return 0 rows and an insert is refused ("new row violates row-level security policy").
  - An explicit foreign `organization_id` is refused by WITH CHECK.
  - A child row that points at another tenant's parent is refused by the composite foreign key `(document_id, organization_id)`.
  - The app role cannot `SET ROLE docpilot`.
- `set_config(..., true)` set at transaction start survives a rolled-back savepoint (the `execute_tool` savepoints are safe).
  A value set inside a rolled-back savepoint reverts.
  After commit or rollback the value is gone, so it must be re-applied on every transaction.
- SQLAlchemy 2.1.4: a `Session` `after_begin` listener fires on every new transaction, including after a handler's `rollback()` and in the test harness's `join_transaction_mode="create_savepoint"` sessions, so it is the one place to apply the tenant.
- `do_orm_execute` plus `with_loader_criteria(TenantMixin, ...)` filters `select(func.count()).select_from(Model)`, `session.get`, and ORM `update`, even on a superuser connection.
  `text()` bypasses it, and RLS catches that.
- A `SECURITY DEFINER` function owned by the superuser owner returns the sum over all tenants while a direct `SUM` under tenant A returns A's only.
- Measured cost of RLS on search:
  - The GIN full-text index cannot serve `tsv @@ query` under RLS, because `ts_match_vq` is not leakproof (as the owner it is a bitmap index scan).
    So lexical search becomes the tenant's rows through the `organization_id` index plus a filter.
  - The HNSW index still serves the dense ordering, with the policy as a filter; a tiny tenant is planned as an exact sort over its own rows.
  - Fine at local scale; it becomes a documented known limit.
- `DROP ROLE` fails with SQLSTATE `2BP01` while another database on the cluster still grants to the role, and `docpilot` and `docpilot_test` share one cluster.
- `argon2-cffi` 25.1.0 (latest; wheel read, not installed): `PasswordHasher` defaults to `profiles.RFC_9106_LOW_MEMORY` (argon2id, t=3, 64 MiB, p=4, 16-byte salt).
  `verify` raises `VerifyMismatchError`, and `check_needs_rehash` exists.
  `pwdlib` 0.3.1 is the alternative and was not chosen (it wraps the same bindings with less control).
- mcp 2.3.0:
  - `mcp.server.auth.provider.TokenVerifier` (`verify_token(token) -> AccessToken | None`), `AccessToken(token, client_id, scopes, expires_at, resource, subject, claims)`, and `mcp.server.auth.middleware.bearer_auth.AuthenticatedUser` exist.
  - In both protocol eras the stateless HTTP transport hands the tool handler `ServerRequestContext.request`, the Starlette request over our ASGI scope.
  - `RequireAuthMiddleware` answers with an OAuth-shaped `{"error", "error_description"}` body, and the OAuth authorization-server parts need a provider we do not have.

### Key design decisions (planner defaults; Anthony can override)

Identity and browser sessions:
- Server-side sessions in an `auth_sessions` table, not JWTs.
  Logout and a password change must take effect at once (a row delete), there is one API process with the database beside it, and the global security guide prescribes exactly this.
- The session token is `secrets.token_urlsafe(32)` (256 bits) in an HttpOnly cookie.
  Only its SHA-256 digest is stored (`bytea`, 32 bytes), with an absolute expiry `SESSION_TTL_HOURS` (default 12, a working day).
  There is no sliding renewal, so a read never writes.
- Cookie: `HttpOnly; SameSite=Lax; Path=/`, `Secure` when `SESSION_COOKIE_SECURE` (default false: the stacks are plain HTTP on loopback).
  The name comes from `SESSION_COOKIE_NAME`, and the demo overlay sets its own name, because cookies ignore ports, so the dev API (8000) and demo API (8100) would otherwise overwrite each other's cookie on `localhost`.
- A cookie, not a bearer token in `localStorage`, for the browser.
  `<img>` and `<iframe>` previews of `GET /documents/{id}/file` cannot send an Authorization header, and the frontend (3000) and API (8000) on `localhost` are same-site, so Lax cookies ride on fetches and subresources.
- CSRF: every cookie-authenticated unsafe request (POST, PATCH, PUT, DELETE, login and logout included) must carry an `Origin` in `CORS_ORIGINS`, or it gets 403.
  `SameSite=Lax` does not cover sibling origins, and every other local app on `localhost:<port>` (8080 belongs to another project) is same-site with the API.
  Bearer requests are exempt, because CSRF is a cookie problem.
- CORS turns on `allow_credentials` with the explicit `CORS_ORIGINS` list.
  A validator refuses `"*"`, and the allowed methods and headers are narrowed to what the frontend sends.
- Passwords: argon2id through `argon2-cffi>=25.1,<26`, pinned to `profiles.RFC_9106_LOW_MEMORY` explicitly, so a library default change never silently changes parameters.
  `check_needs_rehash` runs on each login.
  Hashing runs in the threadpool under a small semaphore (2), since 64 MiB per hash times the threadpool would otherwise be a memory bomb.
  Length is 12 to 128 characters, enforced by the request schema before any hashing.
  An unknown or disabled email is verified against a lazily built dummy hash.
  There is one message, "email or password is incorrect".
- Login rate limits come before hashing: `LOGIN_RATE_LIMIT_PER_MINUTE` (10, per client address) and `LOGIN_ACCOUNT_RATE_LIMIT_PER_MINUTE` (5, keyed by the SHA-256 of the case-folded email).
  Both use the existing in-memory `SlidingWindowLimiter` and answer 429 with `Retry-After`.
  In memory fits the documented single API process; a restart resets it (a known gap).
- No open self-signup.
  Accounts are created by `backend/scripts/accounts.py` (`create-organization`, `create-user`, `add-member`, `set-password`, `create-api-key`), and the demo seed creates the demo user.
  A public signup on a loopback app adds an unauthenticated write path that needs email verification and abuse handling to be correct, and it has no user here.
  Passwords come from `getpass` or `--password-stdin`, never argv.
- Roles: `memberships.role` is `owner` or `member` (TEXT plus CHECK).
  The one gated action is managing API keys (owner only; a member gets 403 and may list the keys).
  An API key is an organization credential that outlives a session and bypasses the cookie and CSRF protections, so issuing one is an admin act.
- Organizations switch per session: the session row pins one organization (the earliest membership at login), and `POST /auth/organization` switches it (rotating the token).
  The UI shows a switcher only when the user has more than one membership, so it costs nothing at one organization.

API keys and MCP:
- Format: `dpk_<12 lowercase base32>_<token_urlsafe(32)>`.
  The prefix is stored unique for lookup and shown in the UI; the SHA-256 digest of the whole token is stored and compared with `hmac.compare_digest`.
  A malformed token is refused without a database query.
  SHA-256, not argon2: a 256-bit random secret cannot be guessed, and a slow hash would only add CPU to every MCP request.
- Each key is scoped to one organization, shown once in the `201` response, revocable (`revoked_at`, row kept for audit), and capped at 25 active keys per organization (409).
  Its `last_used_at` is written at most once a minute.
  A key may call every tenant route except key and session management (403: "an API key cannot manage API keys").
- `/mcp` authenticates with `Authorization: Bearer dpk_...`, using our own `ApiKeyVerifier` that implements the SDK's `TokenVerifier` protocol.
  On success it sets `scope["user"] = AuthenticatedUser(AccessToken(client_id="api-key:<id>", scopes=["mcp:tools"], subject=<org id>, claims={"organization_id": ...}))`, so the SDK's `get_access_token()` sees it, and `on_call_tool` reads the principal from `ctx.request.scope`.
  We keep our own 401 (`{"detail"}` plus `WWW-Authenticate: Bearer realm="doc-pilot"`) instead of `RequireAuthMiddleware`, to keep the API's error body and our check order.
  The OAuth 2.1 authorization-server flow of the MCP spec is out of scope; clients pass a static header.
- `/mcp` check order: 503 (not running) -> Host (421) and Origin (403) -> method (405) -> bearer key (401) -> the rate limit in the `mcp` bucket keyed per API key (429) -> SDK (413 and so on).
  Every request is authenticated, so a key revoked mid-session stops working at its next call.
- Stdio: `python -m app.mcp_server` requires `DOCPILOT_API_KEY` (a `SecretStr` Settings field with that alias).
  It verifies the key at startup (exit 2 with a stderr reason) and again on every tool call (a revoked key gives an `isError` result).
  This was chosen over an `--organization` flag: a key is revocable, auditable (`last_used_at`), and the same credential as HTTP, while a flag is trusted input naming a tenant.
- Of the items Batch M deferred to auth, this batch does API keys and per-key rate buckets.
  These stay deferred and are written down as such: a per-key document scope (`ToolContext.document_ids` remains the seam), non-loopback hosts and TLS, browser origins, a tool-call audit log (Batch G1's `field_events` is its natural home), and MCP resources.

Tenancy:
- Tables: `organizations`, `users` (`email citext` unique, `disabled_at`), `memberships` (PK `(organization_id, user_id)`), `auth_sessions` (FK `(organization_id, user_id)` -> memberships `ON DELETE CASCADE`, so removing a membership ends its sessions), and `api_keys`.
  These identity tables are global, are touched only by `app/auth/`, and have no RLS: a login and a key lookup must read them before any tenant is known.
- `organization_id uuid NOT NULL DEFAULT docpilot_current_organization_id()` goes on all eight content tables: `documents`, `jobs`, `extractions`, `extracted_fields`, `reconciliations`, `document_pages`, `document_chunks`, `ask_runs`.
  It is denormalized rather than derived through `documents`, because a policy with a subquery per row would wreck HNSW and GIN search.
- Composite foreign keys tie each child to its parent's tenant, so no bug can attach tenant B's row to tenant A's document:
  - `(document_id, organization_id)` -> `documents(id, organization_id)` on jobs, extractions, document_pages, document_chunks, and ask_runs (nullable `document_id`: MATCH SIMPLE skips the check when it is NULL).
  - `(extraction_id, organization_id)` -> `extractions` on extracted_fields and reconciliations (`ON DELETE CASCADE` kept).
  - `(page_id, organization_id)` -> `document_pages` on document_chunks.
  - `(ask_run_id, organization_id)` -> `ask_runs` on jobs (`ON DELETE CASCADE` kept).
- One rule for writes: code never sets `organization_id`; the database fills it from the transaction's tenant, and WITH CHECK refuses anything else.
  This is one home for the rule and almost no churn across the roughly 20 insert sites and test fixtures.
- Two independent isolation layers, both fail-closed and each tested alone:
  1. App layer (`app/tenancy.py`):
     - `bind_tenant(session, organization_id)` stores the tenant in `session.info` and, if a transaction is open, applies it at once.
     - An `after_begin` listener runs `SELECT set_config('docpilot.organization_id', :org, true)` on every new transaction.
     - A `do_orm_execute` listener adds `with_loader_criteria(TenantMixin, organization_id == bound)` to every ORM select, update, and delete.
     - It raises `TenantScopeError` before any SQL when a statement touches a tenant model on a session with no tenant bound, so a forgotten binding is loud even on a superuser connection.
  2. Database layer: ENABLE plus FORCE RLS on the eight tables, with policy `tenant_isolation` `FOR ALL USING (organization_id = docpilot_current_organization_id()) WITH CHECK (same)`, under a non-owner, non-superuser, no-BYPASSRLS runtime role.
- The worker's queue claim is the one cross-tenant read.
  A second permissive policy on `jobs` only, `FOR SELECT` and `FOR UPDATE`, admits `current_setting('docpilot.queue_scope', true) = 'on'`.
  `claim_job` and `reclaim_orphaned_jobs` run in a `queue_session()`; the claim then binds the job's organization in the same transaction before it touches the document.
  Every handler, plus `complete_job` and `fail_job`, runs in a tenant session for `job.organization_id`.
  This keeps the tested Python `FOR UPDATE SKIP LOCKED` claim as it is, and jobs hold ids and state, not document text.
- Honest threat statement (README and security overlay): the tenant GUC is set by the app, so RLS here contains application bugs (a query that forgets its filter, a `text()` query, a new route), not an attacker who can run arbitrary SQL as the app role.
  Every query is bound, and there is no injection surface.
  Making the tenant unforgeable (a definer function that sets it from a verified session token) is noted as a hardening option, not built.
- Postgres roles: the group role `docpilot_app` is NOLOGIN and holds every grant.
  Migrations create it if missing.
  Per-environment LOGIN roles are members of it: `docpilot_runtime` (dev, demo, CI) and `docpilot_test_runtime` (the suite).
  Roles are cluster-wide and the dev and test databases share a cluster, so the suite must never change the password the running dev stack logs in with.
  `MIGRATION_DATABASE_URL` (the owner) runs Alembic and the provisioner; `DATABASE_URL` is the runtime role.
- A startup guard `db.verify_tenant_isolation(engine)` refuses to serve when the runtime role is a superuser, has BYPASSRLS, or owns `documents`.
  It runs in the API lifespan, the worker's `main`, and stdio MCP, so a host `backend/.env` that still names the owner fails loudly instead of silently turning RLS off.
  There is no escape hatch.
- Search:
  - `_dense_ranking` also sets `hnsw.iterative_scan = relaxed_order` (pgvector 0.8), so a small tenant among large ones still fills `k` from a post-filtered HNSW scan; the outer query already re-sorts by distance.
  - `_term_idf` becomes per-tenant on its own (counting the tenant's chunks), which is correct: global IDF would leak other tenants' corpus statistics into rankings.
  - The retrieval eval runs with `enable_indexscan = off`, so its numbers do not move.
- Eval CLIs, `seed_labeled_corpus`, and the retrieval and agent evals create an ephemeral organization inside their rolled-back transaction (`tenancy.ephemeral_organization`), so an eval corpus can never mix with real data.
- Maintenance scripts:
  - `reconcile_all.py` and `reindex.py` run per organization (all by default, `--organization <slug>` to narrow), each in its own tenant session.
  - `harvest_corrections.py` requires `--organization`, because it exports tenant data into the public repo's labels, which must be a deliberate choice.
  - `smoke.py` binds `--organization` (default `default`).

Spend and rate limits:
- `DAILY_BUDGET_USD` stays global: it is the one Anthropic key's bill on one machine.
  It is computed by `docpilot_spend_since(timestamptz) RETURNS numeric`, `SECURITY DEFINER`, `SET search_path = pg_catalog, public`, EXECUTE only to `docpilot_app`.
  An aggregate needs no row visibility, so the app gets a number, never another tenant's rows.
  Its 429 says the installation's budget is spent without printing other tenants' figures.
- New `ORGANIZATION_DAILY_BUDGET_USD` (default 0, off) is the noisy-neighbor cap.
  It is computed by the existing `budget.spent_today_usd`, which both layers now make per-organization with no code change; it is checked before the global cap.
- `/stats` `budget`:
  - `daily_budget_usd` is the installation cap (config, not tenant data).
  - `spent_today_usd` becomes this organization's spend.
  - Two fields are added: `organization_daily_budget_usd` (null when off) and `installation_budget_reached` (bool).
  - The demo has one organization, so `demo.py`'s live-take budget check keeps its meaning.
- The ask, upload, and mcp rate limits are keyed by principal (`user:<id>` or `key:<id>`) instead of client address.
  The limiter's job is pacing one client, the principal is now the client, and this fixes the Docker bridge issue that made the per-IP limits effectively global.
  Only login stays keyed by address (plus the account bucket).

Frontend:
- A route group `src/app/(app)/` holds every existing page and the header nav; `src/app/login/page.tsx` sits outside it, so the login page polls nothing (`ReviewQueueLink` would 401-loop).
- One `apiFetch` in `src/lib/api.ts` adds `credentials: "include"` and `cache: "no-store"` to every call (SSE included).
  A 401 sends the browser to `/login?next=<path>`, with `next` accepted only as a same-origin relative path that starts with `/` and not `//` or `/\`.
- There is no `proxy.ts`: the API is the real check, and an optimistic cookie check would need the per-stack cookie name inlined.
- `AccountMenu` (email, organization, a switcher only for more than one membership, "API keys", "Sign out") and `/settings/api-keys` (list, create with the secret shown once, revoke with an inline confirm; read-only for members).

### Every table and query path, and how each is scoped

| Path | Scope after this batch |
| --- | --- |
| `documents`, `jobs`, `extractions`, `extracted_fields`, `reconciliations`, `document_pages`, `document_chunks`, `ask_runs` | `organization_id`, ORM criteria, FORCE RLS; `jobs` also opens to the queue scope for SELECT and UPDATE |
| `organizations`, `users`, `memberships`, `auth_sessions`, `api_keys` | Global identity tables, no RLS; only `app/auth/` queries them, always filtered by the principal's ids |
| `alembic_version` | No grant to `docpilot_app` |
| `routers/documents.py`, `review.py`, `search.py`, `ask.py`, `chat.py`, `stats.py`, `monitoring.py` `/daily` | `get_session` now requires `current_principal` and binds its organization, so a route that touches data cannot run unauthenticated or unscoped |
| `monitoring.py` `/evals`, `stats.py` `last_eval` | Committed eval artifacts (global repo data); authentication required, not tenant-scoped |
| `ask_stream`, `chat_stream` (`stream_answer`) | `get_tenant_session_factory` yields sessions bound to the principal's organization |
| `agent/tools.py` (`search_documents`, `get_page`, `query_extractions` through `records.load_records`), `execute_tool` savepoints | The caller's tenant session; the GUC is set at transaction start, so savepoint rollbacks keep it |
| `mcp_server/server.py` `on_call_tool` | A tenant session for the key's organization (HTTP from `ctx.request.scope`, stdio from the verified `DOCPILOT_API_KEY`) |
| `worker.py` `claim_job`, `reclaim_orphaned_jobs` | `queue_session()` (jobs only), then the claim binds `job.organization_id` |
| `extraction.process_document_job`, `retrieval/index_job.process_index_job`, `evals/online.process_judge_job`, `complete_job`, `fail_job` | A tenant session for the job's organization; the after-rollback re-fetch re-applies it through `after_begin` |
| `budget.spent_today_usd`, `enforce_daily_budget` | Per organization (both layers), plus the global `docpilot_spend_since` |
| `evals/online.record_ask_run` (judge enqueue), `review_routing.refresh_reconciliation`, `lock_extraction` | The caller's tenant session |
| `demo_seed.py`, `scripts/seed_demo.py` | The default organization; reset deletes content rows (RLS-scoped) and the organization's sessions and keys, never identity rows; upserts the demo user |
| `app/evals/corpus.py`, `evals/agent.py`, `evals/retrieval.py` | An ephemeral organization inside the rolled-back transaction |
| `scripts/reconcile_all.py`, `reindex.py`, `harvest_corrections.py` (`app/evals/harvest.py`), `smoke.py` | Per organization as above |

### Migration plan (three revisions after `b4e7c1a9d625`, each with a real downgrade)

1. `<rev>_add_organizations_users_and_credentials` (N1):
   - `CREATE EXTENSION IF NOT EXISTS citext`.
   - The five identity tables, each with `created_at`, CHECKs mirroring the API limits (name 1 to 100 characters, email at most 254, role in (`owner`, `member`), digests exactly 32 bytes, prefix `^[a-z2-7]{12}$`, slug `^[a-z0-9][a-z0-9-]{1,62}$`), and indexes on every foreign key plus `auth_sessions.expires_at`.
   - The default organization as a literal UUID (copied, not imported, from `models.DEFAULT_ORGANIZATION_ID`), slug `default`.
   - The header states retention: sessions are deleted at logout and swept after expiry (at each login, and by `accounts.py prune-sessions`); emails are kept until an operator removes the user.
   - Downgrade drops the tables, then citext.
2. `<rev>_scope_every_table_to_an_organization` (N2):
   - Create `docpilot_current_organization_id()`.
   - Per content table: add `organization_id` nullable, backfill every row to the default organization (everything before this revision is single-tenant), then the large-table pattern from the database guide (`CHECK (organization_id IS NOT NULL) NOT VALID`, `VALIDATE`, `SET NOT NULL`, drop the CHECK), `SET DEFAULT docpilot_current_organization_id()`, and an FK to `organizations` `ON DELETE RESTRICT` (deleting an organization's data is Batch G4's).
   - Add `UNIQUE (id, organization_id)` on the four parents.
   - Drop the single-column FKs (Postgres-generated names such as `extractions_document_id_fkey`; read them from `\d` on `docpilot_test` first), and add the composite FKs `NOT VALID`, then `VALIDATE`.
   - Add indexes, named `ix_<table>_<cols>` like the rest of the schema:
     - `ix_documents_organization_id_created_at`, `ix_ask_runs_organization_id_created_at`, `ix_extractions_organization_id_created_at`, `ix_document_pages_organization_id_created_at` (lists, stats, budget);
     - `ix_document_chunks_organization_id` (the RLS path lexical search takes);
     - `ix_extracted_fields_organization_id_needs_review_reviewed_at` (the queue);
     - `ix_jobs_organization_id` and `ix_reconciliations_organization_id`;
     - composite-FK child indexes led by the parent key (`(document_id, organization_id)` and so on), which also closes the overlay's "no index on `jobs.document_id` or `extractions.document_id`" gap.
   - Downgrade reverses each step and restores the original FKs.
3. `<rev>_enforce_tenant_isolation_with_row_level_security` (N3):
   - A `DO` block creates `docpilot_app` NOLOGIN NOSUPERUSER NOBYPASSRLS if missing.
   - `GRANT USAGE ON SCHEMA public` and SELECT, INSERT, UPDATE, DELETE on the 13 app tables (not `alembic_version`; UUID keys, so no sequences).
   - ENABLE and FORCE RLS, plus the `tenant_isolation` policies.
   - `jobs_queue_claim` (`FOR SELECT`) and `jobs_queue_update` (`FOR UPDATE`) on the queue scope.
   - `docpilot_spend_since` with `REVOKE ALL ... FROM PUBLIC` and `GRANT EXECUTE` to `docpilot_app`.
   - The header names the two GUCs and the threat statement.
   - Downgrade drops policies and the function, sets NO FORCE and DISABLE, and revokes; then a `DO` block tries `DROP ROLE docpilot_app` and on `dependent_objects_still_exist` (2BP01, another database on the cluster still grants to it) raises a NOTICE and keeps the role.
- Provisioning outside Alembic, because a login role and its password do not belong in a migration: `backend/scripts/provision_db_role.py` (logic in `app/db_roles.py`) connects with `MIGRATION_DATABASE_URL`.
  It reads the user and password of `DATABASE_URL`, then creates or alters that LOGIN role (`NOSUPERUSER NOBYPASSRLS NOCREATEROLE INHERIT IN ROLE docpilot_app`), building the statement server-side with `format('%I', '%L')` so the password is never interpolated in Python or logged.
  It refuses when the runtime user equals the owner, or when the two URLs name different servers or databases.
- Existing data: dev, demo, and test rows land in the default organization in revision 2; the dev stack is backed up with `pg_dump` before N3's rebuild (as in M3).

### Tasks

- [x] **N1 Identity schema, password and token primitives, and the accounts CLI (no route changes, zero spend)** - `20a4bbb` - the identity tables and the code that creates and checks credentials, with nothing enforced yet.
  Files:
  - `backend/pyproject.toml` (`argon2-cffi>=25.1,<26`).
  - Revision 1 in `backend/alembic/versions/`.
  - `backend/app/models.py` (`Organization`, `User`, `Membership`, `AuthSession`, `ApiKey`, `DEFAULT_ORGANIZATION_ID`, `MEMBERSHIP_ROLES`).
  - The new `backend/app/auth/` package:
    - `passwords.py` (hasher pinned to RFC 9106 low-memory, `hash_password` and `verify_password` in the threadpool under a semaphore, the dummy verify, `needs_rehash`, `PASSWORD_MIN_CHARS` 12, `PASSWORD_MAX_CHARS` 128);
    - `tokens.py` (session token, API key mint and `parse_api_key`, `digest`);
    - `accounts.py` (create organization, user, membership; set password, which deletes the user's sessions; prune expired sessions);
    - `sessions.py` and `api_keys.py` (create, look up, delete or revoke, as functions over an `AsyncSession`).
  - `backend/scripts/accounts.py` (thin CLI over `app/auth/accounts.py` and `api_keys.py`).
  - `backend/app/config.py` (`session_cookie_name`, `session_cookie_secure`, `session_ttl_hours`, `login_rate_limit_per_minute`, `login_account_rate_limit_per_minute`, each with its comment).
  - `backend/app/demo_seed.py` (`_delete_rows` deletes only content tables, never the identity tables `Base.metadata.sorted_tables` now includes).
  - `backend/.env.example`.
  - Tests: `backend/tests/test_auth_core.py`, which runs the hasher on `profiles.CHEAPEST` by monkeypatch except one test that pins the production parameters, covers token formats and the parser under hypothesis (no input raises or matches without the exact shape), and covers the CLI with `--password-stdin`.
  *Verify:* from `backend/`: `.venv/bin/pip install -e ".[dev]"`; `.venv/bin/ruff check .`; `.venv/bin/pytest tests/test_auth_core.py tests/test_demo_seed.py -q`; `.venv/bin/pytest -q`; and the revision round-trip on the test database only:
  `export DATABASE_URL="$(grep ^DATABASE_URL= .env | cut -d= -f2- | sed 's#/docpilot$#/docpilot_test#')"; .venv/bin/alembic downgrade b4e7c1a9d625 && .venv/bin/alembic upgrade head && .venv/bin/alembic heads` (one head).
  *Result (2026-10-25):* 2052 tests pass (1853 before; 199 new in `test_auth_core.py` and one in `test_demo_seed.py`); revision `aa652338a4ae` round-trips on `docpilot_test` with one head, and `alembic check` finds no drift between the models and the migration. argon2id is pinned to RFC 9106's low-memory parameters (t=3, 64 MiB, p=4), at most two hashes run at once, and an unknown email pays the same one verify as a real one. Review fixes: emails and passwords are NFKC-normalized before storage and comparison (citext and Python fold case differently, and NFC and NFD spellings had created two accounts), a password or cookie with an unencodable character is a miss instead of an exception, emails need exactly one "@" and no whitespace or control characters in Python and in a CHECK, the CLI never echoes a password passed as a flag and turns a constraint refusal into one line with no statement parameters, and the API key parser's property test now catches a parser loosened to `match`, `\w`, or case-insensitive matching on its own.

- [x] **N2 Every content table scoped to an organization in the app layer (zero spend)** - `1db867c` - revision 2 plus the tenancy layer, with every entry point running under a tenant; still single-tenant at runtime (every request runs as the default organization through a placeholder principal that N4 replaces).
  Files:
  - Revision 2.
  - `backend/app/models.py` (`TenantMixin` with the server default and `eager_defaults`, composite `ForeignKeyConstraint`s and `UniqueConstraint`s mirroring the migration, the new indexes).
  - New `backend/app/tenancy.py`: `bind_tenant`, the `after_begin` and `do_orm_execute` listeners registered on `sqlalchemy.orm.Session`, `TenantScopeError`, `tenant_session(factory, org)`, `queue_session(factory)`, `ephemeral_organization`, `organization_ids(session)`.
  - `backend/app/db.py`: `get_session` depends on a `current_principal` placeholder in the new `app/auth/principal.py` (a `Principal` frozen dataclass), and `get_tenant_session_factory` is added.
  - `backend/app/routers/ask.py` and `chat.py` (streams take the tenant factory).
  - `backend/app/worker.py` (`claim_job` and `reclaim_orphaned_jobs` in `queue_session`, the claim binds the job's organization before the document update, `run_once` opens the handler session with `tenant_session(job.organization_id)`).
  - `backend/app/mcp_server/server.py` (`build_server` takes a principal resolver and a tenant session factory; until N5 the resolver returns the default organization), `__main__.py`, `main.py`.
  - `backend/app/retrieval/search.py` (`hnsw.iterative_scan`).
  - `backend/app/evals/agent.py`, `app/evals/retrieval.py`, `app/evals/harvest.py`, `backend/scripts/reconcile_all.py`, `reindex.py`, `harvest_corrections.py`, `smoke.py`, `seed_demo.py`, `app/demo_seed.py` (bind or iterate organizations as in the table above).
  - `backend/tests/conftest.py` (`db_session` binds `DEFAULT_ORGANIZATION_ID`; a `second_organization` fixture creates one in the transaction).
  - `backend/tests/test_worker.py`, `test_telemetry.py`, `test_evals_retrieval.py`, `test_live_smoke.py` (their committed-row sessions bind a tenant).
  - New `backend/tests/test_tenancy.py`:
    - On an owner (superuser, RLS-bypassing) connection, ORM select, `get`, count, update, and delete bound to org A never see or touch org B.
    - An unbound session raises `TenantScopeError` before SQL.
    - The tenant is re-applied after a rollback and survives an `execute_tool` savepoint whose tool failed.
    - An insert takes the bound organization.
    - A child pointing at another tenant's parent is an `IntegrityError` (composite FK).
    - The worker runs two organizations' extract jobs each under its own organization (a fake handler records `current_setting` and the rows it writes).
    - Dense search for a 3-chunk tenant beside a 400-chunk tenant with `enable_seqscan=off` returns that tenant's 3 for `k=3`.
  *Verify:* `.venv/bin/ruff check .`; `.venv/bin/ruff check ../scripts/demos`; `.venv/bin/ruff check ../evals`; `.venv/bin/pytest tests/test_tenancy.py tests/test_worker.py tests/test_mcp.py -q`; `.venv/bin/pytest -q` (no count moves except new tests; the mock eval pins `222 / 224` and `638 / 640` and the retrieval snapshot are unchanged); `.venv/bin/python ../evals/run_retrieval.py --check` passes; revision 2 round-trips on `docpilot_test` as in N1, also with rows present (run the suite once, then downgrade and upgrade).
  *Result (2026-10-25):* 2158 tests pass (2052 before; `test_tenancy.py` is new); the demo-script tests, the mock eval pins, and the retrieval snapshot are unchanged, and the CI retrieval eval's metrics are byte-identical to the commit before. Revision `127aac687508` round-trips on `docpilot_test` empty and with a row in each of the eight tables (backfilled to the default organization; the downgrade restores every original foreign key by name and ON DELETE rule, and is lossy, as its header says). The app layer fails closed: an unbound session refuses any statement on a content table before SQL is sent, and a bound one filters ORM selects, counts, gets, subqueries, CTEs, `exists()`, aliases, unions, and ORM updates and deletes, and refuses what it cannot filter (Core `Table` statements, bulk update or delete by primary key, `INSERT ... SELECT`, `from_statement`, and any write naming `organization_id`). Raw `text()`, raw connection SQL, and identity-map hits are the residue N3's row-level security covers. Review fixes: unions had read every tenant and `from_statement` had run unbound; the Core and bulk forms had run across tenants on a bound session; the pool-leak test now uses a one-connection pool, so it can fail. The small-tenant HNSW test forces the index path and returns 0 hits without `hnsw.iterative_scan`; it reindexes first, because dead entries from rolled-back test inserts had made it fail about half the time.

- [x] **N3 Row-level security under a non-owner runtime role, and the split daily budget (zero spend)** - the commit that checks this box - revision 3, role provisioning, the startup guard, the whole suite running as a non-owner, and the budget split.
  Files:
  - Revision 3.
  - New `backend/app/db_roles.py` and `backend/scripts/provision_db_role.py`.
  - `backend/app/config.py` (`migration_database_url`, `organization_daily_budget_usd`, the `CORS_ORIGINS` no-wildcard validator).
  - `backend/alembic/env.py` (`migration_database_url or database_url`).
  - `backend/app/db.py` (`verify_tenant_isolation`).
  - `backend/app/main.py`, `worker.py`, `mcp_server/__main__.py` (call the guard).
  - `backend/app/budget.py` (`installation_spent_today_usd` through `func.docpilot_spend_since`, an org cap then the global cap with a figure-free message).
  - `backend/app/routers/stats.py` and `backend/app/schemas.py` (`organization_daily_budget_usd`, `installation_budget_reached`).
  - `frontend/src/lib/api.ts` (`Stats` type only).
  - `docker-compose.yml`:
    - `migrate` gets `MIGRATION_DATABASE_URL` (owner) plus `DATABASE_URL` (runtime) and `command: sh -c "alembic upgrade head && python scripts/provision_db_role.py"`;
    - `api` and `worker` get `postgresql+asyncpg://docpilot_runtime:docpilot_runtime@db:5432/docpilot`;
    - `ORGANIZATION_DAILY_BUDGET_USD` is passed to `api`.
  - `docker-compose.demo.yml` (the same for `docpilot_demo`, `seed` included).
  - `backend/tests/conftest.py`:
    - migrate as the owner URL, then `db_roles.ensure_runtime_role` for `docpilot_test_runtime` (fixed dev-grade password) under the existing advisory lock;
    - point `DATABASE_URL` at it before `app.db` imports;
    - assert `verify_tenant_isolation(engine)` so the suite can never silently run with RLS off;
    - add an `owner_engine` fixture for the layer tests.
  - `.github/workflows/ci.yml` `retrieval-eval` job (`MIGRATION_DATABASE_URL` owner and `DATABASE_URL` runtime, then migrate, then `python scripts/provision_db_role.py`, then the eval as runtime).
  - `backend/.env.example` (both URLs, with a note on the startup guard).
  - New `backend/tests/test_rls.py`:
    - Parametrized over the eight tables: a `text()` count bound to A equals A's rows; unbound it is 0; an unbound insert and a foreign `organization_id` are refused; `text()` UPDATE and DELETE of B's rows under A affect 0.
    - The queue scope sees both organizations' jobs and no other table.
    - `docpilot_spend_since` sums both organizations while `spent_today_usd` under A is A's.
    - The function's owner is a superuser or has BYPASSRLS.
    - Schema invariant: every public table with `organization_id` has `relrowsecurity` and `relforcerowsecurity` and a policy, and every table is either such a tenant table or in the identity allow-list.
    - The runtime role is not a superuser, has no BYPASSRLS, owns nothing, and cannot `SET ROLE` to the owner.
    - The guard raises on `owner_engine`.
    - The provisioner refuses an owner runtime URL.
    - The downgrade keeps a role another database still grants to.
  - `backend/tests/test_budget.py` (A's spend counts toward the installation cap and not toward B's organization cap).
  *Verify:*
  - `.venv/bin/ruff check .`; `.venv/bin/pytest tests/test_rls.py tests/test_budget.py tests/test_stats.py -q`; `.venv/bin/pytest -q` (the whole suite as `docpilot_test_runtime`).
  - Revision 3 round-trips on `docpilot_test`.
  - Both `docker compose ... config -q`.
  - Live dev stack: `mkdir -p ~/doc-pilot-backups && docker compose exec -T db pg_dump -U docpilot -Fc docpilot > ~/doc-pilot-backups/docpilot-dev-$(date -u +%Y%m%dT%H%M%S)-before-rls.dump`, then `docker compose up -d --build`; every service healthy; `docker compose exec -T db psql -U docpilot -d docpilot -c "select usename, count(*) from pg_stat_activity where datname = 'docpilot' group by 1"` shows `docpilot_runtime` for api and worker; `docker compose exec -T db psql -U docpilot_runtime -d docpilot -c "select count(*) from documents"` is 0 (no tenant); the app still lists the default organization's documents at `http://localhost:3000`.
  *Result (2026-10-25):* 2267 tests pass, the whole suite now as the non-owner `docpilot_test_runtime` (the startup guard is asserted on its engine, so it can never run with RLS off); the ORM-layer tests run on a superuser connection, so each layer is proven alone. Revision `73a7d8691932` round-trips on `docpilot_test`; its downgrade keeps `docpilot_app` while a login role is a member or another database grants to it. Mutations fail the suite: a `USING (true)` policy, a dropped queue policy, a TRUNCATE grant, a table-level UPDATE grant on `jobs`. Live: the dev database was backed up (`~/doc-pilot-backups/docpilot-dev-20261025T134447-before-rls.dump`), the rebuild migrated it to head and provisioned `docpilot_runtime`, the API and worker connect as that role and are healthy, and the role reads 0 documents with no tenant set. `backend/.env` now names the runtime role in `DATABASE_URL` and the owner in `MIGRATION_DATABASE_URL` (the old file is backed up beside the dump), and the guard passes from the host. Review fixes: the queue-scope UPDATE policy had no WITH CHECK, so a claim could move a job to another organization; `jobs` now grants UPDATE only on the six columns the worker changes. The guard also refuses a role that can `SET ROLE` into a superuser, a BYPASSRLS role, or the owner, and a database missing the content tables. The queue scope ends when the claim binds its tenant. The provisioner sends a SCRAM-SHA-256 verifier, so the runtime role's password never reaches the server, even with `log_statement = all`.

- [ ] **N4 Sign-in for the browser: sessions, CSRF, and auth on every route (zero spend)** - the real `current_principal`, the auth routes, and route-level enforcement.
  Files:
  - `backend/app/auth/principal.py`:
    - `current_principal` resolves `Authorization: Bearer` (API keys arrive in N5; until then any bearer is a 401) or the session cookie through `sessions.lookup` (expired rows are deleted and answer 401; a disabled user or a removed membership is a 401);
    - the Origin check for cookie-authenticated unsafe methods;
    - `Principal.rate_key`.
  - New `backend/app/routers/auth.py`:
    - `POST /auth/login`: both login limits, then the length checks via the schema, then verify, then rehash if needed, then sweep expired sessions, then create a session; `Set-Cookie` and `Cache-Control: no-store`.
    - `POST /auth/logout` (204, deletes the row, clears the cookie).
    - `GET /auth/me`.
    - `POST /auth/organization` (404 for a non-member organization; rotates the token).
  - `backend/app/schemas.py` (`LoginRequest` with email at most 254 and password 12 to 128 characters, `MeOut`, `OrganizationOut`).
  - `backend/app/ratelimit.py` (an `enforce_rate_limit` key argument; `limit_ask` and `limit_upload` key by principal; `limit_login`).
  - `backend/app/main.py` (`dependencies=[Depends(current_principal)]` on every router but auth, the auth router, CORS credentials with explicit methods and headers).
  - `backend/tests/conftest.py`: `client` overrides `current_principal` with an owner of the default organization backed by a user created in `db_session`; an `anon_client` fixture without the override.
  - `AGENTS.md` (one "Running it" line: create your user with `docker compose exec api python scripts/accounts.py create-user ... --organization default --role owner`).
  - New `backend/tests/test_auth.py`:
    - Login sets the cookie with exact flags and `Max-Age`.
    - A wrong password and an unknown email get the same 401 body, and the unknown email still runs a verify.
    - A 129-character password is a 422 with the hasher never called.
    - The per-account and per-address limits answer 429 with `Retry-After` before hashing (hasher call count unchanged).
    - Logout deletes the row; a second logout is a 204.
    - Logout, login, and resolve with no Origin or a foreign Origin get 403.
    - An expired session (row backdated in SQL) is a 401 and its row is gone.
    - `set-password` ends every session, and removing a membership ends its sessions (FK cascade).
    - The organization switch works and a non-member organization is a 404.
    - A CORS preflight from `http://localhost:3000` gets `Access-Control-Allow-Credentials: true`, and one from another origin gets no allow-origin.
  - New `backend/tests/test_route_auth.py`, walking `app.routes`:
    - every route except the allow-list (`/healthz`, `/auth/login`, the OpenAPI and docs routes) answers 401 with no credentials, so a new route that forgets auth fails CI;
    - `/mcp` is listed for N5.
  *Verify:* `.venv/bin/ruff check .`; `.venv/bin/pytest tests/test_auth.py tests/test_route_auth.py tests/test_ratelimit.py -q`; `.venv/bin/pytest -q`; `docker compose up -d --build api`; create a user with the CLI; `curl -si -X POST http://127.0.0.1:8000/auth/login -H 'Origin: http://localhost:3000' -H 'Content-Type: application/json' --data @-` (credentials from stdin) shows the cookie flags; `curl -si http://127.0.0.1:8000/documents` is 401.

- [ ] **N5 API keys: management, bearer access to the REST API, and authenticated MCP over HTTP and stdio (zero spend)** - keys end to end, including both MCP transports.
  Files:
  - New `backend/app/routers/api_keys.py`: `GET /api-keys` returns `{"api_keys": [...]}` (a new endpoint follows the global guide's wrapped list), ordered `created_at DESC, id`; `POST /api-keys` `{name}` returns 201 `{api_key, secret}` (owner only, session only, 409 past 25 active); `DELETE /api-keys/{id}` returns 204, then 404 for a revoked or foreign key.
  - `backend/app/auth/api_keys.py` (verify with the throttled `last_used_at` update).
  - `backend/app/auth/principal.py` (bearer keys).
  - New `backend/app/mcp_server/auth.py` (`ApiKeyVerifier(TokenVerifier)`, `principal_from_request_context`).
  - `backend/app/mcp_server/http.py` (the 401 step with `WWW-Authenticate`, the per-key bucket; the module docstring's order list updated).
  - `backend/app/mcp_server/server.py` (the principal per call; a revoked stdio key gives `isError`).
  - `backend/app/mcp_server/__main__.py` (`DOCPILOT_API_KEY` verified at startup, exit 2 with a reason).
  - `backend/app/config.py` (`mcp_stdio_api_key: SecretStr | None`, alias `DOCPILOT_API_KEY`).
  - `backend/app/schemas.py` (`ApiKeyOut`, `ApiKeyCreated`, `ApiKeyCreateRequest` with a name of 1 to 100 characters).
  - `backend/scripts/accounts.py` (`create-api-key` prints the secret once to stdout).
  - Tests:
    - New `backend/tests/test_api_keys.py`: the secret appears only in the 201 response and never in the list or logs (`caplog`); the stored digest is not the secret; a member gets 403; a bearer key managing keys gets 403; revoke then use gives 401; the cap gives 409; `last_used_at` is written once in a burst.
    - `backend/tests/test_mcp.py` additions in both protocol eras: no header, a malformed key, an unknown key, and a revoked key each get 401 with `WWW-Authenticate`, before the rate limiter; a key of org B sees only B through all three tools; the per-key bucket gives 429; the stdio subprocess refuses a missing or revoked `DOCPILOT_API_KEY` with exit 2 and serves with a valid one (the key must be committed for the subprocess to see it, then deleted in a `finally`).
    - `backend/tests/test_route_auth.py` (`/mcp` 401).
  *Verify:* `.venv/bin/ruff check .`; `.venv/bin/pytest tests/test_api_keys.py tests/test_mcp.py tests/test_route_auth.py -q`; `.venv/bin/pytest -q`; `docker compose up -d --build api`; create a key with the CLI; `claude mcp add --help` read first for the exact header and env flags; then in a scratch directory `claude mcp add --transport http doc-pilot http://127.0.0.1:8000/mcp --header "Authorization: Bearer $KEY"` and the stdio form with `DOCPILOT_API_KEY`, both Connected in `claude mcp list` (a handshake, no model call), removed after; the same URL without the header answers 401.

- [ ] **N6 The tenant isolation suite, a property test, and its own CI step (zero spend)** - one test module that proves tenant B cannot read, list, search, ask about, review, download, see stats or monitoring of, chat about, or reach through MCP any of tenant A's data, at the HTTP and database layers.
  Files:
  - New `backend/tests/test_tenant_isolation.py`, with a `two_tenants` fixture: two organizations, each with an owner, a document with an extraction, a pending field and a reconciliation, a page and chunks indexed with `HashingEmbedder` around a word only that tenant has, an `AskRun`, a chat conversation, a file in `tmp_path`, and an API key.
    As B, with A's ids:
    - `GET /documents`, `/{id}`, and `/{id}/file` are 404 or exclude A.
    - The review queue and count exclude A; resolve is 404 and A's field stays unreviewed.
    - `/search` in all three modes with `k=50` returns no A chunk.
    - `/ask/runs`, `/{id}`, and feedback are 404 or exclude A, with A's run unchanged.
    - Chat, chat stream, and conversations on A's document are 404 before the fake model is called.
    - `/stats` and `/monitoring/daily` equal B's numbers only.
    - `DELETE /api-keys/{A}` is 404.
    - `/ask` with the scripted fake client calling `query_extractions({})`, `search_documents({query: <A's word>})`, and `get_page({A's document, 1})`: the tool results the fake receives hold no A id or text, `get_page` is an error result, and the stored run belongs to B.
    - The same three calls through MCP with B's key in both protocol eras.
    - A route-walk wrong-tenant test: every route with an id path parameter, called as B with A's ids, answers 404 and leaves A's rows unchanged.
    - The worker: two organizations' extract, index, and judge jobs each write only into their own organization.
  - The hypothesis property (the `doc-pilot` profile, `max_examples` about 25): for any assignment of 2 to 6 generated documents with random totals and currencies to A and B, `query_extractions({})` through `execute_tool` under each organization returns exactly that organization's document ids, and its per-currency sums equal the `Decimal` sums of that organization's totals.
  - `.github/workflows/ci.yml` (a "Tenant isolation" step `pytest tests/test_tenant_isolation.py tests/test_rls.py tests/test_route_auth.py -q` before the full `pytest -q`, so the gate is named in the CI log).
  - The testing overlay's harness facts go in N9.
  *Verify:* `.venv/bin/ruff check .`; `.venv/bin/pytest tests/test_tenant_isolation.py -q` twice with identical results; `.venv/bin/pytest -q`; then temporarily drop the `with_loader_criteria` line and confirm the RLS-layer tests still pass while the owner-engine tests fail, and drop one policy and confirm the `test_rls.py` tests fail; restore both (proves each layer is tested alone; nothing is committed from this step).

- [ ] **N7 Frontend: sign-in, sign-out, the account menu, and API key management (zero spend)** - the browser flow over N4 and N5.
  Read first: `frontend/node_modules/next/dist/docs/01-app/03-api-reference/03-file-conventions/route-groups.md`, `layout.md`, `01-app/02-guides/authentication.md`, `redirecting.md`, and the `useSearchParams` reference (it needs a `<Suspense>` boundary on a prerendered page).
  Files:
  - `frontend/src/app/layout.tsx` (html and body only).
  - New `frontend/src/app/(app)/layout.tsx` (the header nav plus `AccountMenu`); every existing route moved into `(app)/` with `git mv` (URLs unchanged).
  - New `frontend/src/app/login/page.tsx`: labelled email and password inputs, `autoComplete="username"` and `"current-password"`, paste allowed, `maxLength` mirroring the API, the submit button disabled while pending, input kept on error, a `role="alert"` banner, and the 429 wording from `Retry-After`.
  - New `frontend/src/app/(app)/settings/api-keys/page.tsx`: list with name, `dpk_<prefix>...`, created, last used, and status; an empty state with its next step; a create form; the one-time secret in a read-only input with Copy and "Store it now; it is not shown again"; revoke with an inline confirm; read-only with a note for members.
  - New `frontend/src/components/AccountMenu.tsx`.
  - `frontend/src/lib/api.ts`:
    - `apiFetch` (credentials, `no-store`, the 401 redirect skipped for `getMe` on the login page);
    - `login`, `logout`, `getMe`, `switchOrganization`, `listApiKeys`, `createApiKey`, `revokeApiKey`, and their types;
    - `safeNextPath`.
  - `frontend/src/app/globals.css` (classes for both themes from existing tokens; no new token unless needed, and then both themes plus an AA check).
  *Verify:* from `frontend/`: `npx eslint src`; `npm run build`; `docker compose up -d --build frontend`; with `chrome-devtools-axi`, light and dark at 360, 768, and 1440 px:
  - An unauthenticated `/review?x=1` lands on `/login?next=%2Freview%3Fx%3D1` and returns there after sign-in.
  - A wrong password shows the alert, with the email kept.
  - `next=//evil.example` falls back to `/`.
  - The document preview image and PDF iframe load with the cookie.
  - A key is created, its secret shown once and gone after Done, then revoked.
  - Sign out returns to `/login`, and Back shows no data.
  - Keyboard alone completes sign-in and key creation.
  - No horizontal scroll at 360 px with the account menu open.

- [ ] **N8 Demo stack and GIF harness signed in, existing scenarios unchanged (zero spend)** - the demo keeps working with auth; no GIF is re-recorded.
  Files:
  - `backend/app/demo_seed.py` (`DEMO_USER_EMAIL` `demo@doc-pilot.local` and `DEMO_USER_PASSWORD`, a 14-character well-known password: public by design, demo database only; the seed upserts the user as owner of the default organization and resets its password; reset deletes content rows and the organization's sessions and keys).
  - `backend/scripts/seed_demo.py` (prints the demo sign-in).
  - `docker-compose.demo.yml` (`SESSION_COOKIE_NAME: docpilot_demo_session`, the runtime URLs).
  - `scripts/demos/demo.py` (an authenticated `urllib` opener with a `CookieJar`, signing in with `Origin: http://localhost:3100` for `ensure_stack_ready` and `budget_state`).
  - `scripts/demos/harness.py` (`sign_in(page, frontend_url, email, password)` through the login page before `page.screencast.start`, so no frame shows it and every recorded scenario starts signed in).
  - `scripts/demos/test_demo.py` and `test_harness.py` (the fakes cover sign-in, and a test fails if `demo.py`'s credentials drift from `app/demo_seed.py`'s, read with `ast`, not imported).
  - `backend/tests/test_demo_seed.py` (the demo user exists and can log in; reset keeps identity rows).
  *Verify:* from `backend/`: `.venv/bin/ruff check ../scripts/demos`; `.venv/bin/pytest ../scripts/demos -q`; `.venv/bin/pytest tests/test_demo_seed.py -q`; `docker compose -f docker-compose.yml -f docker-compose.demo.yml config -q`; `.venv/bin/python ../scripts/demos/demo.py stack up`; `seed --reset` reports 32 documents; `record smoke` (not a live slug, no model call) produces frames in `scripts/demos/out/` that start on the signed-in home page; demo `/stats` `spent_today_usd` reads 0.00 before and after; `stack down` (never `-v` on the dev project).

- [ ] **N9 README Security model for multi-tenant use, MCP and Quickstart sections, overlays, and boards (zero spend)** - the final documentation pass and the "done when" item.
  Files:
  - `README.md`:
    - Security model rewritten: the tenancy model, the two layers and what each catches, the honest GUC threat statement, sessions and CSRF, API keys, budget and limits per organization and installation, what stays local-only (loopback, no TLS, no non-loopback MCP), and known gaps (in-memory login limits, per-tenant lexical scans, FIFO queue across tenants, same-site sibling ports, no upload quota, no account deletion).
    - Replace the "Before the hosted demo ships" paragraph.
    - The MCP section's config snippets with the key header and `DOCPILOT_API_KEY`, and its "What waits for auth" list.
    - The Quickstart (create a user) and Architecture lines.
  - `.claude/skills/doc-pilot-security/SKILL.md`, `doc-pilot-architecture`, `doc-pilot-database` (chain, schema facts, GUCs, roles, measured search plans), `doc-pilot-api-design` (routes, 401 and 403, the wrapped list), `doc-pilot-testing` (runtime role, `owner_engine`, fixtures, `test_route_auth`), `doc-pilot-frontend-ui` (route group, `apiFetch`), `doc-pilot-design-principles` (`app/tenancy.py`, `Principal`), `doc-pilot-repository-structure` (`app/auth/`, new scripts), `doc-pilot-error-handling-logging` (no secret in logs, `TenantScopeError`).
  - `FUTURE.md` (Batch G's "No multi-tenant auth" line now says this batch built it, local-only still).
  - `PRESENT.md`, `PREVIOUS.md`, `UPDATES.md`.
  *Verify:* `.venv/bin/pytest -q` (the README sync tests); every README command run once as written (CLI, curl, `claude mcp add` in a scratch directory); `git diff --stat backend/prompts` empty; `grep -rn "no auth" .claude/skills README.md` returns only intended historical mentions.

### Riskiest step and how to verify the whole change

N3 is the riskiest.
It switches every process and the whole suite from the superuser owner to a non-owner role on a cluster the dev, test, and (separately) demo databases depend on.
FORCE RLS will surface any missing grant or unbound path as an empty result or a refused insert, and Postgres roles are cluster-wide.
Mitigations built into the plan:
- separate login roles per environment, so the suite never changes the dev stack's password;
- the fail-closed startup guard;
- the downgrade that keeps a still-granted role;
- the `pg_dump` backup before the dev rebuild;
- N2 landing the app-layer filter first, so N3 changes only the database layer.

The whole batch is verified by:
- the gates in `AGENTS.md` (`ruff check .`, `../scripts/demos`, `../evals`, `pytest -q`, `pytest ../scripts/demos -q`, `npx eslint src`, `npm run build`, both compose configs);
- `evals/run_retrieval.py --check` unchanged;
- the CI "Tenant isolation" step;
- the N6 layer-removal check;
- the live dev stack signed in through the browser and through `claude mcp list` with a key;
- `demo.py record smoke` signed in;
- spend $0.00 throughout.

### Risks

- Cluster-wide roles shared by `docpilot` and `docpilot_test`: a provisioner bug could change the dev stack's login (separate login names, and a test that the provisioner refuses the owner).
- Lexical search loses its GIN index under RLS (measured); fine for local corpora, documented, and a scale fix would be partitioning or a leakproof-safe search function, not this batch.
- A small tenant's dense recall under HNSW post-filtering depends on `hnsw.iterative_scan` (pinned by an N2 test that forces the index path).
- Test churn: every committed-row session in `test_worker.py`, `test_telemetry.py`, `test_live_smoke.py`, and the eval runners must bind a tenant; `TenantScopeError` makes a missed one loud rather than an empty result.
- The `do_orm_execute` criteria plus `TenantScopeError` touch every ORM query; an entity loaded through an unusual path (a raw `Table`, `text()`) is caught only by RLS, which is intended.
- The host workflow breaks until `backend/.env` names the runtime role (the guard says so); the coder must not edit Anthony's `backend/.env`, only `.env.example` and the README, and must hand Anthony the one-line change.
- Same-site sibling ports: another local app on `localhost` can embed a known document id as an `<img>` with the cookie; ids are random UUIDs, and the gap is documented.
- The pending M5 tape and B12 recordings must pass a key or sign in; the harness covers B12, and M5's `claude mcp add` line needs the header (noted on its board line).

### Open questions for Anthony (defaults let work proceed)

- Session lifetime: 12 hours absolute with no sliding renewal (default), or 7 days?
- Per-organization budget: ship `ORGANIZATION_DAILY_BUDGET_USD` off by default with the global cap unchanged (default), or default it to a share of the global cap?
- API key management: owners only (default), or any member for their own keys?
- Organization switcher: build it, shown only with more than one membership (default), or leave multi-membership to the CLI?
- The batch's demo GIF (FUTURE.md rule): a new `auth-api-keys` scenario (sign in, create a key shown once, revoke) is zero model spend; default is to write the scenario and its test in N8 and record it only after your go-ahead, since this batch was asked not to record.
- Stdio MCP: require `DOCPILOT_API_KEY` (default), or also accept `--organization` for a purely local process?

## Batch M - MCP server over the agent tools (added 2026-10-25)

Planned by the planner against master `63981b7`, from item 3 of the portfolio queue (`project-dossier/PROJECT-QUEUE.md`); worked while Batch B's last two tasks wait (B11b needs approval, B12 needs the v3 routing fix first).
Goal: the three `/ask` tools (`query_extractions`, `search_documents`, `get_page`) served to any MCP client over stdio and streamable HTTP, with the same execution path, caps, and untrusted-data handling as `/ask`.
The MCP server never calls the model: the MCP client is the model, so doc-pilot spends nothing.

Key design decisions (planner defaults; Anthony can override):
- `mcp` 2.3.0, low-level `Server` (not `MCPServer`, which derives schemas from signatures and would fork `TOOL_DEFINITIONS`); the API was read from the installed package, not from memory: FastMCP is gone in 2.x, and a 2026-07-28 stateless protocol era sits beside the 2025-11-25 handshake era.
- One execution path: `loop._run_tool`'s body moves into `tools.execute_tool`, which `/ask` and MCP both call, so the two cannot drift.
- Arguments are validated against the advertised schema in code (the low-level server does not, and MCP clients give no `strict` guarantee); string arguments are capped at 500 characters.
- Results are one framed text block (a fixed untrusted-data notice, then document text between `BEGIN DOCUMENT DATA <nonce>` / `END DOCUMENT DATA <nonce>` with a fresh nonce per call) plus `structuredContent` under an `outputSchema` carrying the `doc-pilot://` source URIs; capped by `MCP_MAX_RESULT_CHARS` (24,000).
- Card numbers in page text are masked when the tools read it (`_line_blocks`), so `/ask` gains it too.
- HTTP is a plain `Route("/mcp")` on the API (a `mount` answers 307 to everything), stateless JSON mode, Host limited to loopback, any `Origin` refused, then the API's own rate limiter in an `mcp` bucket (`MCP_RATE_LIMIT_PER_MINUTE`, 120).
- Deferred to the auth item (portfolio queue item 4): API keys, per-key document scope and rate buckets, non-loopback access, browser origins, a tool-call audit log, MCP resources.

### Tasks

- [x] **M1 One tool-execution path, page-text card masking, shared test fixtures (refactor, zero spend)** - `4758851` - `tools.execute_tool` and `ToolOutcome`; `_run_tool` a thin wrapper; `_line_blocks` masks Luhn-valid card numbers with spans unchanged; `ratelimit.enforce_rate_limit` public; the `spans` fixture in `conftest.py`; the security overlay updated.
  *Verify:* `ruff check .`; `pytest tests/test_agent.py tests/test_telemetry.py tests/test_ratelimit.py -q`; `pytest -q`; `git diff --stat backend/prompts` empty; `TOOL_DEFINITIONS` unchanged.
  *Result (2026-10-25):* 1769 tests pass (1730 before); `TOOL_DEFINITIONS` and the prompts unchanged; the move out of `_run_tool` is verbatim except that a tool failure now logs under `app.agent.tools`. Two review rounds rewrote `pii.mask_card_numbers`, which the evidence-quote sweep shares: the old rule tested the whole joined digit run, so a card printed beside its expiry, auth code, or an amount (`4111 1111 1111 1111 1227 123456`, the usual terminal-slip layout) was sent in full, and two dates joined by " - " were masked about one time in ten because the 16 digits passed Luhn by chance. The rule now drops date and `NN.NN` groups from card candidacy, ranks Luhn-valid 13-19 digit windows card-shaped first (4-4-4-4, 4-6-5, unbroken, and the other issuer groupings), then longest, and masks each to `****` plus its last four; thin and narrow no-break spaces count as separators, and grouping is linear. Over 36,865 strings in every committed label, artifact, fixture, and gold text, the old and new rules differ in 0, so no stored value, label digest, bar verdict, or README table moves. Known limit, documented in the security overlay: a 17-23 digit reference number printed in groups can be partly masked (about 1 in 6 for 20 digits in groups of four).
- [x] **M2 The MCP server: tool adapter, stdio entry point, tests (zero spend)** - `a8bc97e` - `app/mcp_server/server.py` (`build_server`), `__main__.py` (stdio, stdout kept clean), `prompts/mcp_instructions_v1.md`, `McpSource` / `McpToolOutput`, `MCP_MAX_RESULT_CHARS`, `tests/test_mcp.py` over both protocol eras and a stdio subprocess.
  *Verify:* `pip install -e ".[dev]"`; `ruff check .`; `pytest tests/test_mcp.py -q`; `pytest -q` with no network and no API key; `claude mcp list` shows the stdio server Connected (a handshake, no model call).
  *Result (2026-10-25):* 1828 tests pass with `ANTHROPIC_API_KEY` unset (59 new in `tests/test_mcp.py`, every tool test in both protocol eras: the 2025-11-25 handshake and the stateless 2026-07-28 era). The tool list equals `TOOL_DEFINITIONS` minus Anthropic's `strict` key, with read-only annotations and an `outputSchema`, and is byte-identical after a poisoned document is added. Arguments are validated in code against the advertised schema (`NaN`, booleans for integers, extra keys like `__proto__`, and 1 MB strings are all `isError`); an unknown tool is a protocol error; a failing tool or database returns only `Error: the tool failed unexpectedly.` with the traceback on stderr. Everything after the fixed notice, filenames included, sits between `BEGIN` and `END DOCUMENT DATA <code>`, and the code is regenerated if it appears anywhere in the data, so a page cannot close the fence. The result cap was swept from 2,000 characters to the full length: never over, the fence always intact, the structured copy the same cut. A subprocess test proves the server never imports `app.agent.loop`, `app.extraction`, `app.budget`, or `anthropic`. Real stdio, on a throwaway migrated database (dropped after): three tools listed, `query_extractions` summed 500.32 USD and 27.82 EUR over four documents, and stdout held only JSON-RPC frames. The dev database is 2 weeks behind the migrations and empty, so the `claude mcp list` check moves to M3, after the stack is rebuilt. Review fixes: the embedder warm-up starts inside `stdio_server`, after fd 1 points at stderr, and a card-number assertion no longer scans random ids.
- [x] **M3 Streamable HTTP at `/mcp` on the API (zero spend)** - `081868c` - `app/mcp_server/http.py` (`McpHttpEndpoint`, a fresh session manager per lifespan entry), the route and lifespan in `main.py`, `MCP_RATE_LIMIT_PER_MINUTE` in config and compose, HTTP tests for 421, 403, 429 and a twice-entered lifespan.
  *Verify:* `pytest -q`; both `docker compose ... config -q`; `docker compose up -d --build api`; curl `initialize` 200, a foreign Host 421, a foreign Origin 403; `claude mcp list` shows the HTTP server Connected.
  *Result (2026-10-25):* 1853 tests pass with no API key (25 new HTTP tests). The dev stack was rebuilt after a `pg_dump` backup (`~/doc-pilot-backups/docpilot-dev-20261025T022620-before-mcp-m3.dump`); the migrate service brought its database from `d2a7b5c9e013` to head `b4e7c1a9d625`, and every service came back healthy. Live on `127.0.0.1:8000`: `initialize` 200, a foreign Host 421 in both protocol eras, a foreign Origin 403, GET 405; the SDK client lists the three tools over HTTP at 2025-11-25 and 2026-07-28; `claude mcp list` showed both the HTTP and the stdio server Connected (added in a scratch directory and removed after). Only POST is served: in the handshake era a stateless GET opens an SSE stream that stays open, so GET, HEAD, PUT, and DELETE answer 405 before the rate limiter. The reviewer probed 20-odd Host forms (trailing dot, uppercase, no port, unbracketed IPv6, `127.1`, an empty or missing Host: all 421), `Origin: null` (403), chunked bodies over the 64 KiB cap (413), and a shutdown with a call in flight (finishes in 1 s, then 503). Under tracing a call reads `POST /mcp` > `tools/call` > `execute_tool` > retrieval and SQL, with no query text in any span. Review fix: `/mcp/` is routed too, since Starlette would 307 it and clients do not follow a redirect on a POST. Noted for M4: the endpoint is built at import from the real settings, session maker, and embedder, so the test client's dependency overrides do not reach `/mcp`.
- [x] **M4 README section, security model, overlays (zero spend)** - the commit that checks this box - `## MCP server` section with Claude Code and Claude Desktop config snippets, the Security model and Tracing lines, and the architecture, repository-structure, security, api-design, and testing overlays.
  *Verify:* `pytest -q` (README sync tests); every snippet checked with `claude mcp list`.
  *Result (2026-10-25):* README `## MCP server: the same tools for any MCP client` (config for Claude Code over HTTP and stdio and for Claude Desktop, the result shape from a real run, what is enforced, what waits for auth), plus a Security model bullet and a Tracing paragraph; the architecture, repository-structure, api-design, security, and testing overlays record the MCP facts the reviews verified. 1853 tests pass. Both Claude Code commands were checked with `claude mcp list` (Connected); the Claude Desktop entry uses the same `sh -c` command and was not run. The demo stack was seeded (32 documents, replayed, $0.00) and served `/mcp` on 8100 in both protocol eras: `query_extractions` for Northgate summed 488.23 USD over two documents, and `/stats` spend read $0.00 before and after.
- [ ] **M5 Demo GIF: Claude Code answering through the MCP tools** - a `vhs` tape against the demo stack at `127.0.0.1:8100/mcp`, `screenshots/features/mcp-claude-code.gif`, doc-pilot spend $0.00 before and after. ⚠ Needs Anthony's approval: the takes use Anthony's Claude subscription, and recording needs `brew install vhs`.

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
- PII in code, not just the prompt: `payment_method` (brand only) and `payment_card_last4` (exactly four digits); `mask_payment_fields` runs before anything is stored or logged, and corrections get the same rule as 422s. Every evidence quote of every field is also swept for card-shaped numbers (13-19 digits passing Luhn), because the payment line is the natural quote for `amount_paid` and `total` (decided in the B4 review, 2026-10-15).
- Tax lines are `(label, rate, base, amount)`; every printed tax line goes in `tax_lines`.
- The demo seed keeps replaying the committed v1 Sonnet run for 001-025; new documents 026+ replay the newest committed v2 artifact, else label plus gold evidence, tagged and counted.
- Pre-registered acceptance bar for making v2 the default: per model, core accuracy (the 7 v1 fields on 001-025) no worse than its v1 run by more than one field; 0 residual schema violations; Sonnet evidence coverage >= 95%; no `max_tokens` stops; no unmasked card number in the artifact.

### Tasks

- [x] **B1 Versioned extraction schemas (refactor, v1 byte-identical)** - `edbd2b5` - `app/extraction_fields.py` (`FieldSpec`, `ExtractionSchema`, `SCHEMA_V1`, `SCHEMAS`, `schema_for`, `record_extraction_tool`); `schema` threaded through `extract_document`, `_extract_one_block`, `_extract_chunked_pdf`, `merge_page_tool_inputs`, `_schema_violations`, `build_extracted_fields`; module constants removed from every importer; `Settings.extraction_prompt_version` (validated) in config, compose, `.env.example`; a captured v1 tool fixture.
  *Verify:* `ruff check .`; `pytest -q`; `python evals/run.py --mock` predicted = actual 0.988571; the README sync test passes; `git diff --stat backend/prompts` empty; no API key.
  *Result (2026-10-15):* 906 tests pass; the v1 tool, prompts, max tokens, field names, and tie-breaks equal a fixture captured from master before any edit (the reviewer re-derived it from `3d25754`); mock 0.988571 predicted and actual; README and prompts unchanged. Review fixes: readers of committed artifacts degrade to n/a on an unknown or missing prompt version instead of taking down every report and `/monitoring/evals`, and prompt files are read when the schema is built, so a missing one fails at startup rather than after a billed call.
- [x] **B2 Reconciliation rules_v2 (pure engine)** - `c9c7392` - golden `tests/fixtures/rules_v1_reports.json` frozen before any edit; `ReconcileInput` gains discount, shipping, tip, amount_paid, amount_due, dates, `tax_lines`; `total_math` generalized; `tax_lines_sum`, `tax_line_math[i]`, `balance_math`, `due_date_order`, new `non_negative:*`; suggestions over the new money paths.
  *Verify:* reconcile, property, and routing tests pass twice identically; the golden test; regenerating the README tables changes only `rules_v1` to `rules_v2` (Haiku still 50.0%, mutation eval still 1604 of 1628); `reconcile_all.py --dry-run` on the dev stack shows 0 routing changes, after a `pg_dump` backup.
  *Result (2026-10-15):* 987 tests pass. A v1-shaped record keeps its rules_v1 report except the version string: a golden fixture written by the old engine (151 reports: 25 labels, 53 committed-run replays, the staged 016 case, 72 edge shapes) and a 3000-case old-versus-new differential with 0 mismatches. README regeneration changes only `rules_v1` to `rules_v2` (Haiku still 50.0%, mutation eval still 1604 of 1628, 0 wrong fixes). The Mac dev database holds no extractions and predates A4 (backed up to `~/doc-pilot-backups/` anyway), so `reconcile_all.py` ran on the demo database instead: 25 extractions, 0 routing changes in the dry run and after the real run. Review fixes: a rate-only tax line under the tax-included reading is checked on the net amount (it had false-failed real VAT receipts with a wrong fix and false-passed a misread total), suggestion re-runs use only the first tax base (a later base had turned one misread into a wrong fix on a correct field), a negative `amount_due` (overpayment) can be suggested, and the widened tax-included blind spot (a total read as the subtotal with its adjustments) is documented and pinned.
- [x] **B3 Value-quote agreement check and the `quote:` review reason** - `59b9eb3` - printed-amount patterns moved into `app/money.py` (`printed_amounts`); `app/quotes.py` (`quote_agreement`); `quote:<path>` checks in `reconcile.py`; routing, a migration widening the review-reason CHECK, `caught_by_quotes` in `evals/checks.py`.
  *Verify:* `pytest -q`; migration downgrade/upgrade on `docpilot_test` with one head; `run_retrieval.py --check` passes; README regeneration changes nothing.
  *Result (2026-10-15):* 1165 tests pass; migration `a7d2f4c9b108` round-trips on `docpilot_test` with one head; the retrieval gate passes; the rules_v1 golden fixture is unchanged; the README changes only the Ops caption (caught by checks now counts either deterministic reason). With faithful quotes on every leaf the mutation probe catches all 1628 broken amounts with 0 wrong fixes; a misread repeated in the quote, or a missing quote, falls back to exactly the arithmetic's 1604. Review fixes: a 1000x error no longer passes on round thousands ("$1,000" read as 1.000), space-, NBSP-, and apostrophe-grouped amounts read as one number, a zero quoted as "FREE" is skipped rather than flagged, a lone number in a quote (a reference number) is only suggested when the arithmetic corroborates it, and `evidence: null` is told apart from a v1 leaf.
- [x] **B4 Schema v2 contract and persistence (prompt shipped, not the default)** - `2dbed60` - v2 `FieldSpec`s; `prompts/extract_v2.md` and `repair_v2.md`; `_coerce_evidence`; `mask_payment_fields`; v2 persistence in `build_extracted_fields`; `records.py` corrections for the new fields and the PII 422s; `record_lines` for present v2 fields only.
  *Verify:* `ruff check .`; `pytest -q`; a fake-client v2 worker test stores 20 fields; the default is still `extract_v1`; no 13+ digit run survives in `raw_response`.
  *Result (2026-10-15):* 1322 tests pass; v1 tool, prompts, and record lines unchanged (B1 fixture); mock 0.988571; README unchanged; the worker default stays `extract_v1`. A fake-client worker run with `EXTRACTION_PROMPT_VERSION=extract_v2` stores 20 fields with evidence. Review fixes, mostly PII: payment leaves are rebuilt from their allowed keys (a card number in a list, object, or extra key had reached `raw_response`), a non-text `payment_method` correction is a 422, every field's evidence quote is swept for Luhn-valid card numbers, the digit-run rule no longer eats dates and amounts, masking runs for every writer, and reads of an unknown prompt version fall back to v1 instead of failing the endpoint. `max_tokens` is 16384 (evidence roughly doubles the output per line item). Known until B5: a chunked v2 PDF drops scalar evidence (pinned by a test).
- [x] **B5 PDF evidence pages through single-call and chunked paths** - `838630c` - image evidence page 1, out-of-range pages null, `merge_page_tool_inputs` carries each winner's page; v1 merge byte-identical.
  *Verify:* `pytest tests/test_extraction.py -q`; `pytest -q`.
  *Result (2026-10-15):* 1343 tests pass; v1 merge, single-call, repair, and `_chunks` byte-identical (a golden string written against the old code, and the reviewer's 600-example differential per path against `2dbed60`). An image cites page 1, a single-call PDF keeps a quote whose page is out of range with page null, and a chunked PDF cites each winner, and each line and tax-line cell, to its own chunk. When the winning value has no quote, the most confident agreeing chunk's quote is cited instead. Review fixes: payment fields had kept the old strict page rule and lost their quote on a bad page; the evidence property is source-tied (it now catches every value cited to page 1) and fails in seconds, not minutes, after `report_multiple_bugs=False` in the suite's hypothesis profile.
- [x] **B6 Eval dataset v2: labels, 7 new documents, gold evidence** - `b4e7976`, engine fix `a6b02a7` - `make_evals.py` v2 fields (001-025 images never regenerated: `--labels-only`), documents 026-032 (discount and shipping invoice, tip, partial payment, PII card, EUR VAT lines, tax lines only, two-page PDF), `dataset_version` "v2", validators, harvest.
  *Verify:* `git status --porcelain evals/docs evals/text` lists only the new documents; old labels' v1 keys unchanged; every label passes rules_v2 with 0 gold-quote failures; the retrieval corpus stays 25; `pytest -q`.
  *Result (2026-10-15):* 1413 tests pass. 32 labels (dataset v2), the 25 old labels' v1 fields unchanged (sha256 fixture) and no 001-025 image or text touched; 7 new documents, their amounts checked to the cent by the reviewer, every gold quote verbatim and passing; 029's test card number exists only as pixels (labels hold VISA and 1111). The retrieval corpus stays the 25 documents with gold text (`RETRIEVAL_DATASET_VERSION` v1); the published v1 numbers are unchanged; mock 222/224 = 0.991071. Review fixes: 031's tax is null (the v2 prompt forbids a computed total tax; the engine derives it), tax-line labels drop their rate, scenario labels cannot drift from their images, and the mutation eval breaks a number everywhere it was printed once. That last fix exposed a real engine fault: when a tax printed once is misread into both `tax` and its tax line, they still agree, and the only fix that passes every check changes the correct total (87 wrong fixes of 2165). Fixed in the next commit, before any paid run: a tax and its only tax line that print the same number are one reading for the suggestion engine (pages equal and the quotes share a printed number, or no evidence and no printed base), and a computed tax or subtotal proposes fixes to the amounts it was computed from; 0 wrong fixes in the mutation eval, also with realistic quote variants in front of the engine, at the cost of 1086 -> 1081 fixes.
- [x] **B7 v2 eval machinery (offline)** - `7dbf2b4` - scoring matchers for the new fields, evidence in artifacts, mock v2, `--prompt` and `--doc`, replay by artifact schema, the quote eval with its README table, core-accuracy and v2 report tables. From the B3 review: land its three-way quote mutation probe as a mutation-eval variant (faithful quotes, the same digit misread in value and quote, a missing quote on the broken cell), with wrong fixes counted; and replay header-field evidence from artifacts (today only line cells keep theirs). From the B5 review: count evidence coverage by quote presence and track page agreement with gold separately, a null page being a page miss, not a coverage miss. From the B6 review: gold quotes for line items and tax lines carry column padding, so the quote metric must normalize whitespace; payment-line gold quotes are the masked form (`****1111`); `evals/run.py --mock` writes into `evals/results/` by default, where `load_results` would pick it up for the README: make `--mock` default to a scratch location or refuse to write there.
  *Verify:* with no API key: `run.py --mock` (v1 and v2), `--report-only --update-readme`, `run_checks.py --update-readme`; README diff only in marker regions; `pytest -q`.
  *Result (2026-10-15):* 1659 tests pass; the v1 scoring of every committed answer is byte-identical (reviewer: 0 mismatches over 3000 adversarial shapes too). Mock v1 222/224, mock v2 638/640 with 813/813 values quoted. The quote eval (2314 broken amounts) catches all of them with faithful quotes (2275 by the arithmetic alone), is silent when value and quote are misread together, and makes 0 wrong fixes in every variant. The Ops table gains Core (v1 fields on 001-025, shown as `174/175 (99.4%)`) and Caught by quotes. `evals/check_bar.py <v1> <v2>` checks the pre-registered bar term by term, from per-document residual schema violations and stop reasons the artifact now records. Review fixes: `--update-readme` refuses any `--out` but `evals/results` (a smoke run would have replaced the published rows), `--mock` never writes under `evals/results`, a tax-line label keeping its rate is judged on its name, and "Net 30" matches "net 30 days" while "Net 300" does not. The extract_v2 prompt now says a lone "Tax" line is also the total tax.
- [x] **B8 UI for the new fields and evidence, demo seed for 026+** - `8ad1c60` - grouped fields, tax-lines table, folded empty fields, an evidence toggle, quote chips, the review page's `quote:` reason and tax-line corrections; seed replays for 026+ and a staged 028 quote case. From the B2 review: `lib/reconciliation.ts` (`pathLabel`, `fieldOfPath`, `humanize`) must learn `tax_lines[i].*`; `tax_line_math` coefficients (rate/100, e.g. 0.08875) must render as the printed percent (8.875%); a computed fallback base (`tax_lines[i].base`) must not be a button aimed at a missing element; the discount-applied reading's "(discount 10.00 included)" needs wording that says the discount is already taken off. From the tax-mirror fix: a paired tax suggestion names `tax`, and after it is applied the next reconciliation suggests `tax_lines[0].amount`, which the review page cannot prefill today (`draftWithSuggestion` handles only `line_items[i].total`, and `pathLabel` shows "tax lines[0].amount"): prefill and label `tax_lines[i].amount` suggestions ("tax line 1 amount"), and decide whether applying a paired suggestion should also correct the mirrored row in the same review. From B5: `evidence.page` is `number | null` (a single-call PDF page out of range is stored as null with its quote); show "page unknown" or no badge, never "page null". From B3: `api.ts` must type check `kind` `"quote"` and `ReviewReason` `quote:${string}`, and the review page compares reasons by string (`review/page.tsx` reason text).
  *Verify:* eslint, build, backend gates; seed reports 32 documents; in a browser 026 all green, 028 two quote chips, 016 and 018 unchanged; light and dark at 360, 768, 1440; keyboard only; demo spend $0.00.
  *Result (2026-10-15):* 1676 tests pass; eslint, tsc, and build clean. Checked in a browser on the demo stack (32 documents, 7 replayed from their labels until a v2 run is committed): 026 groups its fields, shows the two tax lines as "6.5% of 454.48", passes every check, and reveals each value's page and quote behind one toggle; 028's staged amounts pass the arithmetic and fail their quotes with a chip on each, and keyboard alone corrects the amount due to the printed 314.98, then "Use 250.00" for the amount paid, leaving both checks green; 029 shows only VISA and ****1111; 031's tax is derived; 032 cites page 2; 016 and 018 render exactly as before (the reviewer compared server-rendered markup); no overflow at 360 px in dark. Fixes from the browser and the review: focus moves into the editor on Correct and Use and back on Cancel, the evidence toggle says Hide when open, a field a reviewer set to null keeps its row and history, quotes no longer read dates as amounts, and the review page accepts every tax rate the API accepts.
- [x] **B9 Verify model prices before any paid run (zero spend)** - confirm published prices for `claude-sonnet-5` and `claude-haiku-4-5` and that both are served; update `PRICING_PER_MTOK`; recompute the B10 and B11 caps.
  *Verify:* `pytest -q`; `_compute_cost_usd` equals the published price.
  *Result (2026-10-15):* the published pricing page says Sonnet 5's $2/$10, first announced as introductory, became its standard price (the rise to $3/$15 on 2026-09-01 did not happen); both models are still served; every entry in `PRICING_PER_MTOK` matches the page, now pinned by a test, so recorded costs were right all along. Estimates at these prices (v2 about 4,600 input and 1,650 output tokens per document on Sonnet, 4,200 and 1,400 on Haiku): B10 about $0.08 (cap $0.15); B11 Sonnet about $0.83 (cap $1.00) and Haiku about $0.36 (cap $0.50), about $1.20 together; B11b about $0.57.
- [x] **B10 PAID: v2 smoke on 3 documents (about $0.08, cap $0.15)** - Sonnet v2 on 026, 029, 032 into `evals/results/smoke`. Approved by Anthony 2026-10-16.
  *Verify:* 3 scored, 0 errors, no `max_tokens` stop, 0 residual schema violations, evidence coverage >= 90%, 032's totals cite page 2, no card number in the artifact, spend under the cap.
  *Result (2026-10-16):* every verify term passes: 3 scored, 0 errors, every call stopped on `tool_use` (no `max_tokens`), 0 residual schema violations, 121/121 values quoted with page agreement 100%, 032's subtotal, tax, total, and amount due cite page 2, and no card number in the artifact (029 stored VISA and 1111). Field accuracy 98.3%: the one miss is 029's lone "Tax" line, given a base copied from the subtotal at confidence 0.5 though the line prints none. Quote agreement read 59% only because line-item cells quote the cell where the gold quotes the whole row; every header quote agreed. Spend $0.1445 under the $0.15 cap, but $0.048 per document against the $0.026 estimated: about 8,500 input tokens per call, not 4,600, because the v2 prompt and tool are larger, which puts B11 near $2.30 uncached.
- [x] **B11 PAID: full extract_v3 runs, Sonnet and Haiku (about $1.63 with prompt caching; caps $1.50 and $0.75)** - `run.py --prompt extract_v3`, Sonnet then Haiku, both artifacts committed with the regenerated tables; flip the default to `extract_v3` only if the pre-registered bar holds. Approved by Anthony 2026-10-16. Check the bar with `evals/check_bar.py <v1> <v3>` per model. Once a v3 Sonnet artifact is committed, the demo seed replays it for 026-032 instead of the labels: update the seed tests that encode "no v2 artifact yet" (7 label replays, total cost, pending count, the 028 staged quote case, which needs the committed answer to quote amount_paid).
  *Prep (2026-10-16, zero spend):* `extract_v3` is extract_v2 plus one rule (a tax line's base is only what the line prints, never the subtotal: B10's one miss); the overlay requires a new version for new wording, so v2 stays registered for the smoke artifact. Requests with evidence cache the tool and system prompt (one breakpoint on the system block, 5-minute TTL; v1 requests unchanged): measured with the free token counter, the cached prefix is about 8,100 tokens on Sonnet and 6,250 on Haiku, about 95% of each call. Cache writes are priced at 1.25x and reads at 0.1x in every recorded cost, and the first document runs alone so the cache is written once. Estimate from B10's measured output (about 2,950 tokens per document, which caching does not reduce): Sonnet about $1.10, Haiku about $0.53, about $1.63 together, against $2.30 uncached; the caps leave headroom so a tripped cap cannot fail the bar's complete-run term after the money is spent. Line-item and tax-line quotes now agree when they are whole words of the gold row (B10 re-scores at 100%).
  *v3 smoke (2026-10-16, approved by Anthony, cap $0.15):* Sonnet extract_v3 on 026, 029, 032 scored 3/3 at 100% field accuracy (029's lone tax line now has a null base: B10's one miss is gone), 120/120 values quoted with page and quote agreement 100%, every call stopped on `tool_use`, 0 residual schema violations, no card number. Caching works as designed: 026 wrote 7,905 tokens to the cache, 029 and 032 each read the same 7,905; the recorded costs reconcile exactly with cache writes at 1.25x and reads at 0.1x. Spend $0.1202 (the one-time write and the dense 032 weigh more on three documents than on 32; steady state about $0.033 per Sonnet document, so B11's estimate holds).
  *Verify:* 32 scored each, 0 errors, 0 skipped; `cache_read_input_tokens > 0` on every scored document after the first and a cache write only on the first; the core column compares each model's v1 and v3 runs on the same 25 documents; `check_bar.py` term by term; `pytest -q`.
  *Result (2026-10-17):* both runs complete and the pre-registered bar is met for both (`evals/check_bar.py`, every term PASS). Sonnet: 32/32, 99.84%, core 174/175 = its v1 run (the one miss is 023's hallucinated currency, as under v1, caught by review), 815/815 values quoted with page agreement 100%, $0.841 (cap $1.50); it started inside the v3 smoke run's cache window, so all 32 calls read the cached prefix. Haiku: 32/32, 98.1%, core 168/175 against 167/175 under v1, $0.574 (cap $0.75); of its 12 wrong fields the checks route 58% to review, its own confidence 17%; its three misses on the new documents each break a prompt rule (027 amount paid filled though none is printed, 031 a computed total tax, 032 one tax line). B11 spend $1.41; Batch B's paid total $1.67. Committed with the regenerated tables; the seed now replays the Sonnet v3 run for 026-032.
  ⚠ Default NOT flipped (Anthony's call after the results): under v3 a field the document does not print comes back null at confidence 0.0, and confidence routing sends every such field to review, about 7 per document (52 in the demo queue); the same nulls drag v3's "Conf ✓" column to 0.43. Fix the routing (and the column) before `EXTRACTION_PROMPT_VERSION` defaults to `extract_v3`; a missing vendor or total on a receipt is still worth a review, a missing invoice number is not.
- [ ] **B11b PAID, optional: v1 baselines on dataset v2 (about $0.57)** - only adds v1 on the new documents. ⚠ Needs Anthony's approval.
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
