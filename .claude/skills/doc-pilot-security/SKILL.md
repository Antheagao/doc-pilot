---
name: doc-pilot-security
description: doc-pilot's security facts (sign-in, API keys, two-layer tenant isolation, upload and file-serving checks, CORS, spend caps, prompt-injection boundaries, secrets) and known gaps. Use when touching auth, sessions, tenancy, uploads, served files, user input, prompts or model output, CORS, env secrets, or dependencies in doc-pilot.
---

# doc-pilot security overlay

Load the global `security` guide for the rules; this file holds what exists here.
The auth, session, cookie, and CSRF sections of the global guide apply; "Auth and tenancy" below is what exists here.

## Threat model

- Local-only and multi-tenant: every route but `/healthz`, `/auth/login`, `/auth/logout`, and the docs needs a credential, and every compose port binds `127.0.0.1` (db 5434, api 8000, frontend 3000, Jaeger, demo api 8100 and frontend 3100, load api 8200 with its frontend 3200 and Jaeger ports behind profiles and no database port).
- Never hosted, plain HTTP, no TLS; the README "Security model" says what that rules in and out.
- The GitHub repository `Antheagao/doc-pilot` is public: anything committed is published.
- The README "Security model" section is the narrative; keep it true when a guard changes.

## Auth and tenancy

- `current_principal` (`app/auth/principal.py`) is the dependency on every router but `/auth` (`main.py`).
  A `Bearer dpk_...` header is an API key and decides alone; otherwise the session cookie is looked up by its SHA-256 digest in `auth_sessions`.
  Every failure (missing, unknown, expired, disabled user, removed membership) is one `401` `{"detail": "not authenticated"}` with `WWW-Authenticate: Bearer realm="doc-pilot"`.
- A cookie-authenticated write needs an `Origin` in `CORS_ORIGINS` or it is `403` before any database work (`require_allowed_origin`; login and logout too), because `SameSite=Lax` does not cover sibling `localhost` ports.
  A key is exempt: CSRF is a cookie problem.
- Sessions are server-side rows (digest only), absolute expiry `SESSION_TTL_HOURS` (12), no sliding renewal; the cookie is `HttpOnly; SameSite=Lax; Path=/`, `Secure` by `SESSION_COOKIE_SECURE` (off: plain HTTP), named by `SESSION_COOKIE_NAME` (the demo overlay sets its own, because cookies ignore ports).
  Logout, `set-password`, and removing a membership end sessions at once.
- Passwords (`app/auth/passwords.py`): argon2id pinned to `profiles.RFC_9106_LOW_MEMORY`, NFKC before hashing, 12 to 128 characters (`PASSWORD_MIN_CHARS`, `PASSWORD_MAX_CHARS`, enforced by the schema before hashing), two hashes at once (`HASH_CONCURRENCY`), a dummy verify for an unknown or disabled email, one message for every failure.
  The `422` handler in `main.py` drops `input` and `ctx` on `/auth/login`, so a refused sign-in never echoes a password.
- Sign-in limits run before hashing: per address (`LOGIN_RATE_LIMIT_PER_MINUTE`, 10) and per account (`LOGIN_ACCOUNT_RATE_LIMIT_PER_MINUTE`, 5), the latter keyed by a digest of `accounts.folded_email`, the database's own case fold, for every email, so a `429` never says which accounts exist.
- API keys: `dpk_<12 base32>_<43 urlsafe>`, SHA-256 at rest, constant-time compare, a malformed token costs no query, shown once (the `201` and the CLI's stdout), revocable, at most `MAX_ACTIVE_KEYS` (25) through the API, `last_used_at` at most once a minute.
  Managing keys and sessions is `session_only(...)`: a key gets `403`; creating and revoking also need the owner role.
  A key is an organization credential: it outlives its creator's membership, and only a revoke ends it.
- Tenant isolation, two layers (`app/tenancy.py`, migration `73a7d8691932`):
  - The ORM layer binds a session to an organization (`after_begin` sets `docpilot.organization_id`; `do_orm_execute` filters every select, update, and delete) and raises `TenantScopeError` before SQL when a session has none or a statement cannot be filtered.
  - FORCE RLS on the eight content tables, policy `tenant_isolation`, under the non-owner `docpilot_runtime` (a member of `docpilot_app`).
  - `db.verify_tenant_isolation` runs in the API lifespan, the worker, and stdio MCP, and refuses a superuser, BYPASSRLS, the owner of `documents`, a role that can `SET ROLE` into one of those, or a database missing the tables; there is no switch to skip it.
  - Composite foreign keys keep a child in its parent's organization.
- The tenant setting is written by the app, so RLS contains application bugs (a forgotten filter, a `text()` query, a new route), not an attacker with arbitrary SQL as the app role; an unforgeable setting is a noted hardening option, not built.
- The worker's `queue_session` is the one cross-tenant read: `docpilot.queue_scope` plus the `jobs_queue_claim` and `jobs_queue_update` policies on `jobs` only, ended by `bind_tenant` when the claim binds the job's organization; the role may `UPDATE` `jobs` only in the six worker columns.
- The identity tables (`organizations`, `users`, `memberships`, `auth_sessions`, `api_keys`) have no RLS and are touched only by `app/auth/`, the auth routes, `scripts/accounts.py`, and the demo seed.
- Rules for new code:
  - A router goes through `app.include_router(..., dependencies=authenticated)`; `tests/test_route_auth.py` fails any route that answers without a credential and is not in its `PUBLIC` list.
  - A content table needs `TenantMixin`, a `GRANT` to `docpilot_app`, FORCE RLS and a policy; `tests/test_rls.py` fails until it has them.
  - Never write `organization_id` and never use `text()` or a `Table` statement on a content table.
  - A script that touches tenant data binds an organization (`harvest_corrections.py` requires `--organization`, because it exports tenant data into the public repository).
- The demo user (`demo@doc-pilot.local`, `docpilot-demo!`) is public by design: `demo_seed.py` refuses to create it on any database but `docpilot_demo`.

## What exists today

- Uploads (`routers/documents.py`): the `ALLOWED_MIME_TYPES` allow-list, the `_content_matches_mime` magic-byte check (415 on mismatch), and a stored name of `<uuid><ext>` from `_safe_ext`, never the client filename.
- `MaxBodySizeMiddleware` (`main.py`) aborts bodies over `MAX_UPLOAD_SIZE` (20 MB) before multipart spooling; `_read_limited` is the backstop.
- `GET /documents/{id}/file` serves the stored type, and `SecurityHeadersMiddleware` adds `X-Content-Type-Options: nosniff` to every response.
- CORS: the `CORS_ORIGINS` allow-list with `allow_credentials` on, a `"*"` refused when settings load, methods narrowed to GET, POST, and DELETE, and `Content-Type` the only allowed header (`main.py`).
- `SecurityHeadersMiddleware` also adds `Cache-Control: private, no-store` to every response that sets none, except `/healthz` and the docs.
- Spend: `DAILY_BUDGET_USD` for the installation (`docpilot_spend_since`, a `SECURITY DEFINER` function that returns a number, so a refusal prints no figure) and `ORGANIZATION_DAILY_BUDGET_USD` per organization (off by default, checked first) in `budget.py`; sliding windows keyed by principal (`Principal.rate_key`: `user:<id>` or `key:<id>`; `ratelimit.py`); and per-question `AGENT_MAX_STEPS` and `AGENT_MAX_COST_USD`.
  Uploads and `/ask` plus chat are the only billed entry points.
- PDFs over `pdf_max_pages` (20) are refused before any model call.
- Prompt injection: `extract_v1.md`, `extract_v2.md`, `transcribe_v1.md`, and `agent_v1.md` tell the model document text is data, never instructions; extraction runs under forced tool use with a fixed schema; `evals/docs/025-adversarial-office-order-injection.png` is scored in the extraction and agent evals.
- Card numbers, enforced in code and not left to the prompt (`app/pii.py`, `extraction.mask_card_data`): schema v2 stores `payment_method` (brand or method only) and `payment_card_last4` (exactly four digits), never a longer number.
  `mask_card_data` runs in `_extract_one_block` on the first answer and on any accepted repair answer, before `raw_response`, `_chunks`, eval artifacts, or logs are built from them, and again at the top of `build_extracted_fields`, so the demo seed and replays store the same masked data.
  A digit run is a stretch of digits that spaces, dashes, and dots may separate; it is a card or account number when it holds 13 digits in all or five in a row, so dates, amounts, and phone numbers stay readable.
  The payment leaves are rebuilt from allowed keys only (`value`, `confidence`, `evidence`): `payment_card_last4` is cut to its last four digits (null when it has fewer), `payment_method` keeps text with long runs masked to `****` plus the last four, a value of any other shape (a list, an object) is null, and extra keys are dropped.
  Every evidence quote of every field is swept with the Luhn-checked card rule (13 to 19 digits that pass the check), because the line that shows the amount paid often shows the card; values are never swept, so a sixteen-digit `invoice_number` stays intact.
  A reviewer's correction gets the same rule as a `422` (`records.normalize_correction`): `payment_method` must be text or null without a long digit run, and `payment_card_last4` exactly four digits.
  Card numbers in transcribed page text (`transcribe_v1`) are masked when the agent tools read it, not when it is stored: `_line_blocks` (`agent/tools.py`) runs `pii.mask_card_numbers` on each line that `search_documents` and `get_page` return, so a Luhn-valid 13 to 19 digit number becomes `****` plus its last four digits in what the model is sent, in `Source.lines`, and so in the `cited_text` of an `/ask` citation.
  The same function sweeps evidence quotes at write time, so its rule is shared: it finds a card inside a longer run (a card followed by an expiry, an auth code, or another card), never counts a date or a `NN.NN` amount as part of one, and leaves a number that fails the check alone (an order reference).
  Where stretches of a run overlap, one grouped like a card (4-4-4-4, 4-4-4-4-3, 4-6-5, 4-6-4, 6-13, or unbroken) wins, then the longer, then the leftmost.
  Known limits: a chance Luhn match that takes in digits beside a card can show the wrong last four digits, and a grouped reference number of 17 to 23 digits can be partly masked when a stretch of it passes the check (about 1 in 6 for 20-digit references in groups of four); a future gold quote holding one would be rejected by `evals/dataset.py` until it is masked.
  Character spans still index the stored line, so a masked line may be shorter than its span.
  The stored page text and the `/search` snippets still hold the full number, so masking it at write time stays open (Batch G4).
- Evidence is model output too: `_coerce_evidence` keeps `{page, quote}` only when the page is an integer from 1 to 10,000 and the quote is non-empty text of at most 200 characters, and stores `null` otherwise; the quote is displayed and compared (`app/quotes.py`) but never executed or trusted as the value.
- Model output is untrusted input: `_coerce_leaf` and `_schema_violations` (`extraction.py`) assume `tool_use.input` is adversarial, and the frontend renders values as React text with no `dangerouslySetInnerHTML`.
- Agent citations come from `search_result` blocks (`agent/tools.py`), so cited text is copied from stored pages (with card numbers masked, above), not written by the model.
- MCP (`app/mcp_server/`, README "MCP server"): the same read-only tools as `/ask` for any MCP client, with no model call and no spend.
  Document text reaches the client's model, so a result opens with a fixed untrusted-data notice and holds every document-derived string (page text, extracted fields, and the filename in a title) between `BEGIN DOCUMENT DATA <code>` and `END DOCUMENT DATA <code>` lines; the code is `secrets.token_hex(8)` per call and is replaced if it appears in the data.
  Arguments are checked against the tool's input schema in code (the SDK's low-level server does not) and strings over 500 characters are refused; a result is cut at `MCP_MAX_RESULT_CHARS` with `truncated` set; card numbers in page text are masked (above).
  `/mcp` is stateless and POST-only and checks, in order: the route is up (`503`), loopback Host (`421`) and no Origin (`403`, the frontend included), the method (`405`), the API key (`401` with `WWW-Authenticate`), the per-key `mcp` bucket (`429`), and a body of at most 64 KiB (`413`).
  Stdio needs `DOCPILOT_API_KEY`, verified at startup (exit 2) and before every call; a tool runs in a session bound to the key's organization.
  Still deferred: a document scope per key (`ToolContext.document_ids` is the seam), non-loopback hosts and TLS, browser origins, an audit log of tool calls, MCP resources, and the MCP OAuth flow.
- Spans and Langfuse never carry prompts, documents, questions, or answers (`telemetry.py`, `langfuse_link.py`).
- Secrets: the Anthropic key lives in gitignored `backend/.env`, mirrored to the gitignored root `.env` for compose; `docker compose config` prints it, so never paste that output.
- No credential reaches a log, span, or error: a password is never echoed (the `422` handler, the CLI's `refusal_message`), the provisioner sends a SCRAM verifier and never the runtime role's password, a key's `AccessToken.token` is its public prefix, and `tests/test_api_keys.py` checks the logs and spans for a secret.
- The load stack cannot spend (`docker-compose.load.yml`, `scripts/load/load.py`), by four independent layers, each with its own tests:
  - The overlay runs `api` and `worker` on `MODEL_PROVIDER=scripted` with all four model names `scripted`, and sets `ANTHROPIC_API_KEY` to an explicit empty string on every backend service (`api`, `worker`, `migrate`, `loadgen`), because the base file passes the root `.env`'s real key through and an overlay that only left it out would bill.
  - `Settings` (`app/config.py`) refuses `scripted` beside a non-empty key, and refuses a model name that does not match the provider in either direction, with `hide_input_in_errors` so the refusal never prints the key; a container that somehow got a key dies at import.
  - `load.py` checks before any traffic: `config_problems` on the resolved compose config (any service with a key, a backend service whose key is not explicitly blank, a model setting that is not `scripted`, a budget that is not a tripwire, a missing worker lease) makes `stack up` refuse to start, and `check_stack` (the preflight, re-run after every change in the worker count) asks every running `api` and `worker` container what it resolved and holds, through `docker exec python -c`, and gets a provider, a yes or no for the key (never the key), the database, and the role; anything but scripted, keyless, `docpilot_load`, and `docpilot_runtime` refuses.
  - `DAILY_BUDGET_USD` is one cent in the overlay, so any recorded spend turns uploads and questions into `429`; `read_spend` and `spend_problems` read cost and model names from the database as the owner after every phase and stop the run on anything but `0` and `scripted`, and an artifact with nonzero `spend_usd` fails validation.
- `SCRIPTED_MODEL_ALLOW_API_KEY=true` lifts only the second layer, for a host process that reads `backend/.env` (which holds the real key): the scripted path never builds an Anthropic client and the SDK does not read the key unless one is built; it is not set in any compose file, so never set it in one.
- The load stack's other exceptions are recorded in each artifact and stay on loopback: rate limits off (`ASK_RATE_LIMIT_PER_MINUTE=0`, `UPLOAD_RATE_LIMIT_PER_MINUTE=0`), a lease of several minutes, and a share of answers graded.
  Its sign-in limits stay at their defaults, its tenants' owners get random passwords, and their API keys and passwords go to gitignored `scripts/load/out/tenants.json` (mode 600), never to stdout, an artifact, or the README; `scrub` strips key-shaped text from a harness failure's output.
- The test suite cannot spend: `tests/conftest.py` blanks `ANTHROPIC_API_KEY` unless `RUN_LIVE_SMOKE=1`, and `tests/test_no_spend.py` proves it.

## Known gaps

- Sign-in limits are in memory and per process: a restart resets them, and five wrong tries a minute lock an owner out of sign-in for anyone who knows the email.
  Behind Docker's port mapping every client can share one address, so the per-address limit can be shared too.
- Lexical search scans the organization's chunks: the GIN index cannot serve `@@` under RLS, so it reads the `organization_id` index and filters.
- The queue is FIFO across organizations, so one organization's burst delays the rest.
- Another app on a sibling `localhost` port is same-site: a GET it triggers (an image tag naming a known document id) carries the cookie; ids are random UUIDs.
- No upload quota, no account deletion, and nothing deletes a document's file, pages, and chunks (Batch G4).
- The stream of an `/ask` or chat answer finishes after its key is revoked or its session ends.
- No CSP, `Referrer-Policy`, or frame protection on the API or the Next.js app; `nosniff` and `Cache-Control` are the only security headers.
- `ratelimit._client_key` uses `request.client.host`, now only for sign-in; behind a proxy it needs uvicorn `--proxy-headers` with `--forwarded-allow-ips`.
- Backend dependencies are unpinned ranges in `pyproject.toml` with no lockfile, and CI runs neither `pip-audit` nor `npm audit`.
- Some error `detail` strings echo upstream exception text (`model_error` in `routers/ask.py`).
- Host and Origin checks stop a browser tab and DNS rebinding on `/mcp`; a key is what stops a local program.
