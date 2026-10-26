---
name: doc-pilot-testing
description: doc-pilot's pytest harness on its own docpilot_test Postgres database, model and embedder fakes, live smoke opt-in, eval gates, and known test gaps. Use when writing, placing, or debugging a backend test or eval, or chasing a flaky run in doc-pilot.
---

# doc-pilot testing overlay

Load the global `testing` guide for the rules; this file holds what exists here.
Gate commands live in `AGENTS.md`; the Rust and Playwright subsections of the global guide have no counterpart here yet.

## Harness

- pytest with pytest-asyncio: `asyncio_mode = "auto"` and one session-wide event loop (`backend/pyproject.toml`), because `app.db` builds one asyncpg engine at import.
- The suite has its own database: `tests/conftest.py` builds the owner URL from the dev URL with the database `docpilot_test`, refuses the dev database, and at session start creates the database if missing, runs `alembic upgrade head` as the owner (which installs pgvector), and provisions the login role `docpilot_test_runtime` (`db_roles.ensure_runtime_role`, under an advisory lock so parallel runs do not race).
  `DATABASE_URL` points at that runtime role and `MIGRATION_DATABASE_URL` at the owner before `app.db` is imported, so the whole suite runs as a non-owner that row-level security binds; the autouse session fixture `_row_level_security_binds_the_suite` asserts `verify_tenant_isolation(engine)`, so it can never run with the second layer off.
  The role is the suite's own: roles are cluster-wide, so the suite never touches `docpilot_runtime` or the owner's password.
  The compose worker polls `docpilot`, so it can never claim rows the tests commit.
- `db_session` opens one connection to that database on host port 5434 as the runtime role, begins a transaction, binds `DEFAULT_ORGANIZATION_ID`, and rolls it back after the test; `join_transaction_mode="create_savepoint"` lets app code call `commit()`.
  A setting made in a released savepoint stays for the rest of the test's transaction (in the app a commit ends it), so test `docpilot.queue_scope` on a connection of its own, as `tests/test_rls.py` does.
- Each layer is proved alone.
  `owner_engine` and `owner_session` connect as the database owner, which row-level security never binds, so `tests/test_tenancy.py` shows the ORM filter holding without the policies; `tests/test_rls.py` sends raw `text()` SQL at the policies, the guard, the provisioner, and the downgrade; `owner_engine` is never for what the app does.
- `second_organization` makes a second tenant in the test's transaction, and `two_tenants` (with `tests/tenant_rows.py`) puts one row of every content model in the default and the second organization.
- `tests/test_tenant_isolation.py` is the CI gate named "Tenant isolation" (with `test_rls.py` and `test_route_auth.py`): its `tenants` fixture makes organizations A and B with an owner, data, files, and a key each, `credential` runs every test once by session cookie and once by API key, `isolated_client` is `anon_client` with each request's session bound to the organization its credential names, and `as_a` and `as_b` act as one tenant.
  Every claim has a control that shows the same call works for its own tenant, so a 404 is the isolation and not a broken id.
  When you add a route that takes an id, the route walk there fails until it answers 404 for another tenant's id.
- `client` drives the real app through `httpx.AsyncClient(transport=ASGITransport(app=app))` as an owner of the default organization (`current_principal` is overridden with the `owner` fixture's user) and overrides `get_session`, `get_tenant_session_factory`, `get_identity_session_factory`, and `get_settings` (tmp `upload_dir`, budget and rate limits off) to hand routes the test's transaction.
  `anon_client` does the same without the `current_principal` override, so it reaches the real cookie, key, and Origin checks.
- `tests/test_route_auth.py` walks `app.routes`: every route not in its `PUBLIC` allow-list must answer 401 with no credential, so a route added without auth fails CI.
- The suite cannot spend: `tests/conftest.py` sets `ANTHROPIC_API_KEY` to empty before any settings load unless `RUN_LIVE_SMOKE=1` (`backend/.env` holds the real key), the `client` fixtures hand routes a model client that raises `ModelCallAttempted` if used, and `tests/test_no_spend.py` proves both.
- Hashing in the auth tests runs on argon2's `CHEAPEST` profile by monkeypatch; `tests/test_auth_core.py` has the one test that pins the production parameters.
- There is no per-test schema or database: committed rows persist in `docpilot_test` between runs, so never assert global counts (`tests/test_stats.py` header).
- Worker tests (`tests/test_worker.py`) commit real rows (in the default organization) through the `real_documents` fixture and scope claims with `pending_job_stmt(..., document_id=...)`; `test_skip_locked_never_double_claims` holds two real sessions.
  Every test that commits rows deletes them in a `finally` or fixture teardown: `real_documents`, the stdio MCP test (the subprocess can only see a committed key, deleted afterwards), and the isolation suite's worker test.
- No test reaches the network: `monkeypatch.setattr(extraction_module, "_build_client", lambda settings: fake_client)` (`test_extraction.py`, `test_transcription.py`), fakes built from `SimpleNamespace` with an `AsyncMock` `messages.create`, and `app.dependency_overrides[get_anthropic_client] = lambda: fake` for `/ask` and chat (`test_agent.py`, `test_chat.py`, `test_budget.py`).
- `tests/test_mcp.py` drives the MCP server with the SDK's `Client` in memory, once per protocol era (`mode="legacy"` and `"auto"`), over a `build_server` whose session factory yields `db_session`; the HTTP section serves a `McpHttpEndpoint` in process through `httpx.ASGITransport`, and the SDK client reaches it through `httpx2` (the SDK's own HTTP stack, a separate package from `httpx`).
  The stdio test starts `python -m app.mcp_server` as a subprocess against the test database with `EMBEDDING_BACKEND=hashing`, a blank `ANTHROPIC_API_KEY`, and `DOCPILOT_API_KEY` set to a committed test key; it also checks that a missing or revoked key exits 2.
  The HTTP tests send `Authorization: Bearer <key>` and run in both protocol eras; a key of organization B must see only B through all three tools.
  `main.mcp_endpoint` is built at import from the real settings, `async_session_maker`, and embedder, so the `client` fixture's `dependency_overrides` do not reach `/mcp`: a test that calls a tool through the real app must build its own endpoint (as the others do) or monkeypatch.
  The rate-limit tests swap `ratelimit._limiter` for a fresh `SlidingWindowLimiter(FakeClock())`, and the real-app tests patch `main.warm_up_embedder` to a no-op and enter `app.router.lifespan_context(app)`.
- The `spans` fixture lives in `tests/conftest.py`, not autouse: the tracer provider installs once per process, so it adds the in-memory exporter the first time a test asks and clears it each time; `test_telemetry.py` and `test_mcp.py` request it with `pytestmark = pytest.mark.usefixtures("spans")`.
- Embeddings use `HashingEmbedder` (deterministic, lexical, no download); its numbers are plumbing checks, never quality.
- `tests/test_live_smoke.py` is marked `live` and skips unless `RUN_LIVE_SMOKE=1`; it bills real calls under a cost cap.
- The scripted model (`app/scripted_model.py`) is the second fake behind `_build_client` and the first one that runs in the production image.
  `tests/test_scripted_model.py` covers each request kind, recognition through a metadata nonce (`tests/scripted_helpers.py` makes a picture byte-unique without changing its pixels), unknown bytes, a malformed request raising `BadRequestError`, exact delays through the injected `sleep`, and cost 0 through the real pricing code; `tests/test_scripted_end_to_end.py` goes upload, worker extract and index, `/ask`, `/ask/stream`, and the judge with `anthropic.AsyncAnthropic.__init__` patched to raise, so a real client built anywhere fails the test, and deletes its committed rows in a `finally`.
  `tests/test_no_spend.py` also proves a scripted process is refused beside a key, and `tests/test_worker_scaling.py` runs four workers on one queue across two organizations as the runtime role.
  `tests/test_extraction_query.py` compares `query_extractions` in SQL with a verbatim copy of the old Python implementation in a hypothesis property over adversarial tenants, so the pushdown and its Python fallback cannot drift.
- Randomness is patched, not ranged: `_backoff_delay` tests patch `random.uniform` and assert exact values.
- Property tests use `hypothesis` (`dev` extra): `tests/conftest.py` registers and loads the `doc-pilot` profile with `derandomize=True, deadline=None`, so every run draws the same examples and a failure reproduces anywhere.
  The `.hypothesis/` example database stays gitignored; pin a found regression as an ordinary example test, not as a database entry.
  `tests/test_reconcile_properties.py` is the model: generate a consistent record, break one number, assert what the engine says.

## The load harness

- `scripts/load/` and `scripts/compose_project.py` have pytest suites beside them (`scripts/load/test_artifact.py`, `test_load.py`, `test_locustfile.py` for `workload.py`, `test_phases.py`, `test_report.py`, `test_samplers.py`, and `scripts/test_compose_project.py`), run from `backend/` as `pytest ../scripts -q`.
  They need the backend venv but no Docker, no Locust, and no database: the plan, the arithmetic, the SSE parser, the preflight and config checks, and the artifact validation are pure functions, and the Docker driver is replaced by a fake.
  `locustfile.py` itself needs Locust, which only the `loadgen` image has, so it is covered through `workload.py`.
- `scripts/load/test_report.py` has the README sync tests: the PERF_TABLE region and the two SVGs must equal what the committed artifacts in `scripts/load/results/` render, and the Performance section outside the region must hold no digit at all, so every number on the page is generated.
  Changing an artifact or the renderer means running `python scripts/load/load.py report --update-readme`.
- `python scripts/load/load.py smoke` is the end-to-end check of the whole harness against a real `doc-pilot-load` stack: a few uploads, two workers, a short read run on the hashing embedder, zero errors, a citation on every answer, spend 0 read from the database.
  It writes to a temporary directory and refuses `scripts/load/results/`, and CI runs it as its own path-filtered workflow (`.github/workflows/load-smoke.yml`), so it is not one of the local gates and it needs Docker.
- `scripts/load/onnx_threads_bench.py` is a measuring tool for the embedding thread count, run in the load image; it is not a pytest test.
- A harness test never commits an artifact to `scripts/load/results/`: only a full run does, and a dirty tree or an invalid phase makes it unpublishable.

## Evals as tests

- Extraction: `python evals/run.py --mock [--prompt extract_v2] [--doc STEM]`; `tests/test_evals_runner.py` pins the mock run at `222 / 224` (32 documents times the seven v1 fields) via `predicted_mock_accuracy` and its `caught_by_checks` at `0.5` (the tax perturbation is caught by the arithmetic, the null vendor is not), and the v2 mock at `638 / 640` (32 documents times 20 fields) with `caught_by_checks` and `caught_by_quotes` both `0.5` (the perturbed tax keeps its gold quote, so the quote check names it). `app/evals/scoring.py` holds the per-field match rules (amounts as exact decimals through `money.to_decimal`, `_amounts_match`; identifiers, tax lines, and the evidence counts for v2). A mock run writes to a temporary directory and refuses `evals/results` and everything under it (`tests/test_evals_cli.py`), because `load_results` would report it in the README, and `--update-readme` refuses an `--out` other than `evals/results`. `evals/check_bar.py <v1> <v2>` (logic in `app/evals/bar.py`, `tests/test_evals_bar.py`) checks the pre-registered v2 acceptance bar term by term.
- Reconciliation: `app/evals/checks.py` computes `caught_by_checks` (replayed from each artifact's stored answers whenever a table renders, never stored) and the mutation eval over the labels; `tests/test_evals_checks.py` pins both, plus the quote eval (`run_quote_eval`: the same broken documents with quotes faithful, misread alike, or missing), and `python evals/run_checks.py --update-readme` regenerates the CHECKS_TABLE and QUOTES_TABLE regions offline (no database, key, or model; exits 1 if a label fails the checks).
- README sync: `test_readme_eval_table_and_latest_md_are_current`, `test_readme_checks_table_is_current`, and `test_readme_quotes_table_is_current` fail until the README and `evals/results/latest.md` are regenerated, which is what a rules change needs.
- Retrieval: `tests/test_retrieval_gates.py` compares the offline eval with `evals/retrieval/snapshot_hashing_v1.json`; CI's `retrieval-eval` job runs `evals/run_retrieval.py --check` with the pinned fastembed model.
  A retrieval result records `RETRIEVAL_DATASET_VERSION` (`v1`, the version of the 25 gold texts), not the labels' version, so adding documents without gold text does not make the committed results incomparable.
- Agent and judge: `evals/run_agent.py` and `evals/run_judge_calibration.py` (logic in `app/evals/agent.py`, `judge.py`); live runs stop at `--max-cost`.
- Labels are synthetic and generated record-first (`backend/scripts/make_evals.py`); dataset v2 labels hold the 20 v2 fields and gold evidence, `--labels-only` rewrites them without touching an image (the images of 001-025 are never regenerated), and `tests/test_evals_dataset.py` pins the v1 fields of those 25 by digest. `backend/scripts/harvest_corrections.py` turns review corrections of v2 extractions into labels.

## Known gaps

- Pure-unit runs still need Postgres: the session-start database setup runs for every `pytest` invocation under `backend/`.
- No frontend test runner and no browser E2E; Batch 0 adds Python Playwright as a `demo` extra for recordings, not tests.
- Only the reconciliation engine has property tests; the money module's tests are still examples.
