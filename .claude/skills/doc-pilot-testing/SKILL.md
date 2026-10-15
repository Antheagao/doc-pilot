---
name: doc-pilot-testing
description: doc-pilot's pytest harness on its own docpilot_test Postgres database, model and embedder fakes, live smoke opt-in, eval gates, and known test gaps. Use when writing, placing, or debugging a backend test or eval, or chasing a flaky run in doc-pilot.
---

# doc-pilot testing overlay

Load the global `testing` guide for the rules; this file holds what exists here.
Gate commands live in `AGENTS.md`; the Rust and Playwright subsections of the global guide have no counterpart here yet.

## Harness

- pytest with pytest-asyncio: `asyncio_mode = "auto"` and one session-wide event loop (`backend/pyproject.toml`), because `app.db` builds one asyncpg engine at import.
- The suite has its own database: `tests/conftest.py` points `DATABASE_URL` at `TEST_DATABASE_URL` (default: the dev URL with the database `docpilot_test`) before `app.db` is imported, refuses the dev database, and at session start creates the database if missing and runs `alembic upgrade head` (which installs pgvector).
  The compose worker polls `docpilot`, so it can never claim rows the tests commit.
- `db_session` opens one connection to that database on host port 5434, begins a transaction, and rolls it back after the test; `join_transaction_mode="create_savepoint"` lets app code call `commit()`.
- `client` drives the real app through `httpx.AsyncClient(transport=ASGITransport(app=app))` and overrides `get_session`, `get_session_factory`, and `get_settings` (tmp `upload_dir`, budget and rate limits off).
- There is no per-test schema or database: committed rows persist in `docpilot_test` between runs, so never assert global counts (`tests/test_stats.py` header).
- Worker tests (`tests/test_worker.py`) commit real rows through the `real_documents` fixture and scope claims with `pending_job_stmt(..., document_id=...)`; `test_skip_locked_never_double_claims` holds two real sessions.
- No test reaches the network: `monkeypatch.setattr(extraction_module, "_build_client", lambda settings: fake_client)` (`test_extraction.py`, `test_transcription.py`), fakes built from `SimpleNamespace` with an `AsyncMock` `messages.create`, and `app.dependency_overrides[get_anthropic_client] = lambda: fake` for `/ask` and chat (`test_agent.py`, `test_chat.py`, `test_budget.py`).
- Embeddings use `HashingEmbedder` (deterministic, lexical, no download); its numbers are plumbing checks, never quality.
- `tests/test_live_smoke.py` is marked `live` and skips unless `RUN_LIVE_SMOKE=1`; it bills real calls under a cost cap.
- Randomness is patched, not ranged: `_backoff_delay` tests patch `random.uniform` and assert exact values.
- Property tests use `hypothesis` (`dev` extra): `tests/conftest.py` registers and loads the `doc-pilot` profile with `derandomize=True, deadline=None`, so every run draws the same examples and a failure reproduces anywhere.
  The `.hypothesis/` example database stays gitignored; pin a found regression as an ordinary example test, not as a database entry.
  `tests/test_reconcile_properties.py` is the model: generate a consistent record, break one number, assert what the engine says.

## Evals as tests

- Extraction: `python evals/run.py --mock`; `tests/test_evals_runner.py` pins the mock run at `222 / 224` (32 documents times the seven v1 fields) via `predicted_mock_accuracy` and its `caught_by_checks` at `0.5` (the tax perturbation is caught by the arithmetic, the null vendor is not), and `app/evals/scoring.py` holds the per-field match rules (amounts as exact decimals through `money.to_decimal`, `_amounts_match`).
- Reconciliation: `app/evals/checks.py` computes `caught_by_checks` (replayed from each artifact's stored answers whenever a table renders, never stored) and the mutation eval over the labels; `tests/test_evals_checks.py` pins both, and `python evals/run_checks.py --update-readme` regenerates the CHECKS_TABLE region offline (no database, key, or model; exits 1 if a label fails the checks).
- README sync: `test_readme_eval_table_and_latest_md_are_current` and `test_readme_checks_table_is_current` fail until the README and `evals/results/latest.md` are regenerated, which is what a rules change needs.
- Retrieval: `tests/test_retrieval_gates.py` compares the offline eval with `evals/retrieval/snapshot_hashing_v1.json`; CI's `retrieval-eval` job runs `evals/run_retrieval.py --check` with the pinned fastembed model.
  A retrieval result records `RETRIEVAL_DATASET_VERSION` (`v1`, the version of the 25 gold texts), not the labels' version, so adding documents without gold text does not make the committed results incomparable.
- Agent and judge: `evals/run_agent.py` and `evals/run_judge_calibration.py` (logic in `app/evals/agent.py`, `judge.py`); live runs stop at `--max-cost`.
- Labels are synthetic and generated record-first (`backend/scripts/make_evals.py`); dataset v2 labels hold the 20 v2 fields and gold evidence, `--labels-only` rewrites them without touching an image (the images of 001-025 are never regenerated), and `tests/test_evals_dataset.py` pins the v1 fields of those 25 by digest. `backend/scripts/harvest_corrections.py` turns review corrections of v2 extractions into labels.

## Known gaps

- Pure-unit runs still need Postgres: the session-start database setup runs for every `pytest` invocation under `backend/`.
- No frontend test runner and no browser E2E; Batch 0 adds Python Playwright as a `demo` extra for recordings, not tests.
- Only the reconciliation engine has property tests; the money module's tests are still examples.
