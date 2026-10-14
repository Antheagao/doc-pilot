---
name: doc-pilot-architecture
description: doc-pilot system shape (FastAPI API, Postgres SKIP LOCKED job queue with extract, index, and judge jobs, model calls, retrieval, /ask agent, chat, evals, telemetry, Next.js UI) and where new features and env vars go. Use when adding a module, job kind, model call, env var, or feature batch in doc-pilot.
---

# doc-pilot architecture overlay

Load the global `design-principles` guide for the rules; this file holds what exists here.

## System shape

- Local-only resume demo, never hosted; `infra/terraform/` (Cloud Run and Cloud SQL) is kept as a sample, checked in CI, and never applied.
- The browser calls the API directly at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`, inlined at build time); `CORS_ORIGINS` admits the frontend; there is no proxy and no auth.
- API: `backend/app/main.py` mounts `app/routers/` at the root behind `MaxBodySizeMiddleware` (20 MB) and `SecurityHeadersMiddleware`.
- Worker: `python -m app.worker`, one process, one job at a time; `claim_job` takes the oldest `pending` row with `FOR UPDATE SKIP LOCKED`, and `dispatch_job` routes `jobs.kind` through `HANDLERS`.
- `extract` (`extraction.process_document_job`): forced tool use (`tool_choice` names `record_extraction`), `{value, confidence}` leaves, one schema-repair reprompt, PDFs over `pdf_max_pages_per_call` split per page and merged; fields under `review_threshold` get `needs_review` with `review_reason` `confidence`, then `review_routing.refresh_reconciliation` (inside a savepoint, so a rules failure is logged and costs only the reconciliation, never the billed extraction) stores the `reconciliations` row and re-routes the flags.
- `index` (`retrieval/index_job.process_index_job`), enqueued by the extract job in its own transaction: `transcription.py` (one page per call), `retrieval/chunking.py`, an `Embedder`, then pgvector plus full-text hybrid search fused by RRF (`retrieval/search.py`).
- `judge` (`evals/online.process_judge_job`): grades a sampled `AskRun` for groundedness (`ASK_JUDGE_SAMPLE_RATE`).
- `/ask` (`routers/ask.py`, `agent/loop.py`, `agent/tools.py`): a manual tool-use loop over `search_documents`, `query_extractions`, and `get_page`, capped by `agent_max_steps` and `agent_max_cost_usd`; every answer is stored as an `AskRun`.
- Per-document chat (`routers/chat.py`) reuses `run_agent` with a `ChatScope`; its turns are `AskRun` rows with `document_id` and `conversation_id`.
- Spend guards: `budget.enforce_daily_budget` (`DAILY_BUDGET_USD`) and `ratelimit.limit_ask` / `limit_upload` (in memory, one API process).
- Evals: logic in `backend/app/evals/`, CLIs in `evals/run*.py`, committed artifacts in `evals/results/`, read by `GET /stats` and `GET /monitoring/*`.
- `app/evals/checks.py` replays stored answers and labels through `build_extracted_fields`, `reconcile`, and `route` (no database): `caught_by_checks` per run at render time, and the mutation eval behind `evals/run_checks.py`; `harvest.py` refuses a label that fails the checks.
- Telemetry: OpenTelemetry in `app/telemetry.py`, off unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set; `jobs.traceparent` carries a document's trace across the queue; Langfuse (`app/langfuse_link.py`) is off unless its keys are set.
- Frontend: Next.js 16 in `frontend/`; every page is a client component fetching through `src/lib/api.ts`.

## Where new work goes

- A background step is a new `JOB_KIND_*` in `models.py`, a handler in `HANDLERS`, and an enqueue in the same transaction as the change that causes it (the extract-to-index pattern).
- A model call reuses `_build_client`, `_classify_api_error`, `_ensure_model_priced`, and `PRICING_PER_MTOK` from `extraction.py`, loads a versioned file from `backend/prompts/`, opens a `model_call_span`, and stores its cost on a row that `budget.spent_today_usd` sums.
- A model-touching feature ships with an eval before it ships to the demo.
- Arithmetic over money never goes to a model ("the model reads, code computes", `FUTURE.md`).
- Reconciliation (Batch A): `app/money.py` (exact decimals), `app/reconcile.py` (pure rules and report), `app/review_routing.py` (`route` is the pure flag table; `refresh_reconciliation` is the only database writer and every path that creates or resolves fields calls it); `scripts/reconcile_all.py` backfills old extractions (`--dry-run` previews).
- Planned (`FUTURE.md`): `app/grounding.py` behind an `OcrProvider` (Batch C), `/reports/totals` (Batch D); Batch 0 adds `app/demo_seed.py` and `scripts/demos/`.

## Environment variables

- Every setting is a field on `Settings` in `backend/app/config.py` (pydantic-settings, reads `backend/.env`), bounded with `Field(ge=...)` where it matters, and read through `get_settings()`, which tests override.
- Read straight from `os.environ` instead: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME` (`telemetry.py`) and `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` (`langfuse_link.py`).
- A new variable also goes into `docker-compose.yml` for each service that reads it (`api`, `worker`), and into `docker-compose.demo.yml` when the demo must pin it.
- Known gap: `backend/.env.example` lists only `ANTHROPIC_API_KEY`, `DATABASE_URL`, `EXTRACTION_MODEL`, and `REVIEW_THRESHOLD`.
