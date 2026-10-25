---
name: doc-pilot-repository-structure
description: doc-pilot's folder map - backend app packages, prompts, Alembic versions, scripts, tests, evals, frontend routes and components, infra, boards, and planned paths. Use when placing, naming, or moving a file or folder in doc-pilot.
---

# doc-pilot repository-structure overlay

Load the global `repository-structure` guide for the rules; this file holds what exists here.
The Rust and sqlx sections of the global guide do not apply; their Python and Alembic equivalents are below.

## Top level

| Path | Holds |
| --- | --- |
| `backend/` | The FastAPI package `app/`, `prompts/`, `alembic/`, `scripts/`, `tests/`, `pyproject.toml`, `Dockerfile` |
| `frontend/` | The Next.js app under `src/`, imported through `@/` |
| `evals/` | CLIs `run.py`, `run_retrieval.py`, `run_agent.py`, `run_judge_calibration.py`; data in `docs/`, `labels/`, `text/`, `agent/`, `retrieval/`; committed artifacts in `results/` |
| `infra/` | `terraform/` (a sample, never applied) and `backend.Dockerfile` (the backend image plus eval artifacts) |
| `samples/`, `screenshots/` | PII-free upload samples and README images; feature GIFs are planned at `screenshots/features/<slug>.gif` |
| `scripts/demos/` | The demo pipeline CLI `demo.py` (Batch 0, in progress); recordings go to gitignored `scripts/demos/out/` |
| Root files | `README.md`, `AGENTS.md`, the boards `PRESENT.md`, `PREVIOUS.md`, `FUTURE.md`, `UPDATES.md`, `docker-compose.yml`, `docker-compose.demo.yml` |

## Backend

- `app/` modules: `main.py`, `config.py`, `db.py`, `models.py`, `schemas.py`, `worker.py`, `extraction.py`, `extraction_fields.py`, `transcription.py`, `pdf.py`, `records.py`, `budget.py`, `ratelimit.py`, `telemetry.py`, `langfuse_link.py`.
- Packages: `routers/` (one file per resource), `retrieval/`, `agent/` (`loop.py`, `tools.py`), `mcp_server/` (`server.py`: the tool adapter and result rendering; `http.py`: the `/mcp` ASGI endpoint; `__main__.py`: the stdio entry point), `evals/` (the logic the `evals/*.py` CLIs import).
- All Pydantic request and response models live in `app/schemas.py`, and all ORM models in `app/models.py`.
- Prompts: `backend/prompts/<name>_v<N>.md` (`extract_v1`, `repair_v1`, `transcribe_v1`, `agent_v1`, `document_chat_v1`, `mcp_instructions_v1`, `judge_v1`, `groundedness_v1`).
- Migrations: `backend/alembic/versions/<revision>_<snake_slug>.py`, created by `alembic revision` and chained by `down_revision`.
- Scripts: snake_case `backend/scripts/*.py` CLIs (`fetch_embedding_model`, `harvest_corrections`, `make_evals`, `make_samples`, `reindex`, `smoke`).
- Tests: flat `backend/tests/test_<area>.py` (`test_mcp.py` covers the whole `mcp_server/` package) plus `conftest.py`; eval logic is `test_evals_<suite>.py`, retrieval is `test_retrieval_<part>.py`.
- Planned: `app/money.py`, `app/reconcile.py`, `app/grounding.py`, `app/demo_seed.py`, `backend/scripts/seed_demo.py`, `backend/prompts/extract_v2.md`.

## Frontend

- Routes in `src/app/`: `page.tsx`, `ask/`, `dashboard/`, `documents/[id]/`, `review/`, `search/`.
- Local exception: component files are PascalCase with a default export (`StatusChip.tsx`), except `charts.tsx`, a family of named exports; new components match their neighbors.
- `src/lib/api.ts` is the only library module (types and fetch helpers).
- No `ui/`, `hooks/`, or `_components/` folders exist; the one custom hook, `useLiveProgress`, lives in `AgentRun.tsx`.
- All styles live in `src/app/globals.css`.
