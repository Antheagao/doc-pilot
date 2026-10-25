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

- `app/` modules: `main.py`, `config.py`, `db.py`, `models.py`, `schemas.py`, `worker.py`, `extraction.py`, `extraction_fields.py`, `transcription.py`, `pdf.py`, `records.py`, `budget.py`, `ratelimit.py`, `telemetry.py`, `langfuse_link.py`, `tenancy.py` (binding a session to an organization), and `db_roles.py` (creating the runtime login role).
- Packages: `routers/` (one file per resource, `auth.py` and `api_keys.py` among them), `retrieval/`, `agent/` (`loop.py`, `tools.py`), `auth/` (sign-in and credentials, which owns the identity tables: `passwords.py`, `tokens.py`, `sessions.py`, `api_keys.py`, `accounts.py`, and `principal.py` with `current_principal`), `mcp_server/` (`server.py`: the tool adapter and result rendering; `http.py`: the `/mcp` ASGI endpoint; `auth.py`: API-key authentication for both transports; `__main__.py`: the stdio entry point), `evals/` (the logic the `evals/*.py` CLIs import).
- All Pydantic request and response models live in `app/schemas.py`, and all ORM models in `app/models.py`.
- Prompts: `backend/prompts/<name>_v<N>.md` (`extract_v1`, `repair_v1`, `transcribe_v1`, `agent_v1`, `document_chat_v1`, `mcp_instructions_v1`, `judge_v1`, `groundedness_v1`).
- Migrations: `backend/alembic/versions/<revision>_<snake_slug>.py`, created by `alembic revision` and chained by `down_revision`.
- Scripts: snake_case `backend/scripts/*.py` CLIs (`accounts`: organizations, users, memberships, and API keys; `provision_db_role`: the runtime login role; `fetch_embedding_model`, `harvest_corrections`, `make_evals`, `make_samples`, `reindex`, `smoke`).
- Tests: flat `backend/tests/test_<area>.py` (`test_mcp.py` covers the whole `mcp_server/` package; `test_auth_core.py` the credential primitives and the accounts CLI, `test_auth.py` the sign-in routes, `test_api_keys.py` the key routes, `test_tenancy.py` the ORM layer, `test_rls.py` the database layer, `test_tenant_isolation.py` the end-to-end claim, `test_route_auth.py` the route walk, `test_no_spend.py` the no-model guard) plus `conftest.py` and the shared helper `tenant_rows.py`; eval logic is `test_evals_<suite>.py`, retrieval is `test_retrieval_<part>.py`.
- Planned: `app/grounding.py` (`FUTURE.md`, Batch C).

## Frontend

- Routes in `src/app/`: the route group `(app)/` holds `page.tsx`, `ask/`, `dashboard/`, `documents/[id]/`, `review/`, `search/`, and `settings/api-keys/` with the header layout, and `login/` sits outside it; the root `layout.tsx` is html and body only.
- Local exception: component files are PascalCase with a default export (`StatusChip.tsx`), except `charts.tsx`, a family of named exports; new components match their neighbors.
- `src/lib/api.ts` holds the types, `apiFetch`, and the fetch helpers (auth and API keys included).
- No `ui/`, `hooks/`, or `_components/` folders exist; the one custom hook, `useLiveProgress`, lives in `AgentRun.tsx`.
- All styles live in `src/app/globals.css`.
