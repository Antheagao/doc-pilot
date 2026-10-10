---
name: doc-pilot-pull-requests
description: doc-pilot's commit, board-update, push, and CI facts for master on the public GitHub repo. Use when writing a commit message, staging, pushing, opening or reviewing a PR, or reading CI results in doc-pilot.
---

# doc-pilot pull-requests overlay

Load the global `pull-requests` guide for the rules; this file holds what exists here.
Gate commands and the commit rules live in `AGENTS.md`.

## Repository

- Public repo `github.com/Antheagao/doc-pilot`, default branch `master`, remote `origin`; use `gh-axi`.
- Solo flow: one commit per task, straight on `master`, pushed right after it lands; there is no open PR flow.
- PRs #1 to #18 were stacked `claude/*` branches from cloud sessions, merged with merge commits (six are in history) and brought into local `master` on 2026-10-09; their remote branches still exist.
- Known gap: no branch protection or required checks, merge commits allowed, `delete_branch_on_merge` off, and no git hooks.

## Commits

- Local exception: subjects are plain descriptive sentences with no task-id prefix (`Restart the database and frontend with the Docker daemon too`, `Per-document chat with field citations`); an area prefix appears now and then (`README: ...`, `Terraform: ...`).
- Bodies explain why in prose paragraphs and name what was observed live (`9badd30`).
- Board updates (`PRESENT.md` check-off with the hash, `PREVIOUS.md`, `FUTURE.md`, `UPDATES.md`) ride in the same commit as the task, as in `d429076` and `9badd30`.
- Known gap: step 5 of `.claude/commands/batch-loop.md` still says never to commit the boards; `AGENTS.md` wins.
- Never commit `backend/.env` or the root `.env`; scan the staged diff for key patterns before every push, since the repo is public.

## CI (`.github/workflows/ci.yml`)

- Runs on every push to `master` and on every pull request.
- `backend`: a `pgvector/pgvector:pg16-trixie` service on 5434, `pip install -e .[dev]`, `ruff check .`, `ruff check ../scripts/demos`, `pytest -q` (no migrate step: `tests/conftest.py` creates and migrates `docpilot_test`).
- `retrieval-eval`: the same service image, `alembic upgrade head` on its `docpilot` database, then fetches the pinned fastembed model, then `evals/run_retrieval.py --embedder fastembed --chunk-sizes 0,200 --headers both --check --slack-queries 1`.
- `frontend`: `npm ci`, `npx eslint src`, `npm run build`.
- `compose`: `docker compose config -q`; the demo overlay check is being added with Batch 0 task T0.1.
- `terraform`: `fmt -check`, `init -backend=false`, `validate`, and `test` with mocked providers.
- CI never calls `evals/run.py --mock` itself; `tests/test_evals_runner.py` pins its accuracy inside `pytest`.
- The live smoke suite and live evals never run in CI.
