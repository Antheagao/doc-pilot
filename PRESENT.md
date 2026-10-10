# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## Batch 0 - Demo GIF pipeline and the backlog of merged features (promoted 2026-10-09)

Expanded by the planner against master `d429076`.
Key design decisions:
- Python Playwright as an optional `demo` extra in `backend/pyproject.toml` (one toolchain with the seed and ffmpeg steps; frontend lockfile and image untouched).
- Demos run in a separate compose project `doc-pilot-demo` (`docker-compose.demo.yml`: api 8100, frontend 3100, no host port for its db), so the dev/test db on 5434 is never touched.
- Seed data replays the committed Sonnet eval run (`evals/results/20260731T064853Z_claude-sonnet-5_extract_v1.json`) for honest values, confidences, and costs, with gold page text and in-container embeddings; staged review cases are named constants marked "staged".
- GIFs committed at `screenshots/features/<slug>.gif` (each <= 15 s, < 1 MB); WebM/MP4/poster pairs go to gitignored `scripts/demos/out/`.

### ⚠ BLOCKED: C: drive is full (0.5 GB free, 2026-10-09)

Docker Desktop's disk lives on C:, so image builds fail with `read-only file system` and the Docker CLI hangs.
Nothing that needs Docker (tests on 5434, image rebuilds, demo stack) or a Playwright browser install can run until space is freed on C: or Docker's disk image and `PLAYWRIGHT_BROWSERS_PATH` are moved to D: (527 GB free).

### Tasks

- [x] **T0.0 Restart policies for db, frontend, jaeger** - after a Docker Desktop restart only api and worker came back, and the worker crash-looped on the unresolvable `db` host (observed live).
  *Verify:* `docker compose --profile tracing config --format json` shows `unless-stopped` for db/api/worker/frontend/jaeger and `no` for migrate (passed). Live restart check pending the disk blocker.
- [ ] **T0.1 Isolated demo stack** - `docker-compose.demo.yml` (name `doc-pilot-demo`, `db.ports: !reset []`, api `127.0.0.1:8100`, frontend `127.0.0.1:3100` built with `NEXT_PUBLIC_API_URL=http://localhost:8100`, `CORS_ORIGINS`, `DAILY_BUDGET_USD=${DEMO_DAILY_BUDGET_USD:-1.50}`, judge sampling and Langfuse off, a `seed` service under `profiles: ["seed"]` mounting `./evals:/evals:ro`); `scripts/demos/demo.py stack up|down|status` that always passes `-p doc-pilot-demo` and previews volumes before `down --volumes`, refusing any other project name; CI `config -q` for the overlay; gitignore `scripts/demos/out/`.
  *Verify:* overlay `config -q` exits 0; `demo.py stack up` then `/healthz` on 8100 and `/` on 3100 return 200; `docker port doc-pilot-demo-db-1` prints nothing; dev stack and `pytest` on 5434 unaffected.
- [ ] **T0.2 Deterministic demo seed** - extract `build_extracted_fields(extraction_id, tool_input, review_threshold)` from the worker's persistence loop in `backend/app/extraction.py` (worker calls it unchanged); `backend/app/demo_seed.py` replays the Sonnet artifact (uuid5 ids, images copied into the upload dir, `_demo_replay` provenance in `raw_response`, `created_at` staggered over the 14 days before today, gold pages indexed with `build_embedder`), applies `STAGED_REVIEW_CASES` (016 subtotal 46.25 @ 0.68); `reset_demo_data()` refuses unless `DOCPILOT_DEMO=1`; thin CLI `backend/scripts/seed_demo.py`; `demo.py seed [--reset]`.
  *Verify:* `pytest tests/test_demo_seed.py tests/test_extraction.py -q` and `ruff check app tests scripts` pass; after `demo.py seed --reset`: 25 documents, the expected review count, `search?q=a light for my workspace` ranks 018/022 first, `/documents/<016>/file` returns 200.
- [ ] **T0.3 Recording harness and encoder** - `demo` extra; `scripts/demos/harness.py` (1280x800, light, UTC, injected cursor overlay with click pulse, pacing constants, `fast_forward(factor)` segments with an on-screen speed chip, JSON sidecar); `scripts/demos/encode.py` (trim/setpts/concat, two-pass palette GIF at 12 fps and 960 px with fallbacks to stay < 1 MB, WebM + MP4 + poster); `demo.py record <slug>|all` and `check` (ffprobe: <= 15 s, < 1 MB, 960 px); smoke scenario only.
  *Verify:* `demo.py record smoke` produces gif/webm/mp4/jpg within limits; encoder unit tests pass; the GIF shows the light theme, visible cursor, no blank lead-in.
- [ ] **T0.4 Zero-spend GIFs: hybrid search, review correction** - `search-hybrid` (hybrid "a light for my workspace", then keywords "27.82 euros" hitting the Berlin receipt) and `review-correction` (correct 016's subtotal, badge drops, document shows the struck-through original).
  *Verify:* `demo.py seed --reset && demo.py record search-hybrid review-correction && demo.py check` passes; both GIFs inspected at 100% zoom.
- [ ] **T0.5 Live GIFs: /ask agent steps, per-document chat citations** - recorded against the real agent, with live waits fast-forwarded and disclosed on screen; spend capped by the demo stack's daily budget.
  *Verify:* `demo.py check` passes; summed `ask_runs` cost <= $1.50.
  ⚠ Needs Anthony's approval of this demo spend (expected $0.50-1.00, cap $1.50).
- [ ] **T0.6 Monitoring dashboard GIF** - both halves (eval history from committed artifacts, production spend from the replayed 14 days plus today's live runs), one tooltip, short slow scroll.
  *Verify:* `demo.py check` passes.
- [ ] **T0.7 README Features gallery and recording docs** - "## Features" section with one GIF and one sentence per feature, captions stating what is real (replayed Sonnet run, gold page text, live Opus for ask/chat, fast-forward disclosed); "Recording the demos" subsection.
  *Verify:* every `screenshots/features/*.gif` linked from README exists; `screenshots/features` totals < 5 MB; renders on GitHub.

### Open questions for Anthony (from planning)

- Approve the T0.5 live demo spend (expected $0.50-1.00, hard cap $1.50)?
- Dashboard production half: real data only (sparse answer charts), or allow a clearly captioned synthetic block of `ask_runs`?
- Replay the committed Sonnet run's real values and confidences (planner's recommendation) instead of FUTURE.md's label-perfect records?
- Keep WebM/MP4 pairs out of git (copied into portfolio3 by hand)?
- Optional T0.8: re-record the top `screenshots/demo.gif`, which predates the Search/Ask/Dashboard nav (one live extraction, about $0.02)?
- Prune `search-hybrid.png` / `review-correct.png` from the Screenshots section once GIFs cover them?

## Carry-over (accepted, not blocking)

- H3: a well-formed `line_items` leaf with a non-list value silently stores `[]` (resolved by Batch A2).
- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
