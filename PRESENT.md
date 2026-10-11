# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## Batch 0 - Demo GIF pipeline and the backlog of merged features (promoted 2026-10-09)

Expanded by the planner against master `d429076`.
Key design decisions:
- Python Playwright as an optional `demo` extra in `backend/pyproject.toml` (one toolchain with the seed and ffmpeg steps; frontend lockfile and image untouched).
- Demos run in a separate compose project `doc-pilot-demo` (`docker-compose.demo.yml`: api 8100, frontend 3100, no host port for its db), so the dev/test db on 5434 is never touched.
- Seed data replays the committed Sonnet eval run (`evals/results/20260731T064853Z_claude-sonnet-5_extract_v1.json`) for honest values, confidences, and costs, with gold page text and in-container embeddings; staged review cases are named constants marked "staged".
- GIFs committed at `screenshots/features/<slug>.gif` (each <= 15 s, < 1 MB); WebM/MP4/poster pairs go to gitignored `scripts/demos/out/`.

The 2026-10-09 disk blocker is resolved (Anthony freed C: on 2026-10-10).

### Tasks

- [x] **T0.0 Restart policies for db, frontend, jaeger** - after a Docker Desktop restart only api and worker came back, and the worker crash-looped on the unresolvable `db` host (observed live, and reproduced again on the 2026-10-10 Docker start). - `9badd30`
  *Verify:* `docker compose --profile tracing config --format json` shows `unless-stopped` for db/api/worker/frontend/jaeger and `no` for migrate (passed); after the rebuild, `docker inspect` shows `unless-stopped` on all four running dev containers. Live check passed 2026-10-10: after a machine restart, starting Docker Desktop brought db, api, worker (healthy), and frontend back with no manual `up`, while the demo stack (`restart: "no"`) stayed down.
- [x] **T0.0b Local gates and the worker failure path** (found while verifying, fix before T0.2):
  - Worker bug, reproduced end to end: upload a receipt while the worker's key is invalid and the document sits in `processing` forever with no `last_error`. `_run_handler` (`worker.py:420,429`) and `fail_job` (`worker.py:255`) read `job.id` from a `Job` that the handler's own pre-model-call rollback expired, which raises `MissingGreenlet`; carry `job_id` instead, with a regression test whose handler rolls back and then raises.
  - Tests share the dev database with the running compose worker, which claims the jobs tests commit (`test_skip_locked_never_double_claims` fails while it runs): give tests their own database on the same server, created and migrated at session start, with CI matching.
  - `tzdata` in the `dev` extra, so `ZoneInfo` tests pass on Windows.
  - The 46 ruff findings in `alembic/` (safe autofixes) and CI linting the whole backend (`ruff check .`).
  - A real worker healthcheck (heartbeat file freshness), so `docker compose up --wait` works; today `healthcheck: disable: true` makes `--wait` exit 1 on compose 2.40, the opposite of what its comment says. Then simplify `demo.py stack up` to a single `--wait`.
  - Pin `pgvector/pgvector:pg16-trixie` in compose and CI: the dev volume was initialized under glibc 2.41 and `pg16` (bookworm, 2.36) logs a collation version mismatch.
  *Verify:* with the dev worker running, `ruff check .` and `pytest -q` pass from `backend/` on Windows; repeating the invalid-key upload ends with the document `failed` and a `last_error`; `docker compose up -d --wait` exits 0; no collation warning from `psql`. (passed: 489 tests green with the dev worker running; 3 new handler tests fail with `MissingGreenlet` on the old code; review fixes: readable alembic errors, advisory lock on first-run migrate, engine-binding assertion, demo tests in CI)
- [x] **T0.1 Isolated demo stack** - `docker-compose.demo.yml` (name `doc-pilot-demo`, `db.ports: !reset []`, api `127.0.0.1:8100`, frontend `127.0.0.1:3100` built with `NEXT_PUBLIC_API_URL=http://localhost:8100`, `CORS_ORIGINS`, `DAILY_BUDGET_USD=${DEMO_DAILY_BUDGET_USD:-1.50}`, judge sampling and Langfuse off, a `seed` service under `profiles: ["seed"]` mounting `./evals:/evals:ro`); `scripts/demos/demo.py stack up|down|status` that always passes `-p doc-pilot-demo` and previews volumes before `down --volumes`, refusing any other project name; CI `config -q` for the overlay; gitignore `scripts/demos/out/`.
  *Verify:* overlay `config -q` exits 0; `demo.py stack up` then `/healthz` on 8100 and `/` on 3100 return 200; `docker port doc-pilot-demo-db-1` prints nothing; dev stack and `pytest` on 5434 unaffected. (passed; review: the `down --volumes` guard now checks the resolved config's volume names, `stack up` fails on a dead worker, and dev OTEL and `COMPOSE_PROFILES` no longer leak into the demo; 9 guard tests in `scripts/demos/test_demo.py`)
- [x] **T0.2 Deterministic demo seed** - extract `build_extracted_fields(extraction_id, tool_input, review_threshold)` from the worker's persistence loop in `backend/app/extraction.py` (worker calls it unchanged); `backend/app/demo_seed.py` replays the Sonnet artifact (uuid5 ids, images copied into the upload dir, `_demo_replay` provenance in `raw_response`, `created_at` staggered over the 14 days before today, gold pages indexed with `build_embedder`), applies `STAGED_REVIEW_CASES` (016 subtotal 46.25 @ 0.68); `reset_demo_data()` refuses unless `DOCPILOT_DEMO=1`; thin CLI `backend/scripts/seed_demo.py`; `demo.py seed [--reset]`.
  *Verify:* `pytest tests/test_demo_seed.py tests/test_extraction.py -q` and `ruff check app tests scripts` pass; after `demo.py seed --reset`: 25 documents, the expected review count, `search?q=a light for my workspace` ranks 018/022 first, `/documents/<016>/file` returns 200. (passed: 25 documents, 4 pending reviews (3 real model misses + staged 016), desk-lamp search ranks 018 then 022, today's spend $0; review fixes: the demo database is `docpilot_demo` and reset/seed refuse any other database before any statement, upload cleanup deletes only `<uuid>.<ext>` files, and the staged 016 override is recorded in `_demo_replay.staged`)
- [x] **T0.3 Recording harness and encoder** - `demo` extra; `scripts/demos/harness.py` (1280x800, light, UTC, injected cursor overlay with click pulse, pacing constants, `fast_forward(factor)` segments with an on-screen speed chip, JSON sidecar); `scripts/demos/encode.py` (trim/setpts/concat, two-pass palette GIF at 12 fps and 960 px with fallbacks to stay < 1 MB, WebM + MP4 + poster); `demo.py record <slug>|all` and `check` (ffprobe: <= 15 s, < 1 MB, 960 px); smoke scenario only.
  *Verify:* `demo.py record smoke` produces gif/webm/mp4/jpg within limits; encoder unit tests pass; the GIF shows the light theme, visible cursor, no blank lead-in.
  *Result:* recording uses Playwright `page.screencast` JPEG frames and an ffmpeg concat list instead of `record_video` WebM, whose real-time VP8 flickered on every text edge and pushed the smoke GIF to 2.8-4 MB; the smoke GIF is now 565-907 KB at 12 fps and 960 px. 59 demo-script tests pass without Playwright or ffmpeg.
  *Review (2026-10-10, re-run after the interrupted session):* fixed in the follow-up commit. The size ladder now keeps 960 px and steps fps and palette size (12/256, 10/128, 8/64), and `record` refuses to copy a GIF that `check` would reject, failing before any encode when the timeline is over 15 s. Every concat entry declares `option framerate 1000`, because the 25 fps image default had rounded frame times to 40 ms and dropped frames (82 of 257 shared a tick; now 0). A second `rec.open()` keeps the first lead-in and resets the pointer. `__recSetChip` exists before the overlay installs. ffmpeg output is decoded as UTF-8 on Windows. Concat-list path quoting checked and fine (harness-generated bare names only). Smoke re-recorded on macOS: 553 KB, 960 px, 6.4 s.
- [x] **T0.3b UI polish seen in the recording frames** - money rendered with the currency's minor-unit digits (`5.10`, not `5.1`) through `src/lib/format.ts`, no em dash in UI copy, latency in seconds, timestamps to the minute. Committed with eslint, tsc, and `next build` green. - `216dbf1`, header follow-up in the commit that checks this off
  *Verify:* real-browser check light and dark at 360, 768, and 1440 px on the home, two document pages (016, 024), review, and dashboard against the seeded demo stack (passed 2026-10-10 on macOS: no horizontal overflow, no console errors, amounts with minor-unit digits). The check caught the brand running into the nav at 360 px ("doc-pilotSearch") and a `--` em-dash stand-in in the dashboard intro; both fixed (header gap, smaller phone nav), 12.6 px between brand and nav at 360 px.
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
