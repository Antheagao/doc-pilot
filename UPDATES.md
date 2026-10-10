# doc-pilot — Progress Log

Working toward: AI document intelligence flagship (plan §2 in `C:\projects\references\projects upgrade.md` — planning docs live in the private `references` repo since 2026-08-03).
Update this file at the end of every session: move items between sections, append a dated log entry.

## Status

**Stage: PROJECT DONE (2026-08-03) — deploy permanently cancelled, not deferred.** Anthony ruling 2026-08-03 (recorded in `projects upgrade v2.md`): **no hosting, ever** — for any project. That kills the last blocked item; the queue exit condition (local demo GIF + verified eval table, no deploy) is met and the repo is marked ✅ DONE in the v2 queue. The `<!-- LIVE_DEMO -->` README slot will never be filled; dependency pinning was gated on "before the hosted demo ships" and is moot. Stretch (pgvector RAG, CSV export) stays optional, no longer gated on anything. Feature state: core loop, eval harness with committed live results, review queue + corrections flywheel, failure-handling story, /stats + UI strip, `docker compose up --build` proven end to end, opt-in live smoke suite ($0.10 hard cap), README with GIF up top + compose-first quickstart + "why these choices". 212 mocked tests + 2 live; 3 CI jobs. Portfolio card shipped (portfolio3 `6631444`). Postgres on host port **5434**; API key in gitignored `backend/.env` (mirrored to gitignored root `.env` for compose).

## Done

- [x] 2026-07-30 — Repo created, git initialized, README written; Batch 1 (core loop T1–T7) complete
- [x] 2026-07-31 — Batch 2 (evals E1–E5) complete; live Sonnet/Haiku results in README
- [x] 2026-07-31 — Batch 3 (review queue + corrections→evals harvest) complete via external session; CI + screenshots shipped early

## Next Up

- Nothing required — project done; deploy cancelled permanently (no-hosting ruling 2026-08-03). Former blocked items are moot (deploy, pre-deploy lockfiles) or optional:
- [ ] (optional, ungated) stretch: pgvector RAG search, CSV export
- [ ] (optional hygiene) pin dependencies / lockfiles — images are only built locally now

## Later

- (emptied 2026-08-03 — Batch 5 shipped except the deployed demo, which is cancelled; stretch moved to Next Up as optional)

## Session Log

### 2026-10-09 (batch-loop: Batch 0 promoted, T0.0 done, loop stopped on disk space)
- All 18 PRs marked merged on GitHub (retargeted to master, then master pushed at `98c4ae9`); boards committed in `d429076`.
- Batch 0 (demo GIF pipeline) promoted and expanded by the planner into T0.0-T0.7 on PRESENT.md.
- T0.0: `restart: unless-stopped` on db, frontend, jaeger. Found live: the August compose stack's db had exited and the worker had been crash-looping on DNS since.
- BLOCKED: C: has 0.5 GB free; Docker's disk is on C:, so builds fail read-only and the CLI hangs. Loop stopped until Anthony frees space or moves Docker's disk image to D:.

### 2026-10-09 (stack merge + feature roadmap)
- Merged the 18 stacked cloud-session PRs (#1 retrieval through #18 Langfuse, all CI-green) into local master by fast-forward to `98c4ae9`; no merge commits, linear history kept.
- Not yet pushed: GitHub writes were blocked in this session, so origin/master is still `50b9e05` and the PRs are still open.
- `claude/model-probe` was intentionally not merged (a one-off CI probe; its outcome, the embedding re-pin, is already in the stack). `claude/review-queue` was already in master.
- FUTURE.md rewritten as Batches A-G: deterministic reconciliation, schema v2 with evidence quotes, field highlighting over the document, multi-document totals and export, duplicates and vendor master, more document types, and a local back-office workflow.
- Anthony confirmed: local-only resume demo, never hosted (multi-tenant auth dropped from the plan); OCR is model boxes by default with optional local Tesseract; every feature gets its own demo GIF (new Batch 0 harness, plus a GIF task closing each batch).
- The Status section above predates the stack and is stale; refresh it when Batch A is promoted.

### 2026-08-03 (closeout — no code changes; run from the ecommerce-api session during crash recovery)
- Applied Anthony's portfolio-wide no-hosting ruling (2026-08-03) to this board: deploy is cancelled, not deferred, which makes doc-pilot's queue goal fully met. Marked ✅ DONE in `projects upgrade v2.md` (row #2) with evidence. No repo commits — board files only.

### 2026-08-02 (session 5 — README visuals + portfolio pin assessment)
- Correction to the session-4 note: `screenshots/demo.gif` **was** committed (`874bda5`), not left uncommitted.
- Retook all 4 screenshots (`50b9e05`, pushed): the committed ones predated the stats strip (`e9e0d86`) and design-system pass (`5d9a82b`). Docker Desktop was off — restarted it, brought the compose stack up, uploaded 2 more docs (hardware-invoice + eval doc 016 handwriting, ~$0.02 live). Every field cleared the 0.8 threshold, so the review-flow shots were staged: set doc 016's subtotal to a plausible misread (46.25 @ 0.68) and vendor @ 0.74 in the dev DB via SQL, then performed the subtotal correction through the real UI. Detail shot now shows corrected-with-strikethrough + pending vendor + cost/latency/token footer. README: demo gif now leads (right under the LIVE_DEMO slot), captions rewritten to match.
- Portfolio: assessed doc-pilot against master-plan §4 — passes 5/6 (only the live deploy missing, blocked on hosting choice). Current GitHub pins (Antheagao): portfolio3, character-rater, graph-search-visual, to-do-app, book-notes-app, Shipping_Container_Project. Recommended swap: unpin `to-do-app`, pin `doc-pilot`. **Profile pins cannot be changed via API — Anthony must do the swap in the GitHub UI.** Also flagged: `book-notes-app` is on the plan's own dilution-tier delete list yet still pinned; natural future swap for `ecommerce-api`.
- Compose stack left running.
- Portfolio swap DONE (Anthony clarified "portfolio" = the `portfolio3` site repo, live at anthonymendezswe.com): cloned to `C:\projects\portfolio3`, replaced the To-Do App card in `src/data/projects.ts` with doc-pilot (repo link, 99.4%/$0.01-per-doc metrics in the blurb), converted `screenshots/demo.gif` → `doc-pilot-demo.webm` (53KB) + `.mp4` (134KB) + poster jpg to match the site's video-pair pattern, removed the todo media, build verified, pushed (`6631444`). GitHub profile *pins* still can't be changed via API — unpinning `to-do-app`/pinning `doc-pilot` there remains a manual UI step.

### 2026-08-02 (session 4 — demo gif)
- Recorded `screenshots/demo.gif` (~880 KB, 15s @ 1.6×) against the live compose stack: home/stats strip → upload `samples/grocery-receipt-skewed.png` → "extracting…" spinner → extracted fields with confidence badges (95–98%) → home with updated stats. Playwright headless recording → ffmpeg palette gif. One live extraction: $0.0114 / 4.8s. All fields cleared the review threshold, so the review queue doesn't appear in this take. Gif is uncommitted (Anthony decides); compose stack left running.

### 2026-08-01 (session 3, continued — invocation 7: S1–S4, Batch 5 complete, deploy deferred)
- Anthony deferred hosting ("everything besides that") — deploy moved to FUTURE.md as a blocked item; Batch 5 planned by Opus planner around it (S1 images → S2 compose → S3 smoke → S4 README).
- S1 images (`acc7bd0`): editable install keeps BACKEND_DIR=/app (prompts stay findable), non-root, .env double-locked out of layers. Review caught that empty frontend/public/ isn't git-tracked → fresh clones couldn't build; .gitkeep fix.
- S2 compose (`ecd531f`): five services, migrate-gates-everything, shared uploads volume (absolute storage_path strings make identical mounts mandatory), key on worker only. Proven live: upload through compose → extracted @ $0.0094. Review caught worker inheriting the API healthcheck → permanently unhealthy → would hang `up --wait`; disabled for worker+migrate. Note: `docker compose down -v` during verification wiped the dev DB volume — schema restored via alembic, old dev rows gone (dev data, no loss that matters).
- S3 smoke (`3334294`): opt-in live suite, 2 round-trips vs the committed eval corpus, $0.10 hard cap; live run 2 passed @ $0.019. Review = SHIP with docstring-accuracy fixes (repair calls can double billed requests; cap is per-test fail-fast; don't run alongside the dev worker).
- S4 README (`f142563`): compose-first quickstart, "Why these tech choices", cost figures labeled (smoke sample vs eval mean), invisible LIVE_DEMO slot, eval table byte-identical.
- Batch spend ≈ $0.03. Boards: Batch 5 → PREVIOUS.md; PRESENT.md empty; FUTURE.md holds only deploy (blocked on Anthony) + stretch.

### 2026-08-01 (session 3, continued — batch-loop invocation 6: H4–H6, Batch 4 complete)
- H4 refusal status (`56d9def`): ModelRefusalError → terminal "refused" document status + AA-checked chip + honest detail-page copy. First clean SHIP review of the project (independent isinstance flags dodge the subclass ordering trap).
- H5 backoff (`28ebdc4`): 4xx fail fast, retry-after honored as a floor on the queue's equal-jitter backoff; post-review cap RETRY_AFTER_MAX=900s so a rogue proxy header can't park a doc for a day. SDK hierarchy verified against installed anthropic 0.120.2; tests raise real SDK exceptions.
- H6 chunking (`c999c24`): the batch's riskiest task — extract_document restructured into a dispatcher over _extract_one_block; merge rules with earliest/latest tie-breaks and a 0.5 conflict cap; _chunks provenance without a migration; single-call canary passed unmodified. Review SHIP with 4 low findings, all taken (stale 413 message, OverflowError on 10**400 literals, zero-page refusal, chunked-failure test coverage).
- Batch 4 totals: 212 tests (up from 146 at batch start), $0 live spend, mock accuracy byte-stable. Boards: Batch 4 → PREVIOUS.md, PRESENT.md empty, carry-over notes for the Batch 5 planner recorded there.
- Next invocation: promote Batch 5 (ship signals) via planner. Deploy step will BLOCK on Anthony's hosting choice.

### 2026-07-31 (session 3, continued — batch-loop invocation 5: H1–H3)
- H1 `/stats` (`a432573`): two-round-trip SQL rollup (GROUP BY status, percentile_cont p50/p95), review counts reuse review.py's `_PENDING` so the badge and strip can't drift, last eval artifact projected read-only. Fable review caught a real 500: the artifact projection sat outside the try — a valid-JSON artifact with `summary: {}` killed the endpoint. Fixed + pinned.
- H2 stats strip (`e9e0d86`): fetch-once client component above the document list (docs processed, avg $/doc, p50/p95, pending review, eval accuracy + model/prompt). Review caught fractional-ms rendering (percentile_cont interpolates → "1280.0000000000002ms"); Math.round fix applied directly.
- H3 schema repair (`8a5a999`): the flagship failure-handling pattern — model's own tool_use sent back with an is_error tool_result listing violated paths (repair_v1.md, extract_v1 untouched), strictly-fewer-violations acceptance, all repair failures fall back to first response, both calls billed into one result. Review blocker: violation-list truncation corrupted the acceptance rule (50→25-violation repair rejected as "not better") — counting now untruncated, truncation only in prompt text. claude-api skill consulted for the multi-turn shape.
- Known gap noted on the board (not fixed): well-formed `line_items` leaf with a non-list value yields zero violations and silently stores `[]`.
- 172 backend tests green; `--mock` accuracy 0.988571 unchanged across all three tasks; $0 live spend. Next invocation: H4 → H5 → H6.

### 2026-07-31 (session 3 — pull sync + Batch 4 planning)
- `git pull` brought 5 external-session commits: review queue (`c98df60`), harvest flywheel + CI + screenshots (`a78ed5d`), shared-DB test robustness (`8f63cf5`), security hardening (`046ea79`), design system (`5d9a82b`). Verified all three Batch 3 bullets are genuinely covered in the code (needs_review flagging, review UI + audit columns on extracted_fields, harvest.py flywheel).
- Deviations from the original Batch 3 sketch, accepted as-is: audit trail = review columns on `extracted_fields` (not a separate corrections table); harvested labels go into the main `evals/docs|labels` dirs with `source: "human-review"` (not `evals/from_corrections/`).
- Boards synced: Batches 2+3 moved to PREVIOUS.md; Batch 4 promoted into PRESENT.md and expanded by the Opus planner against the real code into H1–H6 (stats endpoint, stats strip, schema-repair reprompt, refusal status, rate-limit backoff, PDF chunking) with an "already implemented — do not redo" list and a live-spend gate (≤$0.15, optional). FUTURE.md trimmed: Batch 5 now excludes the CI/badge/screenshots already shipped.

### 2026-07-31 (session 2, continued — batch-loop invocation 4: E3–E5, Batch 2 complete)
- E3 runner (`05edaa9`): Fable review pre-spend → billed-but-errored API calls now count against the cost cap and surface as total_error_cost_usd (refusals bill real tokens); CLI exits nonzero on zero-scored runs.
- E4 reporting (`359601e`): README eval table is generated from committed artifacts only, marker-region rewrites proven byte-identical outside the block.
- E5 live runs (`ff1a1d1`): **Sonnet 99.4% @ $0.0125/doc, caught_by_review 1.0; Haiku 95.4% @ $0.0052/doc, caught_by_review 0.0.** The Haiku result is the headline: the cheap tier misses 4.6% of fields and flags none of them — quantified justification for Batch 3's review queue. Total eval spend $0.51.
- Note: an E3 subagent deleted evals/results/ during cleanup and tripped a security warning — verified harmless (directory only ever held that task's own mock artifacts; nothing committed or historical).

### 2026-07-30 (session 2, continued — batch-loop invocation 3: T7, E1, E2)
- T7 wire-up (`5445496`): README quickstart verified by literally following it from a fresh venv (26 tests + live extraction $0.0114/5.2s); mermaid architecture diagram; 3 PII-free samples.
- Batch 1 complete → Batch 2 (evals) promoted; Opus planner expanded it into E1–E5 against the real code (plan includes a pre-spend checklist gating the live runs).
- E1 dataset (`5bb357d`): 25 synthetic labeled docs, record-first generation so labels are correct by construction; deterministic regeneration.
- E2 scorer (`ad7d725`): Fable review caught a float-tolerance bug that would have silently distorted the flagship accuracy table (19.99 vs 20.00 scored wrong; fixed via integer-cent comparison) plus label-validation loopholes; 99 tests green, scorer proven to run with Postgres stopped.
- Next: E3 runner/CLI, E4 README injection, E5 live Sonnet-vs-Haiku runs (≈$0.40 total, checklist-gated).

### 2026-07-30 (session 2, continued — batch-loop invocation 2: T4–T6)
- T4 worker (`edcb524`), T5 VLM extraction (`6f23b9a`), T6 viewer (`09eb4d9`) — each implemented by Sonnet, reviewed by Fable, fixes routed back; all pushed.
- Review catches worth noting: worker survived-poisoned-session + orphan-reclaim (proven live by the reviewer), malformed-tool-input coercion so bad leaves can't burn 3 paid retries, non-retryable failure classification (refusal/truncation fail fast), prompt-injection line in extract_v1, self-healing frontend polling.
- GitHub: remote Antheagao/doc-pilot connected, all commits pushed; batch-loop command now pushes after every task commit (Anthony's request).
- Anthony's API key appeared in tracked `.env.example` mid-session — moved to gitignored `backend/.env` before any commit; never entered git history.
- Live cost datapoint: $0.0086 / 3.7s per synthetic receipt on claude-sonnet-5, all fields correct at 0.95–0.99 confidence.
- Uncommitted leftovers for Anthony: `frontend/CLAUDE.md` + `frontend/AGENTS.md` (auto-generated by create-next-app).

### 2026-07-30 (session 2, continued — first batch-loop run)
- Ran the first `/batch-loop` invocation for real: T1–T3 completed via the tiered agents (coder=Sonnet implemented, deep-reviewer=Fable reviewed T3, fixes routed back to the same coder agent).
- Commits: `a0a4099` (scaffold), `d06ea87` (schema), `a3f85c3` (upload/document endpoints).
- Environment decision: doc-pilot Postgres binds host port **5434** — 5432 and 5433 are occupied by workoutDB containers from another project. Encoded in docker-compose.yml, config.py default, and .env.example.
- Review highlight worth remembering: FastAPI/Starlette spools multipart uploads to disk *before* endpoint code runs, so upload size caps must live in ASGI middleware, not the handler.

### 2026-07-30 (session 2)
- Verified the multi-model agent workflow is installed at the user level (`~/.claude/agents/`): planner=Opus, fast-reader=Haiku, coder=Sonnet, deep-reviewer=Fable, plus the `tiered-task` skill. Main session runs on Fable 5 as orchestrator.
- Created the batch loop system: `PRESENT.md` (Stage 1 broken into 7 concrete tasks T1–T7 with verify criteria), `PREVIOUS.md` (completed log), `FUTURE.md` (Batches 2–5 + stretch), `.claude/commands/batch-loop.md` (`/batch-loop` — plan → work N tasks via tiered agents → commit per task → update boards; continuous via `/loop /batch-loop`).
- Nothing committed yet — task/session files are uncommitted pending Anthony's call; `/batch-loop` commits only code.

### 2026-07-30
- Portfolio plan finalized; doc-pilot confirmed as the AI-engineering flagship (new build).
- CLAUDE.md + UPDATES.md created to carry context across sessions. No code yet.
